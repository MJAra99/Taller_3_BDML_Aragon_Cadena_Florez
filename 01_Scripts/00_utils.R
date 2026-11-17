# ==============================================================================
# 00_utils.R
# Objetivo: funciones comunes del taller: carga y limpieza de datos, folds,
#           MAE, envíos a Kaggle y cálculos espaciales (lectura de capas,
#           distancias, conteos en un radio y cruces con polígonos).
# Entradas: ninguna (se carga con source() desde los demás scripts).
# Salidas:  funciones fun_* en el ambiente.
# ==============================================================================

pacman::p_load(tidyverse, here, caret, sf, dbscan)

semilla <- 2026

# Sistema de coordenadas proyectado en metros para Bogotá:
# MAGNA-SIRGAS 2018 / Origen-Nacional.
crs_metros <- 9377

# ------------------------------------------------------------------------------
# Lectura de los datos crudos de Kaggle
# ------------------------------------------------------------------------------
fun_load_raw <- function() {
  ruta <- here("02_Data", "Raw", "Kaggle")
  list(
    train = read_csv(file.path(ruta, "train.csv"), show_col_types = FALSE),
    test  = read_csv(file.path(ruta, "test.csv"),  show_col_types = FALSE)
  )
}

# ------------------------------------------------------------------------------
# Limpieza de la base mínima
# `ref` es la base de entrenamiento: las medianas para imputar salen siempre
# de train, también cuando se limpia test, para no usar información de test.
# ------------------------------------------------------------------------------
fun_clean_base <- function(df, ref) {
  mediana_banos <- median(ref$bathrooms, na.rm = TRUE)

  df %>%
    mutate(
      # Variable objetivo en log (solo existe en train).
      obj_log_price  = if ("price" %in% names(.)) log(price) else NA_real_,
      # Baños: dummy de faltante e imputación con la mediana de train.
      bin_banos_na   = as.integer(is.na(bathrooms)),
      num_banos      = if_else(is.na(bathrooms), mediana_banos, bathrooms),
      num_habitac    = bedrooms,
      cat_tipo       = factor(property_type, levels = c("Apartamento", "Casa")),
      num_lat        = lat,
      num_lon        = lon,
      num_anio       = year,
      num_mes        = month
    ) %>%
    select(property_id, obj_log_price, bin_banos_na, num_banos, num_habitac,
           cat_tipo, num_lat, num_lon, num_anio, num_mes)
}

# ------------------------------------------------------------------------------
# Folds de validación cruzada aleatoria (índices de entrenamiento para caret)
# ------------------------------------------------------------------------------
fun_folds <- function(y, k = 5, seed = semilla) {
  set.seed(seed)
  createFolds(y, k = k, returnTrain = TRUE)
}

# ------------------------------------------------------------------------------
# MAE en pesos a partir de predicciones en log
# ------------------------------------------------------------------------------
fun_mae_cop <- function(log_obs, log_pred) {
  mean(abs(exp(log_obs) - exp(log_pred)))
}

# MAE en pesos fuera de muestra de un modelo de caret
# (requiere savePredictions = "final" en trainControl).
fun_mae_cv <- function(modelo) {
  fun_mae_cop(modelo$pred$obs, modelo$pred$pred)
}

# ------------------------------------------------------------------------------
# Envío a Kaggle: columnas property_id y price (en pesos)
# ------------------------------------------------------------------------------
fun_submission <- function(ids, log_pred, nombre) {
  ruta <- here("03_Output", "Submissions", paste0(nombre, ".csv"))
  tibble(property_id = ids, price = exp(log_pred)) %>%
    write_csv(ruta)
  message("Envío guardado en: ", ruta)
  invisible(ruta)
}

# ------------------------------------------------------------------------------
# Funciones espaciales
# Todas trabajan en EPSG:9377, de modo que distancias y radios están en metros.
# ------------------------------------------------------------------------------

# Lee una capa descargada en 02_Data/Raw/MapasBogota/capas a partir de su id.
# Acepta GeoJSON, CSV y archivos .zip con un GeoJSON o un Shapefile adentro.
# Un archivo con extensión .geojson que en realidad es un .zip se detecta por
# sus primeros bytes. Las capas con geometría se devuelven en EPSG:9377 y con
# geometrías válidas; las CSV se devuelven como tabla con todas las columnas
# en texto.
fun_leer_capa <- function(id) {
  ruta <- here("02_Data", "Raw", "MapasBogota", "capas")
  archivo <- list.files(ruta, pattern = paste0("^", id, "\\.(geojson|csv|zip)$"),
                        full.names = TRUE)
  if (length(archivo) == 0) stop("No se encontró la capa ", id, " en ", ruta)
  archivo <- archivo[1]

  if (str_detect(archivo, "\\.csv$")) {
    return(read_csv(archivo, col_types = cols(.default = col_character())))
  }

  es_zip <- identical(readBin(archivo, "raw", n = 2), as.raw(c(0x50, 0x4b)))
  if (es_zip) {
    carpeta <- file.path(tempdir(), id)
    unzip(archivo, exdir = carpeta)
    internos <- list.files(carpeta, pattern = "\\.(geojson|json|shp)$",
                           recursive = TRUE, full.names = TRUE)
    if (length(internos) == 0) stop("El .zip de ", id, " no tiene GeoJSON ni Shapefile")
    archivo <- internos[1]
  }

  capa <- read_sf(archivo, quiet = TRUE)
  if (is.na(st_crs(capa))) st_crs(capa) <- 4326
  capa <- capa %>%
    st_transform(crs_metros) %>%
    st_make_valid() %>%
    filter(!st_is_empty(geometry))
  capa
}

# Año de un campo de fecha de ArcGIS. Las fechas llegan como milisegundos
# desde 1970 (número o texto) o como texto con el año al inicio.
fun_anio <- function(x) {
  x_num <- suppressWarnings(as.numeric(x))
  if_else(!is.na(x_num) & x_num > 1e11,
          as.integer(format(as.POSIXct(x_num / 1000, origin = "1970-01-01",
                                       tz = "UTC"), "%Y")),
          as.integer(str_extract(as.character(x), "\\d{4}")))
}

# Coordenadas (x, y) de una capa. Para líneas y polígonos se usa un punto
# sobre la superficie de cada geometría.
fun_coords <- function(capa) {
  geometria <- st_geometry(capa)
  if (!all(st_geometry_type(geometria) %in% c("POINT"))) {
    geometria <- st_point_on_surface(st_zm(geometria))
  }
  st_coordinates(geometria)[, 1:2, drop = FALSE]
}

# Número de elementos de una capa a menos de `radio` metros de cada punto.
fun_conteo_radio <- function(puntos, capa, radio) {
  vecinos <- frNN(x = fun_coords(capa), eps = radio,
                  query = st_coordinates(puntos), sort = FALSE)
  lengths(vecinos$id)
}

# Distancia en metros de cada punto al elemento más cercano de una capa.
# Para capas de puntos usa un árbol de búsqueda; para líneas y polígonos
# mide hasta el borde de la geometría (cero si el punto está adentro).
fun_dist_min <- function(puntos, capa) {
  if (all(st_geometry_type(capa) == "POINT")) {
    return(as.numeric(kNN(x = fun_coords(capa), k = 1,
                          query = st_coordinates(puntos))$dist[, 1]))
  }
  cercano <- st_nearest_feature(puntos, capa)
  as.numeric(st_distance(puntos, capa[cercano, ], by_element = TRUE))
}

# Resumen (`fun`, por defecto la mediana) de un valor de los elementos a
# menos de `radio` metros de cada punto. Si no hay elementos en el radio, se
# usa el valor del elemento más cercano.
fun_resumen_radio <- function(puntos, coords_ref, valores, radio, fun = median) {
  ok <- !is.na(valores)
  coords_ref <- coords_ref[ok, , drop = FALSE]
  valores <- valores[ok]
  consulta <- st_coordinates(puntos)
  vecinos <- frNN(x = coords_ref, eps = radio, query = consulta, sort = FALSE)$id
  cercano <- kNN(x = coords_ref, k = 1, query = consulta)$id[, 1]
  map2_dbl(vecinos, cercano, function(v, c) {
    if (length(v) == 0) valores[c] else fun(valores[v])
  })
}

# Proporción de elementos a menos de `radio` metros que cumplen una condición
# (vector lógico `condicion`). Devuelve NA si no hay elementos en el radio.
fun_proporcion_radio <- function(puntos, coords_ref, condicion, radio) {
  vecinos <- frNN(x = coords_ref, eps = radio, query = st_coordinates(puntos),
                  sort = FALSE)$id
  map_dbl(vecinos, function(v) if (length(v) == 0) NA_real_ else mean(condicion[v]))
}

# Atributos del polígono que contiene a cada punto. Los puntos que no caen en
# ningún polígono (por ejemplo, en el borde de la ciudad) toman los atributos
# del polígono más cercano. Devuelve una tabla con una fila por punto.
fun_poligono <- function(puntos, capa, campos) {
  capa <- capa %>% select(all_of(campos))
  dentro <- st_intersects(puntos, capa)
  indice <- map_int(dentro, ~ if (length(.x) == 0) NA_integer_ else .x[1])
  faltan <- which(is.na(indice))
  if (length(faltan) > 0) {
    indice[faltan] <- st_nearest_feature(puntos[faltan, ], capa)
  }
  capa %>% st_drop_geometry() %>% slice(indice)
}

# Moda de un vector (el valor más frecuente), sin contar los NA.
fun_moda <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA)
  names(sort(table(x), decreasing = TRUE))[1]
}
