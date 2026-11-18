# ==============================================================================
# 05_merge.R
# Objetivo: unir la base limpia con las variables de texto, las espaciales y
#           la altura del terreno, y dejar la base lista para los modelos:
#           - área, baños y parqueaderos con la mejor fuente disponible;
#           - variables derivadas (área por habitación, año del anuncio como
#             factor);
#           - rezago espacial del precio: mediana del precio por m2 y del log
#             del precio de los inmuebles de train más cercanos;
#           - imputación de faltantes: cada variable numérica con faltantes
#             recibe una dummy bin_<variable>_na y sus faltantes se reemplazan
#             por la mediana de train. El área y los baños se imputan con la
#             mediana de train por tipo de propiedad y número de habitaciones.
#           Las medianas se calculan solo con train y se aplican igual a test.
#           Además:
#           - se eliminan de train las copias exactas de un anuncio (todas las
#             columnas originales iguales salvo property_id) y se conserva una;
#           - se crea id_grupo_anuncio, que identifica los anuncios de train con
#             la misma descripción y el mismo precio, para que la validación
#             cruzada aleatoria los ponga en el mismo fold;
#           - se elimina num_ambientes, que coincide con el número de
#             habitaciones en el 99,8 % de los anuncios;
#           - se eliminan las variables constantes en train.
# Entradas: 02_Data/Processed/base_limpia.rds
#           02_Data/Processed/text_features.rds
#           02_Data/Processed/spatial_features.rds
#           02_Data/Processed/elevacion.rds
# Salidas:  02_Data/Processed/db_train.rds
#           02_Data/Processed/db_test.rds
# ==============================================================================

source(here::here("01_Scripts", "00_utils.R"))

ruta_proc <- here("02_Data", "Processed")

base      <- readRDS(file.path(ruta_proc, "base_limpia.rds"))
texto     <- readRDS(file.path(ruta_proc, "text_features.rds")) %>%
  select(-muestra, -num_banos_comp)
espacial  <- readRDS(file.path(ruta_proc, "spatial_features.rds"))
elevacion <- readRDS(file.path(ruta_proc, "elevacion.rds"))

db <- base %>%
  left_join(texto, by = "property_id") %>%
  left_join(espacial, by = "property_id") %>%
  left_join(elevacion, by = "property_id")

stopifnot(nrow(db) == nrow(base))

# ------------------------------------------------------------------------------
# 0. Anuncios repetidos en train
# ------------------------------------------------------------------------------
train_raw <- fun_load_raw()$train

columnas_originales <- setdiff(names(train_raw), "property_id")
ids_copias <- train_raw$property_id[duplicated(train_raw[, columnas_originales])]

grupos <- train_raw %>%
  mutate(id_grupo_anuncio = if_else(
    is.na(description),
    paste0("sin_descripcion_", property_id),
    paste0("grupo_", as.integer(factor(paste(description, price))))
  )) %>%
  select(property_id, id_grupo_anuncio)

db <- db %>%
  filter(!property_id %in% ids_copias) %>%
  left_join(grupos, by = "property_id")

message("Copias exactas eliminadas de train: ", length(ids_copias))
message("Anuncios de train que comparten descripción y precio con otro: ",
        db %>% filter(muestra == "train") %>% add_count(id_grupo_anuncio) %>%
          summarise(sum(n > 1)) %>% pull())

# ------------------------------------------------------------------------------
# 1. Variables combinadas y derivadas
# ------------------------------------------------------------------------------
db <- db %>%
  mutate(
    # Baños: columna original; si falta, el número escrito en el texto.
    num_banos      = coalesce(num_banos, num_banos_texto),
    # Habitaciones: columna original; si es cero o falta, el texto.
    num_habitac    = if_else(is.na(num_habitac) | num_habitac == 0,
                             coalesce(num_habitac_texto, num_habitac), num_habitac),
    # Parqueaderos: número del texto o, si no lo hay, 1 cuando el anuncio
    # menciona parqueadero y 0 cuando no.
    num_parqueaderos = num_parq_comp,
    cat_anio       = factor(num_anio),
    cat_fuente_area = factor(cat_fuente_area)
  )

# ------------------------------------------------------------------------------
# 2. Imputación del área y los baños por tipo de propiedad y habitaciones
# ------------------------------------------------------------------------------
medianas_grupo <- db %>%
  filter(muestra == "train") %>%
  mutate(grupo_habitac = pmin(replace_na(num_habitac, 3), 6)) %>%
  group_by(cat_tipo, grupo_habitac) %>%
  summarise(med_area = median(num_area, na.rm = TRUE),
            med_banos = median(num_banos, na.rm = TRUE), .groups = "drop")

db <- db %>%
  mutate(grupo_habitac = pmin(replace_na(num_habitac, 3), 6)) %>%
  left_join(medianas_grupo, by = c("cat_tipo", "grupo_habitac")) %>%
  mutate(
    bin_area_na  = as.integer(is.na(num_area)),
    num_area     = coalesce(num_area, med_area),
    bin_banos_na = as.integer(is.na(num_banos)),
    num_banos    = coalesce(num_banos, med_banos),
    num_area_por_habitac = num_area / pmax(num_habitac, 1)
  ) %>%
  select(-grupo_habitac, -med_area, -med_banos)

# ------------------------------------------------------------------------------
# 2.1 Rezago espacial del precio
# Para cada inmueble (de train y de test) se buscan los inmuebles de train más
# cercanos y se calcula:
# - num_rez_precio_m2_k10 y num_rez_precio_m2_k30: mediana del precio por m2
#   de los 10 y 30 vecinos más cercanos, usando solo vecinos con área
#   observada (sin imputar);
# - num_rez_log_precio_k10: mediana del log del precio de los 10 vecinos más
#   cercanos;
# - num_rez_dist_k10: distancia en metros al décimo vecino, que indica qué
#   tan densa es la información de precios alrededor del inmueble.
# En train se excluyen como vecinos el propio inmueble y los anuncios de su
# mismo grupo (misma descripción y precio), para que la variable no contenga
# el precio del inmueble. Los inmuebles de test usan como vecinos a todos los
# de train.
# ------------------------------------------------------------------------------

# Índices (sobre `ref`) de los `k` vecinos más cercanos de cada fila de
# `consulta`, sin contar los vecinos del mismo grupo de anuncio. Se piden
# `k_busqueda` candidatos para tener margen después de excluir.
fun_vecinos <- function(consulta, ref, k, k_busqueda = 80) {
  cercanos <- kNN(as.matrix(ref[, c("num_x", "num_y")]), k = min(k_busqueda, nrow(ref) - 1),
                  query = as.matrix(consulta[, c("num_x", "num_y")]))
  grupo_ref <- ref$id_grupo_anuncio
  map(seq_len(nrow(consulta)), function(i) {
    ids <- cercanos$id[i, ]
    dist <- cercanos$dist[i, ]
    validos <- is.na(consulta$id_grupo_anuncio[i]) | grupo_ref[ids] != consulta$id_grupo_anuncio[i]
    list(id = head(ids[validos], k), dist = head(dist[validos], k))
  })
}

ref_todos <- db %>% filter(muestra == "train")
ref_area  <- ref_todos %>% filter(bin_area_na == 0)

vecinos_todos <- fun_vecinos(db, ref_todos, k = 10)
vecinos_area  <- fun_vecinos(db, ref_area,  k = 30)

precio_m2_ref <- ref_area$obj_price / ref_area$num_area

db <- db %>%
  mutate(
    num_rez_precio_m2_k10  = map_dbl(vecinos_area, ~ median(precio_m2_ref[head(.x$id, 10)])),
    num_rez_precio_m2_k30  = map_dbl(vecinos_area, ~ median(precio_m2_ref[.x$id])),
    num_rez_log_precio_k10 = map_dbl(vecinos_todos, ~ median(ref_todos$obj_log_price[.x$id])),
    num_rez_dist_k10       = map_dbl(vecinos_todos, ~ max(.x$dist))
  )

db %>%
  filter(muestra == "train") %>%
  summarise(cor_log_precio_m2_vs_rezago_m2 = cor(log(obj_price / num_area),
                                                 log(num_rez_precio_m2_k10)),
            cor_log_precio_vs_rezago_log_precio = cor(obj_log_price, num_rez_log_precio_k10)) %>%
  print()

# ------------------------------------------------------------------------------
# 3. Imputación del resto de variables numéricas con la mediana de train
# Las variables objetivo y las coordenadas no se imputan.
# ------------------------------------------------------------------------------
no_imputar <- c("obj_price", "obj_log_price", "num_lat", "num_lon", "num_x", "num_y")

numericas_na <- db %>%
  select(where(is.numeric), -any_of(no_imputar)) %>%
  select(where(~ any(is.na(.x)))) %>%
  names()

medianas_train <- db %>%
  filter(muestra == "train") %>%
  summarise(across(all_of(numericas_na), ~ median(.x, na.rm = TRUE)))

for (v in numericas_na) {
  db[[paste0("bin_", str_remove(v, "^num_"), "_na")]] <- as.integer(is.na(db[[v]]))
  db[[v]] <- replace_na(db[[v]], medianas_train[[v]])
}

# Variables categóricas sin dato.
db <- db %>%
  mutate(across(starts_with("cat_") & where(is.character),
                ~ replace_na(.x, "Sin dato")))

# Las variables que toman un solo valor en train no aportan información y se
# eliminan.
constantes <- db %>%
  filter(muestra == "train") %>%
  select(-property_id, -muestra) %>%
  select(where(~ n_distinct(.x) == 1)) %>%
  names()
message("Variables constantes eliminadas: ",
        if (length(constantes) == 0) "ninguna" else paste(constantes, collapse = ", "))
db <- db %>% select(-all_of(constantes))

# ------------------------------------------------------------------------------
# 4. Bases de salida
# ------------------------------------------------------------------------------
db <- db %>% select(-title, -description, -num_ambientes)

db_train <- db %>% filter(muestra == "train") %>% select(-muestra)
db_test  <- db %>% filter(muestra == "test")  %>%
  select(-muestra, -obj_price, -obj_log_price, -id_grupo_anuncio)

saveRDS(db_train, file.path(ruta_proc, "db_train.rds"))
saveRDS(db_test,  file.path(ruta_proc, "db_test.rds"))

# ------------------------------------------------------------------------------
# 5. Resumen en consola
# ------------------------------------------------------------------------------
message("db_train: ", nrow(db_train), " filas y ", ncol(db_train), " columnas")
message("db_test:  ", nrow(db_test), " filas y ", ncol(db_test), " columnas")

# Número de variables por prefijo.
tibble(variable = names(db_train)) %>%
  mutate(prefijo = str_extract(variable, "^[a-z]+_")) %>%
  count(prefijo) %>%
  print()

# Variables con faltantes después de la imputación, sin contar las variables
# objetivo e id_grupo_anuncio, que solo existen en train.
db %>%
  summarise(across(everything(), ~ sum(is.na(.x)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "faltantes") %>%
  filter(faltantes > 0,
         !variable %in% c("obj_price", "obj_log_price", "id_grupo_anuncio")) %>%
  print(n = Inf)
