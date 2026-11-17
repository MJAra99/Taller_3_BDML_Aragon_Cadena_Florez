# ==============================================================================
# 02_descarga_capas.R
# Objetivo: descargar las capas geográficas de Bogotá listadas en
#           capas_seleccionadas.csv y guardar cada una como un archivo GeoJSON
#           en coordenadas WGS84 (EPSG:4326).
#           Hay tres tipos de fuente:
#           - arcgis: capa de un servidor ArcGIS (Mapas Bogotá o Jardín
#             Botánico). Se descarga por bloques de registros. Cuando la
#             columna `recorte` es TRUE, solo se descargan los registros dentro
#             del rectángulo que contiene las propiedades de train y test.
#             Cuando las columnas `campo_anio` y `anios` tienen valor, solo se
#             descargan los registros de esos años (las capas de catastro
#             publican una versión por año). Cuando la columna `geometria` es
#             FALSE, solo se descargan los atributos y se guardan en CSV; se
#             usa en las capas de catastro por manzana, cuya ubicación se toma
#             de 01_valor_referencia con el código de manzana (MANCODIGO). Los
#             bloques que llegan incompletos se dividen en bloques más
#             pequeños. Cuando la columna `simplificar_m` tiene valor, el
#             servidor simplifica las líneas y polígonos con esa tolerancia en
#             metros antes de enviarlos; se usa en las curvas de nivel, cuyas
#             líneas tienen demasiados vértices.
#           - ckan: conjunto de datos de Datos Abiertos Bogotá. Se consulta la
#             lista de recursos del conjunto y se prueban, en este orden, los
#             servicios ArcGIS, el GeoJSON, el Shapefile y el CSV, hasta que
#             uno se descarga.
#           - url: archivo publicado en una dirección fija.
#           Las capas que ya tienen archivo en la carpeta de salida no se
#           vuelven a descargar.
# Entradas: 02_Data/Raw/MapasBogota/capas_seleccionadas.csv
#           02_Data/Raw/Kaggle/{train,test}.csv (coordenadas para el recorte)
# Salidas:  02_Data/Raw/MapasBogota/capas/<id>.geojson (o .csv / .zip)
#           02_Data/Raw/MapasBogota/capas/registro_descarga.csv
# ==============================================================================

pacman::p_load(tidyverse, here, httr, jsonlite)

ruta_capas  <- here("02_Data", "Raw", "MapasBogota", "capas_seleccionadas.csv")
ruta_salida <- here("02_Data", "Raw", "MapasBogota", "capas")
dir.create(ruta_salida, recursive = TRUE, showWarnings = FALSE)

url_ckan <- "https://datosabiertos.bogota.gov.co/api/3/action/package_show"

capas <- read_csv(ruta_capas, show_col_types = FALSE,
                  col_types = cols(.default = col_character(),
                                   geometria = col_logical(),
                                   simplificar_m = col_double(),
                                   recorte = col_logical(),
                                   descargar = col_logical())) %>%
  filter(descargar)

# ------------------------------------------------------------------------------
# Rectángulo de recorte: extremos de las coordenadas de train y test,
# ampliados en 0.02 grados (unos 2 km) para incluir lo que rodea a las
# propiedades del borde.
# ------------------------------------------------------------------------------
coords <- bind_rows(
  read_csv(here("02_Data", "Raw", "Kaggle", "train.csv"),
           col_select = c(lat, lon), show_col_types = FALSE),
  read_csv(here("02_Data", "Raw", "Kaggle", "test.csv"),
           col_select = c(lat, lon), show_col_types = FALSE)
)
margen <- 0.02
rectangulo <- paste(min(coords$lon) - margen, min(coords$lat) - margen,
                    max(coords$lon) + margen, max(coords$lat) + margen,
                    sep = ",")

# ------------------------------------------------------------------------------
# Funciones auxiliares
# ------------------------------------------------------------------------------

# Hace una consulta GET o POST y devuelve el texto de la respuesta. Reintenta
# hasta tres veces con una pausa creciente si el servidor no responde. Si el
# servidor ArcGIS responde con un mensaje de error en el JSON, se detiene con
# ese mensaje.
fun_pedir <- function(url, parametros, metodo = "GET") {
  for (intento in 1:3) {
    Sys.sleep(0.2 * intento)
    respuesta <- tryCatch(
      if (metodo == "GET") {
        GET(url, query = parametros, timeout(120))
      } else {
        POST(url, body = parametros, encode = "form", timeout(300))
      },
      error = function(e) NULL
    )
    if (!is.null(respuesta) && status_code(respuesta) == 200) {
      texto <- content(respuesta, as = "text", encoding = "UTF-8")
      if (str_detect(str_sub(texto, 1, 200), '^\\s*\\{\\s*"error"')) {
        detalle <- tryCatch(fromJSON(texto)$error$message, error = function(e) texto)
        stop("El servidor respondió con un error: ", detalle)
      }
      return(texto)
    }
  }
  stop("Sin respuesta después de tres intentos: ", url)
}

# Construye la condición de años de una capa de ArcGIS. Primero consulta los
# valores distintos del campo del año y conserva los que contienen alguno de
# los años pedidos, escritos tal como los guarda el servidor (número, texto o
# fecha). Si el servidor no entrega los valores distintos, arma la condición
# según el tipo del campo.
fun_condicion_anios <- function(url_capa, info, campo_anio, anios) {
  anios_pedidos <- str_trim(str_split(anios, ",")[[1]])
  tipo_campo <- info$fields$type[info$fields$name == campo_anio]

  if (length(tipo_campo) > 0 && tipo_campo == "esriFieldTypeDate") {
    return(paste0(campo_anio, " >= DATE '", min(anios_pedidos), "-01-01' AND ",
                  campo_anio, " < DATE '", as.integer(max(anios_pedidos)) + 1,
                  "-01-01'"))
  }

  distintos <- tryCatch({
    json <- fromJSON(fun_pedir(file.path(url_capa, "query"),
                               list(where = "1=1", outFields = campo_anio,
                                    returnDistinctValues = "true",
                                    returnGeometry = "false", f = "json")),
                     simplifyVector = FALSE)
    map(json$features, ~ .x$attributes[[campo_anio]])
  }, error = function(e) list())

  if (length(distintos) > 0) {
    message("  valores del campo ", campo_anio, ": ",
            paste(sort(unlist(distintos)), collapse = ", "))
    elegidos <- keep(distintos, ~ any(str_detect(as.character(.x), anios_pedidos)))
    if (length(elegidos) == 0) {
      stop("El campo ", campo_anio, " no tiene los años ", anios)
    }
    valores <- map_chr(elegidos, function(v) {
      if (is.character(v)) paste0("'", v, "'") else format(v, scientific = FALSE)
    })
    return(paste0(campo_anio, " IN (", paste(valores, collapse = ","), ")"))
  }

  if (length(tipo_campo) > 0 && tipo_campo == "esriFieldTypeString") {
    anios_pedidos <- paste0("'", anios_pedidos, "'")
  }
  paste0(campo_anio, " IN (", paste(anios_pedidos, collapse = ","), ")")
}

# Pide los registros de un bloque de identificadores en GeoJSON. Si la
# respuesta llega incompleta (pasa con líneas o polígonos con muchos
# vértices), divide el bloque en dos y pide cada mitad por separado.
fun_pedir_bloque <- function(url_capa, ids_bloque, campos_salida, tolerancia = NA) {
  parametros <- list(objectIds = paste(ids_bloque, collapse = ","),
                     outFields = campos_salida,
                     returnGeometry = "true",
                     outSR = "4326",
                     f = "geojson")
  # La tolerancia va en grados porque la salida está en EPSG:4326
  # (1 grado de latitud mide unos 111.000 m).
  if (!is.na(tolerancia)) parametros$maxAllowableOffset <- tolerancia / 111000
  registros <- tryCatch({
    texto <- fun_pedir(file.path(url_capa, "query"), parametros, metodo = "POST")
    fromJSON(texto, simplifyVector = FALSE)$features
  }, error = function(e) e)

  if (!inherits(registros, "error")) return(registros)
  if (length(ids_bloque) == 1) stop(conditionMessage(registros))
  mitad <- ceiling(length(ids_bloque) / 2)
  c(fun_pedir_bloque(url_capa, ids_bloque[1:mitad], campos_salida, tolerancia),
    fun_pedir_bloque(url_capa, ids_bloque[(mitad + 1):length(ids_bloque)],
                     campos_salida, tolerancia))
}

# Descarga una capa de ArcGIS y la escribe como GeoJSON.
# 1. Lee la descripción de la capa: nombre del campo identificador y número
#    máximo de registros que entrega el servidor por consulta.
# 2. Pide la lista de identificadores de los registros (dentro del rectángulo
#    si `recorte` es TRUE y de los años pedidos si `campo_anio` tiene valor).
# 3. Pide los registros por bloques de identificadores y los escribe uno tras
#    otro en el archivo, sin cargar la capa completa en memoria.
fun_descargar_arcgis <- function(url_capa, archivo, campos = "*", recorte = FALSE,
                                 campo_anio = NA, anios = NA, geometria = TRUE,
                                 simplificar_m = NA) {
  info <- fromJSON(fun_pedir(url_capa, list(f = "pjson")), simplifyVector = TRUE)
  campo_id  <- info$objectIdField %||%
    info$fields$name[info$fields$type == "esriFieldTypeOID"][1]
  tam_bloque <- min(info$maxRecordCount %||% 1000, 2000)

  # "oid" significa que solo se conserva el identificador de cada registro.
  campos_salida <- if (campos == "oid") campo_id else campos

  condicion <- "1=1"
  if (!is.na(campo_anio) && !is.na(anios)) {
    condicion <- fun_condicion_anios(url_capa, info, campo_anio, anios)
    message("  filtro: ", condicion)
  }

  filtro <- list(where = condicion, f = "json", returnIdsOnly = "true")
  if (recorte) {
    filtro <- c(filtro, list(geometry = rectangulo,
                             geometryType = "esriGeometryEnvelope",
                             inSR = "4326",
                             spatialRel = "esriSpatialRelIntersects"))
  }
  ids <- fromJSON(fun_pedir(file.path(url_capa, "query"), filtro,
                            metodo = "POST"))$objectIds
  ids <- sort(ids)
  if (length(ids) == 0) stop("La capa no tiene registros en el área: ", url_capa)

  bloques <- split(ids, ceiling(seq_along(ids) / tam_bloque))
  message("  ", length(ids), " registros en ", length(bloques), " bloques")

  # Sin geometría: se piden solo los atributos de cada bloque y se guardan
  # todos juntos en un CSV.
  if (!geometria) {
    atributos <- map_dfr(seq_along(bloques), function(i) {
      texto <- fun_pedir(file.path(url_capa, "query"),
                         list(objectIds = paste(bloques[[i]], collapse = ","),
                              outFields = campos_salida,
                              returnGeometry = "false",
                              f = "json"),
                         metodo = "POST")
      if (i %% 10 == 0) message("  bloque ", i, " de ", length(bloques))
      as_tibble(fromJSON(texto, simplifyVector = TRUE)$features$attributes) %>%
        mutate(across(everything(), as.character))
    })
    write_csv(atributos, archivo)
    return(length(ids))
  }

  # Escritura del GeoJSON por partes: encabezado, registros de cada bloque
  # separados por comas, y cierre. La conexión al archivo se cierra aunque la
  # descarga se detenga por un error.
  archivo_tmp <- paste0(archivo, ".tmp")
  con <- file(archivo_tmp, open = "w", encoding = "UTF-8")
  on.exit(try(close(con), silent = TRUE), add = TRUE)
  writeLines('{"type":"FeatureCollection","features":[', con, sep = "")
  primero <- TRUE
  for (i in seq_along(bloques)) {
    registros <- fun_pedir_bloque(url_capa, bloques[[i]], campos_salida,
                                  simplificar_m)
    if (length(registros) > 0) {
      json_bloque <- toJSON(registros, auto_unbox = TRUE, digits = NA,
                            null = "null", na = "null")
      # Se quitan los corchetes de la lista para unir los bloques.
      json_bloque <- substr(json_bloque, 2, nchar(json_bloque) - 1)
      writeLines(if (primero) json_bloque else paste0(",", json_bloque),
                 con, sep = "")
      primero <- FALSE
    }
    if (i %% 10 == 0) message("  bloque ", i, " de ", length(bloques))
  }
  writeLines("]}", con, sep = "")
  close(con)
  file.rename(archivo_tmp, archivo)
  length(ids)
}

# Descarga un conjunto de Datos Abiertos Bogotá. Prueba sus recursos en este
# orden: servicios ArcGIS de una capa, GeoJSON, Shapefile y CSV. Si un recurso
# falla, sigue con el siguiente.
fun_descargar_ckan <- function(id_conjunto, archivo_base, campos, recorte) {
  recursos <- fromJSON(fun_pedir(url_ckan, list(id = id_conjunto)),
                       simplifyVector = TRUE)$result$resources %>%
    as_tibble() %>%
    mutate(formato = str_to_upper(format))

  urls_rest <- recursos$url[str_detect(recursos$url, "/(MapServer|FeatureServer)/\\d+/?$")]
  for (url_rest in unique(str_remove(urls_rest, "/$"))) {
    message("  servicio ArcGIS: ", url_rest)
    n <- tryCatch(fun_descargar_arcgis(url_rest, paste0(archivo_base, ".geojson"),
                                       campos, recorte),
                  error = function(e) {
                    message("  no se pudo usar el servicio: ", conditionMessage(e))
                    NULL
                  })
    if (!is.null(n)) return(n)
  }

  extensiones <- c(GEOJSON = ".geojson", SHP = ".zip", CSV = ".csv")
  for (formato_buscado in names(extensiones)) {
    candidatos <- recursos %>% filter(formato == formato_buscado)
    for (url_recurso in candidatos$url) {
      # Algunos recursos GeoJSON se publican comprimidos en .zip; el archivo
      # conserva la extensión de la dirección de descarga.
      extension <- if (str_detect(url_recurso, "\\.zip$")) ".zip" else extensiones[[formato_buscado]]
      archivo <- paste0(archivo_base, extension)
      message("  ", formato_buscado, ": ", url_recurso)
      ok <- tryCatch({
        download.file(url_recurso, archivo, mode = "wb", quiet = TRUE)
        file.exists(archivo) && file.size(archivo) > 0
      }, error = function(e) {
        message("  no se pudo descargar: ", conditionMessage(e))
        FALSE
      })
      if (ok) return(NA_integer_)
      if (file.exists(archivo)) file.remove(archivo)
    }
  }
  stop("Ningún recurso del conjunto se pudo descargar: ", id_conjunto)
}

# ------------------------------------------------------------------------------
# Descarga de todas las capas
# ------------------------------------------------------------------------------
options(timeout = 1800)

registro <- pmap_dfr(capas, function(id, grupo, capa, fuente, origen, campos,
                                     geometria, simplificar_m, recorte,
                                     campo_anio, anios,
                                     descargar, variable) {
  archivo_base <- file.path(ruta_salida, id)
  existente <- list.files(ruta_salida, pattern = paste0("^", id, "\\.(geojson|zip|csv)$"))
  if (length(existente) > 0) {
    message("Ya descargada: ", id)
    return(tibble(id = id, estado = "ya descargada", registros = NA_integer_,
                  archivo = existente[1], fecha = as.character(Sys.time())))
  }

  message("Descargando ", id, " (", capa, ")")
  resultado <- tryCatch({
    n <- switch(
      fuente,
      arcgis = fun_descargar_arcgis(origen,
                                    paste0(archivo_base,
                                           if (geometria) ".geojson" else ".csv"),
                                    campos, recorte, campo_anio, anios,
                                    geometria, simplificar_m),
      ckan   = fun_descargar_ckan(origen, archivo_base, campos, recorte),
      url    = {
        download.file(origen, paste0(archivo_base, ".geojson"),
                      mode = "wb", quiet = TRUE)
        NA_integer_
      }
    )
    tibble(estado = "descargada", registros = as.integer(n))
  }, error = function(e) {
    message("  Error: ", conditionMessage(e))
    tibble(estado = paste("error:", conditionMessage(e)), registros = NA_integer_)
  })

  archivo <- list.files(ruta_salida, pattern = paste0("^", id, "\\.(geojson|zip|csv)$"))
  resultado %>%
    mutate(id = id,
           archivo = if (length(archivo) > 0) archivo[1] else NA_character_,
           fecha = as.character(Sys.time()),
           .before = 1)
})

write_csv(registro, file.path(ruta_salida, "registro_descarga.csv"))

# ------------------------------------------------------------------------------
# Resumen en consola
# ------------------------------------------------------------------------------
registro %>% count(estado = str_extract(estado, "^[a-z ]+")) %>% print()
registro %>% filter(str_detect(estado, "^error")) %>% select(id, estado) %>%
  print(n = Inf)
