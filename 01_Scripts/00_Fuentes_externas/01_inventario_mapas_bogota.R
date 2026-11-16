# ==============================================================================
# 01_inventario_mapas_bogota.R
# Objetivo: construir el inventario de la información geográfica disponible
#           para Bogotá en dos catálogos públicos:
#           (1) el servidor ArcGIS de Catastro Bogotá, que publica las capas
#               de Mapas Bogotá (una fila por capa, con su geometría y campos);
#           (2) el portal Datos Abiertos Bogotá (una fila por conjunto de
#               datos, con la entidad que lo publica y sus formatos).
#           A cada capa se le asigna un nivel de detalle (manzana, lote, UPZ,
#           localidad, etc.) y los años que aparecen en su nombre o descripción.
# Entradas: consultas a
#           https://serviciosgis.catastrobogota.gov.co/arcgis/rest/services
#           https://datosabiertos.bogota.gov.co/api/3/action/package_search
# Salidas:  02_Data/Raw/MapasBogota/inventario_capas_arcgis.csv
#           02_Data/Raw/MapasBogota/inventario_datos_abiertos.csv
# ==============================================================================

pacman::p_load(tidyverse, here, httr, jsonlite)

ruta_salida <- here("02_Data", "Raw", "MapasBogota")
dir.create(ruta_salida, recursive = TRUE, showWarnings = FALSE)

url_arcgis <- "https://serviciosgis.catastrobogota.gov.co/arcgis/rest/services"
url_ckan   <- "https://datosabiertos.bogota.gov.co/api/3/action/package_search"

# ------------------------------------------------------------------------------
# Funciones auxiliares
# ------------------------------------------------------------------------------

# Consulta una URL y devuelve el JSON como lista. Si la consulta falla,
# devuelve NULL y escribe un mensaje con la URL, de modo que una capa con
# error no detiene el inventario.
fun_get_json <- function(url, query = list(f = "pjson")) {
  Sys.sleep(0.2)  # pausa entre consultas para no saturar el servidor
  respuesta <- tryCatch(
    GET(url, query = query, timeout(60)),
    error = function(e) NULL
  )
  if (is.null(respuesta) || status_code(respuesta) != 200) {
    message("Sin respuesta: ", url)
    return(NULL)
  }
  tryCatch(
    fromJSON(content(respuesta, as = "text", encoding = "UTF-8"),
             simplifyVector = FALSE),
    error = function(e) {
      message("JSON no válido: ", url)
      NULL
    }
  )
}

# Quita etiquetas HTML y espacios repetidos de las descripciones.
fun_limpiar_texto <- function(x) {
  x <- if_else(is.na(x), "", x)
  x %>%
    str_replace_all("<[^>]+>", " ") %>%
    str_replace_all("&nbsp;", " ") %>%
    str_squish()
}

# Convierte un elemento del JSON a texto; devuelve NA si el elemento no existe.
fun_valor <- function(x) if (is.null(x)) NA_character_ else as.character(x)

# Asigna el nivel de detalle según las palabras del nombre y la descripción.
# El orden de las condiciones va del nivel más fino al más agregado.
fun_nivel <- function(texto) {
  t <- str_to_lower(texto)
  case_when(
    str_detect(t, "\\bpredio|\\blote")                  ~ "predio o lote",
    str_detect(t, "manzana")                            ~ "manzana",
    str_detect(t, "sector catastral|barrio")            ~ "sector catastral o barrio",
    str_detect(t, "\\bupz\\b|planeamiento zonal")       ~ "UPZ",
    str_detect(t, "\\bupl\\b|planeamiento local")       ~ "UPL",
    str_detect(t, "localidad")                          ~ "localidad",
    TRUE                                                ~ "sin identificar"
  )
}

# Extrae los años (1990 a 2029) mencionados en un texto.
fun_anios <- function(texto) {
  map_chr(str_extract_all(texto, "\\b(199\\d|20[0-2]\\d)\\b"),
          ~ paste(sort(unique(.x)), collapse = ", "))
}

# ------------------------------------------------------------------------------
# 1. Catálogo ArcGIS de Catastro Bogotá
# ------------------------------------------------------------------------------

# 1.1 Lista de servicios: los de la raíz y los de cada carpeta temática.
raiz     <- fun_get_json(url_arcgis)
carpetas <- map_chr(raiz$folders, as.character)

fun_servicios <- function(json) {
  map_dfr(json$services, ~ tibble(servicio = .x$name, tipo = .x$type))
}

servicios <- bind_rows(
  fun_servicios(raiz),
  map_dfr(carpetas, ~ fun_servicios(fun_get_json(file.path(url_arcgis, .x))))
) %>%
  distinct() %>%
  mutate(carpeta = if_else(str_detect(servicio, "/"),
                           str_extract(servicio, "^[^/]+"), "raiz"))

message("Servicios encontrados: ", nrow(servicios))

# 1.2 Capas de cada servicio. Los servicios MapServer y FeatureServer
# describen todas sus capas (con geometría y campos) en el recurso /layers.
# Los demás tipos (ImageServer, GeocodeServer, etc.) quedan con una sola
# fila que describe el servicio.
fun_capas <- function(servicio, tipo) {
  url_servicio <- file.path(url_arcgis, servicio, tipo)

  if (!tipo %in% c("MapServer", "FeatureServer")) {
    info <- fun_get_json(url_servicio)
    return(tibble(
      servicio = servicio, tipo = tipo, id_capa = NA_integer_,
      capa = servicio, tipo_capa = tipo, geometria = NA_character_,
      descripcion = fun_valor(info$serviceDescription),
      campos = NA_character_, n_campos = NA_integer_,
      url = url_servicio
    ))
  }

  json <- fun_get_json(file.path(url_servicio, "layers"))
  if (is.null(json)) {
    return(tibble(
      servicio = servicio, tipo = tipo, id_capa = NA_integer_,
      capa = servicio, tipo_capa = "sin respuesta", geometria = NA_character_,
      descripcion = NA_character_, campos = NA_character_,
      n_campos = NA_integer_, url = url_servicio
    ))
  }

  map_dfr(c(json$layers, json$tables), function(lyr) {
    nombres_campos <- map_chr(lyr$fields %||% list(), function(campo) {
      alias <- fun_valor(campo$alias)
      nombre <- fun_valor(campo$name)
      if (!is.na(alias) && alias != nombre) paste0(nombre, " (", alias, ")") else nombre
    })
    tibble(
      servicio    = servicio,
      tipo        = tipo,
      id_capa     = as.integer(lyr$id),
      capa        = fun_valor(lyr$name),
      tipo_capa   = fun_valor(lyr$type),
      geometria   = str_remove(fun_valor(lyr$geometryType), "esriGeometry"),
      descripcion = fun_valor(lyr$description),
      campos      = paste(nombres_campos, collapse = "; "),
      n_campos    = length(nombres_campos),
      url         = paste0(url_servicio, "/", lyr$id)
    )
  })
}

capas_arcgis <- servicios %>%
  mutate(capas = map2(servicio, tipo, fun_capas)) %>%
  select(carpeta, capas) %>%
  unnest(capas) %>%
  mutate(
    descripcion = fun_limpiar_texto(descripcion),
    nivel       = fun_nivel(paste(capa, descripcion)),
    anios       = fun_anios(paste(servicio, capa, descripcion))
  ) %>%
  select(carpeta, servicio, tipo, id_capa, capa, tipo_capa, geometria, nivel,
         anios, n_campos, campos, descripcion, url) %>%
  arrange(carpeta, servicio, id_capa)

write_csv(capas_arcgis, file.path(ruta_salida, "inventario_capas_arcgis.csv"))
message("Capas ArcGIS guardadas: ", nrow(capas_arcgis))

# ------------------------------------------------------------------------------
# 2. Catálogo de Datos Abiertos Bogotá (API CKAN)
# ------------------------------------------------------------------------------

# La API entrega los conjuntos de datos por páginas; se recorren todas.
tam_pagina <- 500
primera    <- fun_get_json(url_ckan, query = list(rows = 0))
n_total    <- primera$result$count
message("Conjuntos de datos en Datos Abiertos: ", n_total)

paginas <- map(seq(0, n_total - 1, by = tam_pagina), function(inicio) {
  fun_get_json(url_ckan, query = list(rows = tam_pagina, start = inicio))$result$results
})
paginas <- do.call(c, paginas)

datos_abiertos <- map_dfr(paginas, function(d) {
  recursos <- d$resources %||% list()
  formatos <- map_chr(recursos, ~ str_to_upper(fun_valor(.x$format)))
  tibble(
    entidad        = fun_valor(d$organization$title),
    conjunto       = fun_valor(d$title),
    descripcion    = fun_valor(d$notes),
    etiquetas      = paste(map_chr(d$tags %||% list(), ~ fun_valor(.x$name)),
                           collapse = "; "),
    formatos       = paste(sort(unique(formatos[formatos != ""])), collapse = ", "),
    tiene_geodatos = any(str_detect(formatos, "SHP|GEOJSON|KML|REST|WMS|WFS|GDB")),
    url_rest       = paste(unique(na.omit(map_chr(recursos, function(r) {
      u <- fun_valor(r$url)
      if (!is.na(u) && str_detect(u, "arcgis/rest/services")) u else NA_character_
    }))), collapse = "; "),
    actualizado    = fun_valor(d$metadata_modified),
    url            = paste0("https://datosabiertos.bogota.gov.co/dataset/",
                            fun_valor(d$name))
  )
}) %>%
  mutate(
    descripcion = fun_limpiar_texto(descripcion),
    nivel       = fun_nivel(paste(conjunto, descripcion)),
    anios       = fun_anios(paste(conjunto, descripcion))
  ) %>%
  select(entidad, conjunto, nivel, anios, formatos, tiene_geodatos,
         etiquetas, descripcion, actualizado, url_rest, url) %>%
  arrange(entidad, conjunto)

write_csv(datos_abiertos, file.path(ruta_salida, "inventario_datos_abiertos.csv"))
message("Conjuntos de Datos Abiertos guardados: ", nrow(datos_abiertos))

# ------------------------------------------------------------------------------
# 3. Resumen en consola
# ------------------------------------------------------------------------------
capas_arcgis %>% count(carpeta, sort = TRUE) %>% print(n = Inf)
capas_arcgis %>% count(nivel, geometria, sort = TRUE) %>% print(n = Inf)
datos_abiertos %>% count(entidad, sort = TRUE) %>% print(n = Inf)
