# ==============================================================================
# 02_text_features.R
# Objetivo: construir predictores a partir del título y la descripción de cada
#           anuncio de train y test:
#           - números escritos en el texto: área en m2, piso, número de niveles,
#             parqueaderos, habitaciones y baños;
#           - estrato escrito en el texto ("estrato 6");
#           - dummies de características mencionadas (terraza, gimnasio, etc.),
#             de palabras de lujo y de barrios de alto precio;
#           - largo de la descripción;
#           - área final: surface_total o surface_covered cuando existen y son
#             razonables; si no, el área extraída del texto.
#           El texto de Properati viene en minúsculas, sin tildes y sin la letra
#           con tilde (por ejemplo, "bao" por "baño" y "rea" por "área"), y sin
#           separador decimal ("7942mt" corresponde a 79,42 m2). Las expresiones
#           regulares de este script contemplan esas formas.
# Entradas: 02_Data/Raw/Kaggle/{train,test}.csv
# Salidas:  02_Data/Processed/text_features.rds (una fila por property_id)
# ==============================================================================

source(here::here("01_Scripts", "00_utils.R"))
pacman::p_load(stringi)

dir.create(here("02_Data", "Processed"), recursive = TRUE, showWarnings = FALSE)

raw <- fun_load_raw()
db  <- bind_rows(raw$train %>% mutate(muestra = "train"),
                 raw$test  %>% mutate(muestra = "test"))

# ------------------------------------------------------------------------------
# 1. Texto normalizado
# Se unen título y descripción, se pasa a minúsculas, se quitan tildes y se
# reemplazan los signos por espacios. Los números escritos en palabras se
# convierten en dígitos para que las mismas expresiones lean "dos banos" y
# "2 banos".
# ------------------------------------------------------------------------------
numeros_palabra <- c(
  "\\buna?\\b" = "1", "\\bdos\\b" = "2", "\\btres\\b" = "3",
  "\\bcuatro\\b" = "4", "\\bcinco\\b" = "5", "\\bseis\\b" = "6",
  "\\bsiete\\b" = "7", "\\bocho\\b" = "8", "\\bnueve\\b" = "9",
  "\\bdiez\\b" = "10"
)

ordinales_piso <- c(
  "primer" = 1, "primero" = 1, "segundo" = 2, "tercer" = 3, "tercero" = 3,
  "cuarto" = 4, "quinto" = 5, "sexto" = 6, "septimo" = 7, "octavo" = 8,
  "noveno" = 9, "decimo" = 10
)

fun_normalizar <- function(x) {
  x %>%
    replace_na("") %>%
    str_to_lower() %>%
    stri_trans_general("Latin-ASCII") %>%
    str_replace_all("m²", "m2") %>%
    str_replace_all("[^a-z0-9 ]", " ") %>%
    str_squish()
}

db <- db %>%
  mutate(
    texto       = fun_normalizar(paste(title, description)),
    texto_num   = str_replace_all(texto, numeros_palabra),
    num_palabras = str_count(fun_normalizar(description), "\\S+")
  )

# ------------------------------------------------------------------------------
# 2. Números escritos en el texto
# ------------------------------------------------------------------------------

# Extrae todos los números que preceden a un patrón y devuelve el valor que
# indique `resumen` (máximo o primero) dentro del rango [minimo, maximo].
fun_extraer_numero <- function(texto, patron, minimo, maximo, resumen = max) {
  coincidencias <- str_match_all(texto, paste0("\\b(\\d{1,4})\\s?(?:", patron, ")\\b"))
  map_dbl(coincidencias, function(m) {
    valores <- as.numeric(m[, 2])
    valores <- valores[valores >= minimo & valores <= maximo]
    if (length(valores) == 0) NA_real_ else resumen(valores)
  })
}

# 2.1 Área en m2. Se aceptan las unidades m2, mt2, mts2, mt, mts, metros y
# metros cuadrados. Para cada número se leen las dos palabras anteriores y las
# dos siguientes, y se descartan:
# - las áreas de espacios parciales (terraza, balcón, patio, jardín, depósito,
#   garaje), porque no son el área del inmueble;
# - las medidas de frente o fondo de un lote ("6 mts de frente"), que son
#   longitudes y no áreas.
# Si quedan varias áreas (por ejemplo, área construida y área privada), se
# toma la mayor. Los valores de cuatro dígitos entre 1.500 y 9.999 se dividen
# por 100, porque corresponden a áreas escritas con decimales sin separador
# ("7942mt" = 79,42 m2).
unidades_area <- "m2|m 2|mt2|mts2|mtrs2|mt|mts|mtrs|metros cuadrados|metros|mtrs cuadrados"
contexto_parcial <- "terraza|balcon|patio|jardin|deposito|garaje|parqueadero|frente|fondo"

fun_area_texto <- function(texto) {
  patron <- paste0("((?:\\S+ ){0,2})\\b(\\d{2,4})\\s?(?:", unidades_area,
                   ")\\b((?: \\S+){0,2})")
  coincidencias <- str_match_all(texto, patron)
  map_dbl(coincidencias, function(m) {
    if (nrow(m) == 0) return(NA_real_)
    parcial <- str_detect(paste(m[, 2], m[, 4]), contexto_parcial)
    valores <- as.numeric(m[!parcial, 3])
    valores <- if_else(valores >= 1500 & valores <= 9999, valores / 100, valores)
    valores <- valores[valores >= 15 & valores <= 1000]
    if (length(valores) == 0) NA_real_ else max(valores)
  })
}

# 2.2 Piso del apartamento: "piso 7", "7 piso" u ordinal antes o después de
# la palabra piso ("septimo piso", "piso septimo").
fun_piso <- function(texto) {
  piso_num <- coalesce(
    as.numeric(str_match(texto, "\\bpiso (\\d{1,2})\\b")[, 2]),
    as.numeric(str_match(texto, "\\b(\\d{1,2}) piso\\b")[, 2])
  )
  lista_ordinales <- paste(names(ordinales_piso), collapse = "|")
  ordinal <- coalesce(
    str_match(texto, paste0("\\b(", lista_ordinales, ") piso\\b"))[, 2],
    str_match(texto, paste0("\\bpiso (", lista_ordinales, ")\\b"))[, 2]
  )
  piso <- coalesce(piso_num, unname(ordinales_piso[ordinal]))
  if_else(piso >= 1 & piso <= 40, piso, NA_real_)
}

db <- db %>%
  mutate(
    num_area_texto   = fun_area_texto(texto),
    num_piso_texto   = fun_piso(texto),
    num_niveles_texto = fun_extraer_numero(texto_num, "pisos|niveles", 2, 5),
    num_parq_texto   = fun_extraer_numero(texto_num,
                                          "parqueaderos?|garajes?|parqueos?", 1, 6),
    num_habitac_texto = fun_extraer_numero(texto_num,
                                           "habitaciones|alcobas|cuartos|dormitorios|habitacion|alcoba",
                                           1, 10),
    num_banos_texto  = fun_extraer_numero(texto_num, "banos|baos|bano|bao", 1, 10),
    # Un área del texto menor a 10 m2 por habitación no es el área del
    # inmueble completo y se trata como faltante.
    num_area_texto   = if_else(!is.na(num_area_texto) & bedrooms > 0 &
                                 num_area_texto < 10 * bedrooms,
                               NA_real_, num_area_texto),
    # Áreas de tres dígitos escritas con un decimal sin separador ("677 m2"
    # por 67,7 m2). Cuando el área por habitación supera el percentil 99 de
    # las áreas de surface_total (100 m2 por habitación en apartamentos y
    # 180 m2 en casas) y el valor dividido por 10 sigue siendo un área
    # válida, se divide por 10.
    limite_area_hab  = if_else(property_type == "Casa", 180, 100),
    num_area_texto   = if_else(!is.na(num_area_texto) & num_area_texto >= 150 &
                                 num_area_texto / pmax(bedrooms, 1, na.rm = TRUE) > limite_area_hab,
                               num_area_texto / 10, num_area_texto),
    # Estrato escrito en el texto ("estrato 6"); si hay varios, el primero.
    num_estrato_texto = as.numeric(str_match(texto, "\\bestrato (\\d)\\b")[, 2]),
    num_estrato_texto = if_else(num_estrato_texto %in% 1:6, num_estrato_texto, NA_real_)
  ) %>%
  select(-limite_area_hab)

# ------------------------------------------------------------------------------
# 3. Dummies de características mencionadas
# Cada dummy vale 1 si el texto contiene alguna de las expresiones.
# ------------------------------------------------------------------------------
palabras_clave <- c(
  bin_parqueadero  = "parqueadero|garaje|parqueo",
  bin_terraza      = "terraza|balcon",
  bin_gimnasio     = "gimnasio|\\bgym\\b",
  bin_piscina      = "piscina",
  bin_ascensor     = "ascensor",
  bin_chimenea     = "chimenea",
  bin_vigilancia   = "vigilancia|porteria|portero|seguridad 24",
  bin_deposito     = "deposito",
  bin_estudio      = "estudio",
  bin_remodelado   = "remodelad|renovad",
  bin_vista        = "\\bvista",
  bin_duplex       = "duplex",
  bin_penthouse    = "penthouse|pent house",
  bin_exterior     = "\\bexterior",
  bin_conjunto     = "conjunto cerrado|conjunto residencial",
  bin_bbq          = "\\bbbq\\b|barbecue|asador",
  bin_salon_comunal = "salon comunal|salon social",
  bin_jardin       = "jardin|zonas verdes",
  bin_estrenar     = "para estrenar|nuevo|sobre planos",
  bin_parque_cerca = "frente al parque|cerca al parque|cerca de parque",
  bin_lujo         = "lujo|exclusiv|alto nivel|premium",
  bin_vista_ciudad = "vista (a la|sobre la|de la) ciudad|vista panoramica",
  # Barrios de alto precio de Chapinero y sus alrededores mencionados en el
  # anuncio. Chicó y Rosales tienen dummy propia; el resto se agrupa porque
  # cada uno aparece en muy pocos anuncios de train.
  bin_chico        = "\\bchico\\b",
  bin_rosales      = "\\brosales\\b",
  bin_barrio_alto  = paste("\\bchico\\b|\\brosales\\b|la cabrera|el retiro|el nogal|antiguo country",
                           "el refugio|parque (de la )?93|virrey|emaus|quinta camacho",
                           "zona g\\b|zona t\\b|chapinero alto", sep = "|")
)

for (nombre in names(palabras_clave)) {
  db[[nombre]] <- as.integer(str_detect(db$texto, palabras_clave[[nombre]]))
}

# ------------------------------------------------------------------------------
# 4. Área final
# Orden de preferencia: surface_total, surface_covered y área del texto. Las
# superficies fuera del rango [15, 1000] m2 se tratan como faltantes.
# `cat_fuente_area` indica de dónde sale el valor.
# ------------------------------------------------------------------------------
fun_area_valida <- function(x) if_else(x >= 15 & x <= 1000, x, NA_real_)

db <- db %>%
  mutate(
    area_total    = fun_area_valida(surface_total),
    area_cubierta = fun_area_valida(surface_covered),
    num_area      = coalesce(area_total, area_cubierta, num_area_texto),
    cat_fuente_area = case_when(
      !is.na(area_total)     ~ "surface_total",
      !is.na(area_cubierta)  ~ "surface_covered",
      !is.na(num_area_texto) ~ "texto",
      TRUE                   ~ "sin dato"
    ),
    # Habitaciones, baños y parqueaderos completados con el texto.
    num_banos_comp = coalesce(bathrooms, num_banos_texto),
    num_parq_comp  = coalesce(num_parq_texto,
                              if_else(bin_parqueadero == 1, 1, 0))
  )

# ------------------------------------------------------------------------------
# 5. Base de salida
# ------------------------------------------------------------------------------
text_features <- db %>%
  select(property_id, muestra, num_area, cat_fuente_area, num_area_texto,
         num_piso_texto, num_niveles_texto, num_parq_texto, num_parq_comp,
         num_habitac_texto, num_banos_texto, num_banos_comp, num_estrato_texto,
         num_palabras, starts_with("bin_"))

saveRDS(text_features, here("02_Data", "Processed", "text_features.rds"))

# ------------------------------------------------------------------------------
# 6. Resumen en consola
# ------------------------------------------------------------------------------

# Porcentaje de anuncios con dato en cada variable, por muestra.
text_features %>%
  group_by(muestra) %>%
  summarise(across(c(num_area, num_area_texto, num_piso_texto, num_parq_texto,
                     num_habitac_texto, num_banos_texto, num_estrato_texto),
                   ~ round(100 * mean(!is.na(.x)), 1)),
            across(starts_with("bin_"), ~ round(100 * mean(.x), 1))) %>%
  pivot_longer(-muestra, names_to = "variable", values_to = "porcentaje") %>%
  pivot_wider(names_from = muestra, values_from = porcentaje) %>%
  print(n = Inf)

# Fuente del área final, por muestra.
text_features %>% count(muestra, cat_fuente_area) %>% print()

# Concordancia entre el área del texto y surface_total cuando existen las dos.
db %>%
  filter(!is.na(area_total), !is.na(num_area_texto)) %>%
  summarise(n = n(),
            correlacion = cor(area_total, num_area_texto),
            mediana_diferencia_pct = median(abs(num_area_texto / area_total - 1)) * 100) %>%
  print()
