# ==============================================================================
# 01_clean_base.R
# Objetivo: limpiar las variables originales de Kaggle y unir train y test en
#           una sola base con una fila por propiedad:
#           - tipos de dato y nombres con los prefijos del taller;
#           - fecha del anuncio;
#           - superficies fuera del rango [15, 1000] m2 tratadas como faltantes;
#           - habitaciones, ambientes y baños fuera de rangos razonables
#             tratados como faltantes;
#           - coordenadas en EPSG:9377 (metros), para los cálculos espaciales.
# Entradas: 02_Data/Raw/Kaggle/{train,test}.csv
# Salidas:  02_Data/Processed/base_limpia.rds
# ==============================================================================

source(here::here("01_Scripts", "00_utils.R"))

dir.create(here("02_Data", "Processed"), recursive = TRUE, showWarnings = FALSE)

raw <- fun_load_raw()

base <- bind_rows(raw$train %>% mutate(muestra = "train"),
                  raw$test  %>% mutate(muestra = "test")) %>%
  mutate(
    # Variable objetivo (solo en train).
    obj_price     = price,
    obj_log_price = log(price),
    # Fecha del anuncio.
    num_anio      = as.integer(year),
    num_mes       = as.integer(month),
    fecha         = make_date(year, month, 1),
    # Tipo de propiedad.
    cat_tipo      = factor(property_type, levels = c("Apartamento", "Casa")),
    bin_casa      = as.integer(property_type == "Casa"),
    # Superficies: valores fuera de [15, 1000] m2 se tratan como faltantes.
    num_surface_total   = if_else(between(surface_total, 15, 1000), surface_total, NA_real_),
    num_surface_covered = if_else(between(surface_covered, 15, 1000), surface_covered, NA_real_),
    # Habitaciones, ambientes y baños: valores fuera de rango como faltantes.
    num_habitac   = if_else(between(bedrooms, 0, 15), bedrooms, NA_real_),
    num_ambientes = if_else(between(rooms, 1, 20), rooms, NA_real_),
    num_banos     = if_else(between(bathrooms, 1, 15), bathrooms, NA_real_),
    # Coordenadas.
    num_lat       = lat,
    num_lon       = lon
  )

# Coordenadas proyectadas en metros (EPSG:9377).
coords_m <- base %>%
  st_as_sf(coords = c("lon", "lat"), crs = 4326, remove = FALSE) %>%
  st_transform(crs_metros) %>%
  st_coordinates()

base <- base %>%
  mutate(num_x = coords_m[, 1], num_y = coords_m[, 2]) %>%
  select(property_id, muestra, obj_price, obj_log_price, fecha, num_anio,
         num_mes, cat_tipo, bin_casa, num_surface_total, num_surface_covered,
         num_habitac, num_ambientes, num_banos, num_lat, num_lon, num_x, num_y,
         title, description)

stopifnot(!anyDuplicated(base$property_id))
saveRDS(base, here("02_Data", "Processed", "base_limpia.rds"))

# ------------------------------------------------------------------------------
# Resumen en consola
# ------------------------------------------------------------------------------
base %>% count(muestra, cat_tipo) %>% print()
base %>% count(muestra, num_anio) %>% print()
base %>%
  group_by(muestra) %>%
  summarise(across(c(num_surface_total, num_surface_covered, num_habitac,
                     num_ambientes, num_banos),
                   ~ round(100 * mean(is.na(.x)), 1))) %>%
  print()
