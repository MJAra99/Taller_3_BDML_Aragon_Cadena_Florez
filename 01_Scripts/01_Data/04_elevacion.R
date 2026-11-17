# ==============================================================================
# 04_elevacion.R
# Objetivo: obtener la altura del terreno (metros sobre el nivel del mar) en la
#           ubicación de cada propiedad, a partir del modelo digital de
#           elevación de Amazon Web Services (Terrain Tiles) que descarga el
#           paquete elevatr. Con zoom 12, cada celda del modelo mide unos 30 m
#           en Bogotá. La consulta se hace sobre las coordenadas únicas, porque
#           muchas propiedades comparten ubicación.
# Entradas: 02_Data/Processed/base_limpia.rds
# Salidas:  02_Data/Processed/elevacion.rds (property_id, num_altura_msnm)
# ==============================================================================

source(here::here("01_Scripts", "00_utils.R"))
pacman::p_load(elevatr)

base <- readRDS(here("02_Data", "Processed", "base_limpia.rds"))

coords_unicas <- base %>%
  distinct(num_lon, num_lat) %>%
  mutate(id_coord = row_number())

puntos_unicos <- coords_unicas %>%
  st_as_sf(coords = c("num_lon", "num_lat"), crs = 4326, remove = FALSE)

message("Consultando la altura de ", nrow(puntos_unicos), " coordenadas únicas")
altura <- get_elev_point(puntos_unicos, src = "aws", z = 12)

elevacion <- base %>%
  select(property_id, num_lon, num_lat) %>%
  left_join(coords_unicas, by = c("num_lon", "num_lat")) %>%
  left_join(st_drop_geometry(altura) %>% select(id_coord, num_altura_msnm = elevation),
            by = "id_coord") %>%
  select(property_id, num_altura_msnm)

saveRDS(elevacion, here("02_Data", "Processed", "elevacion.rds"))

# ------------------------------------------------------------------------------
# Resumen en consola
# ------------------------------------------------------------------------------
elevacion %>%
  left_join(base %>% select(property_id, muestra), by = "property_id") %>%
  group_by(muestra) %>%
  summarise(pct_na = round(100 * mean(is.na(num_altura_msnm)), 1),
            minimo = min(num_altura_msnm, na.rm = TRUE),
            mediana = median(num_altura_msnm, na.rm = TRUE),
            maximo = max(num_altura_msnm, na.rm = TRUE)) %>%
  print()
