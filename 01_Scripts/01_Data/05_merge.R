# ==============================================================================
# 05_merge.R
# Objetivo: unir la base limpia con las variables de texto, las espaciales y
#           la altura del terreno, y dejar la base lista para los modelos:
#           - área, baños y parqueaderos con la mejor fuente disponible;
#           - variables derivadas (área por habitación, año del anuncio como
#             factor);
#           - imputación de faltantes: cada variable numérica con faltantes
#             recibe una dummy bin_<variable>_na y sus faltantes se reemplazan
#             por la mediana de train. El área y los baños se imputan con la
#             mediana de train por tipo de propiedad y número de habitaciones.
#           Las medianas se calculan solo con train y se aplican igual a test.
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

# Las dummies de faltantes que no varían (todas cero) no aportan y se eliminan.
dummies_constantes <- db %>%
  select(starts_with("bin_") & ends_with("_na")) %>%
  select(where(~ n_distinct(.x) == 1)) %>%
  names()
db <- db %>% select(-all_of(dummies_constantes))

# ------------------------------------------------------------------------------
# 4. Bases de salida
# ------------------------------------------------------------------------------
db <- db %>% select(-title, -description)

db_train <- db %>% filter(muestra == "train") %>% select(-muestra)
db_test  <- db %>% filter(muestra == "test")  %>% select(-muestra, -obj_price, -obj_log_price)

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

# Variables con faltantes después de la imputación (solo deberían quedar las
# coordenadas, si alguna falta).
db %>%
  summarise(across(everything(), ~ sum(is.na(.x)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "faltantes") %>%
  filter(faltantes > 0, !variable %in% c("obj_price", "obj_log_price")) %>%
  print(n = Inf)
