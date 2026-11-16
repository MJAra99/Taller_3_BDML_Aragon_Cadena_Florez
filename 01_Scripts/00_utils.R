# ==============================================================================
# 00_utils.R
# Objetivo: funciones comunes del taller (limpieza, folds, MAE y envíos a Kaggle).
# Entradas: ninguna (se carga con source() desde los demás scripts).
# Salidas:  funciones fun_* en el ambiente.
# ==============================================================================

pacman::p_load(tidyverse, here, caret)

semilla <- 2026

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
