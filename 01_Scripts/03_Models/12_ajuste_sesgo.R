# ==============================================================================
# 12_ajuste_sesgo.R
# Objetivo: corregir la subestimación del XGBoost con pesos sobre la base con
#           texto como datos (envío 16_xgb_pesos_texto).
#           El modelo se entrena con el error cuadrático del log del precio. Al
#           volver a pesos con exp(), la predicción se acerca a la mediana del
#           precio y no a su promedio, y queda por debajo del precio observado
#           en las zonas caras como Chapinero. Este script:
#           1. Reconstruye las predicciones fuera de muestra del modelo 16 con
#              los 5 folds de localidades completas por fuera, con los mismos
#              hiperparámetros, pesos y faltantes nativos de
#              10_xgb_pesos_texto.R.
#           2. Multiplica esas predicciones por factores entre 1,00 y 1,15 y
#              calcula el MAE en pesos de cada factor: en todo train, en los
#              anuncios de Chapinero y ponderado por los pesos de parecido con
#              test (w), que aproxima el error en test.
#           3. Genera dos envíos, cada uno con un criterio para escoger el
#              factor, aplicado a las predicciones de test del modelo 16:
#              - 18_xgb_pesos_texto_sesgo_ponderado: el factor con el menor MAE
#                ponderado por w, que usa todo train pero da más peso a los
#                anuncios parecidos a test;
#              - 19_xgb_pesos_texto_sesgo_chapinero: el factor con el menor MAE
#                en los anuncios de Chapinero, que es la localidad de test pero
#                tiene solo 305 anuncios en train.
#           Subir las predicciones reduce el MAE, pero aumenta la proporción de
#           inmuebles cuyo precio se sobreestima; esa proporción se reporta para
#           cada factor.
# Entradas: 02_Data/Processed/db_{train,test}.rds
#           02_Data/Processed/pesos_adversarial_texto.rds
#           03_Output/Models/16_xgb_pesos_texto.rds
# Salidas:  03_Output/Submissions/18_xgb_pesos_texto_sesgo_ponderado.csv,
#           19_xgb_pesos_texto_sesgo_chapinero.csv
#           03_Output/Models/18_xgb_pesos_texto_sesgo_ponderado.rds,
#           19_xgb_pesos_texto_sesgo_chapinero.rds
#           03_Output/Tables/ajuste_sesgo_factores.csv
#           03_Output/Models/resultados_modelos.csv (filas de los modelos)
# ==============================================================================

source(here::here("01_Scripts", "00_utils.R"))
pacman::p_load(xgboost)

modelo_16 <- readRDS(here("03_Output", "Models", "16_xgb_pesos_texto.rds"))
hp <- modelo_16$hiperparam

# ------------------------------------------------------------------------------
# 1. Datos con faltantes nativos, como en 10_xgb_pesos_texto.R
# Cada dummy bin_<x>_na de 05_merge.R marca las filas donde num_<x> fue
# imputada con la mediana; en esas filas num_<x> vuelve a NA.
# ------------------------------------------------------------------------------
datos <- fun_datos_modelo()

fun_restaurar_na <- function(df) {
  df <- df %>% mutate(num_vref_m2_200m = na_if(num_vref_m2_200m, 0))
  banderas <- str_subset(names(df), "^bin_.+_na$")
  for (b in banderas) {
    v <- paste0("num_", str_remove(str_remove(b, "^bin_"), "_na$"))
    if (v %in% names(df)) df[[v]][df[[b]] == 1] <- NA
  }
  df
}

train_x <- fun_restaurar_na(datos$train)
test_x  <- fun_restaurar_na(datos$test)
n_train <- nrow(train_x)

fml_x  <- reformulate(datos$predictores, intercept = FALSE)
todo   <- bind_rows(train_x[datos$predictores], test_x[datos$predictores])
x_todo <- model.matrix(fml_x, model.frame(fml_x, data = todo, na.action = na.pass))
x_train <- x_todo[seq_len(n_train), , drop = FALSE]
y_train <- train_x$obj_log_price
precio  <- train_x$obj_price

# Pesos de parecido con test (w) y peso de entrenamiento w^gamma normalizado a
# media 1, con el gamma escogido en el modelo 16.
w <- train_x %>%
  select(property_id) %>%
  left_join(readRDS(here("02_Data", "Processed", "pesos_adversarial_texto.rds")),
            by = "property_id") %>%
  pull(peso)
stopifnot(!anyNA(w))
peso <- w^hp$gamma_peso
peso <- peso / mean(peso)

# ------------------------------------------------------------------------------
# 2. Predicciones fuera de muestra con localidades completas por fuera
# ------------------------------------------------------------------------------
parametros <- list(objective = "reg:squarederror", eval_metric = "mae",
                   eta = hp$eta, subsample = 0.8, colsample_bytree = 0.7,
                   max_depth = hp$max_depth, min_child_weight = hp$min_child_weight,
                   nthread = max(1, parallel::detectCores() - 1))

folds_localidad <- fun_folds_cv(datos$train, "localidad")
pred_oof <- rep(NA_real_, n_train)
for (i in seq_along(folds_localidad)) {
  f <- folds_localidad[[i]]
  message("Fold ", i, " de ", length(folds_localidad))
  set.seed(semilla)
  m <- xgb.train(params = parametros,
                 data = xgb.DMatrix(x_train[f$train, , drop = FALSE],
                                    label = y_train[f$train], weight = peso[f$train]),
                 nrounds = hp$rondas, verbose = 0)
  pred_oof[f$val] <- predict(m, xgb.DMatrix(x_train[f$val, , drop = FALSE]))
}

# ------------------------------------------------------------------------------
# 3. MAE por factor de ajuste
# ------------------------------------------------------------------------------
es_chap <- train_x$cat_localidad == "CHAPINERO"

factores <- map_dfr(seq(1, 1.15, by = 0.01), function(factor) {
  pred <- factor * exp(pred_oof)
  tibble(factor = factor,
         mae_millones           = mean(abs(pred - precio)) / 1e6,
         mae_chapinero_millones = mean(abs(pred[es_chap] - precio[es_chap])) / 1e6,
         mae_ponderado_millones = weighted.mean(abs(pred - precio), w) / 1e6,
         porc_sobreestima       = 100 * mean(pred > precio),
         porc_sobreestima_chap  = 100 * mean(pred[es_chap] > precio[es_chap]))
})
write_csv(factores, here("03_Output", "Tables", "ajuste_sesgo_factores.csv"))
print(factores, n = Inf)

# ------------------------------------------------------------------------------
# 4. Envíos: predicciones de test del modelo 16 multiplicadas por el factor
# ------------------------------------------------------------------------------
fun_envio_factor <- function(envio, factor_elegido, criterio) {
  message(envio, ": factor ", factor_elegido, " (", criterio, ")")
  pred_test <- modelo_16$pred_test + log(factor_elegido)
  fun_submission(datos$test$property_id, pred_test, envio)
  mae <- tibble(esquema = "localidad",
                mae_cop = factores$mae_millones[factores$factor == factor_elegido] * 1e6)
  hiperparam_texto <- paste0(fun_texto_hiperparam(hp), ", factor = ", factor_elegido)
  saveRDS(list(envio = envio, familia = "XGBoost", hiperparam = hp,
               factor = factor_elegido, criterio = criterio, tabla_factores = factores,
               mae = mae, pred_test = pred_test),
          here("03_Output", "Models", paste0(envio, ".rds")))
  fun_registrar_resultado(envio, "XGBoost", hiperparam_texto, mae, envio)
}

factor_ponderado <- factores$factor[which.min(factores$mae_ponderado_millones)]
factor_chapinero <- factores$factor[which.min(factores$mae_chapinero_millones)]

fun_envio_factor("18_xgb_pesos_texto_sesgo_ponderado", factor_ponderado,
                 "menor MAE ponderado por el parecido con test")
fun_envio_factor("19_xgb_pesos_texto_sesgo_chapinero", factor_chapinero,
                 "menor MAE en los anuncios de Chapinero")

message("Listo. Sube a Kaggle 18_xgb_pesos_texto_sesgo_ponderado.csv y ",
        "19_xgb_pesos_texto_sesgo_chapinero.csv")
