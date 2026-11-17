# ==============================================================================
# 06_xgboost.R
# Objetivo: boosting de árboles (XGBoost) del log del precio con todos los
#           predictores de la base completa. La validación cruzada se programa
#           directamente con xgboost, porque caret::xgbTree no es compatible
#           con la versión 3 del paquete.
#           1. Para cada combinación de la grilla (profundidad y tamaño mínimo
#              de las hojas) se entrena en cada fold del esquema de selección
#              con parada temprana: se detiene cuando el MAE del fold de
#              validación no mejora en 50 rondas.
#           2. Se escoge la combinación con el menor MAE en pesos sobre todas
#              las predicciones fuera de muestra juntas. El número de rondas
#              final es el promedio de las rondas óptimas de los folds.
#           3. Con esos hiperparámetros y ese número de rondas fijo se calcula
#              el MAE en pesos con los cuatro esquemas de validación.
#           4. Se entrena el modelo final con todo train y se predice test.
# Entradas: 02_Data/Processed/db_{train,test}.rds
# Salidas:  03_Output/Submissions/09_xgb_completa.csv
#           03_Output/Models/09_xgb_completa.rds
#           03_Output/Models/resultados_modelos.csv (fila del modelo)
# ==============================================================================

source(here::here("01_Scripts", "00_utils.R"))
pacman::p_load(xgboost)

envio   <- "09_xgb_completa"
familia <- "XGBoost"

datos <- fun_datos_modelo()
folds <- fun_todos_folds(datos$train)

# ------------------------------------------------------------------------------
# 1. Matrices de predictores
# Las variables categóricas se convierten en dummies con los mismos niveles en
# train y test.
# ------------------------------------------------------------------------------
fml_x <- reformulate(datos$predictores, intercept = FALSE)
x_todo <- model.matrix(fml_x, data = bind_rows(datos$train[datos$predictores],
                                               datos$test[datos$predictores]))
n_train <- nrow(datos$train)
x_train <- x_todo[seq_len(n_train), , drop = FALSE]
x_test  <- x_todo[-seq_len(n_train), , drop = FALSE]
y_train <- datos$train$obj_log_price

# ------------------------------------------------------------------------------
# 2. Funciones de entrenamiento
# ------------------------------------------------------------------------------
parametros_base <- list(objective = "reg:squarederror",
                        eval_metric = "mae",
                        eta = 0.05,
                        subsample = 0.8,
                        colsample_bytree = 0.7,
                        nthread = max(1, parallel::detectCores() - 1))

# Ronda óptima de un modelo con parada temprana (base 1).
fun_mejor_ronda <- function(modelo) {
  ronda <- attributes(modelo)$early_stop$best_iteration
  if (is.null(ronda)) ronda <- as.integer(xgb.attr(modelo, "best_iteration")) + 1
  as.integer(ronda)
}

# Entrena en cada fold. Con `rondas = NULL` usa parada temprana y devuelve la
# ronda óptima de cada fold; con un número de rondas fijo entrena exactamente
# esas rondas. Devuelve el MAE en pesos de las predicciones fuera de muestra y
# las rondas usadas.
fun_cv_xgb <- function(parametros, folds, rondas = NULL, max_rondas = 3000) {
  pred <- rep(NA_real_, n_train)
  rondas_fold <- integer(length(folds))
  for (i in seq_along(folds)) {
    f <- folds[[i]]
    d_train <- xgb.DMatrix(x_train[f$train, , drop = FALSE], label = y_train[f$train])
    d_val   <- xgb.DMatrix(x_train[f$val, , drop = FALSE],   label = y_train[f$val])
    set.seed(semilla)
    if (is.null(rondas)) {
      modelo <- xgb.train(params = parametros, data = d_train, nrounds = max_rondas,
                          evals = list(val = d_val), early_stopping_rounds = 50,
                          verbose = 0)
      rondas_fold[i] <- fun_mejor_ronda(modelo)
    } else {
      modelo <- xgb.train(params = parametros, data = d_train, nrounds = rondas,
                          verbose = 0)
      rondas_fold[i] <- rondas
    }
    pred[f$val] <- predict(modelo, d_val, iterationrange = c(1, rondas_fold[i]))
  }
  validas <- !is.na(pred)
  list(mae_cop = fun_mae_log(y_train[validas], pred[validas]),
       rondas = rondas_fold)
}

# ------------------------------------------------------------------------------
# 3. Afinación con el esquema de selección
# ------------------------------------------------------------------------------
grid <- expand.grid(max_depth = c(4, 6, 8), min_child_weight = c(1, 10))

message("Afinando con el esquema: ", etiquetas_cv[[esquema_seleccion]])
tabla_afinacion <- pmap_dfr(grid, function(max_depth, min_child_weight) {
  parametros <- c(parametros_base, list(max_depth = max_depth,
                                        min_child_weight = min_child_weight))
  r <- fun_cv_xgb(parametros, folds[[esquema_seleccion]])
  message("  max_depth = ", max_depth, ", min_child_weight = ", min_child_weight,
          ": MAE ", round(r$mae_cop / 1e6, 1), " millones, rondas ",
          paste(r$rondas, collapse = "/"))
  tibble(max_depth = max_depth, min_child_weight = min_child_weight,
         rondas = round(mean(r$rondas)), mae_cop = r$mae_cop)
}) %>%
  arrange(mae_cop)

mejor <- tabla_afinacion[1, ]
parametros_mejor <- c(parametros_base, list(max_depth = mejor$max_depth,
                                            min_child_weight = mejor$min_child_weight))
hiperparam <- data.frame(max_depth = mejor$max_depth,
                         min_child_weight = mejor$min_child_weight,
                         eta = parametros_base$eta,
                         rondas = mejor$rondas)
hiperparam_texto <- fun_texto_hiperparam(hiperparam)
message("Hiperparámetros escogidos: ", hiperparam_texto)

# ------------------------------------------------------------------------------
# 4. MAE con los cuatro esquemas (hiperparámetros y rondas fijos)
# ------------------------------------------------------------------------------
message("Calculando el MAE con los cuatro esquemas")
mae <- imap_dfr(folds, function(f, esquema) {
  tibble(esquema = esquema,
         mae_cop = fun_cv_xgb(parametros_mejor, f, rondas = mejor$rondas)$mae_cop)
})

# ------------------------------------------------------------------------------
# 5. Modelo final, envío y registro
# ------------------------------------------------------------------------------
message("Ajuste final con todo train")
set.seed(semilla)
final <- xgb.train(params = parametros_mejor,
                   data = xgb.DMatrix(x_train, label = y_train),
                   nrounds = mejor$rondas, verbose = 0)
pred_test <- predict(final, xgb.DMatrix(x_test))
fun_submission(datos$test$property_id, pred_test, envio)

importancia <- xgb.importance(model = final)

saveRDS(list(envio = envio, familia = familia, hiperparam = hiperparam,
             tabla_afinacion = tabla_afinacion, mae = mae, pred_test = pred_test,
             importancia = importancia),
        here("03_Output", "Models", paste0(envio, ".rds")))
fun_registrar_resultado(envio, familia, hiperparam_texto, mae, envio)

print(tabla_afinacion)
print(mae %>% mutate(mae_millones = round(mae_cop / 1e6, 1)))
print(head(importancia, 20))
message("Descripción para Kaggle: ", fun_descripcion_kaggle(familia, hiperparam_texto, mae))
