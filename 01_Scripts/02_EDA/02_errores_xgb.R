# ==============================================================================
# 02_errores_xgb.R
# Objetivo: cobertura espacial de test y análisis de los errores del XGBoost
#           (09_xgb_completa).
#           1. Cobertura espacial: distancia de cada inmueble de test al
#              inmueble de train más cercano y número de inmuebles de train a
#              menos de 500 m.
#           2. Errores fuera de muestra del XGBoost con el esquema de
#              localidades completas por fuera: MAE y error medio
#              (sobreestimación o subestimación) por localidad, estrato, tipo
#              de inmueble y decil de precio.
#           3. Importancia de las variables del XGBoost.
# Entradas: 02_Data/Processed/db_{train,test}.rds
#           03_Output/Models/09_xgb_completa.rds
# Salidas:  03_Output/Tables/eda_cobertura_test.csv, eda_errores_*.csv,
#           eda_importancia_xgb.csv
#           03_Output/Figures/eda_cobertura_test.png, eda_observado_predicho.png,
#           eda_error_decil_precio.png, eda_importancia_xgb.png
#           03_Output/Predictions/oof_xgb_localidad.rds
# ==============================================================================

source(here::here("01_Scripts", "00_utils.R"))
pacman::p_load(xgboost, scales)

ruta_tablas  <- here("03_Output", "Tables")
ruta_figuras <- here("03_Output", "Figures")
ruta_pred    <- here("03_Output", "Predictions")
walk(c(ruta_tablas, ruta_figuras, ruta_pred), dir.create, showWarnings = FALSE,
     recursive = TRUE)

datos <- fun_datos_modelo()
train <- datos$train
test  <- datos$test

guardar_figura <- function(grafico, nombre, ancho = 8, alto = 5) {
  ggsave(file.path(ruta_figuras, paste0(nombre, ".png")), grafico,
         width = ancho, height = alto, dpi = 300, bg = "white")
}

# ------------------------------------------------------------------------------
# 1. Cobertura espacial de test
# ------------------------------------------------------------------------------
coords_train <- as.matrix(train[, c("num_x", "num_y")])
coords_test  <- as.matrix(test[, c("num_x", "num_y")])

cobertura <- tibble(
  property_id = test$property_id,
  dist_train_mas_cercano = kNN(coords_train, k = 1, query = coords_test)$dist[, 1],
  n_train_500m = lengths(frNN(coords_train, eps = 500, query = coords_test,
                              sort = FALSE)$id)
)
resumen_cobertura <- cobertura %>%
  summarise(dist_mediana_m = median(dist_train_mas_cercano),
            dist_p90_m = quantile(dist_train_mas_cercano, 0.9),
            porc_sin_train_500m = 100 * mean(n_train_500m == 0),
            n_train_500m_mediana = median(n_train_500m))
write_csv(resumen_cobertura, file.path(ruta_tablas, "eda_cobertura_test.csv"))
print(resumen_cobertura)

g_cobertura <- cobertura %>%
  ggplot(aes(x = n_train_500m + 1)) +
  geom_histogram(bins = 40, fill = "#0072B2") +
  scale_x_log10() +
  labs(title = "Inmuebles de train cerca de cada inmueble de test",
       x = "Número de inmuebles de train a menos de 500 m + 1 (escala logarítmica)",
       y = "Inmuebles de test") +
  theme_minimal()
guardar_figura(g_cobertura, "eda_cobertura_test")

# ------------------------------------------------------------------------------
# 2. Errores fuera de muestra del XGBoost
# Se reentrena el XGBoost con los hiperparámetros escogidos en 06_xgboost.R en
# cada fold de localidades completas por fuera. Cada inmueble recibe la
# predicción de un modelo que no vio su localidad. Error = predicción -
# precio observado: positivo es sobreestimación, negativo es subestimación.
# ------------------------------------------------------------------------------
xgb_guardado <- readRDS(here("03_Output", "Models", "09_xgb_completa.rds"))
hp <- xgb_guardado$hiperparam

fml_x <- reformulate(datos$predictores, intercept = FALSE)
x_todo <- model.matrix(fml_x, data = bind_rows(train[datos$predictores],
                                               test[datos$predictores]))
x_train <- x_todo[seq_len(nrow(train)), , drop = FALSE]
y_train <- train$obj_log_price

parametros <- list(objective = "reg:squarederror", eval_metric = "mae",
                   eta = hp$eta, subsample = 0.8, colsample_bytree = 0.7,
                   max_depth = hp$max_depth, min_child_weight = hp$min_child_weight,
                   nthread = max(1, parallel::detectCores() - 1))

folds_localidad <- fun_folds_cv(train, "localidad")
pred_oof <- rep(NA_real_, nrow(train))
for (i in seq_along(folds_localidad)) {
  f <- folds_localidad[[i]]
  message("Fold ", i, " de ", length(folds_localidad))
  set.seed(semilla)
  modelo <- xgb.train(params = parametros,
                      data = xgb.DMatrix(x_train[f$train, , drop = FALSE],
                                         label = y_train[f$train]),
                      nrounds = hp$rondas, verbose = 0)
  pred_oof[f$val] <- predict(modelo, xgb.DMatrix(x_train[f$val, , drop = FALSE]))
}

errores <- train %>%
  transmute(property_id, cat_localidad, cat_tipo, obj_price,
            estrato = round(num_estrato_200m),
            pred_price = exp(pred_oof),
            error = pred_price - obj_price,
            error_abs = abs(error),
            error_pct = 100 * error / obj_price,
            decil_precio = ntile(obj_price, 10))
saveRDS(errores, file.path(ruta_pred, "oof_xgb_localidad.rds"))

# Resumen de errores por una variable de agrupación.
fun_resumen_errores <- function(df, ...) {
  df %>%
    group_by(...) %>%
    summarise(n = n(),
              precio_medio_millones = mean(obj_price) / 1e6,
              mae_millones = mean(error_abs) / 1e6,
              error_medio_millones = mean(error) / 1e6,
              error_pct_mediano = median(error_pct),
              porc_sobreestima = 100 * mean(error > 0),
              .groups = "drop")
}

errores_total     <- fun_resumen_errores(errores)
errores_localidad <- fun_resumen_errores(errores, cat_localidad) %>% arrange(desc(mae_millones))
errores_estrato   <- fun_resumen_errores(errores, estrato)
errores_tipo      <- fun_resumen_errores(errores, cat_tipo)
errores_decil     <- fun_resumen_errores(errores, decil_precio)

write_csv(errores_total,     file.path(ruta_tablas, "eda_errores_total.csv"))
write_csv(errores_localidad, file.path(ruta_tablas, "eda_errores_localidad.csv"))
write_csv(errores_estrato,   file.path(ruta_tablas, "eda_errores_estrato.csv"))
write_csv(errores_tipo,      file.path(ruta_tablas, "eda_errores_tipo.csv"))
write_csv(errores_decil,     file.path(ruta_tablas, "eda_errores_decil_precio.csv"))

print(errores_total, width = Inf)
print(errores_localidad, n = 25, width = Inf)
print(errores_estrato, width = Inf)
print(errores_tipo, width = Inf)
print(errores_decil, width = Inf)

g_obs_pred <- errores %>%
  ggplot(aes(x = obj_price / 1e6, y = pred_price / 1e6,
             color = cat_localidad == "CHAPINERO")) +
  geom_point(size = 0.3, alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0) +
  scale_color_manual(values = c(`TRUE` = "#D55E00", `FALSE` = "grey60"),
                     labels = c(`TRUE` = "Chapinero", `FALSE` = "Resto de Bogotá")) +
  guides(color = guide_legend(override.aes = list(size = 2, alpha = 1))) +
  labs(title = "Precio observado y predicho fuera de muestra (XGBoost, localidades por fuera)",
       x = "Precio observado (millones)", y = "Precio predicho (millones)", color = NULL) +
  theme_minimal() +
  theme(legend.position = "bottom")
guardar_figura(g_obs_pred, "eda_observado_predicho", ancho = 7, alto = 7)

g_decil <- errores_decil %>%
  ggplot(aes(x = factor(decil_precio), y = error_medio_millones)) +
  geom_col(fill = "#0072B2") +
  geom_hline(yintercept = 0) +
  labs(title = "Error medio por decil de precio (XGBoost, localidades por fuera)",
       subtitle = "Positivo: el modelo sobreestima el precio; negativo: lo subestima",
       x = "Decil de precio observado", y = "Error medio (millones)") +
  theme_minimal()
guardar_figura(g_decil, "eda_error_decil_precio")

# ------------------------------------------------------------------------------
# 3. Importancia de las variables del XGBoost (ganancia)
# ------------------------------------------------------------------------------
importancia <- as_tibble(xgb_guardado$importancia)
write_csv(importancia, file.path(ruta_tablas, "eda_importancia_xgb.csv"))

g_importancia <- importancia %>%
  slice_max(Gain, n = 25) %>%
  mutate(Feature = fct_reorder(Feature, Gain)) %>%
  ggplot(aes(x = Gain, y = Feature)) +
  geom_col(fill = "#0072B2") +
  labs(title = "Variables más importantes del XGBoost",
       x = "Ganancia relativa", y = NULL) +
  theme_minimal()
guardar_figura(g_importancia, "eda_importancia_xgb", alto = 7)

message("Tablas en ", ruta_tablas, " y figuras en ", ruta_figuras)
