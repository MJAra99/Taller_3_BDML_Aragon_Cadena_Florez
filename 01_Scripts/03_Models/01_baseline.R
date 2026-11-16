# ==============================================================================
# 01_baseline.R
# Objetivo: primeros cuatro envíos a Kaggle con la base mínima (sin variables
#           externas ni de texto): regresión lineal, Elastic Net, CART y
#           Random Forest. Todos predicen el log del precio y se evalúan con
#           los mismos 5 folds aleatorios.
# Entradas: 02_Data/Raw/Kaggle/{train,test}.csv
# Salidas:  03_Output/Submissions/01_lm_base.csv, 02_enet_base.csv,
#           03_cart_base.csv, 04_rf_base.csv
#           03_Output/Models/baseline_resumen.csv (MAE de validación cruzada)
# ==============================================================================

source(here::here("01_Scripts", "00_utils.R"))
pacman::p_load(glmnet, rpart, ranger)

# ------------------------------------------------------------------------------
# 1. Datos
# ------------------------------------------------------------------------------
raw      <- fun_load_raw()
db_train <- fun_clean_base(raw$train, ref = raw$train)
db_test  <- fun_clean_base(raw$test,  ref = raw$train)

fml <- obj_log_price ~ num_habitac + num_banos + bin_banos_na + cat_tipo +
  num_lat + num_lon + num_anio + num_mes

# Mismos folds para los cuatro modelos.
folds <- fun_folds(db_train$obj_log_price)
ctrl  <- trainControl(method = "cv", index = folds,
                      savePredictions = "final")

# ------------------------------------------------------------------------------
# 2. Regresión lineal
# ------------------------------------------------------------------------------
set.seed(semilla)
m_lm <- train(fml, data = db_train, method = "lm",
              trControl = ctrl, metric = "MAE")

# ------------------------------------------------------------------------------
# 3. Elastic Net (predictores estandarizados)
# ------------------------------------------------------------------------------
set.seed(semilla)
m_enet <- train(fml, data = db_train, method = "glmnet",
                trControl = ctrl, metric = "MAE",
                preProcess = c("center", "scale"),
                tuneGrid = expand.grid(alpha  = seq(0, 1, by = 0.25),
                                       lambda = 10^seq(-4, -1, length.out = 10)))

# ------------------------------------------------------------------------------
# 4. CART (se ajusta el parámetro de complejidad cp)
# ------------------------------------------------------------------------------
set.seed(semilla)
m_cart <- train(fml, data = db_train, method = "rpart",
                trControl = ctrl, metric = "MAE",
                tuneGrid = expand.grid(cp = 10^seq(-5, -2, length.out = 10)))

# ------------------------------------------------------------------------------
# 5. Random Forest (grilla pequeña de mtry y tamaño mínimo de hoja)
# ------------------------------------------------------------------------------
set.seed(semilla)
m_rf <- train(fml, data = db_train, method = "ranger",
              trControl = ctrl, metric = "MAE",
              num.trees = 300,
              tuneGrid = expand.grid(mtry = c(2, 4, 6),
                                     splitrule = "variance",
                                     min.node.size = c(5, 20)))

# ------------------------------------------------------------------------------
# 6. Resumen de validación cruzada (MAE en pesos) y envíos
# ------------------------------------------------------------------------------
modelos <- list(`01_lm_base`   = m_lm,
                `02_enet_base` = m_enet,
                `03_cart_base` = m_cart,
                `04_rf_base`   = m_rf)

resumen <- imap_dfr(modelos, function(m, nombre) {
  tibble(envio         = nombre,
         hiperparam    = paste(names(m$bestTune),
                               sapply(m$bestTune, as.character), sep = " = ",
                               collapse = ", "),
         mae_cv_cop    = fun_mae_cv(m))
})
print(resumen)
write_csv(resumen, here("03_Output", "Models", "baseline_resumen.csv"))

iwalk(modelos, function(m, nombre) {
  fun_submission(db_test$property_id, predict(m, newdata = db_test), nombre)
})
