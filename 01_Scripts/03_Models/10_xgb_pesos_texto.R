# ==============================================================================
# 10_xgb_pesos_texto.R
# Objetivo: XGBoost del log del precio en el que cada anuncio de train pesa
#           según qué tanto se parece a las propiedades de test (Chapinero).
#           Es el mismo procedimiento de 07_xgb_pesos.R, aplicado a la base que
#           incluye, además de las variables de 09_xgb_pesos_base_final.R, las
#           variables de texto como datos de 01_Data/06_texto_como_datos.R
#           (predicciones del Lasso sobre las palabras y proporciones de los
#           tópicos LDA). Los envíos y las salidas llevan el sufijo _texto para
#           no sobrescribir los de los scripts anteriores.
#           La exploración (02_EDA/01_chapinero_vs_bogota.R) mostró que test
#           difiere mucho del resto de Bogotá en estrato, catastro, comercio y
#           tipo de vivienda, pero se parece a los 305 anuncios de train en
#           Chapinero. Los pesos llevan esa idea al entrenamiento.
#
#           1. Faltantes nativos: XGBoost maneja los NA por sí mismo, así que
#              las variables imputadas con la mediana en 05_merge.R (área,
#              baños, etc.) vuelven a NA. Los ceros del valor de referencia del
#              m2 de terreno también son NA.
#           2. Adversarial validation: un clasificador XGBoost intenta
#              distinguir train de test con características de la vivienda,
#              del texto y del entorno. NO usa coordenadas, localidad, UPZ ni
#              variables que identifican la ubicación casi directamente
#              (galerías, zonas turísticas, distritos creativos, altura,
#              distancias a hitos), porque con ellas separa train y test
#              perfectamente y los pesos no sirven. Las probabilidades de train
#              son fuera de muestra (5 folds).
#           3. Pesos: w = p / (1 - p), recortados en el percentil 99 y
#              normalizados a media 1. Se reportan el tamaño de muestra
#              efectivo y el balance antes y después de ponderar.
#           4. XGBoost ponderado: el peso de entrenamiento es w^gamma, con
#              gamma en {0; 0,5; 1}. gamma = 0 es el modelo sin pesos (control,
#              con faltantes nativos). Se afina con el esquema de localidades
#              y se escoge por el MAE ponderado por w en validación, que estima
#              el error en una muestra con la composición de test. También se
#              reporta el MAE en los anuncios de train en Chapinero cuando
#              Chapinero está fuera del entrenamiento, que es lo más parecido
#              a predecir test.
#           5. Se generan dos envíos: el mejor sin pesos (control) y el mejor
#              con pesos, para comparar en Kaggle si la ponderación ayuda.
# Entradas: 02_Data/Processed/db_{train,test}.rds
# Salidas:  02_Data/Processed/pesos_adversarial_texto.rds (property_id, p, peso)
#           03_Output/Submissions/15_xgb_na_nativo_texto.csv, 16_xgb_pesos_texto.csv
#           03_Output/Models/15_xgb_na_nativo_texto.rds, 16_xgb_pesos_texto.rds
#           03_Output/Models/resultados_modelos.csv (filas de los modelos)
#           03_Output/Tables/pesos_*_texto.csv, xgb_pesos_*_texto.csv
#           03_Output/Figures/pesos_*_texto.{pdf,png},
#           xgb_pesos_afinacion_texto.{pdf,png}
# ==============================================================================

source(here::here("01_Scripts", "00_utils.R"))
pacman::p_load(xgboost, scales, patchwork)

ruta_fig <- here("03_Output", "Figures")
ruta_tab <- here("03_Output", "Tables")
dir.create(ruta_fig, recursive = TRUE, showWarnings = FALSE)
dir.create(ruta_tab, recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------------------------
# 0. Estilo de las gráficas (el mismo de la exploración)
# ------------------------------------------------------------------------------
col_resto      <- "#2a78d6"
col_chap_train <- "#1baf7a"
col_test       <- "#eb6834"
col_gris       <- "#c3c2b7"
col_texto      <- "#52514e"

theme_set(
  theme_minimal(base_size = 11) +
    theme(panel.grid.minor   = element_blank(),
          panel.grid.major   = element_line(color = "#e6e5e0", linewidth = 0.3),
          plot.title         = element_text(face = "bold", size = 12),
          plot.title.position = "plot",
          plot.subtitle      = element_text(color = col_texto, size = 9.5),
          plot.caption       = element_text(color = col_texto, size = 8, hjust = 0),
          plot.caption.position = "plot",
          axis.text          = element_text(color = col_texto),
          strip.text         = element_text(face = "bold", hjust = 0),
          legend.position    = "top",
          legend.justification = "left",
          legend.title       = element_blank())
)

dispositivo_pdf <- if (capabilities("cairo")) cairo_pdf else "pdf"

fun_guardar <- function(p, nombre, ancho = 8, alto = 5) {
  ggsave(file.path(ruta_fig, paste0(nombre, ".pdf")), p,
         width = ancho, height = alto, device = dispositivo_pdf)
  ggsave(file.path(ruta_fig, paste0(nombre, ".png")), p,
         width = ancho, height = alto, dpi = 300, bg = "white")
}

nucleos <- max(1, parallel::detectCores() - 1)

# Ronda óptima de un modelo con parada temprana (base 1), igual que en
# 06_xgboost.R.
fun_mejor_ronda <- function(modelo) {
  ronda <- attributes(modelo)$early_stop$best_iteration
  if (is.null(ronda)) ronda <- as.integer(xgb.attr(modelo, "best_iteration")) + 1
  as.integer(ronda)
}

# AUC a partir de rangos (equivale al estadístico de Mann-Whitney).
fun_auc <- function(y, p) {
  r  <- rank(p)
  n1 <- sum(y == 1)
  n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# ------------------------------------------------------------------------------
# 1. Datos con faltantes nativos
# Cada dummy bin_<x>_na de 05_merge.R marca las filas donde num_<x> fue
# imputada con la mediana; en esas filas num_<x> vuelve a NA.
# ------------------------------------------------------------------------------
datos <- fun_datos_modelo()
folds <- fun_todos_folds(datos$train)

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
n_test  <- nrow(test_x)

message("Faltantes restaurados (porcentaje en train):")
print(round(100 * colMeans(is.na(train_x[datos$predictores]))[
  colMeans(is.na(train_x[datos$predictores])) > 0], 1))

# Matrices del modelo de precios, con los mismos predictores de 06_xgboost.R.
# `na.pass` es necesario: por defecto model.matrix elimina las filas con NA.
fml_x  <- reformulate(datos$predictores, intercept = FALSE)
todo   <- bind_rows(train_x[datos$predictores], test_x[datos$predictores])
x_todo <- model.matrix(fml_x, model.frame(fml_x, data = todo, na.action = na.pass))
x_train <- x_todo[seq_len(n_train), , drop = FALSE]
x_test  <- x_todo[-seq_len(n_train), , drop = FALSE]
y_train <- train_x$obj_log_price

# ------------------------------------------------------------------------------
# 2. Adversarial validation: ¿qué tanto se parece cada anuncio de train a test?
# ------------------------------------------------------------------------------
vars_adv <- c(
  # Vivienda
  "bin_casa", "num_area", "num_habitac", "num_banos", "num_parqueaderos",
  "num_piso_texto", "num_area_por_habitac",
  # Texto
  "num_palabras", "bin_terraza", "bin_chimenea", "bin_vista", "bin_exterior",
  "bin_remodelado", "bin_ascensor", "bin_gimnasio", "bin_conjunto",
  "bin_jardin", "bin_bbq", "bin_vigilancia", "bin_piscina", "bin_duplex",
  "bin_penthouse", "bin_estudio", "bin_deposito",
  # Entorno: calidad del vecindario y del mercado, no hitos de ubicación
  "num_estrato_200m", "num_vref_m2_200m", "num_avaluo_ph_m2_200m",
  "num_pisos_200m", "num_predios_mz_200m", "num_prop_comer_300m",
  "num_prop_resid_300m", "num_restaurantes_500m", "num_bares_500m",
  "num_estab_500m", "num_gastro_300m", "num_ips_1km", "num_colegios_1km",
  "num_luminarias_ha", "num_atracos_celda", "num_arboles_300m",
  "num_dist_parque", "num_paraderos_300m"
)
faltan <- setdiff(vars_adv, names(train_x))
if (length(faltan) > 0) message("Variables no encontradas (se omiten): ", paste(faltan, collapse = ", "))
vars_adv <- intersect(vars_adv, names(train_x))

x_adv <- as.matrix(bind_rows(train_x[vars_adv], test_x[vars_adv]))
y_adv <- c(rep(0, n_train), rep(1, n_test))

param_adv <- list(objective = "binary:logistic", eval_metric = "auc",
                  eta = 0.05, max_depth = 4, min_child_weight = 5,
                  subsample = 0.8, colsample_bytree = 0.7, nthread = nucleos)

set.seed(semilla)
folds_adv <- createFolds(factor(y_adv), k = 5)
p_adv <- rep(NA_real_, length(y_adv))
rondas_adv <- integer(length(folds_adv))

message("Clasificador adversarial (5 folds)")
for (i in seq_along(folds_adv)) {
  val <- folds_adv[[i]]
  d_tr  <- xgb.DMatrix(x_adv[-val, , drop = FALSE], label = y_adv[-val])
  d_val <- xgb.DMatrix(x_adv[val, , drop = FALSE],  label = y_adv[val])
  set.seed(semilla)
  m <- xgb.train(params = param_adv, data = d_tr, nrounds = 2000,
                 evals = list(val = d_val), early_stopping_rounds = 50, verbose = 0)
  rondas_adv[i] <- fun_mejor_ronda(m)
  p_adv[val] <- predict(m, d_val, iterationrange = c(1, rondas_adv[i]))
}

auc_adv <- fun_auc(y_adv, p_adv)
message("AUC del clasificador adversarial (fuera de muestra): ", round(auc_adv, 3))

# Clasificador con todos los datos, solo para ver qué variables separan
# train de test.
set.seed(semilla)
m_adv_final <- xgb.train(params = param_adv,
                         data = xgb.DMatrix(x_adv, label = y_adv),
                         nrounds = round(mean(rondas_adv)), verbose = 0)
importancia_adv <- xgb.importance(model = m_adv_final)

# ------------------------------------------------------------------------------
# 3. Pesos
# w = p / (1 - p) es proporcional a la razón de densidades test / train. Se
# recorta en el percentil 99 para que unos pocos anuncios no dominen y se
# normaliza a media 1.
# ------------------------------------------------------------------------------
p_train <- pmin(pmax(p_adv[seq_len(n_train)], 1e-4), 1 - 1e-4)
w_bruto <- p_train / (1 - p_train)
tope    <- quantile(w_bruto, 0.99)
w       <- pmin(w_bruto, tope)
w       <- w / mean(w)

ess <- sum(w)^2 / sum(w^2)
message("Tamaño de muestra efectivo con pesos: ", round(ess), " de ", n_train,
        " (", round(100 * ess / n_train, 1), " %)")

saveRDS(tibble(property_id = train_x$property_id, p_test = p_train, peso = w),
        here("02_Data", "Processed", "pesos_adversarial_texto.rds"))

# Participación de cada localidad en el número de anuncios y en el peso total.
pesos_loc <- tibble(localidad = str_to_title(train_x$cat_localidad), w = w) %>%
  group_by(localidad) %>%
  summarise(anuncios = n(),
            pct_anuncios = 100 * n() / n_train,
            pct_peso = 100 * sum(w) / sum(.env$w),
            peso_medio = mean(w), .groups = "drop") %>%
  arrange(desc(pct_peso))
write_csv(pesos_loc, file.path(ruta_tab, "pesos_por_localidad_texto.csv"))
print(pesos_loc, n = 20)

# Balance antes y después de ponderar (diferencia estandarizada con la
# desviación estándar combinada sin ponderar, para que sean comparables).
vars_balance <- c(
  bin_casa              = "Es casa",
  num_habitac           = "Habitaciones",
  num_area              = "Área (m²)",
  num_estrato_200m      = "Estrato (200 m)",
  num_vref_m2_200m      = "Valor de referencia m² terreno (200 m)",
  num_avaluo_ph_m2_200m = "Avalúo PH por m² (200 m)",
  num_pisos_200m        = "Pisos de las construcciones (200 m)",
  num_prop_comer_300m   = "Proporción de uso comercial (300 m)",
  num_prop_resid_300m   = "Proporción de uso residencial (300 m)",
  num_restaurantes_500m = "Restaurantes (500 m)",
  num_bares_500m        = "Bares (500 m)",
  num_ips_1km           = "IPS (1 km)",
  num_colegios_1km      = "Colegios (1 km)",
  # Estas dos no entran al clasificador: sirven para ver si la ponderación
  # también acerca variables que no usó.
  num_galerias_1km      = "Galerías (1 km) *",
  num_dist_zona_turistica = "Distancia a zona turística *"
)

balance_pesos <- imap_dfr(vars_balance, function(etiqueta, v) {
  a <- test_x[[v]]
  b <- train_x[[v]]
  ok <- !is.na(b)
  sd_comb <- sqrt((var(a, na.rm = TRUE) + var(b, na.rm = TRUE)) / 2)
  tibble(variable = v, etiqueta = etiqueta,
         sin_pesos = (mean(a, na.rm = TRUE) - mean(b, na.rm = TRUE)) / sd_comb,
         con_pesos = (mean(a, na.rm = TRUE) - weighted.mean(b[ok], w[ok])) / sd_comb)
})
write_csv(balance_pesos, file.path(ruta_tab, "pesos_balance_texto.csv"))
print(balance_pesos)

# ------------------------------------------------------------------------------
# 4. Figuras de los pesos
# ------------------------------------------------------------------------------
# 4.1 Distribución de la probabilidad de "parecerse a test".
p_prob <- tibble(p = p_adv,
                 grupo = c(if_else(train_x$cat_localidad == "CHAPINERO",
                                   "Chapinero (train)", "Resto de Bogotá (train)"),
                           rep("Test (Chapinero)", n_test))) %>%
  mutate(grupo = factor(grupo, levels = c("Resto de Bogotá (train)",
                                          "Chapinero (train)", "Test (Chapinero)"))) %>%
  ggplot(aes(p, color = grupo, fill = grupo)) +
  geom_density(alpha = 0.12, linewidth = 0.8, adjust = 1.2) +
  scale_color_manual(values = c(col_resto, col_chap_train, col_test)) +
  scale_fill_manual(values = c(col_resto, col_chap_train, col_test)) +
  labs(title = "¿Qué tanto se parece cada anuncio a test?",
       subtitle = paste0("Probabilidad fuera de muestra de pertenecer a test según el clasificador adversarial. AUC = ",
                         number(auc_adv, accuracy = 0.001, decimal.mark = ",")),
       x = "Probabilidad de pertenecer a test", y = "Densidad",
       caption = "El clasificador no usa coordenadas, localidad, UPZ ni hitos de ubicación.")
fun_guardar(p_prob, "pesos_probabilidad_test_texto", ancho = 8, alto = 4.5)

# 4.2 Variables que más separan train de test.
p_imp_adv <- importancia_adv %>%
  slice_max(Gain, n = 15) %>%
  mutate(Feature = fct_reorder(Feature, Gain)) %>%
  ggplot(aes(Gain, Feature)) +
  geom_col(fill = col_resto, width = 0.65) +
  scale_x_continuous(labels = label_percent(), expand = expansion(mult = c(0, 0.05))) +
  labs(title = "Variables que más distinguen a test de train",
       subtitle = "Importancia (ganancia) en el clasificador adversarial",
       x = "Participación en la ganancia total", y = NULL)
fun_guardar(p_imp_adv, "pesos_importancia_adversarial_texto", ancho = 7.5, alto = 5)

# 4.3 Dónde queda el peso: participación por localidad.
p_loc <- pesos_loc %>%
  slice_max(pct_peso, n = 10) %>%
  mutate(localidad = fct_reorder(localidad, pct_peso)) %>%
  pivot_longer(c(pct_anuncios, pct_peso), names_to = "medida", values_to = "pct") %>%
  mutate(medida = dplyr::recode(medida, pct_anuncios = "% de los anuncios de train",
                         pct_peso = "% del peso total")) %>%
  ggplot(aes(pct, localidad, fill = medida)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.7) +
  scale_fill_manual(values = c("% de los anuncios de train" = col_gris,
                               "% del peso total" = col_test)) +
  labs(title = "Con los pesos, el entrenamiento se inclina hacia las zonas parecidas a Chapinero",
       subtitle = paste0("Las 10 localidades con más peso. Tamaño de muestra efectivo: ",
                         number(ess, big.mark = "."), " de ", number(n_train, big.mark = "."), " anuncios"),
       x = "%", y = NULL)
fun_guardar(p_loc, "pesos_por_localidad_texto", ancho = 8, alto = 5)

# 4.4 Balance antes y después de ponderar.
p_bal <- balance_pesos %>%
  mutate(etiqueta = fct_reorder(etiqueta, abs(sin_pesos))) %>%
  pivot_longer(c(sin_pesos, con_pesos), names_to = "momento", values_to = "dif") %>%
  mutate(momento = dplyr::recode(momento, sin_pesos = "Sin pesos", con_pesos = "Con pesos")) %>%
  ggplot(aes(dif, etiqueta)) +
  annotate("rect", xmin = -0.25, xmax = 0.25, ymin = -Inf, ymax = Inf, fill = "#f0efec") +
  geom_vline(xintercept = 0, color = col_texto, linewidth = 0.3) +
  geom_line(aes(group = etiqueta), color = col_gris, linewidth = 0.8) +
  geom_point(aes(color = momento, shape = momento), size = 2.6) +
  scale_color_manual(values = c("Sin pesos" = col_resto, "Con pesos" = col_test)) +
  scale_shape_manual(values = c("Sin pesos" = 16, "Con pesos" = 17)) +
  labs(title = "Los pesos acercan train a test",
       subtitle = "Diferencia estandarizada de medias entre test y train, antes y después de ponderar",
       x = "Diferencia estandarizada (positivo: mayor en test)", y = NULL,
       caption = "* Variables que no usa el clasificador. La banda gris marca |d| < 0,25.")
fun_guardar(p_bal, "pesos_balance_texto", ancho = 8, alto = 5.5)

# ------------------------------------------------------------------------------
# 5. XGBoost ponderado: afinación con el esquema de localidades
# En cada fold, el peso de entrenamiento es w^gamma (normalizado a media 1) y
# la parada temprana usa el MAE ponderado por w en el fold de validación.
# ------------------------------------------------------------------------------
es_chap <- train_x$cat_localidad == "CHAPINERO"

parametros_base <- list(objective = "reg:squarederror", eval_metric = "mae",
                        eta = 0.05, subsample = 0.8, colsample_bytree = 0.7,
                        nthread = nucleos)

fun_peso_entreno <- function(gamma) {
  pe <- w^gamma
  pe / mean(pe)
}

# Entrena en cada fold. Con `rondas = NULL` usa parada temprana; con un número
# fijo entrena exactamente esas rondas. Devuelve las predicciones fuera de
# muestra (log) y las rondas.
fun_cv_xgb <- function(parametros, folds, peso, rondas = NULL, max_rondas = 3000) {
  pred <- rep(NA_real_, n_train)
  rondas_fold <- integer(length(folds))
  for (i in seq_along(folds)) {
    f <- folds[[i]]
    d_tr  <- xgb.DMatrix(x_train[f$train, , drop = FALSE], label = y_train[f$train],
                         weight = peso[f$train])
    d_val <- xgb.DMatrix(x_train[f$val, , drop = FALSE], label = y_train[f$val],
                         weight = w[f$val])
    set.seed(semilla)
    if (is.null(rondas)) {
      m <- xgb.train(params = parametros, data = d_tr, nrounds = max_rondas,
                     evals = list(val = d_val), early_stopping_rounds = 50, verbose = 0)
      rondas_fold[i] <- fun_mejor_ronda(m)
    } else {
      m <- xgb.train(params = parametros, data = d_tr, nrounds = rondas, verbose = 0)
      rondas_fold[i] <- rondas
    }
    pred[f$val] <- predict(m, d_val, iterationrange = c(1, rondas_fold[i]))
  }
  list(pred = pred, rondas = rondas_fold)
}

# Métricas en pesos de las predicciones fuera de muestra:
# - mae: MAE sin ponderar (comparable con resultados_modelos.csv);
# - mae_pond: MAE ponderado por w, estimación del MAE en una muestra como test;
# - mae_chap: MAE en los anuncios de train en Chapinero;
# - sesgo_chap y pct_sobre_chap: error promedio (predicho - observado) y
#   porcentaje de sobreestimaciones en Chapinero, para la Etapa 4.
fun_metricas <- function(pred) {
  ok  <- !is.na(pred)
  err <- exp(pred) - exp(y_train)
  tibble(mae            = mean(abs(err[ok])),
         mae_pond       = weighted.mean(abs(err[ok]), w[ok]),
         mae_chap       = mean(abs(err[ok & es_chap])),
         sesgo_chap     = mean(err[ok & es_chap]),
         pct_sobre_chap = 100 * mean(err[ok & es_chap] > 0))
}

grid <- expand.grid(gamma_peso = c(0, 0.5, 1),
                    max_depth = c(6, 8),
                    min_child_weight = c(1, 5))

message("Afinando con el esquema: ", etiquetas_cv[[esquema_seleccion]],
        " (", nrow(grid), " combinaciones)")
tabla_afinacion <- pmap_dfr(grid, function(gamma_peso, max_depth, min_child_weight) {
  parametros <- c(parametros_base, list(max_depth = max_depth,
                                        min_child_weight = min_child_weight))
  r <- fun_cv_xgb(parametros, folds[[esquema_seleccion]], fun_peso_entreno(gamma_peso))
  met <- fun_metricas(r$pred)
  message(sprintf("  gamma = %.1f, max_depth = %d, min_child_weight = %d: MAE %.1f | ponderado %.1f | Chapinero %.1f millones",
                  gamma_peso, max_depth, min_child_weight,
                  met$mae / 1e6, met$mae_pond / 1e6, met$mae_chap / 1e6))
  bind_cols(tibble(gamma_peso, max_depth, min_child_weight,
                   rondas = round(mean(r$rondas))), met)
}) %>%
  arrange(mae_pond)

write_csv(tabla_afinacion, file.path(ruta_tab, "xgb_pesos_afinacion_texto.csv"))
print(tabla_afinacion %>% mutate(across(c(mae, mae_pond, mae_chap, sesgo_chap), ~ round(.x / 1e6, 1))))

# Figura: efecto de gamma en las tres métricas de error.
p_afin <- tabla_afinacion %>%
  mutate(config = paste0("max_depth = ", max_depth, ", min_child_weight = ", min_child_weight)) %>%
  pivot_longer(c(mae, mae_pond, mae_chap), names_to = "metrica", values_to = "valor") %>%
  mutate(metrica = factor(dplyr::recode(metrica,
                                 mae = "MAE sin ponderar",
                                 mae_pond = "MAE ponderado (composición de test)",
                                 mae_chap = "MAE en Chapinero (train, fuera de muestra)"),
                          levels = c("MAE sin ponderar",
                                     "MAE ponderado (composición de test)",
                                     "MAE en Chapinero (train, fuera de muestra)"))) %>%
  ggplot(aes(gamma_peso, valor / 1e6, color = config)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.2) +
  facet_wrap(~ metrica, scales = "free_y") +
  scale_x_continuous(breaks = c(0, 0.5, 1)) +
  scale_color_manual(values = c("#2a78d6", "#eb6834", "#1baf7a", "#eda100")) +
  labs(title = "¿Ayuda ponderar por parecido a test?",
       subtitle = "Validación cruzada dejando localidades completas por fuera. gamma = 0: sin pesos; gamma = 1: pesos completos",
       x = "Exponente de los pesos (gamma)", y = "Millones de COP") +
  guides(color = guide_legend(nrow = 2))
fun_guardar(p_afin, "xgb_pesos_afinacion_texto", ancho = 11, alto = 4.8)

# ------------------------------------------------------------------------------
# 6. Modelos finales: el mejor sin pesos (control) y el mejor con pesos
# Para cada uno: MAE sin ponderar con los cuatro esquemas (para la tabla de
# resultados), métricas ponderadas por esquema, ajuste con todo train, envío
# y registro.
# ------------------------------------------------------------------------------
fun_finalizar <- function(config, envio) {
  parametros <- c(parametros_base, list(max_depth = config$max_depth,
                                        min_child_weight = config$min_child_weight))
  peso <- fun_peso_entreno(config$gamma_peso)
  hiperparam <- data.frame(gamma_peso = config$gamma_peso,
                           max_depth = config$max_depth,
                           min_child_weight = config$min_child_weight,
                           eta = parametros_base$eta,
                           rondas = config$rondas)
  hiperparam_texto <- fun_texto_hiperparam(hiperparam)
  message("\n", envio, ": ", hiperparam_texto)

  metricas_esquemas <- imap_dfr(folds, function(f, esquema) {
    r <- fun_cv_xgb(parametros, f, peso, rondas = config$rondas)
    bind_cols(tibble(esquema = esquema), fun_metricas(r$pred))
  })
  mae <- metricas_esquemas %>% select(esquema, mae_cop = mae)

  set.seed(semilla)
  final <- xgb.train(params = parametros,
                     data = xgb.DMatrix(x_train, label = y_train, weight = peso),
                     nrounds = config$rondas, verbose = 0)
  pred_test <- predict(final, xgb.DMatrix(x_test))
  fun_submission(datos$test$property_id, pred_test, envio)

  importancia <- xgb.importance(model = final)
  saveRDS(list(envio = envio, familia = "XGBoost", hiperparam = hiperparam,
               tabla_afinacion = tabla_afinacion, mae = mae,
               metricas_esquemas = metricas_esquemas, pred_test = pred_test,
               importancia = importancia, auc_adversarial = auc_adv, ess = ess),
          here("03_Output", "Models", paste0(envio, ".rds")))
  fun_registrar_resultado(envio, "XGBoost", hiperparam_texto, mae, envio)

  print(metricas_esquemas %>% mutate(across(c(mae, mae_pond, mae_chap, sesgo_chap),
                                            ~ round(.x / 1e6, 1))))
  message("Descripción para Kaggle: ",
          fun_descripcion_kaggle("XGBoost", hiperparam_texto, mae))
  invisible(list(final = final, metricas = metricas_esquemas, importancia = importancia))
}

mejor_control <- tabla_afinacion %>% filter(gamma_peso == 0) %>% slice(1)
mejor_pesos   <- tabla_afinacion %>% filter(gamma_peso > 0)  %>% slice(1)

res_control <- fun_finalizar(mejor_control, "15_xgb_na_nativo_texto")
res_pesos   <- fun_finalizar(mejor_pesos,   "16_xgb_pesos_texto")

# Comparación final en consola.
comparacion <- bind_rows(
  res_control$metricas %>% mutate(modelo = "15_xgb_na_nativo_texto"),
  res_pesos$metricas   %>% mutate(modelo = "16_xgb_pesos_texto")
) %>%
  select(modelo, esquema, mae, mae_pond, mae_chap, sesgo_chap, pct_sobre_chap)
write_csv(comparacion, file.path(ruta_tab, "xgb_pesos_comparacion_texto.csv"))
print(comparacion %>% mutate(across(c(mae, mae_pond, mae_chap, sesgo_chap), ~ round(.x / 1e6, 1))),
      n = 20)

message("Listo. Sube a Kaggle 15_xgb_na_nativo_texto.csv y 16_xgb_pesos_texto.csv y anota el puntaje público de cada uno.")
