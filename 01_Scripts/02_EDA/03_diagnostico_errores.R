# ==============================================================================
# 03_diagnostico_errores.R
# Objetivo: entender dónde y por qué se equivoca el mejor XGBoost (envío 16,
#           con pesos y texto como datos; el 18 es el mismo modelo x 1,03) y
#           qué tanto de eso se traslada a test.
#           1. Reconstruye las predicciones fuera de muestra del modelo 16 con
#              dos esquemas: aleatoria por grupos de anuncios y localidades
#              completas por fuera (mismos hiperparámetros, pesos y faltantes
#              nativos de 03_Models/10_xgb_pesos_texto.R).
#           2. ¿Qué validación sigue a Kaggle? Compara el MAE de cada esquema
#              (sin ponderar, ponderado por parecido con test y solo en
#              Chapinero) con el puntaje público de los envíos.
#           3. Sesgo: subestimación o sobreestimación por decil del precio
#              OBSERVADO y por decil del precio PREDICHO. Lo primero siempre
#              muestra regresión a la media; lo segundo es lo que importa para
#              comprar, porque al decidir solo se conoce la predicción.
#           4. Curva de calibración (observado promedio vs predicho).
#           5. Error y sesgo por segmentos donde test difiere de train: área
#              faltante, precisión de las coordenadas, tipo de inmueble,
#              estrato y parecido con test. Cada segmento muestra su peso en
#              train y en test.
#           6. MAE esperado en test: los errores fuera de muestra se
#              reponderan para que train tenga la composición de test en esos
#              segmentos, y se compara con Kaggle.
#           7. Factor de ajuste del sesgo: MAE fuera de muestra por factor
#              frente a los tres puntos que da Kaggle (1; 1,03; 1,08), con el
#              factor óptimo que implican esos tres puntos.
#           8. Mapa del sesgo (hexágonos) con la ubicación de test.
#           9. Lente de negocio: distribución de predicho / observado y
#              porcentaje de sobreestimaciones grandes.
#           10. Los 40 peores errores con el texto del anuncio, para revisar a
#               mano si son errores del modelo o de los datos.
# Entradas: 02_Data/Processed/db_{train,test}.rds
#           02_Data/Processed/pesos_adversarial_texto.rds
#           02_Data/Raw/Kaggle/{train,test}.csv (precisión de coordenadas y texto)
#           03_Output/Models/16_xgb_pesos_texto.rds
#           03_Output/Models/resultados_modelos.csv
#           03_Output/Tables/xgb_pesos_comparacion_{base_final,texto}.csv
#           02_Data/Raw/Kaggle/kaggle_puntajes.csv (puntajes de Kaggle)
# Salidas:  03_Output/Predictions/oof_16_{aleatoria,localidad}.rds
#           03_Output/Tables/diag_*.csv
#           03_Output/Figures/diag_*.{pdf,png}
# ==============================================================================

source(here::here("01_Scripts", "00_utils.R"))
pacman::p_load(xgboost, scales, patchwork, hexbin)

ruta_fig  <- here("03_Output", "Figures")
ruta_tab  <- here("03_Output", "Tables")
ruta_pred <- here("03_Output", "Predictions")
walk(c(ruta_fig, ruta_tab, ruta_pred), dir.create, recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------------------------
# 0. Estilo de las gráficas (el mismo de la exploración)
# ------------------------------------------------------------------------------
col_resto      <- "#2a78d6"
col_chap_train <- "#1baf7a"
col_test       <- "#eb6834"
col_gris       <- "#c3c2b7"
col_texto      <- "#52514e"
col_esquemas   <- c("Aleatoria" = col_resto, "Localidades" = col_test)

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

millones   <- label_number(scale = 1e-6, big.mark = ".", decimal.mark = ",", accuracy = 1)
porcentaje <- label_percent(accuracy = 1, decimal.mark = ",")
nucleos    <- max(1, parallel::detectCores() - 1)

# ------------------------------------------------------------------------------
# 1. Datos, pesos e hiperparámetros del modelo 16
# ------------------------------------------------------------------------------
datos <- fun_datos_modelo()

# Igual que en 10_xgb_pesos_texto.R: las variables imputadas vuelven a NA.
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

fml_x   <- reformulate(datos$predictores, intercept = FALSE)
todo    <- bind_rows(train_x[datos$predictores], test_x[datos$predictores])
x_todo  <- model.matrix(fml_x, model.frame(fml_x, data = todo, na.action = na.pass))
x_train <- x_todo[seq_len(n_train), , drop = FALSE]
y_train <- train_x$obj_log_price
precio  <- train_x$obj_price

pesos_adv <- readRDS(here("02_Data", "Processed", "pesos_adversarial_texto.rds"))
w <- train_x %>% select(property_id) %>% left_join(pesos_adv, by = "property_id") %>% pull(peso)
p_test_train <- train_x %>% select(property_id) %>% left_join(pesos_adv, by = "property_id") %>% pull(p_test)
stopifnot(!anyNA(w))

hp <- readRDS(here("03_Output", "Models", "16_xgb_pesos_texto.rds"))$hiperparam
peso_entreno <- w^hp$gamma_peso
peso_entreno <- peso_entreno / mean(peso_entreno)

# Precisión de las coordenadas en los datos crudos: número de decimales de la
# latitud y la longitud tal como vienen escritas en el csv (sin ceros a la
# derecha). Con 3 decimales la ubicación es una celda de unos 110 m; con 2, de
# más de 1 km.
fun_decimales <- function(x) {
  d <- str_extract(x, "(?<=\\.)\\d+")
  d <- str_remove(replace_na(d, ""), "0+$")
  nchar(d)
}

fun_crudos <- function(archivo) {
  read_csv(here("02_Data", "Raw", "Kaggle", archivo), show_col_types = FALSE,
           col_types = cols(lat = col_character(), lon = col_character(),
                            .default = col_guess())) %>%
    transmute(property_id,
              num_dec_coord = pmin(fun_decimales(lat), fun_decimales(lon)),
              title, description)
}
crudo_train <- fun_crudos("train.csv")
crudo_test  <- fun_crudos("test.csv")

fun_grupo_precision <- function(d) {
  factor(case_when(d <= 2 ~ "2 decimales o menos (~1 km)",
                   d == 3 ~ "3 decimales (~110 m)",
                   TRUE   ~ "4 o más decimales"),
         levels = c("2 decimales o menos (~1 km)", "3 decimales (~110 m)", "4 o más decimales"))
}

# Segmentos donde test difiere de train (se usan en train y en test).
fun_segmentos <- function(df, crudo) {
  df %>%
    select(property_id, cat_tipo, bin_area_na, num_estrato_200m) %>%
    left_join(crudo %>% select(property_id, num_dec_coord), by = "property_id") %>%
    mutate(seg_area      = if_else(bin_area_na == 1, "Sin área", "Con área"),
           seg_precision = fun_grupo_precision(num_dec_coord),
           seg_tipo      = as.character(cat_tipo),
           seg_estrato   = case_when(is.na(num_estrato_200m) ~ "Sin dato",
                                     num_estrato_200m < 3.5 ~ "Estrato < 3,5",
                                     num_estrato_200m < 4.5 ~ "Estrato 3,5 a 4,5",
                                     num_estrato_200m < 5.5 ~ "Estrato 4,5 a 5,5",
                                     TRUE                   ~ "Estrato 5,5 o más"))
}
seg_train <- fun_segmentos(train_x, crudo_train)
seg_test  <- fun_segmentos(test_x,  crudo_test)

# ------------------------------------------------------------------------------
# 2. Predicciones fuera de muestra del modelo 16 (rondas fijas)
# ------------------------------------------------------------------------------
parametros <- list(objective = "reg:squarederror", eval_metric = "mae",
                   eta = hp$eta, subsample = 0.8, colsample_bytree = 0.7,
                   max_depth = hp$max_depth, min_child_weight = hp$min_child_weight,
                   nthread = nucleos)

fun_oof <- function(esquema) {
  folds <- fun_folds_cv(datos$train, esquema)
  pred <- rep(NA_real_, n_train)
  for (i in seq_along(folds)) {
    f <- folds[[i]]
    message("  ", esquema, ": fold ", i, " de ", length(folds))
    set.seed(semilla)
    m <- xgb.train(params = parametros,
                   data = xgb.DMatrix(x_train[f$train, , drop = FALSE],
                                      label = y_train[f$train], weight = peso_entreno[f$train]),
                   nrounds = hp$rondas, verbose = 0)
    pred[f$val] <- predict(m, xgb.DMatrix(x_train[f$val, , drop = FALSE]))
  }
  pred
}

message("Predicciones fuera de muestra del modelo 16")
oof <- map_dfr(c(aleatoria = "aleatoria", localidad = "localidad"), function(esq) {
  pred_log <- fun_oof(esq)
  r <- tibble(property_id = train_x$property_id,
              esquema     = esq,
              obs         = precio,
              pred        = exp(pred_log))
  saveRDS(r, file.path(ruta_pred, paste0("oof_16_", esq, ".rds")))
  r
}) %>%
  mutate(esquema   = dplyr::recode(esquema, aleatoria = "Aleatoria", localidad = "Localidades"),
         error     = pred - obs,                 # positivo = sobreestima
         error_pct = pred / obs - 1,
         error_abs = abs(error),
         w         = rep(w, 2),
         p_test    = rep(p_test_train, 2),
         chapinero = rep(train_x$cat_localidad == "CHAPINERO", 2)) %>%
  left_join(seg_train, by = "property_id") %>%
  filter(!is.na(pred))

# ------------------------------------------------------------------------------
# 3. ¿Qué validación sigue a Kaggle?
# Puntajes público y privado de cada envío, copiados de la página de envíos
# de la competencia (02_Data/Raw/Kaggle/kaggle_puntajes.csv).
# ------------------------------------------------------------------------------
kaggle <- read_csv(here("02_Data", "Raw", "Kaggle", "kaggle_puntajes.csv"),
                   show_col_types = FALSE)

# MAE de los pares con y sin pesos (13 vs 14 y 15 vs 16) con las tres métricas
# de cada esquema, frente a Kaggle. Si un esquema ordena bien los pares, sirve
# para escoger modelos.
comparaciones <- bind_rows(
  read_csv(file.path(ruta_tab, "xgb_pesos_comparacion_base_final.csv"), show_col_types = FALSE),
  read_csv(file.path(ruta_tab, "xgb_pesos_comparacion_texto.csv"), show_col_types = FALSE)
) %>%
  left_join(kaggle, by = c("modelo" = "envio")) %>%
  mutate(par = if_else(str_detect(modelo, "texto"), "Base + texto (15 vs 16)",
                       "Base final (13 vs 14)"),
         version = if_else(str_detect(modelo, "pesos"), "Con pesos", "Sin pesos"))

tabla_sigue_kaggle <- comparaciones %>%
  select(par, version, esquema, mae, mae_pond, mae_chap, kaggle_publico) %>%
  pivot_longer(c(mae, mae_pond, mae_chap), names_to = "metrica", values_to = "valor") %>%
  group_by(par, esquema, metrica) %>%
  summarise(
    dif_cv     = valor[version == "Con pesos"] - valor[version == "Sin pesos"],
    dif_kaggle = kaggle_publico[version == "Con pesos"] - kaggle_publico[version == "Sin pesos"],
    .groups = "drop") %>%
  mutate(acierta = sign(dif_cv) == sign(dif_kaggle))
write_csv(tabla_sigue_kaggle, file.path(ruta_tab, "diag_cv_vs_kaggle.csv"))
print(tabla_sigue_kaggle %>% mutate(across(c(dif_cv, dif_kaggle), ~ round(.x / 1e6, 1))), n = Inf)

p_sigue <- tabla_sigue_kaggle %>%
  mutate(metrica = dplyr::recode(metrica, mae = "MAE", mae_pond = "MAE ponderado por parecido con test",
                          mae_chap = "MAE en Chapinero (train)"),
         esquema = dplyr::recode(esquema, aleatoria = "Aleatoria", bloques = "Bloques 2 km",
                          bloques_buffer = "Bloques + buffer", localidad = "Localidades"),
         esquema = factor(esquema, levels = rev(c("Aleatoria", "Bloques 2 km",
                                                  "Bloques + buffer", "Localidades")))) %>%
  ggplot(aes(dif_cv / 1e6, esquema, color = acierta)) +
  geom_vline(xintercept = 0, color = col_texto, linewidth = 0.3) +
  geom_point(size = 3) +
  facet_grid(par ~ metrica) +
  scale_color_manual(values = c(`TRUE` = col_chap_train, `FALSE` = col_test),
                     labels = c(`TRUE` = "Ordena igual que Kaggle", `FALSE` = "Ordena al revés")) +
  labs(title = "¿Qué validación anticipa si los pesos ayudan en Kaggle?",
       subtitle = "Cambio en el MAE de CV al agregar pesos (millones). En Kaggle, los pesos mejoraron en los dos pares",
       x = "MAE con pesos - MAE sin pesos (millones; negativo = los pesos ayudan)", y = NULL)
fun_guardar(p_sigue, "diag_cv_vs_kaggle", ancho = 10, alto = 5)

# ------------------------------------------------------------------------------
# 4. Resumen del error fuera de muestra
# ------------------------------------------------------------------------------
fun_resumen <- function(df, peso = NULL) {
  pe <- if (is.null(peso)) rep(1, nrow(df)) else peso
  tibble(n                 = nrow(df),
         mae_millones      = weighted.mean(df$error_abs, pe) / 1e6,
         sesgo_millones    = weighted.mean(df$error, pe) / 1e6,
         sesgo_pct_mediano = 100 * median(df$error_pct),
         pct_sobreestima   = 100 * weighted.mean(df$error > 0, pe),
         pct_sobre_10      = 100 * weighted.mean(df$error_pct > 0.10, pe),
         pct_sub_10        = 100 * weighted.mean(df$error_pct < -0.10, pe))
}

resumen <- oof %>%
  group_by(esquema) %>%
  group_modify(~ bind_rows(
    fun_resumen(.x) %>% mutate(muestra = "Todo train"),
    fun_resumen(.x, .x$w) %>% mutate(muestra = "Ponderado por parecido con test"),
    fun_resumen(filter(.x, chapinero)) %>% mutate(muestra = "Chapinero (train)")
  )) %>%
  ungroup() %>%
  relocate(muestra, .after = esquema)
write_csv(resumen, file.path(ruta_tab, "diag_resumen.csv"))
print(resumen, width = Inf)

# ------------------------------------------------------------------------------
# 5. Sesgo por decil del precio observado y del predicho
# ------------------------------------------------------------------------------
sesgo_deciles <- oof %>%
  group_by(esquema) %>%
  mutate(`Decil del precio observado` = ntile(obs, 10),
         `Decil del precio predicho`  = ntile(pred, 10)) %>%
  pivot_longer(starts_with("Decil"), names_to = "condiciona", values_to = "decil") %>%
  group_by(esquema, condiciona, decil) %>%
  summarise(sesgo_pct   = mean(error_pct),
            mediana_pct = median(error_pct),
            p25 = quantile(error_pct, 0.25), p75 = quantile(error_pct, 0.75),
            pct_sobre   = mean(error > 0),
            .groups = "drop")
write_csv(sesgo_deciles, file.path(ruta_tab, "diag_sesgo_deciles.csv"))

p_deciles <- ggplot(sesgo_deciles, aes(decil, mediana_pct, color = esquema, fill = esquema)) +
  geom_hline(yintercept = 0, color = col_texto, linewidth = 0.3) +
  geom_ribbon(aes(ymin = p25, ymax = p75), alpha = 0.12, color = NA) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 1.8) +
  facet_wrap(~ condiciona) +
  scale_x_continuous(breaks = 1:10) +
  scale_y_continuous(labels = porcentaje) +
  scale_color_manual(values = col_esquemas) +
  scale_fill_manual(values = col_esquemas) +
  labs(title = "Por precio observado, el modelo comprime los extremos; por precio predicho, casi no",
       subtitle = "Error porcentual (predicho / observado - 1): mediana y rango intercuartil por decil",
       x = "Decil", y = "Error porcentual (positivo = sobreestima)",
       caption = paste("Condicionar en el precio observado siempre muestra regresión a la media.",
                       "Para comprar solo se conoce la predicción: el panel derecho es el relevante."))
fun_guardar(p_deciles, "diag_sesgo_deciles", ancho = 10, alto = 5)

# ------------------------------------------------------------------------------
# 6. Curva de calibración
# ------------------------------------------------------------------------------
calibracion <- oof %>%
  group_by(esquema) %>%
  mutate(bin = ntile(pred, 20)) %>%
  group_by(esquema, bin) %>%
  summarise(pred_medio = mean(pred), obs_medio = mean(obs),
            obs_mediano = median(obs), .groups = "drop")

calibracion_chap <- oof %>%
  filter(chapinero) %>%
  group_by(esquema) %>%
  mutate(bin = ntile(pred, 6)) %>%
  group_by(esquema, bin) %>%
  summarise(pred_medio = mean(pred), obs_medio = mean(obs), .groups = "drop")

p_calib <- ggplot(calibracion, aes(pred_medio, obs_medio)) +
  geom_abline(slope = 1, intercept = 0, color = col_gris, linewidth = 0.6) +
  geom_line(color = col_resto, linewidth = 0.9) +
  geom_point(color = col_resto, size = 1.8) +
  geom_point(data = calibracion_chap, color = col_chap_train, size = 2.6, shape = 17) +
  facet_wrap(~ esquema) +
  scale_x_continuous(labels = millones) +
  scale_y_continuous(labels = millones) +
  labs(title = "Calibración: precio observado promedio según el precio predicho",
       subtitle = "Azul: todo train en 20 grupos de predicción. Verde: Chapinero (train) en 6 grupos. Sobre la diagonal = subestima",
       x = "Precio predicho promedio (millones)", y = "Precio observado promedio (millones)")
fun_guardar(p_calib, "diag_calibracion", ancho = 10, alto = 5)

# ------------------------------------------------------------------------------
# 7. Error por segmentos donde test difiere de train
# ------------------------------------------------------------------------------
segmentos <- c(seg_area = "Área", seg_precision = "Precisión de coordenadas",
               seg_tipo = "Tipo", seg_estrato = "Estrato del entorno (200 m)")

tabla_segmentos <- imap_dfr(segmentos, function(etiqueta, v) {
  comp_test <- seg_test %>%
    count(nivel = as.character(.data[[v]])) %>%
    mutate(pct_test = n / sum(n)) %>%
    select(-n)
  oof %>%
    mutate(nivel = as.character(.data[[v]])) %>%
    group_by(esquema, nivel) %>%
    summarise(n = n(),
              mae_millones = mean(error_abs) / 1e6,
              mae_pct      = median(abs(error_pct)),
              sesgo_pct    = median(error_pct),
              pct_sobre    = mean(error > 0),
              .groups = "drop") %>%
    group_by(esquema) %>%
    mutate(pct_train = n / sum(n)) %>%
    ungroup() %>%
    left_join(comp_test, by = "nivel") %>%
    mutate(segmento = etiqueta, pct_test = replace_na(pct_test, 0))
})
write_csv(tabla_segmentos, file.path(ruta_tab, "diag_segmentos.csv"))
print(tabla_segmentos %>% filter(esquema == "Aleatoria"), n = Inf, width = Inf)

datos_seg <- tabla_segmentos %>%
  filter(esquema == "Aleatoria") %>%
  mutate(nivel_etq = paste0(nivel, "\ntrain ", percent(pct_train, 1), " · test ", percent(pct_test, 1)))

p_seg_mae <- ggplot(datos_seg, aes(mae_pct, fct_rev(nivel_etq))) +
  geom_col(fill = col_resto, width = 0.6) +
  facet_grid(segmento ~ ., scales = "free_y", space = "free_y") +
  scale_x_continuous(labels = porcentaje, expand = expansion(mult = c(0, 0.08))) +
  labs(title = "Error absoluto mediano (%)", x = NULL, y = NULL) +
  theme(strip.text.y = element_text(angle = 0, hjust = 0))
p_seg_sesgo <- ggplot(datos_seg, aes(sesgo_pct, fct_rev(nivel_etq),
                                     fill = sesgo_pct > 0)) +
  geom_vline(xintercept = 0, color = col_texto, linewidth = 0.3) +
  geom_col(width = 0.6, show.legend = FALSE) +
  facet_grid(segmento ~ ., scales = "free_y", space = "free_y") +
  scale_fill_manual(values = c(`TRUE` = col_test, `FALSE` = col_resto)) +
  scale_x_continuous(labels = porcentaje) +
  labs(title = "Sesgo mediano (% ; positivo = sobreestima)", x = NULL, y = NULL) +
  theme(axis.text.y = element_blank(), strip.text.y = element_blank())
p_seg <- (p_seg_mae | p_seg_sesgo) +
  plot_annotation(
    title = "Dónde se equivoca el modelo y cuánto pesan esos segmentos en test",
    subtitle = "Validación aleatoria por grupos de anuncios. Debajo de cada nivel: su participación en train y en test",
    theme = theme(plot.title = element_text(face = "bold", size = 12)))
fun_guardar(p_seg, "diag_segmentos", ancho = 11, alto = 8)

# Sesgo y error según qué tanto se parece cada anuncio a test (deciles de la
# probabilidad del clasificador adversarial).
parecido <- oof %>%
  group_by(esquema) %>%
  mutate(decil_parecido = ntile(p_test, 10)) %>%
  group_by(esquema, decil_parecido) %>%
  summarise(mae_pct = median(abs(error_pct)), sesgo_pct = median(error_pct),
            precio_medio = mean(obs), .groups = "drop")
write_csv(parecido, file.path(ruta_tab, "diag_parecido_test.csv"))

p_parecido <- parecido %>%
  pivot_longer(c(mae_pct, sesgo_pct), names_to = "medida", values_to = "valor") %>%
  mutate(medida = dplyr::recode(medida, mae_pct = "Error absoluto mediano",
                         sesgo_pct = "Sesgo mediano (positivo = sobreestima)")) %>%
  ggplot(aes(decil_parecido, valor, color = esquema)) +
  geom_hline(yintercept = 0, color = col_texto, linewidth = 0.3) +
  geom_line(linewidth = 0.9) + geom_point(size = 1.8) +
  facet_wrap(~ medida, scales = "free_y") +
  scale_x_continuous(breaks = 1:10) +
  scale_y_continuous(labels = porcentaje) +
  scale_color_manual(values = col_esquemas) +
  labs(title = "Entre más se parece un anuncio a test, más lo subestima el modelo?",
       subtitle = "Por decil de la probabilidad de pertenecer a test (clasificador adversarial)",
       x = "Decil de parecido con test (10 = más parecido)", y = NULL)
fun_guardar(p_parecido, "diag_parecido_test", ancho = 10, alto = 4.5)

# ------------------------------------------------------------------------------
# 8. MAE esperado en test: reponderar train a la composición de test
# Celdas: tipo x área faltante x precisión x estrato. Cada anuncio de train
# recibe el peso (participación de su celda en test) / (participación en
# train). Se compara también con los pesos adversariales y con su producto.
# ------------------------------------------------------------------------------
celdas <- c("seg_tipo", "seg_area", "seg_precision", "seg_estrato")
comp_celdas_test <- seg_test %>%
  count(!!!syms(celdas), name = "n_test") %>%
  mutate(pct_test = n_test / sum(n_test))

oof <- oof %>%
  group_by(esquema) %>%
  mutate(n_esquema = n()) %>%
  group_by(esquema, !!!syms(celdas)) %>%
  mutate(pct_train = n() / n_esquema) %>%
  ungroup() %>%
  left_join(comp_celdas_test %>% select(-n_test), by = celdas) %>%
  mutate(w_celda = replace_na(pct_test, 0) / pct_train)

esperado <- oof %>%
  group_by(esquema) %>%
  summarise(
    `Sin reponderar`               = mean(error_abs),
    `Composición de test (celdas)` = weighted.mean(error_abs, w_celda),
    `Pesos adversariales`          = weighted.mean(error_abs, w),
    `Celdas x adversariales`       = weighted.mean(error_abs, w_celda * w),
    `Solo Chapinero (train)`       = mean(error_abs[chapinero]),
    .groups = "drop") %>%
  pivot_longer(-esquema, names_to = "reponderacion", values_to = "mae") %>%
  mutate(kaggle_16 = kaggle$kaggle_publico[kaggle$envio == "16_xgb_pesos_texto"])
write_csv(esperado, file.path(ruta_tab, "diag_mae_esperado_test.csv"))
print(esperado %>% mutate(across(c(mae, kaggle_16), ~ round(.x / 1e6, 1))), n = Inf)

p_esperado <- esperado %>%
  mutate(reponderacion = fct_reorder(reponderacion, mae)) %>%
  ggplot(aes(mae, reponderacion, color = esquema)) +
  geom_point(size = 3) +
  geom_vline(aes(xintercept = kaggle_16), color = col_texto, linetype = "dashed") +
  scale_x_continuous(labels = millones) +
  scale_color_manual(values = col_esquemas) +
  labs(title = "¿Cuánto del error en Kaggle se explica por la composición de test?",
       subtitle = "MAE fuera de muestra del modelo 16 reponderado para parecerse a test. Línea punteada: Kaggle (envío 16)",
       x = "MAE (millones)", y = NULL,
       caption = "Celdas: tipo x área faltante x precisión de coordenadas x estrato del entorno.")
fun_guardar(p_esperado, "diag_mae_esperado_test", ancho = 9, alto = 4.5)


# ------------------------------------------------------------------------------
# 9. Factor de ajuste del sesgo: CV frente a Kaggle
# Con los tres puntajes de Kaggle (factor 1 = envío 16; 1,03 = 18; 1,08 = 19)
# se ajusta una parábola y se obtiene el factor que minimiza el MAE en test.
# ------------------------------------------------------------------------------
puntos_kaggle <- tibble(
  factor = c(1, 1.03, 1.08),
  mae = kaggle$kaggle_publico[match(c("16_xgb_pesos_texto",
                                      "18_xgb_pesos_texto_sesgo_ponderado",
                                      "19_xgb_pesos_texto_sesgo_chapinero"), kaggle$envio)])
parabola <- lm(mae ~ factor + I(factor^2), data = puntos_kaggle)
factor_kaggle <- -coef(parabola)[["factor"]] / (2 * coef(parabola)[["I(factor^2)"]])
message("Factor que minimiza el MAE de Kaggle según los tres envíos: ", round(factor_kaggle, 3))

rejilla <- seq(0.95, 1.15, by = 0.005)
curvas <- oof %>%
  filter(esquema == "Aleatoria") %>%
  { d <- .; map_dfr(rejilla, function(f) {
      e <- abs(f * d$pred - d$obs)
      tibble(factor = f,
             `Todo train`                    = mean(e),
             `Ponderado por parecido`        = weighted.mean(e, d$w),
             `Composición de test (celdas)`  = weighted.mean(e, d$w_celda),
             `Chapinero (train)`             = mean(e[d$chapinero]))
    }) } %>%
  pivot_longer(-factor, names_to = "muestra", values_to = "mae")

optimos <- curvas %>% group_by(muestra) %>% slice_min(mae, n = 1, with_ties = FALSE) %>% ungroup()
write_csv(bind_rows(optimos, tibble(factor = factor_kaggle, muestra = "Kaggle (parábola)",
                                    mae = predict(parabola, tibble(factor = factor_kaggle)))),
          file.path(ruta_tab, "diag_factor_optimo.csv"))
print(optimos)

curva_kaggle <- tibble(factor = seq(0.99, 1.10, by = 0.002)) %>%
  mutate(mae = predict(parabola, newdata = .))

p_factor <- ggplot(curvas, aes(factor, mae, color = muestra)) +
  geom_line(linewidth = 0.9) +
  geom_point(data = optimos, size = 2.5) +
  geom_line(data = curva_kaggle, color = col_texto, linetype = "dashed", inherit.aes = FALSE,
            aes(factor, mae)) +
  geom_point(data = puntos_kaggle, inherit.aes = FALSE, aes(factor, mae),
             color = "black", size = 3, shape = 18) +
  geom_vline(xintercept = factor_kaggle, color = col_texto, linetype = "dotted") +
  annotate("text", x = factor_kaggle, y = max(puntos_kaggle$mae), hjust = -0.05,
           label = paste0("Óptimo Kaggle ~ ", number(factor_kaggle, accuracy = 0.001, decimal.mark = ",")),
           size = 3, color = col_texto) +
  scale_y_continuous(labels = millones) +
  scale_color_manual(values = c(col_resto, col_gris, col_chap_train, col_test)) +
  labs(title = "¿Cuánto hay que subir las predicciones?",
       subtitle = "MAE fuera de muestra (validación aleatoria) al multiplicar las predicciones por un factor. Rombos: Kaggle",
       x = "Factor", y = "MAE (millones)",
       caption = "Línea punteada: parábola por los tres puntajes de Kaggle (envíos 16, 18 y 19).")
fun_guardar(p_factor, "diag_factor_sesgo", ancho = 9, alto = 5)

# ------------------------------------------------------------------------------
# 10. Mapa del sesgo
# ------------------------------------------------------------------------------
puntos_test <- test_x %>% select(num_x, num_y)
caja <- puntos_test %>%
  summarise(xmin = min(num_x) - 4000, xmax = max(num_x) + 4000,
            ymin = min(num_y) - 4000, ymax = max(num_y) + 6000)

p_mapa <- oof %>%
  filter(esquema == "Aleatoria") %>%
  left_join(train_x %>% select(property_id, num_x, num_y), by = "property_id") %>%
  filter(between(num_x, caja$xmin, caja$xmax), between(num_y, caja$ymin, caja$ymax)) %>%
  ggplot(aes(num_x, num_y)) +
  geom_point(data = puntos_test, color = col_gris, size = 0.3, alpha = 0.5) +
  stat_summary_hex(aes(z = pmax(pmin(error_pct, 0.3), -0.3)), fun = "median",
                   bins = 35, alpha = 0.9) +
  scale_fill_gradient2(low = col_resto, mid = "white", high = col_test, midpoint = 0,
                       labels = porcentaje, name = "Sesgo mediano") +
  coord_equal() +
  labs(title = "Dónde subestima (azul) y sobreestima (naranja) el modelo, alrededor de Chapinero",
       subtitle = "Sesgo mediano por hexágono, validación aleatoria. Puntos grises: inmuebles de test",
       x = NULL, y = NULL) +
  theme(axis.text = element_blank(), panel.grid.major = element_blank(),
        legend.position = "right")
fun_guardar(p_mapa, "diag_mapa_sesgo", ancho = 8, alto = 8)

# ------------------------------------------------------------------------------
# 11. Lente de negocio: predicho / observado
# ------------------------------------------------------------------------------
negocio <- oof %>%
  mutate(grupo = if_else(chapinero, "Chapinero (train)", "Resto de Bogotá")) %>%
  group_by(esquema, grupo) %>%
  summarise(pct_sobre_10 = mean(error_pct > 0.10),
            pct_sobre_20 = mean(error_pct > 0.20),
            pct_sub_10   = mean(error_pct < -0.10),
            pct_sub_20   = mean(error_pct < -0.20), .groups = "drop")
write_csv(negocio, file.path(ruta_tab, "diag_negocio.csv"))
print(negocio)

p_negocio <- oof %>%
  mutate(grupo = if_else(chapinero, "Chapinero (train)", "Resto de Bogotá")) %>%
  ggplot(aes(pmin(pmax(error_pct, -0.6), 0.6), color = grupo, fill = grupo)) +
  annotate("rect", xmin = 0.1, xmax = Inf, ymin = -Inf, ymax = Inf, fill = col_test, alpha = 0.07) +
  geom_vline(xintercept = 0, color = col_texto, linewidth = 0.3) +
  geom_density(alpha = 0.15, linewidth = 0.8, adjust = 1.1) +
  facet_wrap(~ esquema) +
  scale_x_continuous(labels = porcentaje) +
  scale_color_manual(values = c(col_chap_train, col_resto)) +
  scale_fill_manual(values = c(col_chap_train, col_resto)) +
  labs(title = "Distribución del error: la zona sombreada es donde se pierde capital",
       subtitle = "Predicho / observado - 1 (recortado a ±60 %). Sombreado: sobreestimaciones de más del 10 %",
       x = "Error porcentual", y = "Densidad")
fun_guardar(p_negocio, "diag_negocio", ancho = 10, alto = 4.5)

# ------------------------------------------------------------------------------
# 12. Los 40 peores errores (validación aleatoria), para revisar a mano
# ------------------------------------------------------------------------------
peores <- oof %>%
  filter(esquema == "Aleatoria") %>%
  slice_max(error_abs, n = 40) %>%
  left_join(crudo_train %>% select(property_id, title, description), by = "property_id") %>%
  left_join(train_x %>% select(property_id, cat_localidad, num_area, num_habitac, num_banos),
            by = "property_id") %>%
  transmute(property_id, cat_localidad, seg_tipo, num_area, num_habitac, num_banos,
            obs_millones = obs / 1e6, pred_millones = pred / 1e6,
            error_pct = round(100 * error_pct, 1), seg_precision,
            title, description = str_trunc(description, 300))
write_csv(peores, file.path(ruta_tab, "diag_peores_errores.csv"))

message("Listo. Figuras diag_*.png en 03_Output/Figures y tablas diag_*.csv en 03_Output/Tables.")
