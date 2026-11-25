# ==============================================================================
# 03_figuras_presentaciones.R
# Objetivo: figuras, tablas y cifras de las presentaciones (best_equipo_01 y
#           compare_equipo_01) que no generan los demás scripts.
#           1. Cifras que citan las diapositivas, como macros de LaTeX
#              (pres_cifras.tex): MAE del mejor envío, precio típico, balance
#              entre train y test, número de variables por grupo, etc.
#           2. Escalera del error: puntaje privado de Kaggle de los envíos que
#              marcan cada mejora.
#           3. Importancia de las variables del XGBoost 20 (miembro del mejor
#              ensamble) agrupada por tipo de información.
#           4. Mejor puntaje privado de Kaggle por familia de algoritmos.
#           5. Tabla de los envíos 05 a 09 (misma base, cinco familias): MAE de
#              los cuatro esquemas de validación cruzada y de Kaggle.
#           6. Puesto de cada familia según cada validación y según Kaggle.
#           7. Por qué pierden los modelos simples: error medio por decil de
#              precio de la regresión lineal frente al mejor ensamble, y número
#              de valores distintos que predicen CART y XGBoost en test.
#           8. Validación cruzada frente a Kaggle, y puntaje público frente a
#              privado, para todos los envíos.
#           9. Tabla de ventajas y desventajas de cada familia.
# Entradas: 02_Data/Raw/Kaggle/{test.csv,kaggle_puntajes.csv}
#           02_Data/Processed/db_{train,test}.rds
#           03_Output/Models/{resultados_modelos.csv,20_xgb_mejorado.rds}
#           03_Output/Predictions/oof_29_ensamble_embeddings_localidad.rds
#           03_Output/Tables/negocio_{semaforo_resumen,mejor_margen}.csv
#           03_Output/Submissions/{07_cart_completa,09_xgb_completa,
#           29_ensamble_embeddings}.csv
# Salidas:  03_Output/Figures/pres_*.{pdf,png}
#           03_Output/Tables/pres_*.{tex,csv}
# ==============================================================================

source(here::here("01_Scripts", "00_utils.R"))
pacman::p_load(scales, patchwork)

ruta_figuras <- here("03_Output", "Figures")
ruta_tablas  <- here("03_Output", "Tables")
walk(c(ruta_figuras, ruta_tablas), dir.create, showWarnings = FALSE, recursive = TRUE)

guardar <- function(grafico, nombre, ancho = 8, alto = 5) {
  for (ext in c("pdf", "png")) {
    ggsave(file.path(ruta_figuras, paste0(nombre, ".", ext)), grafico,
           width = ancho, height = alto, dpi = 300, bg = "white")
  }
  message("Figura guardada: ", nombre)
}

tema <- theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(color = "grey30", size = 11),
        plot.caption = element_text(color = "grey40", size = 9, hjust = 0),
        panel.grid.minor = element_blank(),
        legend.position = "bottom")

millones <- function(x, d = 1) trimws(format(round(x / 1e6, d), nsmall = d, decimal.mark = ",", big.mark = "."))
pct      <- function(x, d = 0) paste0(trimws(format(round(100 * x, d), nsmall = d, decimal.mark = ",")), "\\%")

# Escribe un archivo de texto en UTF-8, para que las tildes lleguen bien a LaTeX.
escribir <- function(lineas, archivo) writeLines(enc2utf8(lineas), file.path(ruta_tablas, archivo), useBytes = TRUE)

# Familia de cada envío según su nombre.
fun_familia <- function(envio) {
  case_when(
    str_detect(envio, "ensamble") ~ "Ensamble de XGBoost",
    str_detect(envio, "_lm_")     ~ "Regresión lineal",
    str_detect(envio, "_enet_")   ~ "Elastic Net",
    str_detect(envio, "_cart_")   ~ "CART",
    str_detect(envio, "_rf_")     ~ "Random Forest",
    str_detect(envio, "_nnet_")   ~ "Red neuronal",
    str_detect(envio, "_xgb_")    ~ "XGBoost",
    TRUE ~ NA_character_
  )
}

kaggle <- read_csv(here("02_Data", "Raw", "Kaggle", "kaggle_puntajes.csv"), show_col_types = FALSE) %>%
  mutate(familia = fun_familia(envio))
resultados <- read_csv(here("03_Output", "Models", "resultados_modelos.csv"),
                       col_types = cols(.default = col_character())) %>%
  mutate(across(starts_with("mae_cv_"), as.numeric))

train <- readRDS(here("02_Data", "Processed", "db_train.rds"))
test  <- readRDS(here("02_Data", "Processed", "db_test.rds"))

# ------------------------------------------------------------------------------
# 1. Cifras para las diapositivas
# ------------------------------------------------------------------------------
mejor <- kaggle %>% slice_min(kaggle_privado, n = 1)
pred_29 <- read_csv(here("03_Output", "Submissions", "29_ensamble_embeddings.csv"),
                    show_col_types = FALSE)
precio_tipico_test <- median(pred_29$price)

es_chap <- train$cat_localidad == "CHAPINERO"
test_raw <- read_csv(here("02_Data", "Raw", "Kaggle", "test.csv"),
                     col_types = cols(lat = col_character(), lon = col_character(), .default = col_guess()))
train_raw <- read_csv(here("02_Data", "Raw", "Kaggle", "train.csv"),
                      col_types = cols(lat = col_character(), lon = col_character(), .default = col_guess()))
fun_decimales <- function(x) nchar(str_remove(x, "^[^.]*\\.?"))
decimales_lat <- fun_decimales(test_raw$lat)

# Grupos de variables según su origen.
fun_grupo <- function(v) {
  case_when(
    str_detect(v, "^num_txt_|^num_palabras|^num_estrato_texto|^num_barrios_texto|^bin_b_|^bin_(chico|rosales|barrio_alto|lujo|vista|vista_ciudad|terraza|gimnasio|piscina|ascensor|chimenea|vigilancia|deposito|estudio|remodelado|duplex|penthouse|exterior|conjunto|bbq|salon_comunal|jardin|estrenar|parqueadero|parque_cerca)$") ~ "Texto del anuncio",
    str_detect(v, "^num_rez_|^num_comp_") ~ "Precio de vecinos y comparables",
    str_detect(v, "vref|avaluo|estrato|pisos_200m|predios_mz") ~ "Valor del suelo (Catastro)",
    str_detect(v, "area|banos|habitac|surface|parq|piso|niveles|cat_tipo|bin_casa|cat_fuente_area|ambientes") ~ "Tamaño y estructura",
    str_detect(v, "^num_x$|^num_y$|dec_coord|coord_imprecisa|anio|mes") ~ "Ubicación exacta y fecha",
    TRUE ~ "Entorno urbano y servicios"
  )
}
# Resultados del análisis de negocio (04_Results/02_negocio.R) y margen de
# seguridad de Chapinero con las predicciones fuera de muestra del envío 29.
oof_29 <- readRDS(here("03_Output", "Predictions", "oof_29_ensamble_embeddings_localidad.rds"))
razon_chap <- train %>%
  filter(cat_localidad == "CHAPINERO") %>%
  inner_join(oof_29, by = "property_id") %>%
  mutate(razon = exp(pred_log) / obj_price) %>%
  pull(razon)
semaforo <- read_csv(here("03_Output", "Tables", "negocio_semaforo_resumen.csv"), show_col_types = FALSE)
mejor_margen <- read_csv(here("03_Output", "Tables", "negocio_mejor_margen.csv"), show_col_types = FALSE) %>%
  filter(descuento == max(descuento))
fun_semaforo <- function(color) semaforo$porc[str_detect(semaforo$semaforo, paste0("^", color))] / 100

grupos_base <- tibble(variable = setdiff(names(train), no_predictores)) %>%
  filter(!str_detect(variable, "_na$")) %>%
  mutate(grupo = fun_grupo(variable)) %>%
  count(grupo)
n_grupo <- function(g) grupos_base$n[grupos_base$grupo == g]

cifras <- c(
  maeprivado      = millones(mejor$kaggle_privado),
  maepublico      = millones(mejor$kaggle_publico),
  mejorenvio      = str_replace_all(mejor$envio, "_", "\\\\_"),
  preciotipico    = millones(precio_tipico_test, 0),
  maepctprecio    = pct(mejor$kaggle_privado / precio_tipico_test),
  ntrain          = format(nrow(train), big.mark = "."),
  ntest           = format(nrow(test), big.mark = "."),
  nchaptrain      = format(sum(es_chap), big.mark = "."),
  pcttestchap     = pct(mean(test$cat_localidad == "CHAPINERO")),
  preciomdchap    = millones(median(train$obj_price[es_chap]), 0),
  preciomdresto   = millones(median(train$obj_price[!es_chap]), 0),
  aptotest        = pct(mean(test$cat_tipo == "Apartamento")),
  aptotrain       = pct(mean(train$cat_tipo == "Apartamento")),
  sinareatest     = pct(mean(test$bin_area_na == 1)),
  sinareatrain    = pct(mean(train$bin_area_na == 1)),
  sinareachap     = pct(mean(train$bin_area_na[es_chap] == 1)),
  coordredondeada = pct(mean(decimales_lat <= 3)),
  coordredtrain   = pct(mean(fun_decimales(train_raw$lat) <= 3)),
  nvarexternas    = n_grupo("Entorno urbano y servicios") + n_grupo("Valor del suelo (Catastro)"),
  nvartexto       = n_grupo("Texto del anuncio"),
  nenvios         = nrow(kaggle),
  maecv           = millones(resultados$mae_cv_localidad[resultados$modelo == mejor$envio]),
  margenchap      = pct(1 - 1 / quantile(razon_chap, 0.9)),
  sobreestimachap = pct(mean(razon_chap > 1)),
  sobrediezchap   = pct(mean(razon_chap > 1.1)),
  pctverde        = pct(fun_semaforo("Verde")),
  pctamarillo     = pct(fun_semaforo("Amarillo")),
  pctrojo         = pct(fun_semaforo("Rojo")),
  mejormargen     = pct(mejor_margen$margen),
  gananciacien    = millones(mejor_margen$ganancia_por_100_ofertas * 1e6, 0),
  negociosmargen  = pct(mejor_margen$porc_negocios / 100),
  pagademasmargen = pct(mejor_margen$porc_compras_pagando_de_mas / 100)
)
escribir(sprintf("\\newcommand{\\%s}{%s}", names(cifras), cifras), "pres_cifras.tex")
print(cifras)

# ------------------------------------------------------------------------------
# 2. Escalera del error con el puntaje privado
# ------------------------------------------------------------------------------
pasos <- tribble(
  ~envio,                              ~paso,
  "01_lm_base",                        "Base\nmínima",
  "05_lm_completa",                    "+ entorno\ny texto",
  "09_xgb_completa",                   "+ XGBoost",
  "12_xgb_base_final",                 "+ precio de\nvecinos",
  "16_xgb_pesos_texto",                "+ texto como\ndatos",
  "21_xgb_mejorado_sesgo",             "+ barrios y\najuste de nivel",
  "27_ensamble_calibrado_reanuncios",  "+ ensamble y\nre-anuncios",
  "29_ensamble_embeddings",            "+ más\nsemillas"
) %>%
  left_join(kaggle, by = "envio") %>%
  mutate(orden = row_number(),
         paso = factor(paso, levels = paso),
         mejora = lag(kaggle_privado) - kaggle_privado)
write_csv(pasos %>% select(envio, paso, kaggle_privado, mejora), file.path(ruta_tablas, "pres_escalera.csv"))

g_escalera <- pasos %>%
  ggplot(aes(paso, kaggle_privado / 1e6)) +
  geom_col(fill = c(rep("grey70", 2), rep("#2166AC", 6)), width = 0.7) +
  geom_text(aes(label = millones(kaggle_privado)), vjust = -0.4, size = 4) +
  geom_text(data = filter(pasos, !is.na(mejora)),
            aes(y = 15, label = paste0("-", millones(mejora))), color = "white", size = 3.6, fontface = "bold") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
  labs(title = "Cada idea, cuánto bajó el error",
       subtitle = "MAE privado de Kaggle (millones de pesos). En blanco: cuánto bajó frente al paso anterior.",
       x = NULL, y = "MAE privado (millones)",
       caption = "Gris: regresión lineal. Azul: XGBoost y ensambles de XGBoost.") +
  tema +
  theme(axis.text.x = element_text(size = 11), panel.grid.major.x = element_blank())
guardar(g_escalera, "pres_escalera", ancho = 11, alto = 5.2)

# ------------------------------------------------------------------------------
# 3. Importancia por grupo de variables (XGBoost 20)
# ------------------------------------------------------------------------------
importancia <- readRDS(here("03_Output", "Models", "20_xgb_mejorado.rds"))$importancia %>%
  as_tibble() %>%
  mutate(grupo = fun_grupo(Feature))
importancia_grupo <- importancia %>%
  group_by(grupo) %>%
  summarise(ganancia = sum(Gain),
            principales = paste(head(Feature[order(-Gain)], 3), collapse = ", "),
            .groups = "drop") %>%
  mutate(ganancia = ganancia / sum(ganancia)) %>%
  arrange(ganancia) %>%
  mutate(grupo = factor(grupo, levels = grupo))
write_csv(importancia_grupo, file.path(ruta_tablas, "pres_importancia_grupos.csv"))

g_importancia <- importancia_grupo %>%
  ggplot(aes(ganancia, grupo)) +
  geom_col(fill = "#2166AC", width = 0.65) +
  geom_text(aes(label = paste0(format(round(100 * ganancia), nsmall = 0), "%")),
            hjust = -0.15, size = 4.5) +
  scale_x_continuous(labels = label_percent(), expand = expansion(mult = c(0, 0.15))) +
  labs(title = "Qué información usa el modelo para fijar el precio",
       subtitle = "Participación de cada grupo en la ganancia total del XGBoost 20 (miembro del mejor ensamble)",
       x = "Participación en la ganancia", y = NULL) +
  tema
guardar(g_importancia, "pres_importancia_grupos", ancho = 9.5, alto = 5)

# ------------------------------------------------------------------------------
# 4. Mejor puntaje privado por familia
# ------------------------------------------------------------------------------
mejor_familia <- kaggle %>%
  group_by(familia) %>%
  slice_min(kaggle_privado, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  arrange(desc(kaggle_privado)) %>%
  mutate(familia = factor(familia, levels = familia))
write_csv(mejor_familia, file.path(ruta_tablas, "pres_mejor_familia.csv"))

g_familias <- mejor_familia %>%
  ggplot(aes(kaggle_privado / 1e6, familia)) +
  geom_segment(aes(x = 0, xend = kaggle_privado / 1e6, yend = familia), color = "grey75", linewidth = 1) +
  geom_point(aes(color = str_detect(familia, "XGBoost")), size = 5) +
  geom_text(aes(label = paste0(millones(kaggle_privado), "  (", str_extract(envio, "^\\d+"), ")")),
            hjust = -0.25, size = 4) +
  scale_color_manual(values = c(`TRUE` = "#2166AC", `FALSE` = "grey45"), guide = "none") +
  scale_x_continuous(limits = c(0, max(mejor_familia$kaggle_privado) / 1e6 * 1.3), breaks = seq(0, 300, 50)) +
  labs(title = "Seis familias, un ganador claro: el boosting",
       subtitle = "Mejor MAE privado de Kaggle de cada familia (millones de pesos; entre paréntesis, el envío)",
       x = "MAE privado (millones)", y = NULL) +
  tema
guardar(g_familias, "pres_familias", ancho = 9.5, alto = 5)

# ------------------------------------------------------------------------------
# 5. Tabla de los envíos 05 a 09: misma base, cinco familias
# ------------------------------------------------------------------------------
misma_base <- resultados %>%
  filter(modelo %in% c("05_lm_completa", "06_enet_completa", "07_cart_completa",
                       "08_rf_completa", "09_xgb_completa")) %>%
  left_join(kaggle %>% select(envio, kaggle_privado), by = c("modelo" = "envio")) %>%
  transmute(familia = fun_familia(modelo), aleatoria = mae_cv_aleatoria, bloques = mae_cv_bloques,
            buffer = mae_cv_bloques_buffer, localidad = mae_cv_localidad, kaggle = kaggle_privado)
write_csv(misma_base, file.path(ruta_tablas, "pres_misma_base.csv"))

fun_celda <- function(x, columna) {
  texto <- millones(x, 0)
  ifelse(x == min(columna), paste0("\\textbf{", texto, "}"), texto)
}
filas_tex <- misma_base %>%
  mutate(across(aleatoria:kaggle, ~ fun_celda(.x, .x))) %>%
  pmap_chr(function(familia, aleatoria, bloques, buffer, localidad, kaggle)
    paste(familia, aleatoria, bloques, buffer, localidad, kaggle, sep = " & ") %>% paste0(" \\\\"))
escribir(c("\\begin{tabular}{lrrrr|r}", "\\toprule",
           " & \\multicolumn{4}{c|}{Validación cruzada} & Kaggle \\\\",
           "Familia & Aleatoria & Bloques 2 km & Bloques + buffer & Localidades & privado \\\\",
           "\\midrule", filas_tex, "\\bottomrule", "\\end{tabular}"), "pres_misma_base.tex")

# ------------------------------------------------------------------------------
# 6. Puesto de cada familia según cada validación y según Kaggle
# ------------------------------------------------------------------------------
puestos <- misma_base %>%
  pivot_longer(c(aleatoria, bloques, localidad, kaggle), names_to = "medida", values_to = "mae") %>%
  group_by(medida) %>%
  mutate(puesto = rank(mae)) %>%
  ungroup() %>%
  mutate(medida = factor(medida, levels = c("aleatoria", "bloques", "localidad", "kaggle"),
                         labels = c("CV aleatoria", "CV bloques 2 km", "CV localidades", "Kaggle privado")))

colores_familia <- c("Regresión lineal" = "#B2182B", "Elastic Net" = "#EF8A62", "CART" = "#999999",
                     "Random Forest" = "#1A9850", "XGBoost" = "#2166AC")
g_puestos <- puestos %>%
  ggplot(aes(medida, puesto, group = familia, color = familia)) +
  geom_line(linewidth = 1.3) +
  geom_point(size = 4) +
  geom_text(data = filter(puestos, medida == "Kaggle privado"),
            aes(label = familia), hjust = -0.15, size = 4.2, show.legend = FALSE) +
  scale_y_reverse(breaks = 1:5, labels = paste0(1:5, "°")) +
  scale_x_discrete(expand = expansion(add = c(0.3, 1.3))) +
  scale_color_manual(values = colores_familia, guide = "none") +
  labs(title = "La validación aleatoria escoge el modelo equivocado",
       subtitle = "Puesto de cada familia (misma base, envíos 05 a 09) según cada validación y según Kaggle",
       x = NULL, y = "Puesto") +
  tema
guardar(g_puestos, "pres_puestos_validacion", ancho = 10, alto = 5)

# ------------------------------------------------------------------------------
# 7. Por qué pierden los modelos simples
# ------------------------------------------------------------------------------
# (a) Error medio por decil de precio con localidades por fuera: regresión
# lineal estimada aquí con los predictores de la base final y los mismos
# folds, frente al mejor ensamble.
datos <- fun_datos_modelo()
folds_localidad <- fun_folds_cv(datos$train, "localidad")
x_lm <- model.matrix(reformulate(datos$predictores), data = datos$train)
y_lm <- datos$train$obj_log_price
pred_lm <- rep(NA_real_, nrow(datos$train))
for (f in folds_localidad) {
  coeficientes <- lm.fit(x_lm[f$train, , drop = FALSE], y_lm[f$train])$coefficients
  coeficientes[is.na(coeficientes)] <- 0
  pred_lm[f$val] <- as.numeric(x_lm[f$val, , drop = FALSE] %*% coeficientes)
}
deciles <- bind_rows(
  tibble(property_id = datos$train$property_id, pred_log = pred_lm, modelo = "Regresión lineal"),
  oof_29 %>% mutate(modelo = "Ensamble de XGBoost (29)")
) %>%
  inner_join(train %>% select(property_id, obj_price), by = "property_id") %>%
  mutate(decil = ntile(obj_price, 10), .by = modelo) %>%
  group_by(modelo, decil) %>%
  summarise(error_medio = mean(exp(pred_log) / obj_price - 1), .groups = "drop")
write_csv(deciles, file.path(ruta_tablas, "pres_error_decil_modelos.csv"))

g_deciles <- deciles %>%
  ggplot(aes(factor(decil), error_medio, fill = modelo)) +
  geom_hline(yintercept = 0) +
  geom_col(position = position_dodge(width = 0.75), width = 0.7) +
  scale_y_continuous(labels = label_percent()) +
  scale_fill_manual(values = c("Regresión lineal" = "#B2182B", "Ensamble de XGBoost (29)" = "#2166AC"),
                    name = NULL) +
  labs(title = "Los dos comprimen los precios; la regresión lineal más",
       subtitle = "Error medio (predicción / precio - 1) por decil de precio, con localidades por fuera",
       x = "Decil de precio observado (1 = más barato)", y = "Error medio") +
  tema

# (b) Valores distintos que predicen CART y XGBoost para los anuncios de test.
predicciones_test <- bind_rows(
  read_csv(here("03_Output", "Submissions", "07_cart_completa.csv"), show_col_types = FALSE) %>%
    mutate(modelo = "CART (07)"),
  read_csv(here("03_Output", "Submissions", "09_xgb_completa.csv"), show_col_types = FALSE) %>%
    mutate(modelo = "XGBoost (09)")
)
distintos <- predicciones_test %>% group_by(modelo) %>% summarise(n = n_distinct(price))
g_cart <- predicciones_test %>%
  left_join(distintos, by = "modelo") %>%
  mutate(modelo = sprintf("%s: %s precios distintos", modelo, format(n, big.mark = ".", decimal.mark = ","))) %>%
  ggplot(aes(price / 1e6)) +
  geom_histogram(bins = 150, fill = "grey30") +
  facet_wrap(~ modelo, ncol = 1, scales = "free_y") +
  scale_x_continuous(limits = c(0, 2500)) +
  labs(title = "Un solo árbol predice en escalones",
       subtitle = "Precios predichos para los anuncios de test",
       x = "Precio predicho (millones)", y = "Anuncios") +
  tema +
  theme(strip.text = element_text(face = "bold", hjust = 0))

guardar(g_deciles | g_cart, "pres_modelos_simples", ancho = 13, alto = 5.2)

# ------------------------------------------------------------------------------
# 8. Validación cruzada frente a Kaggle, y público frente a privado
# ------------------------------------------------------------------------------
cv_kaggle <- resultados %>%
  select(envio = modelo, aleatoria = mae_cv_aleatoria, localidad = mae_cv_localidad) %>%
  inner_join(kaggle, by = "envio") %>%
  filter(localidad < 500e6) %>%                       # la red 17 explota con localidades por fuera
  pivot_longer(c(aleatoria, localidad), names_to = "esquema", values_to = "mae_cv") %>%
  filter(!is.na(mae_cv))
correlaciones <- cv_kaggle %>%
  group_by(esquema) %>%
  summarise(rho = cor(mae_cv, kaggle_privado, method = "spearman"),
            sesgo = mean(kaggle_privado - mae_cv), .groups = "drop") %>%
  mutate(etiqueta = sprintf("CV %s\nKaggle es %s millones más alto; correlación de rangos: %s",
                            if_else(esquema == "aleatoria", "aleatoria", "con localidades por fuera"),
                            millones(sesgo, 0), format(round(rho, 2), decimal.mark = ",")))
write_csv(correlaciones, file.path(ruta_tablas, "pres_cv_vs_kaggle.csv"))

g_cv_kaggle <- cv_kaggle %>%
  left_join(correlaciones %>% select(esquema, etiqueta), by = "esquema") %>%
  ggplot(aes(mae_cv / 1e6, kaggle_privado / 1e6, color = familia)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  geom_point(size = 3, alpha = 0.85) +
  facet_wrap(~ etiqueta) +
  scale_color_brewer(palette = "Dark2", name = NULL) +
  coord_equal(xlim = c(70, 250), ylim = c(70, 250)) +
  labs(title = "La validación espacial se parece más a Kaggle",
       subtitle = "Cada punto es un envío. Línea punteada: la validación acierta exactamente el error de Kaggle.",
       x = "MAE de validación cruzada (millones)", y = "MAE privado de Kaggle (millones)",
       caption = "Se excluye la red 17, cuyo error con localidades por fuera es de 348 millones.") +
  tema +
  theme(strip.text = element_text(size = 10, face = "bold"))
guardar(g_cv_kaggle, "pres_cv_vs_kaggle", ancho = 12, alto = 6.5)

pares <- c("10_xgb_na_nativo", "11_xgb_pesos", "15_xgb_na_nativo_texto", "16_xgb_pesos_texto")
g_pub_priv <- kaggle %>%
  ggplot(aes(kaggle_publico / 1e6, kaggle_privado / 1e6)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  geom_point(aes(color = envio %in% pares), size = 3) +
  geom_text(data = filter(kaggle, envio %in% pares),
            aes(label = str_extract(envio, "^\\d+")), nudge_x = -3, nudge_y = 3, size = 4, color = "#B2182B") +
  scale_color_manual(values = c(`TRUE` = "#B2182B", `FALSE` = "#2166AC"), guide = "none") +
  coord_equal() +
  labs(title = "Público frente a privado: el orden se mantiene arriba",
       subtitle = "Rojo: pares con y sin pesos que se invierten (10 y 11; 15 y 16)",
       x = "MAE público (millones)", y = "MAE privado (millones)") +
  tema
guardar(g_pub_priv, "pres_publico_privado", ancho = 7, alto = 6.5)

# ------------------------------------------------------------------------------
# 9. Tabla de ventajas y desventajas por familia
# - Kaggle privado: mejor envío de la familia.
# - Pérdida fuera de zona: cuánto sube el MAE al pasar de la validación
#   aleatoria a la de localidades por fuera (envíos 05 a 09 con la misma base;
#   para la red neuronal, el envío 17).
# - Las tres columnas cualitativas resumen las características de cada familia:
#   si captura interacciones y no linealidades, si es interpretable y si es
#   rápida de estimar.
# ------------------------------------------------------------------------------
perdida <- resultados %>%
  filter(modelo %in% c("05_lm_completa", "06_enet_completa", "07_cart_completa",
                       "08_rf_completa", "09_xgb_completa", "17_nnet_texto")) %>%
  transmute(familia = fun_familia(modelo), perdida = mae_cv_localidad / mae_cv_aleatoria - 1)

lleno <- "$\\bullet\\bullet\\bullet$"; medio <- "$\\bullet\\bullet\\circ$"; bajo <- "$\\bullet\\circ\\circ$"
ventajas <- tribble(
  ~familia,             ~interacciones, ~interpretable, ~rapidez,
  "Regresión lineal",   bajo,           lleno,          lleno,
  "Elastic Net",        bajo,           lleno,          lleno,
  "CART",               medio,          lleno,          lleno,
  "Random Forest",      lleno,          bajo,           bajo,
  "XGBoost",            lleno,          medio,          medio,
  "Red neuronal",       medio,          bajo,           bajo
) %>%
  left_join(mejor_familia %>% mutate(familia = as.character(familia)) %>%
              select(familia, kaggle_privado), by = "familia") %>%
  left_join(perdida, by = "familia")
write_csv(ventajas, file.path(ruta_tablas, "pres_ventajas.csv"))

filas_ventajas <- ventajas %>%
  pmap_chr(function(familia, interacciones, interpretable, rapidez, kaggle_privado, perdida)
    paste(familia, millones(kaggle_privado, 0), paste0("+", pct(perdida)), interacciones,
          interpretable, rapidez, sep = " & ") %>% paste0(" \\\\"))
escribir(c("\\begin{tabular}{lrrccc}", "\\toprule",
           "Familia & Kaggle privado & Pérdida fuera de zona & Interacciones & Interpretable & Rápido \\\\",
           "\\midrule", filas_ventajas, "\\bottomrule", "\\end{tabular}"), "pres_ventajas.tex")

message("Figuras en ", ruta_figuras, " y tablas en ", ruta_tablas)
