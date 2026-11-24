# ==============================================================================
# 01_mapas.R
# Objetivo: mapas de Bogotá para las presentaciones. Todos tienen de fondo las
#           localidades urbanas de Bogotá (sin Sumapaz, que es rural) con
#           Chapinero resaltado, y los de detalle se acercan a Chapinero con
#           los límites de sus UPZ (Unidades de Planeamiento Zonal).
#           1. Ubicación de los anuncios de train y de test.
#           2. Precio por m2 mediano de train por UPZ.
#           3. Detalle de Chapinero: precio por m2 predicho para test por el
#              mejor envío (29_ensamble_embeddings).
#           4. Detalle de Chapinero: error fuera de muestra del mejor modelo
#              en train (validación con localidades completas por fuera), en
#              porcentaje del precio observado.
#           5. Detalle de Chapinero: cobertura de test, es decir, número de
#              anuncios de train a menos de 500 m de cada anuncio de test.
#           6. Folds de la validación cruzada espacial: localidades completas
#              por fuera y bloques de 2 km.
#           7. MAE por localidad, con la localidad por fuera, de tres modelos:
#              regresión lineal, red neuronal (24_nnet_robusta) y el mejor
#              ensamble (29_ensamble_embeddings).
#           8. Peso promedio por UPZ de los anuncios de train según su parecido
#              con test (pesos de 10_xgb_pesos_texto.R).
#           9. Contexto urbano: parques, estaciones de TransMilenio y
#              estaciones de la primera línea del metro.
#           Los precios por m2 usan solo anuncios con área observada (sin
#           imputar). En los mapas por UPZ, las UPZ con menos de 20 anuncios
#           quedan en gris.
# Entradas: 02_Data/Processed/db_{train,test}.rds
#           02_Data/Processed/pesos_adversarial_texto.rds
#           02_Data/Raw/MapasBogota/capas/{44_localidades,45_upz,25_parques,
#           28a_estaciones_tm,32_metro}.geojson (de 02_descarga_capas.R)
#           03_Output/Submissions/29_ensamble_embeddings.csv
#           03_Output/Predictions/oof_29_ensamble_embeddings_localidad.rds
#           03_Output/Predictions/oof_24_nnet_robusta_localidad.rds
# Salidas:  03_Output/Figures/mapa_*.{pdf,png}
#           03_Output/Tables/mapa_mae_localidad_modelos.csv
# ==============================================================================

source(here::here("01_Scripts", "00_utils.R"))
pacman::p_load(sf, scales, patchwork)

ruta_figuras <- here("03_Output", "Figures")
ruta_tablas  <- here("03_Output", "Tables")
walk(c(ruta_figuras, ruta_tablas), dir.create, showWarnings = FALSE, recursive = TRUE)

# Colores
color_train <- "grey55"
color_test  <- "#D55E00"
color_chap  <- "black"

guardar_mapa <- function(grafico, nombre, ancho = 7, alto = 8) {
  for (ext in c("pdf", "png")) {
    ggsave(file.path(ruta_figuras, paste0(nombre, ".", ext)), grafico,
           width = ancho, height = alto, dpi = 300, bg = "white")
  }
  message("Mapa guardado: ", nombre)
}

# ------------------------------------------------------------------------------
# 0. Datos y capas de base
# ------------------------------------------------------------------------------
train <- readRDS(here("02_Data", "Processed", "db_train.rds"))
test  <- readRDS(here("02_Data", "Processed", "db_test.rds"))

localidades <- fun_leer_capa("44_localidades") %>%
  transmute(localidad = str_to_upper(stringi::stri_trans_general(LOCNOMBRE, "Latin-ASCII"))) %>%
  filter(localidad != "SUMAPAZ")
chapinero <- localidades %>% filter(localidad == "CHAPINERO")

upz <- fun_leer_capa("45_upz") %>%
  transmute(cat_upz_codigo = as.character(CODIGO_UPZ), nombre_upz = NOMBRE)

# Nombre de localidad sin tildes, para unir con las capas.
fun_sin_tildes <- function(x) str_to_upper(stringi::stri_trans_general(x, "Latin-ASCII"))
train <- train %>% mutate(localidad = fun_sin_tildes(cat_localidad))
test  <- test  %>% mutate(localidad = fun_sin_tildes(cat_localidad))

# Ventana de la ciudad: los anuncios de train con un margen de 1 km.
ventana_ciudad <- list(x = range(train$num_x) + c(-1000, 1000),
                       y = range(train$num_y) + c(-1000, 1000))

# Ventana de Chapinero: el 98% central de los anuncios de test con un margen
# de 800 m. No se usa todo el polígono de Chapinero porque la localidad incluye
# los cerros orientales, donde no hay anuncios.
ventana_chap <- list(x = unname(quantile(test$num_x, c(0.01, 0.99))) + c(-800, 800),
                     y = unname(quantile(test$num_y, c(0.01, 0.99))) + c(-800, 800))

# Capas de base: localidades en gris claro con sus nombres y Chapinero con
# borde negro. `nombres = FALSE` omite los nombres de las localidades.
capa_ciudad <- function(nombres = TRUE) {
  capas <- list(
    geom_sf(data = localidades, fill = "grey96", color = "grey70", linewidth = 0.3),
    geom_sf(data = chapinero, fill = NA, color = color_chap, linewidth = 0.8)
  )
  if (nombres) {
    capas <- c(capas, list(
      geom_sf_text(data = st_point_on_surface(localidades),
                   aes(label = str_to_title(localidad)),
                   size = 2.2, color = "grey35", check_overlap = TRUE)
    ))
  }
  capas
}

# Capas de base del detalle de Chapinero: UPZ con sus nombres y Chapinero con
# borde negro.
capa_chapinero <- function() {
  upz_ventana <- st_crop(upz, c(xmin = ventana_chap$x[1], xmax = ventana_chap$x[2],
                                ymin = ventana_chap$y[1], ymax = ventana_chap$y[2]))
  list(
    geom_sf(data = localidades, fill = "grey97", color = "grey60", linewidth = 0.4),
    geom_sf(data = upz_ventana, fill = NA, color = "grey75", linewidth = 0.25),
    geom_sf(data = chapinero, fill = NA, color = color_chap, linewidth = 0.9),
    geom_sf_text(data = st_point_on_surface(upz_ventana),
                 aes(label = str_wrap(str_to_title(nombre_upz), 12)),
                 size = 1.9, color = "grey30", lineheight = 0.8, check_overlap = TRUE)
  )
}

coord_ciudad <- function() coord_sf(xlim = ventana_ciudad$x, ylim = ventana_ciudad$y,
                                    crs = crs_metros, datum = NA, expand = FALSE)
coord_chap   <- function() coord_sf(xlim = ventana_chap$x, ylim = ventana_chap$y,
                                    crs = crs_metros, datum = NA, expand = FALSE)

tema_mapa <- theme_void(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 13),
        plot.subtitle = element_text(color = "grey30", size = 9.5),
        plot.caption = element_text(color = "grey40", size = 8, hjust = 0),
        legend.position = "bottom",
        legend.title = element_text(size = 9),
        legend.text = element_text(size = 8),
        plot.margin = margin(8, 8, 8, 8))

# Inmuebles como puntos sf.
fun_puntos <- function(df) st_as_sf(df, coords = c("num_x", "num_y"), crs = crs_metros, remove = FALSE)

# ------------------------------------------------------------------------------
# 1. Ubicación de train y test
# ------------------------------------------------------------------------------
n_chap_train <- sum(train$localidad == "CHAPINERO")
mapa_1 <- ggplot() +
  capa_ciudad() +
  geom_point(data = train, aes(num_x, num_y), color = color_train, size = 0.15, alpha = 0.25) +
  geom_point(data = test, aes(num_x, num_y), color = color_test, size = 0.2, alpha = 0.4) +
  coord_ciudad() +
  labs(title = "El modelo aprende de Bogotá y debe predecir en Chapinero",
       subtitle = paste0("Gris: ", comma(nrow(train), big.mark = ".", decimal.mark = ","), " anuncios de train. ",
                         "Naranja: ", comma(nrow(test), big.mark = ".", decimal.mark = ","), " anuncios de test.\n",
                         "Solo ", n_chap_train, " anuncios de train están en Chapinero (borde negro)."),
       caption = "Fuente: Properati 2019-2021 y límites de localidades de Catastro Bogotá.") +
  tema_mapa
guardar_mapa(mapa_1, "mapa_01_train_test")

# ------------------------------------------------------------------------------
# 2. Precio por m2 mediano de train por UPZ
# ------------------------------------------------------------------------------
precio_upz <- train %>%
  filter(bin_area_na == 0) %>%
  group_by(cat_upz_codigo) %>%
  summarise(n = n(), precio_m2 = median(obj_price / num_area) / 1e6, .groups = "drop") %>%
  mutate(precio_m2 = if_else(n >= 20, precio_m2, NA_real_))

mapa_2 <- ggplot() +
  geom_sf(data = localidades, fill = "grey92", color = NA) +
  geom_sf(data = left_join(upz, precio_upz, by = "cat_upz_codigo"),
          aes(fill = precio_m2), color = "white", linewidth = 0.15) +
  geom_sf(data = localidades, fill = NA, color = "grey40", linewidth = 0.3) +
  geom_sf(data = chapinero, fill = NA, color = color_chap, linewidth = 0.9) +
  scale_fill_viridis_c(option = "magma", direction = -1, na.value = "grey88",
                       name = "Precio por m2 mediano (millones de pesos)",
                       guide = guide_colorbar(barwidth = 12, barheight = 0.5,
                                              title.position = "top")) +
  coord_ciudad() +
  labs(title = "Chapinero está entre las zonas más caras de la ciudad",
       subtitle = "Precio por m2 mediano de los anuncios de train por UPZ. Gris: UPZ con menos de 20 anuncios.",
       caption = "Solo anuncios con área observada. Borde negro: Chapinero.") +
  tema_mapa
guardar_mapa(mapa_2, "mapa_02_precio_m2_upz")

# ------------------------------------------------------------------------------
# 3. Precio por m2 predicho para test (mejor envío)
# ------------------------------------------------------------------------------
pred_test <- read_csv(here("03_Output", "Submissions", "29_ensamble_embeddings.csv"),
                      show_col_types = FALSE)
test_pred <- test %>%
  inner_join(pred_test, by = "property_id") %>%
  filter(bin_area_na == 0) %>%
  mutate(precio_m2_pred = price / num_area / 1e6)

limite_m2 <- quantile(test_pred$precio_m2_pred, c(0.02, 0.98))
mapa_3 <- ggplot() +
  capa_chapinero() +
  geom_point(data = test_pred,
             aes(num_x, num_y, color = pmin(pmax(precio_m2_pred, limite_m2[1]), limite_m2[2])),
             size = 0.6, alpha = 0.7) +
  scale_color_viridis_c(option = "magma", direction = -1,
                        name = "Precio por m2 predicho (millones de pesos)",
                        guide = guide_colorbar(barwidth = 12, barheight = 0.5,
                                               title.position = "top")) +
  coord_chap() +
  labs(title = "Precio por m2 que predice el mejor modelo en Chapinero",
       subtitle = "Cada punto es un anuncio de test con área observada.\nPredicción del envío 29 (ensamble con embeddings).",
       caption = "Colores recortados entre los percentiles 2 y 98. Líneas grises: UPZ. Borde negro: Chapinero.") +
  tema_mapa
guardar_mapa(mapa_3, "mapa_03_chapinero_prediccion_test", ancho = 7, alto = 8.5)

# ------------------------------------------------------------------------------
# 4. Error fuera de muestra del mejor modelo alrededor de Chapinero
# Error = predicción / precio observado - 1. Positivo: sobreestima.
# ------------------------------------------------------------------------------
oof_29 <- readRDS(here("03_Output", "Predictions", "oof_29_ensamble_embeddings_localidad.rds"))
errores <- train %>%
  inner_join(oof_29, by = "property_id") %>%
  mutate(error_pct = exp(pred_log) / obj_price - 1)

errores_ventana <- errores %>%
  filter(between(num_x, ventana_chap$x[1], ventana_chap$x[2]),
         between(num_y, ventana_chap$y[1], ventana_chap$y[2]))
resumen_chap <- errores %>%
  filter(localidad == "CHAPINERO") %>%
  summarise(mae = mean(abs(exp(pred_log) - obj_price)) / 1e6,
            sobreestima = 100 * mean(error_pct > 0),
            error_mediano = 100 * median(error_pct))

mapa_4 <- ggplot() +
  capa_chapinero() +
  geom_point(data = arrange(errores_ventana, abs(error_pct)),
             aes(num_x, num_y, color = pmin(pmax(error_pct, -0.4), 0.4)),
             size = 0.9, alpha = 0.85) +
  scale_color_gradient2(low = "#2166AC", mid = "grey92", high = "#B2182B", midpoint = 0,
                        labels = label_percent(), limits = c(-0.4, 0.4),
                        name = "Error de predicción (azul: subestima; rojo: sobreestima)",
                        guide = guide_colorbar(barwidth = 12, barheight = 0.5,
                                               title.position = "top")) +
  coord_chap() +
  labs(title = "Dónde se equivoca el mejor modelo alrededor de Chapinero",
       subtitle = sprintf(paste0("Anuncios de train, predichos sin ver su localidad. En Chapinero: MAE de %.0f millones, ",
                                 "sobreestima el %.0f%%\nde los inmuebles y el error mediano es de %+.1f%%."),
                          resumen_chap$mae, resumen_chap$sobreestima, resumen_chap$error_mediano),
       caption = "Validación cruzada con localidades completas por fuera. Errores recortados en ±40%.") +
  tema_mapa
guardar_mapa(mapa_4, "mapa_04_chapinero_error_oof", ancho = 7, alto = 8.5)

# ------------------------------------------------------------------------------
# 5. Cobertura de test: anuncios de train a menos de 500 m
# ------------------------------------------------------------------------------
vecinos_500 <- frNN(as.matrix(train[, c("num_x", "num_y")]), eps = 500,
                    query = as.matrix(test[, c("num_x", "num_y")]), sort = FALSE)$id
cobertura <- test %>%
  mutate(n_train_500m = lengths(vecinos_500),
         categoria = cut(n_train_500m, breaks = c(-Inf, 0, 9, 29, Inf),
                         labels = c("Ninguno", "1 a 9", "10 a 29", "30 o más")))

mapa_5 <- ggplot() +
  capa_chapinero() +
  geom_point(data = cobertura, aes(num_x, num_y, color = categoria), size = 0.6, alpha = 0.8) +
  scale_color_manual(values = c("Ninguno" = "#B2182B", "1 a 9" = "#F4A582",
                                "10 a 29" = "#92C5DE", "30 o más" = "#2166AC"),
                     name = "Anuncios de train a menos de 500 m",
                     guide = guide_legend(override.aes = list(size = 2.5, alpha = 1))) +
  coord_chap() +
  labs(title = "Dónde el modelo tiene poca información para predecir",
       subtitle = sprintf("Anuncios de test según cuántos anuncios de train hay cerca. %.0f%% de test tiene menos de 10.",
                          100 * mean(cobertura$n_train_500m < 10)),
       caption = "Líneas grises: UPZ. Borde negro: Chapinero.") +
  tema_mapa
guardar_mapa(mapa_5, "mapa_05_chapinero_cobertura", ancho = 7, alto = 8.5)

# ------------------------------------------------------------------------------
# 6. Folds de la validación cruzada espacial
# ------------------------------------------------------------------------------
fun_fold_por_fila <- function(folds) {
  fold <- integer(nrow(train))
  for (i in seq_along(folds)) fold[folds[[i]]$val] <- i
  factor(fold)
}
folds_localidad <- fun_folds_cv(train, "localidad")
folds_bloques   <- fun_folds_cv(train, "bloques")
train_folds <- train %>%
  mutate(fold_localidad = fun_fold_por_fila(folds_localidad),
         fold_bloques   = fun_fold_por_fila(folds_bloques))

fold_de_localidad <- train_folds %>%
  count(localidad, fold_localidad) %>%
  group_by(localidad) %>%
  slice_max(n, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  select(localidad, fold = fold_localidad)

colores_fold <- c("#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E")

mapa_6a <- ggplot() +
  geom_sf(data = left_join(localidades, fold_de_localidad, by = "localidad"),
          aes(fill = fold), color = "white", linewidth = 0.3) +
  geom_sf(data = chapinero, fill = NA, color = color_chap, linewidth = 0.8) +
  scale_fill_manual(values = colores_fold, na.value = "grey90", na.translate = FALSE, name = "Fold") +
  coord_ciudad() +
  labs(title = "Localidades completas por fuera",
       subtitle = "Cada color se deja por fuera una vez para validar.") +
  tema_mapa

mapa_6b <- ggplot() +
  capa_ciudad(nombres = FALSE) +
  geom_point(data = train_folds, aes(num_x, num_y, color = fold_bloques), size = 0.15, alpha = 0.5) +
  scale_color_manual(values = colores_fold, name = "Fold",
                     guide = guide_legend(override.aes = list(size = 2.5, alpha = 1))) +
  coord_ciudad() +
  labs(title = "Bloques de 2 km",
       subtitle = "Cada anuncio toma el fold de su bloque.") +
  tema_mapa

mapa_6 <- (mapa_6a | mapa_6b) +
  plot_annotation(title = "Validación cruzada espacial: el modelo se evalúa en zonas que no vio",
                  caption = "Borde negro: Chapinero.",
                  theme = theme(plot.title = element_text(face = "bold", size = 14),
                                plot.caption = element_text(color = "grey40", size = 8, hjust = 0)))
guardar_mapa(mapa_6, "mapa_06_folds_espaciales", ancho = 12, alto = 8)

# ------------------------------------------------------------------------------
# 7. MAE por localidad de tres modelos, con la localidad por fuera
# La regresión lineal se estima aquí con los mismos folds de localidades y los
# predictores de la base final; las otras dos predicciones fuera de muestra
# vienen de los scripts de cada modelo.
# ------------------------------------------------------------------------------
# La matriz de diseño se arma con todo train para que las dummies de categorías
# que solo aparecen en la localidad por fuera existan en todos los folds; los
# coeficientes que no se pueden estimar en un fold (columnas sin variación) se
# toman como cero.
datos <- fun_datos_modelo()
x_lm <- model.matrix(reformulate(datos$predictores), data = datos$train)
y_lm <- datos$train$obj_log_price
pred_lm <- rep(NA_real_, nrow(datos$train))
for (f in folds_localidad) {
  coeficientes <- lm.fit(x_lm[f$train, , drop = FALSE], y_lm[f$train])$coefficients
  coeficientes[is.na(coeficientes)] <- 0
  pred_lm[f$val] <- as.numeric(x_lm[f$val, , drop = FALSE] %*% coeficientes)
}
oof_24 <- readRDS(here("03_Output", "Predictions", "oof_24_nnet_robusta_localidad.rds"))

predicciones <- bind_rows(
  tibble(property_id = datos$train$property_id, pred_log = pred_lm,
         modelo = "Regresión lineal"),
  oof_24 %>% mutate(modelo = "Red neuronal (24)"),
  oof_29 %>% mutate(modelo = "Ensamble (29, mejor envío)")
) %>%
  mutate(modelo = factor(modelo, levels = c("Regresión lineal", "Red neuronal (24)",
                                            "Ensamble (29, mejor envío)")))

mae_localidad <- predicciones %>%
  inner_join(train %>% select(property_id, localidad, obj_price), by = "property_id") %>%
  group_by(modelo, localidad) %>%
  summarise(n = n(), mae_millones = mean(abs(exp(pred_log) - obj_price)) / 1e6,
            .groups = "drop")
write_csv(mae_localidad, file.path(ruta_tablas, "mapa_mae_localidad_modelos.csv"))

mae_total <- mae_localidad %>%
  group_by(modelo) %>%
  summarise(mae = weighted.mean(mae_millones, n)) %>%
  mutate(etiqueta = sprintf("%s\nMAE total: %.0f millones", modelo, mae),
         etiqueta = factor(etiqueta, levels = etiqueta[order(modelo)]))
limite_mae <- quantile(mae_localidad$mae_millones, 0.95)

mapa_7 <- ggplot() +
  geom_sf(data = localidades %>%
            inner_join(mae_localidad, by = "localidad", relationship = "many-to-many") %>%
            left_join(mae_total %>% select(modelo, etiqueta), by = "modelo"),
          aes(fill = pmin(mae_millones, limite_mae)), color = "white", linewidth = 0.3) +
  geom_sf(data = chapinero, fill = NA, color = color_chap, linewidth = 0.8) +
  facet_wrap(~ etiqueta, nrow = 1) +
  scale_fill_viridis_c(option = "rocket", direction = -1,
                       name = "MAE con la localidad por fuera (millones de pesos)",
                       guide = guide_colorbar(barwidth = 14, barheight = 0.5,
                                              title.position = "top")) +
  coord_ciudad() +
  labs(title = "El ensamble reduce el error en casi todas las localidades",
       subtitle = "Error absoluto medio de cada localidad cuando el modelo se entrena sin ella.",
       caption = sprintf("Colores recortados en %.0f millones. Borde negro: Chapinero.", limite_mae)) +
  tema_mapa +
  theme(strip.text = element_text(size = 10, face = "bold", margin = margin(b = 4)))
guardar_mapa(mapa_7, "mapa_07_mae_localidad_modelos", ancho = 13, alto = 7)

# ------------------------------------------------------------------------------
# 8. Peso de los anuncios de train según su parecido con test
# ------------------------------------------------------------------------------
pesos <- readRDS(here("02_Data", "Processed", "pesos_adversarial_texto.rds"))
peso_upz <- train %>%
  inner_join(pesos, by = "property_id") %>%
  group_by(cat_upz_codigo) %>%
  summarise(n = n(), peso = mean(peso), .groups = "drop") %>%
  mutate(peso = if_else(n >= 20, peso, NA_real_))

mapa_8 <- ggplot() +
  geom_sf(data = localidades, fill = "grey92", color = NA) +
  geom_sf(data = left_join(upz, peso_upz, by = "cat_upz_codigo"),
          aes(fill = peso), color = "white", linewidth = 0.15) +
  geom_sf(data = localidades, fill = NA, color = "grey40", linewidth = 0.3) +
  geom_sf(data = chapinero, fill = NA, color = color_chap, linewidth = 0.9) +
  scale_fill_viridis_c(option = "mako", direction = -1, trans = "log10",
                       labels = label_number(accuracy = 0.1), na.value = "grey88",
                       name = "Peso promedio en el entrenamiento (1 = peso normal)",
                       guide = guide_colorbar(barwidth = 12, barheight = 0.5,
                                              title.position = "top")) +
  coord_ciudad() +
  labs(title = "El modelo con pesos aprende más de las zonas parecidas a test",
       subtitle = "Peso promedio por UPZ de los anuncios de train, según qué tanto se parecen a los de test.",
       caption = "Escala logarítmica. Gris: UPZ con menos de 20 anuncios. Borde negro: Chapinero.") +
  tema_mapa
guardar_mapa(mapa_8, "mapa_08_pesos_upz")

# ------------------------------------------------------------------------------
# 9. Contexto urbano: parques, TransMilenio y metro
# ------------------------------------------------------------------------------
ventana_sf <- st_as_sfc(st_bbox(c(xmin = ventana_ciudad$x[1], xmax = ventana_ciudad$x[2],
                                  ymin = ventana_ciudad$y[1], ymax = ventana_ciudad$y[2]),
                                crs = st_crs(crs_metros)))
parques <- fun_leer_capa("25_parques") %>% st_intersection(ventana_sf)
estaciones_tm <- fun_leer_capa("28a_estaciones_tm")
metro <- fun_leer_capa("32_metro")

mapa_9 <- ggplot() +
  capa_ciudad() +
  geom_sf(data = parques, fill = "#7FBC41", color = NA, alpha = 0.8) +
  geom_sf(data = st_centroid(estaciones_tm), aes(color = "Estación de TransMilenio"), size = 0.7) +
  geom_sf(data = st_centroid(metro), aes(color = "Estación de la línea 1 del metro"),
          size = 1.4, shape = 17) +
  geom_sf(data = chapinero, fill = NA, color = color_chap, linewidth = 0.9) +
  scale_color_manual(values = c("Estación de TransMilenio" = "#B2182B",
                                "Estación de la línea 1 del metro" = "#5E3C99"),
                     name = NULL,
                     guide = guide_legend(override.aes = list(size = 2.5))) +
  coord_ciudad() +
  labs(title = "Fuentes externas: parques y transporte",
       subtitle = "Verde: parques del Sistema Distrital de Parques. Puntos: estaciones de transporte masivo.",
       caption = "Fuente: Mapas Bogotá (IDECA). Borde negro: Chapinero.") +
  tema_mapa
guardar_mapa(mapa_9, "mapa_09_contexto_urbano")

message("Mapas guardados en ", ruta_figuras)
