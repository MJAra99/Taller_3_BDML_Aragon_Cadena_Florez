# ==============================================================================
# 02_negocio.R
# Objetivo: mapas y gráficas para la decisión de compra de la empresa. En este
#           negocio sobreestimar el precio destruye capital (se paga de más) y
#           subestimarlo, en el peor caso, hace perder el negocio. Todo se
#           calcula con las predicciones fuera de muestra del mejor modelo
#           (29_ensamble_embeddings, validación con localidades completas por
#           fuera) sobre los anuncios de train, cuyo precio sí se observa.
#           La razón r = predicción / precio observado resume el error: r > 1
#           es sobreestimación.
#           1. Mapa del riesgo de sobreestimación por UPZ: porcentaje de
#              anuncios con sobreestimación mayor al 10%.
#           2. Mapa del margen de seguridad por UPZ: cuánto por debajo de la
#              predicción hay que ofertar para que solo 1 de cada 10 ofertas
#              pague más que el precio observado. Si q90 es el percentil 90 de
#              r en la UPZ, el margen es 1 - 1 / q90.
#           3. Semáforo de compra para los anuncios de test, con el margen de
#              seguridad local (calculado con los 30 anuncios de train más
#              cercanos) y la cobertura (anuncios de train a menos de 500 m):
#              - verde: 30 o más anuncios de train a menos de 500 m y margen
#                local de hasta 15%; el modelo es confiable con un margen bajo;
#              - rojo: menos de 10 anuncios de train a menos de 500 m o margen
#                local mayor a 25%; el modelo no tiene información suficiente o
#                se equivoca mucho en la zona, y hace falta un avalúo;
#              - amarillo: el resto; se puede comprar con el margen local.
#           4. Curva de decisión del margen de seguridad. La empresa oferta la
#              predicción menos un margen m y revende al precio observado. El
#              vendedor acepta si la oferta es al menos su precio menos un
#              descuento d que está dispuesto a ceder (por vender rápido y sin
#              intermediarios); se comparan d de 5%, 10% y 15%. Para cada
#              margen se calcula la ganancia neta (precio observado menos
#              precio pagado) por cada 100 ofertas, el porcentaje de negocios
#              cerrados y el porcentaje de compras en las que se paga más que
#              el precio observado. Este último muestra la selección adversa:
#              los vendedores aceptan justo cuando el modelo sobreestima.
#           5. Kaggle frente a negocio: al multiplicar las predicciones por un
#              factor, cómo cambian el MAE y el porcentaje de sobreestimación.
#           Las curvas 4 y 5 se calculan en los anuncios de Chapinero de train y
#           en todo train ponderado por el parecido con test (pesos de
#           10_xgb_pesos_texto.R), que aproxima la composición de test.
# Entradas: 02_Data/Processed/db_{train,test}.rds
#           02_Data/Processed/pesos_adversarial_texto.rds
#           02_Data/Raw/MapasBogota/capas/{44_localidades,45_upz}.geojson
#           03_Output/Predictions/oof_29_ensamble_embeddings_localidad.rds
#           03_Output/Submissions/29_ensamble_embeddings.csv
# Salidas:  03_Output/Figures/negocio_*.{pdf,png}
#           03_Output/Tables/negocio_*.csv
# ==============================================================================

source(here::here("01_Scripts", "00_utils.R"))
pacman::p_load(sf, scales, patchwork)

ruta_figuras <- here("03_Output", "Figures")
ruta_tablas  <- here("03_Output", "Tables")
walk(c(ruta_figuras, ruta_tablas), dir.create, showWarnings = FALSE, recursive = TRUE)

descuentos_vendedor   <- c(0.05, 0.10, 0.15)   # descuentos que acepta el vendedor
riesgo_aceptado       <- 0.10   # proporción de ofertas que pueden pagar de más

guardar <- function(grafico, nombre, ancho = 7, alto = 8) {
  for (ext in c("pdf", "png")) {
    ggsave(file.path(ruta_figuras, paste0(nombre, ".", ext)), grafico,
           width = ancho, height = alto, dpi = 300, bg = "white")
  }
  message("Figura guardada: ", nombre)
}

# ------------------------------------------------------------------------------
# 0. Datos, predicciones fuera de muestra y capas de base
# ------------------------------------------------------------------------------
fun_sin_tildes <- function(x) str_to_upper(stringi::stri_trans_general(x, "Latin-ASCII"))

train <- readRDS(here("02_Data", "Processed", "db_train.rds")) %>%
  mutate(localidad = fun_sin_tildes(cat_localidad))
test  <- readRDS(here("02_Data", "Processed", "db_test.rds")) %>%
  mutate(localidad = fun_sin_tildes(cat_localidad))

oof <- train %>%
  inner_join(readRDS(here("03_Output", "Predictions", "oof_29_ensamble_embeddings_localidad.rds")),
             by = "property_id") %>%
  inner_join(readRDS(here("02_Data", "Processed", "pesos_adversarial_texto.rds")),
             by = "property_id") %>%
  mutate(pred = exp(pred_log),
         razon = pred / obj_price,
         es_chapinero = localidad == "CHAPINERO")

localidades <- fun_leer_capa("44_localidades") %>%
  transmute(localidad = fun_sin_tildes(LOCNOMBRE)) %>%
  filter(localidad != "SUMAPAZ")
chapinero <- localidades %>% filter(localidad == "CHAPINERO")
upz <- fun_leer_capa("45_upz") %>%
  transmute(cat_upz_codigo = as.character(CODIGO_UPZ), nombre_upz = NOMBRE)

ventana_ciudad <- list(x = range(train$num_x) + c(-1000, 1000),
                       y = range(train$num_y) + c(-1000, 1000))
# Ventana de Chapinero: el 98% central de los anuncios de test con 800 m de
# margen (la localidad incluye los cerros orientales, donde no hay anuncios).
ventana_chap <- list(x = unname(quantile(test$num_x, c(0.01, 0.99))) + c(-800, 800),
                     y = unname(quantile(test$num_y, c(0.01, 0.99))) + c(-800, 800))

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

tema_grafica <- theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 13),
        plot.subtitle = element_text(color = "grey30", size = 9.5),
        plot.caption = element_text(color = "grey40", size = 8, hjust = 0),
        legend.position = "bottom",
        panel.grid.minor = element_blank())

# Mapa de la ciudad con una variable por UPZ (UPZ con menos de 20 anuncios en
# gris) y Chapinero con borde negro.
fun_mapa_upz <- function(valores, variable, escala, titulo, subtitulo, nota) {
  ggplot() +
    geom_sf(data = localidades, fill = "grey92", color = NA) +
    geom_sf(data = left_join(upz, valores, by = "cat_upz_codigo"),
            aes(fill = .data[[variable]]), color = "white", linewidth = 0.15) +
    geom_sf(data = localidades, fill = NA, color = "grey40", linewidth = 0.3) +
    geom_sf(data = chapinero, fill = NA, color = "black", linewidth = 0.9) +
    escala +
    coord_ciudad() +
    labs(title = titulo, subtitle = subtitulo, caption = nota) +
    tema_mapa
}

# ------------------------------------------------------------------------------
# 1. y 2. Riesgo de sobreestimación y margen de seguridad por UPZ
# ------------------------------------------------------------------------------
riesgo_upz <- oof %>%
  group_by(cat_upz_codigo, cat_upz) %>%
  summarise(n = n(),
            sobreestima_10 = mean(razon > 1.10),
            margen = 1 - 1 / quantile(razon, 1 - riesgo_aceptado),
            .groups = "drop") %>%
  mutate(across(c(sobreestima_10, margen), ~ if_else(n >= 20, .x, NA_real_)))
write_csv(riesgo_upz %>% arrange(desc(margen)), file.path(ruta_tablas, "negocio_riesgo_upz.csv"))

resumen_chap <- oof %>%
  filter(es_chapinero) %>%
  summarise(sobreestima_10 = mean(razon > 1.10),
            margen = 1 - 1 / quantile(razon, 1 - riesgo_aceptado))

mapa_riesgo <- fun_mapa_upz(
  riesgo_upz, "sobreestima_10",
  scale_fill_distiller(palette = "Reds", direction = 1, labels = label_percent(),
                       na.value = "grey88",
                       name = "Anuncios sobreestimados en más del 10%",
                       guide = guide_colorbar(barwidth = 12, barheight = 0.5,
                                              title.position = "top")),
  titulo = "Dónde el modelo haría pagar de más",
  subtitulo = sprintf(paste0("Porcentaje de anuncios de train cuya predicción supera el precio en más del 10%%.\n",
                             "En Chapinero: %.0f%% de los anuncios."), 100 * resumen_chap$sobreestima_10),
  nota = "Predicciones del mejor modelo sin ver la localidad del anuncio. Gris: UPZ con menos de 20 anuncios.")
guardar(mapa_riesgo, "negocio_01_riesgo_sobreestimacion_upz")

mapa_margen <- fun_mapa_upz(
  riesgo_upz, "margen",
  scale_fill_distiller(palette = "YlOrRd", direction = 1, labels = label_percent(),
                       na.value = "grey88",
                       name = "Margen bajo la predicción para que solo 1 de cada 10 ofertas pague de más",
                       guide = guide_colorbar(barwidth = 14, barheight = 0.5,
                                              title.position = "top")),
  titulo = "Cuánto ofertar por debajo de la predicción en cada zona",
  subtitulo = sprintf(paste0("Si la empresa oferta la predicción menos este margen, paga más que el precio\n",
                             "en solo 1 de cada 10 compras. En Chapinero el margen es de %.0f%%."),
                      100 * resumen_chap$margen),
  nota = "Margen = 1 - 1 / percentil 90 de (predicción / precio). Gris: UPZ con menos de 20 anuncios.")
guardar(mapa_margen, "negocio_02_margen_seguridad_upz")

# ------------------------------------------------------------------------------
# 3. Semáforo de compra para los anuncios de test
# ------------------------------------------------------------------------------
coords_train <- as.matrix(oof[, c("num_x", "num_y")])
coords_test  <- as.matrix(test[, c("num_x", "num_y")])

vecinos_30 <- kNN(coords_train, k = 30, query = coords_test)$id
margen_local <- apply(vecinos_30, 1, function(ids) 1 - 1 / quantile(oof$razon[ids], 1 - riesgo_aceptado))
n_500m <- lengths(frNN(coords_train, eps = 500, query = coords_test, sort = FALSE)$id)

semaforo <- test %>%
  select(property_id, num_x, num_y, localidad) %>%
  mutate(margen_local = margen_local,
         n_train_500m = n_500m,
         semaforo = case_when(
           n_train_500m < 10 | margen_local > 0.25 ~ "Rojo: avalúo antes de ofertar",
           n_train_500m >= 30 & margen_local <= 0.15 ~ "Verde: ofertar con margen bajo",
           TRUE ~ "Amarillo: ofertar con margen local"
         ),
         semaforo = factor(semaforo, levels = c("Verde: ofertar con margen bajo",
                                                "Amarillo: ofertar con margen local",
                                                "Rojo: avalúo antes de ofertar")))
write_csv(semaforo %>% select(-num_x, -num_y), file.path(ruta_tablas, "negocio_semaforo_test.csv"))

conteo_semaforo <- semaforo %>% count(semaforo) %>% mutate(porc = 100 * n / sum(n))
write_csv(conteo_semaforo, file.path(ruta_tablas, "negocio_semaforo_resumen.csv"))
print(conteo_semaforo)
etiquetas_semaforo <- setNames(sprintf("%s (%.0f%%)", conteo_semaforo$semaforo, conteo_semaforo$porc),
                               conteo_semaforo$semaforo)

upz_ventana <- st_crop(upz, c(xmin = ventana_chap$x[1], xmax = ventana_chap$x[2],
                              ymin = ventana_chap$y[1], ymax = ventana_chap$y[2]))
mapa_semaforo <- ggplot() +
  geom_sf(data = localidades, fill = "grey97", color = "grey60", linewidth = 0.4) +
  geom_sf(data = upz_ventana, fill = NA, color = "grey75", linewidth = 0.25) +
  geom_sf(data = chapinero, fill = NA, color = "black", linewidth = 0.9) +
  geom_point(data = semaforo, aes(num_x, num_y, color = semaforo), size = 0.6, alpha = 0.8) +
  geom_sf_text(data = st_point_on_surface(upz_ventana),
               aes(label = str_wrap(str_to_title(nombre_upz), 12)),
               size = 1.9, color = "grey20", lineheight = 0.8, check_overlap = TRUE) +
  scale_color_manual(values = c("Verde: ofertar con margen bajo" = "#1A9850",
                                "Amarillo: ofertar con margen local" = "#F2B701",
                                "Rojo: avalúo antes de ofertar" = "#D73027"),
                     labels = etiquetas_semaforo, name = NULL,
                     guide = guide_legend(ncol = 1, override.aes = list(size = 3, alpha = 1))) +
  coord_chap() +
  labs(title = "¿Dónde puede la empresa comprar usando el modelo?",
       subtitle = paste0("Anuncios de test según la información cercana y el error del modelo en su zona.\n",
                         "Verde: 30 o más anuncios de train a menos de 500 m y margen local de hasta 15%.\n",
                         "Rojo: menos de 10 anuncios a menos de 500 m o margen local mayor a 25%."),
       caption = paste0("Margen local: margen para que solo 1 de cada 10 ofertas pague de más,\n",
                        "calculado con los 30 anuncios de train más cercanos.")) +
  tema_mapa
guardar(mapa_semaforo, "negocio_03_semaforo_test", ancho = 7, alto = 8.8)

# ------------------------------------------------------------------------------
# 4. Curva de decisión del margen de seguridad
# Se calcula en todo train ponderado por el parecido con test; la tabla también
# guarda el resultado en los anuncios de Chapinero de train.
# ------------------------------------------------------------------------------
# Resultados de ofertar pred * (1 - margen) cuando el vendedor acepta hasta un
# descuento `descuento`, con pesos `w` por anuncio.
fun_resultado_margen <- function(df, w, margen, descuento) {
  w <- rep_len(w, nrow(df))
  oferta <- df$pred * (1 - margen)
  acepta <- oferta >= df$obj_price * (1 - descuento)
  pierde <- acepta & oferta > df$obj_price
  tibble(margen = margen, descuento = descuento,
         porc_negocios = 100 * weighted.mean(acepta, w),
         porc_compras_pagando_de_mas = if (any(acepta)) 100 * sum(w * pierde) / sum(w * acepta) else NA_real_,
         ganancia_por_100_ofertas = 100 * sum(w * acepta * (df$obj_price - oferta)) / sum(w) / 1e6)
}

margenes <- seq(0, 0.30, by = 0.01)
escenarios <- expand_grid(margen = margenes, descuento = descuentos_vendedor)
curva <- bind_rows(
  pmap_dfr(escenarios, function(margen, descuento)
    fun_resultado_margen(oof, oof$peso, margen, descuento)) %>%
    mutate(grupo = "Train ponderado por parecido con test"),
  pmap_dfr(escenarios, function(margen, descuento)
    fun_resultado_margen(filter(oof, es_chapinero), 1, margen, descuento)) %>%
    mutate(grupo = "Chapinero (train)")
)
write_csv(curva, file.path(ruta_tablas, "negocio_curva_margen.csv"))

curva_graf <- curva %>%
  filter(grupo == "Train ponderado por parecido con test") %>%
  mutate(escenario = factor(sprintf("El vendedor cede hasta %.0f%%", 100 * descuento),
                            levels = sprintf("El vendedor cede hasta %.0f%%", 100 * descuentos_vendedor)))
mejor_margen <- curva_graf %>%
  group_by(escenario) %>%
  slice_max(ganancia_por_100_ofertas, n = 1, with_ties = FALSE) %>%
  ungroup()
write_csv(mejor_margen, file.path(ruta_tablas, "negocio_mejor_margen.csv"))
print(mejor_margen)

# Resumen de cada escenario para el subtítulo de la figura.
texto_escenarios <- mejor_margen %>%
  mutate(texto = if_else(
    ganancia_por_100_ofertas > 0,
    sprintf("Si el vendedor cede hasta %.0f%%, el mejor margen es %.0f%%: gana %s millones por cada 100 ofertas y cierra el %.0f%% de los negocios.",
            100 * descuento, 100 * margen,
            trimws(format(round(ganancia_por_100_ofertas), big.mark = ".", decimal.mark = ",")), porc_negocios),
    sprintf("Si el vendedor cede hasta %.0f%%, ningún margen hasta 30%% es rentable: solo acepta cuando el modelo le sobreestima la vivienda.",
            100 * descuento))) %>%
  pull(texto) %>%
  paste(collapse = "\n")

colores_escenario <- c("#B2182B", "#F2B701", "#1A9850")

g_ganancia <- curva_graf %>%
  ggplot(aes(margen, ganancia_por_100_ofertas, color = escenario)) +
  geom_hline(yintercept = 0, color = "grey40") +
  geom_line(linewidth = 1) +
  geom_point(data = filter(mejor_margen, ganancia_por_100_ofertas > 0), size = 3, show.legend = FALSE) +
  geom_text(data = filter(mejor_margen, ganancia_por_100_ofertas > 0),
            aes(label = sprintf("margen %.0f%%", 100 * margen)),
            vjust = -1.1, size = 3.1, show.legend = FALSE) +
  scale_x_continuous(labels = label_percent()) +
  scale_y_continuous(labels = label_comma(big.mark = ".", decimal.mark = ",")) +
  scale_color_manual(values = colores_escenario, name = NULL) +
  labs(title = "Ganancia neta por cada 100 ofertas",
       subtitle = "Precio de reventa (precio observado) menos el precio pagado.",
       x = "Margen de la oferta bajo la predicción", y = "Millones de pesos") +
  tema_grafica

g_seleccion <- curva_graf %>%
  ggplot(aes(margen, porc_compras_pagando_de_mas, color = escenario)) +
  geom_line(linewidth = 1) +
  geom_line(aes(y = porc_negocios), linewidth = 0.7, linetype = "dashed", show.legend = FALSE) +
  scale_x_continuous(labels = label_percent()) +
  scale_y_continuous(labels = function(x) paste0(x, "%"), limits = c(0, 100)) +
  scale_color_manual(values = colores_escenario, name = NULL) +
  labs(title = "Selección adversa",
       subtitle = "El vendedor acepta sobre todo cuando el modelo le sobreestima la vivienda.\nContinua: compras en las que se paga más que el precio. Punteada: negocios cerrados.",
       x = "Margen de la oferta bajo la predicción", y = "Porcentaje") +
  tema_grafica

g_curva <- (g_ganancia | g_seleccion) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")
g_curva <- g_curva +
  plot_annotation(
    title = "Regla de compra: ofertar la predicción menos un margen de seguridad",
    subtitle = texto_escenarios,
    caption = "Predicciones fuera de muestra del mejor modelo con localidades completas por fuera; anuncios de train ponderados por parecido con test.",
    theme = theme(plot.title = element_text(face = "bold", size = 14),
                  plot.subtitle = element_text(color = "grey30", size = 10),
                  plot.caption = element_text(color = "grey40", size = 8, hjust = 0)))
guardar(g_curva, "negocio_04_curva_margen", ancho = 13, alto = 6)

# ------------------------------------------------------------------------------
# 5. Kaggle frente a negocio: multiplicar las predicciones por un factor
# ------------------------------------------------------------------------------
fun_resultado_factor <- function(df, w, factor) {
  w <- rep_len(w, nrow(df))
  pred <- factor * df$pred
  tibble(factor = factor,
         mae_millones = weighted.mean(abs(pred - df$obj_price), w) / 1e6,
         porc_sobreestima = 100 * weighted.mean(pred > df$obj_price, w))
}
factores <- seq(0.80, 1.15, by = 0.01)
curva_factor <- bind_rows(
  map_dfr(factores, ~ fun_resultado_factor(filter(oof, es_chapinero), 1, .x)) %>%
    mutate(grupo = "Chapinero (train)"),
  map_dfr(factores, ~ fun_resultado_factor(oof, oof$peso, .x)) %>%
    mutate(grupo = "Train ponderado por parecido con test")
)
write_csv(curva_factor, file.path(ruta_tablas, "negocio_kaggle_vs_negocio.csv"))

mejor_factor <- curva_factor %>% group_by(grupo) %>% slice_min(mae_millones, n = 1, with_ties = FALSE)

g_mae <- curva_factor %>%
  ggplot(aes(factor, mae_millones, linetype = grupo)) +
  annotate("rect", xmin = -Inf, xmax = 1, ymin = -Inf, ymax = Inf, fill = "#2166AC", alpha = 0.06) +
  annotate("rect", xmin = 1, xmax = Inf, ymin = -Inf, ymax = Inf, fill = "#B2182B", alpha = 0.06) +
  geom_vline(xintercept = 1, color = "grey40") +
  geom_line(linewidth = 0.9, color = "grey20") +
  geom_point(data = mejor_factor, color = "#B2182B", size = 3) +
  labs(title = "Lo que premia Kaggle: el MAE",
       subtitle = "El MAE baja al subir un poco las predicciones.",
       x = "Factor que multiplica la predicción", y = "MAE (millones de pesos)", linetype = NULL) +
  tema_grafica

g_sobre <- curva_factor %>%
  ggplot(aes(factor, porc_sobreestima, linetype = grupo)) +
  annotate("rect", xmin = -Inf, xmax = 1, ymin = -Inf, ymax = Inf, fill = "#2166AC", alpha = 0.06) +
  annotate("rect", xmin = 1, xmax = Inf, ymin = -Inf, ymax = Inf, fill = "#B2182B", alpha = 0.06) +
  geom_vline(xintercept = 1, color = "grey40") +
  geom_line(linewidth = 0.9, color = "#B2182B") +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  labs(title = "Lo que le cuesta a la empresa: pagar de más",
       subtitle = "Subir las predicciones aumenta las compras sobreestimadas.",
       x = "Factor que multiplica la predicción", y = "Anuncios sobreestimados", linetype = NULL) +
  tema_grafica

g_factor <- (g_mae | g_sobre) +
  plot_annotation(
    title = "El objetivo de la competencia no es el objetivo del negocio",
    subtitle = paste0("Los envíos 18 y 19 multiplicaron las predicciones por 1,03 y 1,10 y bajaron el MAE en Kaggle, ",
                      "pero para comprar\nconviene lo contrario: ofertar por debajo de la predicción (zona azul)."),
    caption = "Predicciones fuera de muestra del mejor modelo con localidades completas por fuera.",
    theme = theme(plot.title = element_text(face = "bold", size = 14),
                  plot.subtitle = element_text(color = "grey30", size = 10),
                  plot.caption = element_text(color = "grey40", size = 8, hjust = 0)))
guardar(g_factor, "negocio_05_kaggle_vs_negocio", ancho = 12, alto = 5.5)

message("Figuras en ", ruta_figuras, " y tablas en ", ruta_tablas)
