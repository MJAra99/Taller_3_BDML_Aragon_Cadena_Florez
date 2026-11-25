# ==============================================================================
# 01_chapinero_vs_bogota.R
# Objetivo: describir en qué se diferencia Chapinero (donde está test) del resto
#           de Bogotá (donde está casi todo train):
#           - tablas: resumen por localidad y tabla de balance con diferencias
#             estandarizadas entre test y train;
#           - precio por m2 por localidad y por UPZ;
#           - características de las viviendas (tipo, área, habitaciones,
#             estrato);
#           - entorno: parques, comercio, transporte, catastro, comparados con
#             las demás localidades;
#           - menciones en el texto de los anuncios;
#           - relación entre el precio por m2 y la valorización catastral de
#             las manzanas cercanas, y soporte común de test dentro de train;
#           - mapa del precio por m2 con la ubicación de test.
#           Se comparan tres grupos:
#             * Test (Chapinero): todo test (94 % en Chapinero), sin precio;
#             * Chapinero (train): los 305 anuncios de train en Chapinero;
#             * Resto de Bogotá (train): los demás anuncios de train.
#           El precio por m2 solo se calcula con el área observada (no con el
#           área imputada en 05_merge.R, que es la mediana y distorsiona la
#           distribución).
# Entradas: 02_Data/Processed/db_{train,test}.rds
# Salidas:  03_Output/Figures/eda_*.{pdf,png}
#           03_Output/Tables/eda_*.{csv,tex}
# ==============================================================================

source(here::here("01_Scripts", "00_utils.R"))
pacman::p_load(scales, patchwork, kableExtra, hexbin)

ruta_fig <- here("03_Output", "Figures")
ruta_tab <- here("03_Output", "Tables")
dir.create(ruta_fig, recursive = TRUE, showWarnings = FALSE)
dir.create(ruta_tab, recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------------------------
# 0. Estilo de las gráficas y funciones de exportación
# Colores: azul para el resto de Bogotá, naranja para test (Chapinero) y verde
# agua para los anuncios de train en Chapinero. Gris para las demás
# localidades cuando se resalta un solo grupo.
# ------------------------------------------------------------------------------
col_resto      <- "#2a78d6"
col_chap_train <- "#1baf7a"
col_test       <- "#eb6834"
col_gris       <- "#c3c2b7"
col_texto      <- "#52514e"

colores_grupo <- c("Resto de Bogotá (train)" = col_resto,
                   "Chapinero (train)"       = col_chap_train,
                   "Test (Chapinero)"        = col_test)

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

# Exporta una tabla a csv (para revisar) y a LaTeX (para las diapositivas).
fun_tabla_tex <- function(df, nombre, titulo, digitos = 1) {
  write_csv(df, file.path(ruta_tab, paste0(nombre, ".csv")))
  tex <- kbl(df, format = "latex", booktabs = TRUE, digits = digitos,
             caption = titulo, linesep = "",
             format.args = list(big.mark = ".", decimal.mark = ",")) %>%
    kable_styling(latex_options = c("hold_position", "scale_down"))
  writeLines(as.character(tex), file.path(ruta_tab, paste0(nombre, ".tex")))
}

# Valores de una variable sin las observaciones imputadas: si existe la dummy
# bin_<variable>_na creada en 05_merge.R, las filas imputadas quedan en NA.
fun_valores <- function(df, variable) {
  x <- df[[variable]]
  bandera <- paste0("bin_", sub("^num_", "", variable), "_na")
  if (bandera %in% names(df)) x[df[[bandera]] == 1] <- NA
  x
}

# Diferencia estandarizada: diferencia de medias sobre la desviación estándar
# combinada. Por convención, |d| > 0,25 indica un desbalance importante.
fun_dif_std <- function(a, b) {
  (mean(a, na.rm = TRUE) - mean(b, na.rm = TRUE)) /
    sqrt((var(a, na.rm = TRUE) + var(b, na.rm = TRUE)) / 2)
}

# ------------------------------------------------------------------------------
# 1. Datos
# ------------------------------------------------------------------------------
train <- readRDS(here("02_Data", "Processed", "db_train.rds"))
test  <- readRDS(here("02_Data", "Processed", "db_test.rds"))

db <- bind_rows(train %>% mutate(muestra = "train"),
                test  %>% mutate(muestra = "test")) %>%
  mutate(
    grupo = case_when(muestra == "test"            ~ "Test (Chapinero)",
                      cat_localidad == "CHAPINERO" ~ "Chapinero (train)",
                      TRUE                         ~ "Resto de Bogotá (train)"),
    grupo = factor(grupo, levels = names(colores_grupo)),
    localidad = str_to_title(cat_localidad),
    upz       = str_to_title(cat_upz),
    # Área y precio por m2 solo con el área observada.
    area_real     = if_else(bin_area_na == 0, num_area, NA_real_),
    precio_m2     = obj_price / area_real,
    # El valor de referencia del terreno tiene ceros que corresponden a
    # manzanas sin dato.
    num_vref_m2_200m   = na_if(num_vref_m2_200m, 0),
    # El avalúo por manzana tiene una cola muy larga; se usa en logaritmo.
    log_avaluo_mz_200m = log(num_avaluo_mz_200m)
  )

# Precio por m2 sin los extremos (errores de área o de precio), para gráficas.
lim_pm2 <- quantile(db$precio_m2, c(0.005, 0.995), na.rm = TRUE)
db <- db %>%
  mutate(precio_m2 = if_else(between(precio_m2, lim_pm2[1], lim_pm2[2]),
                             precio_m2, NA_real_))

# Revisión rápida de calidad de los datos.
message("Observaciones por grupo:")
print(count(db, grupo))
message("Porcentaje con área imputada por grupo:")
print(db %>% group_by(grupo) %>% summarise(pct_area_imputada = 100 * mean(bin_area_na)))
message("Anuncios con valor de referencia del m2 en cero (tratados como NA): ",
        sum(is.na(db$num_vref_m2_200m)))
message("Avalúo por manzana: mediana ", round(median(db$num_avaluo_mz_200m) / 1e6),
        " millones, máximo ", round(max(db$num_avaluo_mz_200m) / 1e6), " millones")

# ------------------------------------------------------------------------------
# 2. Tabla 1: resumen por localidad
# ------------------------------------------------------------------------------
resumen_loc <- db %>%
  mutate(unidad = if_else(muestra == "test", "Test (Chapinero)", localidad)) %>%
  group_by(unidad) %>%
  summarise(
    `Anuncios`                    = n(),
    `Precio mediano (mill.)`      = median(obj_price, na.rm = TRUE) / 1e6,
    `Precio m² mediano (mill.)`   = median(precio_m2, na.rm = TRUE) / 1e6,
    `Área mediana (m²)`           = median(area_real, na.rm = TRUE),
    `Casas (%)`                   = 100 * mean(bin_casa),
    `Habitaciones (media)`        = mean(num_habitac, na.rm = TRUE),
    `Estrato (media)`             = mean(num_estrato_200m, na.rm = TRUE),
    `Valor ref. m² terreno (mill.)` = median(num_vref_m2_200m, na.rm = TRUE) / 1e6,
    .groups = "drop"
  ) %>%
  filter(`Anuncios` >= 50) %>%
  arrange(desc(`Precio m² mediano (mill.)`)) %>%
  rename(Localidad = unidad)

fun_tabla_tex(resumen_loc, "eda_tabla_resumen_localidad",
              "Características de los anuncios por localidad (localidades con al menos 50 anuncios)")
print(resumen_loc)

# ------------------------------------------------------------------------------
# 3. Tabla 2 y figura: balance entre test y train
# Se compara la media de test con la del resto de Bogotá y con la de los
# anuncios de train en Chapinero. Si Chapinero (train) se parece a test y el
# resto de Bogotá no, esos 305 anuncios son las observaciones más valiosas de
# train.
# ------------------------------------------------------------------------------
vars_balance <- tribble(
  ~variable,                 ~etiqueta,                                  ~bloque,
  "bin_casa",                "Es casa",                                  "Vivienda",
  "area_real",               "Área (m²)",                                "Vivienda",
  "num_habitac",             "Habitaciones",                             "Vivienda",
  "num_banos",               "Baños",                                    "Vivienda",
  "num_parqueaderos",        "Parqueaderos",                             "Vivienda",
  "num_area_por_habitac",    "Área por habitación",                      "Vivienda",
  "num_piso_texto",          "Piso del apartamento",                     "Vivienda",
  "bin_terraza",             "Menciona terraza o balcón",                "Texto",
  "bin_chimenea",            "Menciona chimenea",                        "Texto",
  "bin_vista",               "Menciona vista",                           "Texto",
  "bin_exterior",            "Menciona exterior",                        "Texto",
  "bin_remodelado",          "Menciona remodelado",                      "Texto",
  "bin_ascensor",            "Menciona ascensor",                        "Texto",
  "bin_gimnasio",            "Menciona gimnasio",                        "Texto",
  "bin_conjunto",            "Menciona conjunto cerrado",                "Texto",
  "bin_jardin",              "Menciona jardín o zonas verdes",           "Texto",
  "bin_bbq",                 "Menciona BBQ",                             "Texto",
  "bin_vigilancia",          "Menciona vigilancia",                      "Texto",
  "num_palabras",            "Palabras en la descripción",               "Texto",
  "num_estrato_200m",        "Estrato (200 m)",                          "Entorno",
  "num_vref_m2_200m",        "Valor de referencia m² terreno (200 m)",   "Entorno",
  "num_avaluo_ph_m2_200m",   "Avalúo PH por m² (200 m)",                 "Entorno",
  "log_avaluo_mz_200m",      "Log avalúo por manzana (200 m)",           "Entorno",
  "num_pisos_200m",          "Pisos de las construcciones (200 m)",      "Entorno",
  "num_predios_mz_200m",     "Predios por manzana (200 m)",              "Entorno",
  "num_prop_comer_300m",     "Proporción de uso comercial (300 m)",      "Entorno",
  "num_prop_resid_300m",     "Proporción de uso residencial (300 m)",    "Entorno",
  "num_dist_parque",         "Distancia al parque más cercano",          "Entorno",
  "num_area_parque",         "Área del parque más cercano",              "Entorno",
  "num_arboles_300m",        "Árboles (300 m)",                          "Entorno",
  "num_restaurantes_500m",   "Restaurantes (500 m)",                     "Entorno",
  "num_bares_500m",          "Bares (500 m)",                            "Entorno",
  "num_estab_500m",          "Establecimientos de comercio (500 m)",     "Entorno",
  "num_cultura_1km",         "Equipamientos culturales (1 km)",          "Entorno",
  "num_galerias_1km",        "Galerías (1 km)",                          "Entorno",
  "num_ies_1km",             "Instituciones de educación superior (1 km)", "Entorno",
  "num_colegios_1km",        "Colegios (1 km)",                          "Entorno",
  "num_ips_1km",             "IPS (1 km)",                               "Entorno",
  "num_dist_estacion_tm",    "Distancia a estación de TransMilenio",     "Entorno",
  "num_dist_cc",             "Distancia a centro comercial",             "Entorno",
  "num_dist_zona_turistica", "Distancia a zona turística",               "Entorno",
  "num_atracos_celda",       "Atracos en la celda",                      "Entorno",
  "num_luminarias_ha",       "Luminarias por hectárea",                  "Entorno",
  "num_altura_msnm",         "Altura (msnm)",                            "Entorno",
  "num_espacio_publico_upz", "Espacio público (UPZ)",                    "Entorno"
)

balance <- pmap_dfr(vars_balance, function(variable, etiqueta, bloque) {
  x       <- fun_valores(db, variable)
  x_test  <- x[db$grupo == "Test (Chapinero)"]
  x_chap  <- x[db$grupo == "Chapinero (train)"]
  x_resto <- x[db$grupo == "Resto de Bogotá (train)"]
  tibble(bloque, etiqueta, variable,
         media_test        = mean(x_test,  na.rm = TRUE),
         media_chap_train  = mean(x_chap,  na.rm = TRUE),
         media_resto       = mean(x_resto, na.rm = TRUE),
         dif_std_resto     = fun_dif_std(x_test, x_resto),
         dif_std_chap      = fun_dif_std(x_test, x_chap))
}) %>%
  arrange(desc(abs(dif_std_resto)))

fun_tabla_tex(
  balance %>%
    transmute(Bloque = bloque, Variable = etiqueta,
              `Test (Chapinero)` = media_test,
              `Chapinero (train)` = media_chap_train,
              `Resto de Bogotá (train)` = media_resto,
              `Dif. est. vs. resto` = dif_std_resto,
              `Dif. est. vs. Chapinero train` = dif_std_chap),
  "eda_tabla_balance",
  "Balance de características: test frente a train (diferencias estandarizadas)",
  digitos = 2
)
print(balance, n = 50)

# Figura: las 25 variables con mayor desbalance frente al resto de Bogotá.
p_balance <- balance %>%
  slice_max(abs(dif_std_resto), n = 25) %>%
  mutate(etiqueta = fct_reorder(etiqueta, dif_std_resto)) %>%
  pivot_longer(c(dif_std_resto, dif_std_chap),
               names_to = "comparacion", values_to = "dif") %>%
  mutate(comparacion = dplyr::recode(comparacion,
                              dif_std_resto = "Test vs. resto de Bogotá (train)",
                              dif_std_chap  = "Test vs. Chapinero (train)")) %>%
  ggplot(aes(dif, etiqueta, color = comparacion, shape = comparacion)) +
  annotate("rect", xmin = -0.25, xmax = 0.25, ymin = -Inf, ymax = Inf,
           fill = "#f0efec") +
  geom_vline(xintercept = 0, color = col_texto, linewidth = 0.3) +
  geom_point(size = 2.5) +
  facet_grid(bloque ~ ., scales = "free_y", space = "free_y") +
  scale_color_manual(values = c("Test vs. resto de Bogotá (train)" = col_resto,
                                "Test vs. Chapinero (train)"       = col_chap_train)) +
  scale_shape_manual(values = c("Test vs. resto de Bogotá (train)" = 16,
                                "Test vs. Chapinero (train)"       = 17)) +
  labs(title = "¿Qué hace distinta a Chapinero?",
       subtitle = "Diferencia estandarizada de medias entre test y train. La banda gris marca |d| < 0,25.",
       x = "Diferencia estandarizada (positivo: mayor en test)", y = NULL,
       caption = "Las 25 variables con mayor diferencia frente al resto de Bogotá. Variables imputadas excluidas del cálculo.") +
  theme(strip.text.y = element_text(angle = 0))

fun_guardar(p_balance, "eda_balance_test_train", ancho = 8.5, alto = 7.5)

# ------------------------------------------------------------------------------
# 4. Precio por m2 por localidad y por UPZ (solo train)
# ------------------------------------------------------------------------------
d_pm2_loc <- db %>%
  filter(muestra == "train", !is.na(precio_m2)) %>%
  add_count(localidad) %>%
  filter(n >= 50) %>%
  mutate(localidad = fct_reorder(localidad, precio_m2, median),
         es_chap   = localidad == "Chapinero")

n_loc <- d_pm2_loc %>% distinct(localidad, n)

p_pm2_loc <- ggplot(d_pm2_loc, aes(precio_m2 / 1e6, localidad, fill = es_chap)) +
  geom_boxplot(outlier.shape = NA, width = 0.6, linewidth = 0.3, color = col_texto) +
  geom_text(data = n_loc, aes(x = 22, y = localidad, label = paste0("n = ", n)),
            inherit.aes = FALSE, hjust = 1, size = 2.8, color = col_texto) +
  scale_x_log10(breaks = c(1, 2, 3, 5, 8, 12, 20)) +
  coord_cartesian(xlim = c(1, 22)) +
  scale_fill_manual(values = c(`TRUE` = col_test, `FALSE` = col_gris), guide = "none") +
  labs(title = "Chapinero tiene el precio por m² más alto de la muestra",
       subtitle = "Precio por m² de los anuncios de train por localidad (millones de COP, escala logarítmica)",
       x = "Precio por m² (millones de COP)", y = NULL,
       caption = "Solo anuncios con área observada. Localidades con al menos 50 anuncios. Cajas sin valores atípicos.")

fun_guardar(p_pm2_loc, "eda_precio_m2_localidad", ancho = 8, alto = 6)

# Precio por m2 en las UPZ donde está test: cuántos anuncios de train hay en
# cada una para "anclar" las predicciones.
upz_test <- db %>%
  filter(muestra == "test") %>%
  count(upz, name = "n_test") %>%
  filter(n_test >= 50)

d_upz <- db %>%
  filter(muestra == "train", !is.na(precio_m2), upz %in% upz_test$upz) %>%
  add_count(upz, name = "n_train") %>%
  left_join(upz_test, by = "upz") %>%
  mutate(etiqueta_upz = paste0(upz, "\ntrain: ", n_train, " | test: ", n_test),
         etiqueta_upz = fct_reorder(etiqueta_upz, precio_m2, median),
         en_chapinero = localidad == "Chapinero")

p_upz <- ggplot(d_upz, aes(precio_m2 / 1e6, etiqueta_upz, fill = en_chapinero)) +
  geom_boxplot(outlier.shape = NA, width = 0.6, linewidth = 0.3, color = col_texto) +
  geom_jitter(height = 0.15, size = 0.6, alpha = 0.25, color = col_texto) +
  scale_x_log10(breaks = c(2, 3, 5, 8, 12, 20)) +
  scale_fill_manual(values = c(`TRUE` = col_test, `FALSE` = col_gris),
                    labels = c(`TRUE` = "UPZ de Chapinero", `FALSE` = "UPZ de otras localidades")) +
  labs(title = "Pocos anuncios de train en las UPZ donde está test",
       subtitle = "Precio por m² de train en las UPZ con al menos 50 anuncios de test (millones de COP)",
       x = "Precio por m² (millones de COP, escala logarítmica)", y = NULL,
       caption = "Las etiquetas muestran el número de anuncios de train con área observada y el número de anuncios de test en cada UPZ.")

fun_guardar(p_upz, "eda_precio_m2_upz_test", ancho = 8, alto = 5)

# ------------------------------------------------------------------------------
# 5. Cómo son las viviendas: tipo, habitaciones, área y estrato
# ------------------------------------------------------------------------------
p_tipo <- db %>%
  group_by(grupo) %>%
  summarise(pct_casas = 100 * mean(bin_casa)) %>%
  ggplot(aes(pct_casas, fct_rev(grupo), fill = grupo)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = paste0(round(pct_casas, 1), " %")), hjust = -0.15, size = 3,
            color = col_texto) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.2))) +
  scale_fill_manual(values = colores_grupo) +
  labs(title = "A. Casas (% de los anuncios)", x = NULL, y = NULL) +
  theme(axis.text.x = element_blank(), panel.grid.major = element_blank())

p_habitac <- db %>%
  filter(!is.na(num_habitac), num_habitac >= 1) %>%
  mutate(habitac = factor(pmin(num_habitac, 5), levels = 1:5,
                          labels = c("1", "2", "3", "4", "5+"))) %>%
  count(grupo, habitac) %>%
  group_by(grupo) %>%
  mutate(pct = 100 * n / sum(n)) %>%
  ggplot(aes(habitac, pct, fill = grupo)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.75) +
  scale_fill_manual(values = colores_grupo) +
  labs(title = "B. Habitaciones (% de cada grupo)", x = "Habitaciones", y = NULL)

p_area <- db %>%
  filter(!is.na(area_real)) %>%
  ggplot(aes(area_real, color = grupo)) +
  geom_density(linewidth = 0.8, adjust = 1.2) +
  scale_x_log10(breaks = c(25, 50, 100, 200, 400, 800)) +
  scale_color_manual(values = colores_grupo) +
  labs(title = "C. Área observada (m², escala logarítmica)", x = "Área (m²)", y = "Densidad")

p_estrato <- db %>%
  filter(!is.na(num_estrato_200m)) %>%
  mutate(estrato = factor(round(num_estrato_200m))) %>%
  count(grupo, estrato) %>%
  group_by(grupo) %>%
  mutate(pct = 100 * n / sum(n)) %>%
  ggplot(aes(estrato, pct, fill = grupo)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.75) +
  scale_fill_manual(values = colores_grupo) +
  labs(title = "D. Estrato predominante a 200 m (% de cada grupo)", x = "Estrato", y = NULL)

p_viviendas <- (p_tipo | p_habitac) / (p_area | p_estrato) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "En test casi todo son apartamentos de estrato 6, con menos habitaciones que en el resto de Bogotá",
    caption = "Test (Chapinero): todo test. Chapinero (train): 305 anuncios de train en Chapinero. Área solo cuando es observada.",
    theme = theme(plot.title = element_text(face = "bold", size = 13))
  ) &
  theme(legend.position = "top")

fun_guardar(p_viviendas, "eda_caracteristicas_viviendas", ancho = 10, alto = 7.5)

# ------------------------------------------------------------------------------
# 6. Entorno: Chapinero frente a las demás localidades
# Cada panel es una variable. Se recortan los valores por encima del percentil
# 99 de cada variable para que las cajas sean legibles.
# ------------------------------------------------------------------------------
vars_entorno <- c(
  num_estrato_200m      = "Estrato (200 m)",
  num_vref_m2_200m      = "Valor ref. m² terreno (millones, 200 m)",
  num_avaluo_ph_m2_200m = "Avalúo PH por m² (millones, 200 m)",
  num_dist_parque       = "Distancia al parque más cercano (m)",
  num_arboles_300m      = "Árboles (300 m)",
  num_restaurantes_500m = "Restaurantes (500 m)",
  num_bares_500m        = "Bares (500 m)",
  num_estab_500m        = "Establecimientos de comercio (500 m)",
  num_dist_estacion_tm  = "Distancia a estación de TransMilenio (m)"
)

d_entorno <- db %>%
  mutate(unidad = case_when(muestra == "test" ~ "Test (Chapinero)",
                            localidad == "Chapinero" ~ "Chapinero (train)",
                            TRUE ~ localidad),
         num_vref_m2_200m      = num_vref_m2_200m / 1e6,
         num_avaluo_ph_m2_200m = num_avaluo_ph_m2_200m / 1e6) %>%
  add_count(unidad) %>%
  filter(n >= 300) %>%
  select(unidad, all_of(names(vars_entorno))) %>%
  pivot_longer(-unidad, names_to = "variable", values_to = "valor") %>%
  filter(!is.na(valor)) %>%
  group_by(variable) %>%
  mutate(valor = pmin(valor, quantile(valor, 0.99))) %>%
  ungroup() %>%
  mutate(etiqueta = factor(vars_entorno[variable], levels = vars_entorno),
         resaltado = case_when(unidad == "Test (Chapinero)"  ~ "Test (Chapinero)",
                               unidad == "Chapinero (train)" ~ "Chapinero (train)",
                               TRUE ~ "Otras localidades (train)"))

# Las localidades se ordenan por el valor de referencia del m2 de terreno.
orden_unidades <- d_entorno %>%
  filter(variable == "num_vref_m2_200m") %>%
  group_by(unidad) %>%
  summarise(m = median(valor)) %>%
  arrange(m) %>%
  pull(unidad)

p_entorno <- d_entorno %>%
  mutate(unidad = factor(unidad, levels = orden_unidades)) %>%
  ggplot(aes(valor, unidad, fill = resaltado)) +
  geom_boxplot(outlier.shape = NA, width = 0.65, linewidth = 0.25, color = col_texto) +
  facet_wrap(~ etiqueta, scales = "free_x", ncol = 3) +
  scale_fill_manual(values = c("Test (Chapinero)"          = col_test,
                               "Chapinero (train)"         = col_chap_train,
                               "Otras localidades (train)" = col_gris)) +
  labs(title = "Entorno de las propiedades: Chapinero frente a las demás localidades",
       subtitle = "Localidades con al menos 300 anuncios, ordenadas por el valor de referencia del m² de terreno",
       x = NULL, y = NULL,
       caption = "Valores recortados en el percentil 99 de cada variable. Cajas sin valores atípicos.") +
  theme(axis.text.y = element_text(size = 8))

fun_guardar(p_entorno, "eda_entorno_localidades", ancho = 11, alto = 9)

# ------------------------------------------------------------------------------
# 7. Lo que dicen los anuncios: menciones en título y descripción
# ------------------------------------------------------------------------------
vars_texto <- vars_balance %>% filter(bloque == "Texto", str_starts(variable, "bin_"))

d_texto <- db %>%
  group_by(grupo) %>%
  summarise(across(all_of(vars_texto$variable), ~ 100 * mean(.x, na.rm = TRUE))) %>%
  pivot_longer(-grupo, names_to = "variable", values_to = "pct") %>%
  left_join(vars_texto, by = "variable") %>%
  mutate(etiqueta = str_remove(etiqueta, "^Menciona "),
         etiqueta = str_to_sentence(etiqueta))

orden_texto <- d_texto %>%
  select(etiqueta, grupo, pct) %>%
  pivot_wider(names_from = grupo, values_from = pct) %>%
  mutate(dif = `Test (Chapinero)` - `Resto de Bogotá (train)`) %>%
  arrange(dif) %>%
  pull(etiqueta)

p_texto <- d_texto %>%
  mutate(etiqueta = factor(etiqueta, levels = orden_texto)) %>%
  ggplot(aes(pct, etiqueta)) +
  geom_line(aes(group = etiqueta), color = col_gris, linewidth = 0.8) +
  geom_point(aes(color = grupo), size = 2.8) +
  scale_color_manual(values = colores_grupo) +
  labs(title = "Los anuncios de Chapinero hablan de vista, chimenea y exterior; menos de conjunto y jardín",
       subtitle = "Porcentaje de anuncios que mencionan cada característica en el título o la descripción",
       x = "% de anuncios", y = NULL,
       caption = "Ordenado por la diferencia entre test y el resto de Bogotá.")

fun_guardar(p_texto, "eda_menciones_texto", ancho = 8.5, alto = 5.5)

# ------------------------------------------------------------------------------
# 8. Precio por m2 y valorización catastral de las manzanas cercanas
# Cada panel relaciona el precio por m2 de train con una medida catastral.
# La banda naranja marca el rango p10-p90 de esa medida en test: muestra si
# test cae dentro de la nube de train (soporte común) o en su extremo.
# ------------------------------------------------------------------------------
vars_catastro <- c(
  num_vref_m2_200m      = "Valor de referencia del m² de terreno (millones, 200 m)",
  num_avaluo_ph_m2_200m = "Avalúo catastral PH por m² (millones, 200 m)",
  num_avaluo_mz_200m    = "Avalúo catastral por manzana (millones, 200 m)"
)

fun_scatter_catastro <- function(variable, etiqueta) {
  d <- db %>%
    filter(muestra == "train", !is.na(precio_m2), !is.na(.data[[variable]])) %>%
    mutate(x = .data[[variable]] / 1e6)
  rango_test <- quantile(db[[variable]][db$muestra == "test"] / 1e6,
                         c(0.1, 0.9), na.rm = TRUE)
  ggplot(d, aes(x, precio_m2 / 1e6)) +
    annotate("rect", xmin = rango_test[1], xmax = rango_test[2],
             ymin = -Inf, ymax = Inf, fill = col_test, alpha = 0.12) +
    geom_point(data = ~ filter(.x, grupo == "Resto de Bogotá (train)"),
               color = col_resto, alpha = 0.06, size = 0.5) +
    geom_point(data = ~ filter(.x, grupo == "Chapinero (train)"),
               color = col_chap_train, alpha = 0.7, size = 1.1) +
    geom_smooth(aes(color = grupo), method = "lm", formula = y ~ x,
                se = TRUE, linewidth = 0.9, alpha = 0.15) +
    scale_x_log10() +
    scale_y_log10(breaks = c(1, 2, 3, 5, 8, 12, 20)) +
    scale_color_manual(values = colores_grupo[1:2]) +
    labs(x = etiqueta, y = "Precio por m² (millones)")
}

paneles_catastro <- imap(vars_catastro, ~ fun_scatter_catastro(.y, .x))

p_catastro <- wrap_plots(paneles_catastro, ncol = 3) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "El precio por m² sigue a la valorización catastral, pero test vive en el extremo de la distribución",
    subtitle = "Anuncios de train, escalas logarítmicas. Rectas MCO por grupo. Banda naranja: rango p10–p90 de test.",
    caption = "Medidas catastrales: mediana de las manzanas a menos de 200 m, del año catastral del anuncio.",
    theme = theme(plot.title = element_text(face = "bold", size = 12))
  ) &
  theme(legend.position = "top")

fun_guardar(p_catastro, "eda_precio_vs_catastro", ancho = 12, alto = 4.8)

# Soporte común: distribución de las medidas catastrales y del estrato en los
# tres grupos.
p_soporte <- db %>%
  transmute(grupo,
            `Valor ref. m² terreno (millones)` = num_vref_m2_200m / 1e6,
            `Avalúo PH por m² (millones)`      = num_avaluo_ph_m2_200m / 1e6,
            `Log avalúo por manzana`           = log_avaluo_mz_200m) %>%
  pivot_longer(-grupo, names_to = "variable", values_to = "valor") %>%
  filter(!is.na(valor)) %>%
  group_by(variable) %>%
  filter(between(valor, quantile(valor, 0.005), quantile(valor, 0.995))) %>%
  ungroup() %>%
  ggplot(aes(valor, color = grupo, fill = grupo)) +
  geom_density(alpha = 0.12, linewidth = 0.8, adjust = 1.2) +
  facet_wrap(~ variable, scales = "free", ncol = 3) +
  scale_color_manual(values = colores_grupo) +
  scale_fill_manual(values = colores_grupo) +
  labs(title = "Soporte común: ¿hay en train propiedades con un entorno catastral como el de test?",
       x = NULL, y = "Densidad")

fun_guardar(p_soporte, "eda_soporte_comun_catastro", ancho = 11, alto = 4)

# ------------------------------------------------------------------------------
# 9. ¿Las relaciones son las mismas en Chapinero?
# Correlación de Spearman entre el log del precio por m2 y cada variable de
# entorno, por separado en Chapinero (train) y en el resto de Bogotá. Con
# solo 305 anuncios en Chapinero las correlaciones de ese grupo son ruidosas
# (error estándar aproximado de 0,06).
# ------------------------------------------------------------------------------
vars_cor <- vars_balance %>% filter(bloque == "Entorno")

cors <- db %>%
  filter(muestra == "train", !is.na(precio_m2)) %>%
  group_by(grupo) %>%
  summarise(across(all_of(vars_cor$variable),
                   ~ cor(.x, log(precio_m2), method = "spearman",
                         use = "pairwise.complete.obs"))) %>%
  pivot_longer(-grupo, names_to = "variable", values_to = "rho") %>%
  left_join(vars_cor, by = "variable")

orden_cor <- cors %>%
  filter(grupo == "Resto de Bogotá (train)") %>%
  arrange(rho) %>%
  pull(etiqueta)

p_cor <- cors %>%
  mutate(etiqueta = factor(etiqueta, levels = orden_cor)) %>%
  ggplot(aes(rho, etiqueta)) +
  geom_vline(xintercept = 0, color = col_texto, linewidth = 0.3) +
  geom_line(aes(group = etiqueta), color = col_gris, linewidth = 0.8) +
  geom_point(aes(color = grupo), size = 2.6) +
  scale_color_manual(values = colores_grupo[1:2]) +
  labs(title = "Dentro de Chapinero, el catastro y el estrato explican mucho menos del precio",
       subtitle = "Correlación de Spearman entre el precio por m² y cada variable de entorno (train)",
       x = "Correlación de Spearman", y = NULL,
       caption = "Chapinero (train) tiene 305 anuncios: error estándar aproximado de 0,06 en cada correlación.")

fun_guardar(p_cor, "eda_correlaciones_chapinero_vs_resto", ancho = 8.5, alto = 7.5)

# ------------------------------------------------------------------------------
# 10. Mapa: precio por m2 de train y ubicación de test
# Hexágonos con la mediana del precio por m2 de train; puntos naranja con las
# propiedades de test. Coordenadas en EPSG:9377 (metros).
# ------------------------------------------------------------------------------
p_mapa <- ggplot() +
  stat_summary_hex(data = db %>% filter(muestra == "train", !is.na(precio_m2)),
                   aes(num_x, num_y, z = precio_m2 / 1e6),
                   fun = median, bins = 60) +
  geom_point(data = db %>% filter(muestra == "test"),
             aes(num_x, num_y), color = col_test, size = 0.3, alpha = 0.15) +
  scale_fill_gradient(low = "#cde2fb", high = "#0d366b", trans = "log10",
                      breaks = c(2, 4, 8, 16),
                      name = "Precio m² mediano\n(millones COP)") +
  coord_equal() +
  labs(title = "Train cubre la ciudad; test se concentra en Chapinero",
       subtitle = "Hexágonos: mediana del precio por m² de train. Puntos naranja: propiedades de test.",
       x = NULL, y = NULL) +
  theme(axis.text = element_blank(), panel.grid.major = element_blank(),
        legend.position = "right", legend.title = element_text(size = 9))

fun_guardar(p_mapa, "eda_mapa_precio_m2_test", ancho = 7, alto = 8.5)

message("Listo: figuras en 03_Output/Figures y tablas en 03_Output/Tables (prefijo eda_).")
