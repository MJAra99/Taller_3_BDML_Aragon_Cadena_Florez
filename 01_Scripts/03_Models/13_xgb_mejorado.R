# ==============================================================================
# 13_xgb_mejorado.R
# Objetivo: XGBoost con pesos (como el envío 16) más cinco mejoras que salen
#           del diagnóstico de errores (02_EDA/03_diagnostico_errores.R):
#           1. Precisión de las coordenadas. En test, el 52 % de las
#              coordenadas viene redondeado a 3 decimales (~110 m) o menos, y
#              el 10 % a 2 decimales (~1 km). Se agregan el número de
#              decimales y, para los inmuebles con 2 decimales o menos,
#              versiones suavizadas de las variables de entorno: el promedio
#              de esa variable entre los inmuebles con coordenadas precisas a
#              menos de 600 m (train y test; no usa el precio).
#           2. Barrios en el texto. Con coordenadas imprecisas, el barrio que
#              menciona el anuncio es la mejor pista de la ubicación. Se agrega
#              una dummy por barrio de Chapinero y alrededores con al menos 10
#              menciones en train y 10 en test, y una dummy agregada del
#              corredor de alto valor (93, Virrey, zonas G y T, Andino).
#           3. Área rescatada del texto. El 44 % de test no tiene área. Se
#              buscan menciones que el regex de 02_text_features.R no captura
#              ("area de 135 m2", "98 m de construccion"), descartando las que
#              hablan de balcones, terrazas, patios, depósitos o lotes.
#           4. Comparable exacto, como haría un avaluador: precio mediano (log)
#              de los anuncios de train con las mismas coordenadas, las mismas
#              habitaciones, los mismos baños y el mismo tipo, pero con otra
#              descripción, y cuántos hay. En train nunca se usa el mismo
#              anuncio ni sus copias.
#           5. Test es 97 % apartamentos y train 75 %. Las casas pueden pesar
#              menos en el entrenamiento (peso_casa).
#           Escogencia: el diagnóstico mostró que, en los pares 13/14 y 15/16,
#           la validación aleatoria con el MAE ponderado por parecido con test
#           ordena los modelos igual que Kaggle y la de localidades no. Por eso
#           aquí los hiperparámetros se escogen con ese criterio. Los cuatro
#           esquemas se siguen reportando.
#           Posprocesamiento: (a) los anuncios de test con la misma descripción
#           reciben el promedio de sus predicciones (en train, los anuncios
#           repetidos tienen el mismo precio); (b) se recorta al rango de
#           precios de train [300, 1.650] millones; (c) el envío 21 multiplica
#           por el factor que sugieren los tres puntajes de Kaggle de los
#           envíos 16, 18 y 19 (si existe diag_factor_optimo.csv; si no, 1,05).
# Entradas: 02_Data/Processed/db_{train,test}.rds
#           02_Data/Processed/pesos_adversarial_texto.rds
#           02_Data/Raw/Kaggle/{train,test}.csv
#           03_Output/Tables/diag_factor_optimo.csv (opcional)
# Salidas:  02_Data/Processed/features_mejorado.rds
#           03_Output/Submissions/20_xgb_mejorado.csv, 21_xgb_mejorado_sesgo.csv
#           03_Output/Models/20_xgb_mejorado.rds
#           03_Output/Models/resultados_modelos.csv (filas de los modelos)
#           03_Output/Predictions/oof_20_{esquema}.rds
#           03_Output/Tables/xgb_mejorado_{afinacion,ablacion,nuevas_vars,barrios_menciones}.csv
#           03_Output/Figures/xgb_mejorado_importancia.{pdf,png}
# ==============================================================================

source(here::here("01_Scripts", "00_utils.R"))
pacman::p_load(xgboost, scales, stringi)

ruta_fig  <- here("03_Output", "Figures")
ruta_tab  <- here("03_Output", "Tables")
ruta_pred <- here("03_Output", "Predictions")
walk(c(ruta_fig, ruta_tab, ruta_pred), dir.create, recursive = TRUE, showWarnings = FALSE)

nucleos <- max(1, parallel::detectCores() - 1)
limites_precio <- c(3e8, 1.65e9)

col_resto <- "#2a78d6"
col_texto <- "#52514e"
theme_set(theme_minimal(base_size = 11) +
            theme(panel.grid.minor = element_blank(),
                  plot.title = element_text(face = "bold", size = 12),
                  plot.title.position = "plot",
                  plot.subtitle = element_text(color = col_texto, size = 9.5)))
dispositivo_pdf <- if (capabilities("cairo")) cairo_pdf else "pdf"
fun_guardar <- function(p, nombre, ancho = 8, alto = 5) {
  ggsave(file.path(ruta_fig, paste0(nombre, ".pdf")), p, width = ancho, height = alto,
         device = dispositivo_pdf)
  ggsave(file.path(ruta_fig, paste0(nombre, ".png")), p, width = ancho, height = alto,
         dpi = 300, bg = "white")
}

fun_mejor_ronda <- function(modelo) {
  ronda <- attributes(modelo)$early_stop$best_iteration
  if (is.null(ronda)) ronda <- as.integer(xgb.attr(modelo, "best_iteration")) + 1
  as.integer(ronda)
}

# ------------------------------------------------------------------------------
# 1. Datos con faltantes nativos (igual que 10_xgb_pesos_texto.R)
# Además, el área por habitación vuelve a NA cuando el área era imputada.
# ------------------------------------------------------------------------------
datos <- fun_datos_modelo()

fun_restaurar_na <- function(df) {
  df <- df %>% mutate(num_vref_m2_200m = na_if(num_vref_m2_200m, 0))
  banderas <- str_subset(names(df), "^bin_.+_na$")
  for (b in banderas) {
    v <- paste0("num_", str_remove(str_remove(b, "^bin_"), "_na$"))
    if (v %in% names(df)) df[[v]][df[[b]] == 1] <- NA
  }
  df %>% mutate(num_area_por_habitac = if_else(bin_area_na == 1, NA_real_, num_area_por_habitac))
}

train_x <- fun_restaurar_na(datos$train)
test_x  <- fun_restaurar_na(datos$test)
n_train <- nrow(train_x)
n_test  <- nrow(test_x)

# ------------------------------------------------------------------------------
# 2. Datos crudos: coordenadas como texto, descripción y llaves
# ------------------------------------------------------------------------------
fun_normalizar <- function(x) {
  x %>%
    replace_na("") %>%
    str_to_lower() %>%
    stri_trans_general("Latin-ASCII") %>%
    str_replace_all("m²", "m2") %>%
    str_replace_all("[^a-z0-9 ]", " ") %>%
    str_squish()
}

fun_decimales <- function(x) {
  d <- str_extract(x, "(?<=\\.)\\d+")
  nchar(str_remove(replace_na(d, ""), "0+$"))
}

fun_crudos <- function(archivo) {
  read_csv(here("02_Data", "Raw", "Kaggle", archivo), show_col_types = FALSE,
           col_types = cols(lat = col_character(), lon = col_character(),
                            .default = col_guess())) %>%
    transmute(property_id,
              num_dec_coord = pmin(fun_decimales(lat), fun_decimales(lon)),
              llave_comp = paste(lat, lon, bedrooms, replace_na(as.character(bathrooms), "na"),
                                 property_type, sep = "_"),
              desc_norm = fun_normalizar(description),
              texto     = fun_normalizar(paste(title, description)))
}

crudo <- bind_rows(
  train_x %>% select(property_id, num_x, num_y, num_habitac, num_area, obj_log_price, id_grupo_anuncio) %>%
    left_join(fun_crudos("train.csv"), by = "property_id") %>% mutate(muestra = "train"),
  test_x %>% select(property_id, num_x, num_y, num_habitac, num_area) %>%
    left_join(fun_crudos("test.csv"), by = "property_id") %>% mutate(muestra = "test")
)
stopifnot(!anyNA(crudo$num_dec_coord))

# ------------------------------------------------------------------------------
# 3. Mejora 1: precisión de las coordenadas y entorno suavizado
# ------------------------------------------------------------------------------
vars_suavizar <- intersect(c("num_vref_m2_200m", "num_avaluo_ph_m2_200m", "num_estrato_200m",
                             "num_avaluo_mz_200m", "num_pisos_200m", "num_prop_comer_300m",
                             "num_prop_resid_300m", "num_gastro_300m"),
                           names(train_x))
entorno <- bind_rows(train_x %>% select(property_id, all_of(vars_suavizar)),
                     test_x  %>% select(property_id, all_of(vars_suavizar)))
crudo <- crudo %>% left_join(entorno, by = "property_id")

precisos   <- which(crudo$num_dec_coord >= 4)
imprecisos <- which(crudo$num_dec_coord <= 2)
vecinos <- frNN(as.matrix(crudo[precisos, c("num_x", "num_y")]), eps = 600,
                query = as.matrix(crudo[imprecisos, c("num_x", "num_y")]), sort = FALSE)$id

for (v in vars_suavizar) {
  nuevo <- crudo[[v]]
  valores_ref <- crudo[[v]][precisos]
  suav <- map_dbl(vecinos, ~ if (length(.x) == 0) NA_real_ else mean(valores_ref[.x], na.rm = TRUE))
  nuevo[imprecisos] <- coalesce(suav, nuevo[imprecisos])
  crudo[[paste0(v, "_suav")]] <- nuevo
}
message("Inmuebles con coordenadas de 2 decimales o menos: train ",
        sum(crudo$num_dec_coord <= 2 & crudo$muestra == "train"), ", test ",
        sum(crudo$num_dec_coord <= 2 & crudo$muestra == "test"))

# ------------------------------------------------------------------------------
# 4. Mejora 2: barrios mencionados en el texto
# El texto ya está sin tildes y en minúscula. Ya existen bin_chico y
# bin_rosales en la base; aquí se separan los Chicós y se agregan más barrios.
# Varios barrios muy mencionados en test (La Cabrera, El Nogal, Andino, Zona G)
# tienen entre 10 y 30 menciones en train: por eso el umbral es bajo (10 en
# train y 10 en test).
# ------------------------------------------------------------------------------
barrios <- c(
  quinta_camacho   = "quinta camacho",
  chapinero_alto   = "chapinero alto",
  chapinero_central = "chapinero central",
  el_retiro        = "\\bel retiro\\b",
  antiguo_country  = "antiguo country",
  la_cabrera       = "\\bla cabrera\\b|\\bcabrera\\b",
  el_nogal         = "\\bnogal\\b",
  bosque_calderon  = "bosque calderon",
  pardo_rubio      = "pardo rubio",
  marly            = "\\bmarly\\b",
  emaus            = "\\bemaus\\b",
  granada          = "\\bgranada\\b",
  la_salle         = "\\bla salle\\b",
  seminario        = "\\bseminario\\b",
  sucre            = "\\bsucre\\b",
  porciuncula      = "porciuncula",
  el_lago          = "\\bel lago\\b|\\blago gaitan\\b",
  virrey           = "\\bvirrey\\b",
  parque_93        = "parque (de la )?93|\\bla 93\\b",
  zona_g           = "\\bzona g\\b",
  zona_t           = "\\bzona t\\b|zona rosa",
  chico_navarra    = "chico navarra",
  chico_reservado  = "chico reservado",
  chico_norte      = "chico norte",
  chico_alto       = "chico alto",
  chico_oriental   = "chico oriental",
  rosales_alto     = "rosales alto|alto de los rosales",
  bellavista       = "\\bbellavista\\b",
  cerros           = "\\bcerros\\b|circunvalar",
  la_merced        = "\\bla merced\\b",
  teusaquillo      = "\\bteusaquillo\\b",
  galerias         = "\\bgalerias\\b",
  usaquen          = "\\busaquen\\b",
  santa_barbara    = "santa barbara",
  santa_bibiana    = "santa bibiana",
  la_carolina      = "la carolina",
  multicentro      = "\\bmulticentro\\b",
  andino           = "\\bandino\\b",
  # Agregado: corredor comercial de alto valor (93, Virrey, zonas G y T,
  # Andino, El Retiro). Muchos de estos barrios casi no aparecen en train,
  # así que juntarlos le da al modelo suficientes casos para aprender algo.
  corredor_premium = "\\bvirrey\\b|parque (de la )?93|\\bzona g\\b|\\bzona t\\b|zona rosa|\\bandino\\b|\\bel retiro\\b|\\bnogal\\b|\\bcabrera\\b"
)

dummies_barrio <- imap_dfc(barrios, function(patron, nombre) {
  tibble(!!paste0("bin_b_", nombre) := as.integer(str_detect(crudo$texto, patron)))
})
menciones <- tibble(variable = names(dummies_barrio),
                    train = colSums(dummies_barrio[crudo$muestra == "train", ]),
                    test  = colSums(dummies_barrio[crudo$muestra == "test", ]))
vars_barrio <- menciones %>% filter(train >= 10, test >= 10) %>% pull(variable)
write_csv(menciones, file.path(ruta_tab, "xgb_mejorado_barrios_menciones.csv"))
message("Barrios del texto que entran al modelo: ", length(vars_barrio), " de ", length(barrios))
crudo <- bind_cols(crudo, dummies_barrio[vars_barrio])
crudo$num_barrios_texto <- rowSums(dummies_barrio)

# ------------------------------------------------------------------------------
# 5. Mejora 3: área rescatada del texto (solo donde el área falta)
# ------------------------------------------------------------------------------
unidad <- "(?:m2|mt2|mts2|mtrs2|mts|mt|mtrs|metros cuadrados|metros|m)\\b"
# Se descartan las distancias ("a 200 metros del parque").
no_distancia <- "(?! (?:del?|al|a la) (?:parque|centro|estacion|transmilenio|universidad|calle|carrera|avenida|la |el ))"
patron_antes   <- paste0("(?:area|construccion|construida|construidos|privada|privados|total|de)\\D{0,15}?\\b(\\d{2,3}) ?",
                         unidad, no_distancia)
patron_despues <- paste0("\\b(\\d{2,3}) ?", unidad, "(?: cuadrados)?(?: de)? (?:construccion|construidos|area|privados|aprox|totales)")
excluir <- "balcon|terraza|patio|deposito|jardin|lote|frente|fondo|altura|garaje|parqueadero|bodega|piso \\d"

fun_area_rescate <- function(texto, patron) {
  pos <- str_locate(texto, patron)[, "start"]
  num <- as.numeric(str_match(texto, patron)[, 2])
  contexto <- str_sub(texto, pmax(1, pos - 35), pmax(1, pos - 1))
  if_else(!is.na(num) & !str_detect(replace_na(contexto, ""), excluir), num, NA_real_)
}

crudo <- crudo %>%
  mutate(area_rescate = coalesce(fun_area_rescate(texto, patron_despues),
                                 fun_area_rescate(texto, patron_antes)),
         area_rescate = if_else(between(area_rescate, 20, 600) &
                                  between(area_rescate / pmax(num_habitac, 1, na.rm = TRUE), 12, 150),
                                area_rescate, NA_real_),
         bin_area_rescatada = as.integer(is.na(num_area) & !is.na(area_rescate)))
message("Áreas rescatadas: train ", sum(crudo$bin_area_rescatada[crudo$muestra == "train"]),
        " de ", sum(is.na(crudo$num_area) & crudo$muestra == "train"), " sin área; test ",
        sum(crudo$bin_area_rescatada[crudo$muestra == "test"]),
        " de ", sum(is.na(crudo$num_area) & crudo$muestra == "test"))

# ------------------------------------------------------------------------------
# 6. Mejora 4: comparable exacto
# Para cada anuncio, los grupos de train con la misma llave (coordenadas
# textuales, habitaciones, baños y tipo) y OTRA descripción. Cada descripción
# cuenta una vez (mediana del log del precio dentro de la descripción). La
# regla es la misma en train y en test, para que la variable signifique lo
# mismo en los dos.
# ------------------------------------------------------------------------------
ref_comp <- crudo %>%
  filter(muestra == "train", desc_norm != "") %>%
  group_by(llave_comp, desc_norm) %>%
  summarise(lp = median(obj_log_price), .groups = "drop")

comparables <- crudo %>%
  select(property_id, llave_comp, desc_norm) %>%
  inner_join(ref_comp, by = "llave_comp", suffix = c("", "_ref"), relationship = "many-to-many") %>%
  filter(desc_norm != desc_norm_ref) %>%
  group_by(property_id) %>%
  summarise(num_comp_log_precio = median(lp), num_comp_n = n(), .groups = "drop")

crudo <- crudo %>%
  left_join(comparables, by = "property_id") %>%
  mutate(num_comp_n = replace_na(num_comp_n, 0L))
message("Con comparable exacto: train ",
        percent(mean(crudo$num_comp_n[crudo$muestra == "train"] > 0), 0.1), ", test ",
        percent(mean(crudo$num_comp_n[crudo$muestra == "test"] > 0), 0.1))

# ------------------------------------------------------------------------------
# 7. Base del modelo con las variables nuevas
# ------------------------------------------------------------------------------
vars_nuevas <- c("num_dec_coord", paste0(vars_suavizar, "_suav"), vars_barrio,
                 "num_barrios_texto", "bin_area_rescatada", "num_comp_log_precio", "num_comp_n")

features <- crudo %>%
  mutate(bin_coord_imprecisa = as.integer(num_dec_coord <= 3)) %>%
  select(property_id, all_of(vars_nuevas), bin_coord_imprecisa, area_rescate)
saveRDS(features, here("02_Data", "Processed", "features_mejorado.rds"))
vars_nuevas <- c(vars_nuevas, "bin_coord_imprecisa")

fun_agregar <- function(df) {
  df %>%
    left_join(features, by = "property_id") %>%
    mutate(num_area = coalesce(num_area, area_rescate),
           bin_area_na = if_else(bin_area_rescatada == 1, 0L, as.integer(bin_area_na)),
           num_area_por_habitac = coalesce(num_area_por_habitac, num_area / pmax(num_habitac, 1))) %>%
    select(-area_rescate)
}
train_x <- fun_agregar(train_x)
test_x  <- fun_agregar(test_x)

tabla_nuevas <- tibble(variable = vars_nuevas,
                       media_train = map_dbl(vars_nuevas, ~ mean(train_x[[.x]], na.rm = TRUE)),
                       media_test  = map_dbl(vars_nuevas, ~ mean(test_x[[.x]], na.rm = TRUE)),
                       pct_na_train = map_dbl(vars_nuevas, ~ 100 * mean(is.na(train_x[[.x]]))),
                       pct_na_test  = map_dbl(vars_nuevas, ~ 100 * mean(is.na(test_x[[.x]]))))
write_csv(tabla_nuevas, file.path(ruta_tab, "xgb_mejorado_nuevas_vars.csv"))
print(tabla_nuevas, n = Inf)

fun_matrices <- function(predictores) {
  fml <- reformulate(predictores, intercept = FALSE)
  todo <- bind_rows(train_x[predictores], test_x[predictores])
  x <- model.matrix(fml, model.frame(fml, data = todo, na.action = na.pass))
  list(train = x[seq_len(n_train), , drop = FALSE], test = x[-seq_len(n_train), , drop = FALSE])
}
pred_base   <- datos$predictores
pred_mejor  <- c(datos$predictores, vars_nuevas)
x_base  <- fun_matrices(pred_base)
x_mejor <- fun_matrices(pred_mejor)
y_train <- train_x$obj_log_price
precio  <- train_x$obj_price

# ------------------------------------------------------------------------------
# 8. Pesos: parecido con test (w) y casas
# ------------------------------------------------------------------------------
w <- train_x %>%
  select(property_id) %>%
  left_join(readRDS(here("02_Data", "Processed", "pesos_adversarial_texto.rds")), by = "property_id") %>%
  pull(peso)
stopifnot(!anyNA(w))
es_casa <- train_x$cat_tipo == "Casa"
es_chap <- train_x$cat_localidad == "CHAPINERO"

fun_peso_entreno <- function(gamma, peso_casa) {
  pe <- w^gamma * if_else(es_casa, peso_casa, 1)
  pe / mean(pe)
}

# ------------------------------------------------------------------------------
# 9. Validación cruzada
# ------------------------------------------------------------------------------
folds <- fun_todos_folds(datos$train)

parametros_base <- list(objective = "reg:squarederror", eval_metric = "mae",
                        eta = 0.05, subsample = 0.8, colsample_bytree = 0.7,
                        nthread = nucleos)

# Predicciones fuera de muestra (log). Con `rondas = NULL` usa parada temprana
# con el MAE ponderado por w en validación.
fun_cv_xgb <- function(x, parametros, folds, peso, rondas = NULL, max_rondas = 3000) {
  pred <- rep(NA_real_, n_train)
  rondas_fold <- integer(length(folds))
  for (i in seq_along(folds)) {
    f <- folds[[i]]
    d_tr  <- xgb.DMatrix(x[f$train, , drop = FALSE], label = y_train[f$train], weight = peso[f$train])
    d_val <- xgb.DMatrix(x[f$val, , drop = FALSE], label = y_train[f$val], weight = w[f$val])
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

fun_metricas <- function(pred_log, factor = 1) {
  ok  <- !is.na(pred_log)
  err <- factor * exp(pred_log) - precio
  tibble(mae            = mean(abs(err[ok])),
         mae_pond       = weighted.mean(abs(err[ok]), w[ok]),
         mae_chap       = mean(abs(err[ok & es_chap])),
         sesgo_pond     = weighted.mean(err[ok], w[ok]),
         sesgo_chap     = mean(err[ok & es_chap]),
         pct_sobre_chap = 100 * mean(err[ok & es_chap] > 0))
}

# 9.1 Afinación con la validación aleatoria y el MAE ponderado.
grid <- expand.grid(gamma_peso = c(0.5, 1), peso_casa = c(1, 0.3),
                    max_depth = c(6, 8), min_child_weight = c(1, 5))

message("Afinando con la validación aleatoria (", nrow(grid), " combinaciones)")
afinacion <- pmap_dfr(grid, function(gamma_peso, peso_casa, max_depth, min_child_weight) {
  par <- c(parametros_base, list(max_depth = max_depth, min_child_weight = min_child_weight))
  r <- fun_cv_xgb(x_mejor$train, par, folds$aleatoria, fun_peso_entreno(gamma_peso, peso_casa))
  met <- fun_metricas(r$pred)
  message(sprintf("  gamma %.1f, casa %.1f, depth %d, mcw %d: MAE pond %.1f M, MAE %.1f M",
                  gamma_peso, peso_casa, max_depth, min_child_weight,
                  met$mae_pond / 1e6, met$mae / 1e6))
  tibble(gamma_peso, peso_casa, max_depth, min_child_weight,
         rondas = round(mean(r$rondas))) %>% bind_cols(met)
}) %>%
  arrange(mae_pond)
write_csv(afinacion, file.path(ruta_tab, "xgb_mejorado_afinacion.csv"))
print(afinacion %>% mutate(across(starts_with(c("mae", "sesgo")), ~ round(.x / 1e6, 1))), n = Inf)

mejor <- afinacion[1, ]
par_mejor <- c(parametros_base, list(max_depth = mejor$max_depth,
                                     min_child_weight = mejor$min_child_weight))
peso_mejor <- fun_peso_entreno(mejor$gamma_peso, mejor$peso_casa)
hiperparam <- data.frame(gamma_peso = mejor$gamma_peso, peso_casa = mejor$peso_casa,
                         max_depth = mejor$max_depth, min_child_weight = mejor$min_child_weight,
                         eta = parametros_base$eta, rondas = mejor$rondas)
hiperparam_texto <- fun_texto_hiperparam(hiperparam)
message("Hiperparámetros escogidos: ", hiperparam_texto)

# 9.2 Ablación: los mismos hiperparámetros y pesos con y sin las variables
# nuevas, para medir cuánto aportan.
message("Ablación: con y sin variables nuevas")
oof_base <- fun_cv_xgb(x_base$train, par_mejor, folds$aleatoria, peso_mejor, rondas = mejor$rondas)$pred

# 9.3 Los cuatro esquemas con hiperparámetros y rondas fijos.
message("MAE con los cuatro esquemas")
oof_esquemas <- imap(folds, function(f, esquema) {
  fun_cv_xgb(x_mejor$train, par_mejor, f, peso_mejor, rondas = mejor$rondas)$pred
})

# Promedio de las predicciones dentro de cada grupo de descripción idéntica
# (lo mismo que se hará en test). En la validación aleatoria los grupos de
# anuncios repetidos quedan en el mismo fold, así que no hay fuga.
fun_promediar_duplicados <- function(pred_log, desc) {
  grupo <- if_else(nchar(desc) >= 40, desc, paste0("unico_", seq_along(desc)))
  ave(pred_log, grupo, FUN = function(z) mean(z, na.rm = TRUE))
}
desc_train <- crudo$desc_norm[match(train_x$property_id, crudo$property_id)]
desc_test  <- crudo$desc_norm[match(test_x$property_id, crudo$property_id)]
oof_aleatoria_prom <- fun_promediar_duplicados(oof_esquemas$aleatoria, desc_train)

ablacion <- bind_rows(
  fun_metricas(oof_base) %>% mutate(version = "Sin variables nuevas (base del 16)"),
  fun_metricas(oof_esquemas$aleatoria) %>% mutate(version = "Con variables nuevas"),
  fun_metricas(oof_aleatoria_prom) %>% mutate(version = "Con variables nuevas + promedio de duplicados")
) %>% relocate(version)
write_csv(ablacion, file.path(ruta_tab, "xgb_mejorado_ablacion.csv"))
print(ablacion %>% mutate(across(starts_with(c("mae", "sesgo")), ~ round(.x / 1e6, 1))), width = Inf)

# ------------------------------------------------------------------------------
# 10. Factor de ajuste del sesgo
# ------------------------------------------------------------------------------
ruta_factor <- file.path(ruta_tab, "diag_factor_optimo.csv")
factor_sesgo <- if (file.exists(ruta_factor)) {
  read_csv(ruta_factor, show_col_types = FALSE) %>%
    filter(muestra == "Kaggle (parábola)") %>% pull(factor) %>% round(2)
} else {
  1.05
}
if (length(factor_sesgo) != 1 || is.na(factor_sesgo)) factor_sesgo <- 1.05
message("Factor para el envío 21: ", factor_sesgo)

curva_factor <- map_dfr(seq(1, 1.12, by = 0.01), ~ fun_metricas(oof_aleatoria_prom, .x) %>%
                          mutate(factor = .x, .before = 1))
print(curva_factor %>% mutate(across(starts_with(c("mae", "sesgo")), ~ round(.x / 1e6, 1))), n = Inf)

# ------------------------------------------------------------------------------
# 11. Modelo final, envíos y registro
# ------------------------------------------------------------------------------
message("Ajuste final con todo train")
set.seed(semilla)
final <- xgb.train(params = par_mejor,
                   data = xgb.DMatrix(x_mejor$train, label = y_train, weight = peso_mejor),
                   nrounds = mejor$rondas, verbose = 0)
pred_test <- predict(final, xgb.DMatrix(x_mejor$test))
pred_test <- fun_promediar_duplicados(pred_test, desc_test)

fun_submission(test_x$property_id, pred_test, "20_xgb_mejorado", limites = limites_precio)
fun_submission(test_x$property_id, pred_test + log(factor_sesgo), "21_xgb_mejorado_sesgo",
               limites = limites_precio)

importancia <- xgb.importance(model = final)

# Predicciones fuera de muestra, para el ensamble.
iwalk(oof_esquemas, function(p, esquema) {
  saveRDS(tibble(property_id = train_x$property_id, pred_log = p),
          file.path(ruta_pred, paste0("oof_20_", esquema, ".rds")))
})

saveRDS(list(envio = "20_xgb_mejorado", familia = "XGBoost", hiperparam = hiperparam,
             vars_nuevas = vars_nuevas, afinacion = afinacion, ablacion = ablacion,
             factor_sesgo = factor_sesgo, pred_test = pred_test, importancia = importancia),
        here("03_Output", "Models", "20_xgb_mejorado.rds"))

# Registro con el MAE sin ponderar de cada esquema (comparable con las filas
# anteriores de resultados_modelos.csv). El 21 usa las mismas predicciones
# fuera de muestra multiplicadas por el factor.
mae_20 <- imap_dfr(oof_esquemas, ~ tibble(esquema = .y, mae_cop = fun_metricas(.x)$mae))
mae_21 <- imap_dfr(oof_esquemas, ~ tibble(esquema = .y, mae_cop = fun_metricas(.x, factor_sesgo)$mae))
fun_registrar_resultado("20_xgb_mejorado", "XGBoost", hiperparam_texto, mae_20, "20_xgb_mejorado")
fun_registrar_resultado("21_xgb_mejorado_sesgo", "XGBoost",
                        paste0(hiperparam_texto, ", factor = ", factor_sesgo),
                        mae_21, "21_xgb_mejorado_sesgo")
print(mae_20 %>% mutate(mae_millones = round(mae_cop / 1e6, 1)))

# Importancia de las 25 variables principales, resaltando las nuevas.
p_imp <- importancia %>%
  slice_max(Gain, n = 25) %>%
  mutate(nueva = Feature %in% vars_nuevas | str_detect(Feature, "^bin_b_|_suav$|^num_comp"),
         Feature = fct_reorder(Feature, Gain)) %>%
  ggplot(aes(Gain, Feature, fill = nueva)) +
  geom_col(width = 0.65) +
  scale_fill_manual(values = c(`FALSE` = "#c3c2b7", `TRUE` = col_resto),
                    labels = c(`FALSE` = "Variables del modelo 16", `TRUE` = "Variables nuevas"),
                    name = NULL) +
  scale_x_continuous(labels = label_percent(), expand = expansion(mult = c(0, 0.05))) +
  labs(title = "Importancia de las variables en el XGBoost mejorado",
       subtitle = "Participación en la ganancia total, 25 variables principales",
       x = NULL, y = NULL) +
  theme(legend.position = "top", legend.justification = "left")
fun_guardar(p_imp, "xgb_mejorado_importancia", ancho = 8, alto = 7)

message("Listo. Sube 20_xgb_mejorado.csv y 21_xgb_mejorado_sesgo.csv a Kaggle. ",
        "Descripción sugerida: XGBoost con pesos, ", hiperparam_texto,
        ", + precisión de coordenadas, barrios del texto, área rescatada, comparable exacto; ",
        "escogido con CV aleatoria ponderada; duplicados promediados y recorte al rango de train. ",
        "MAE CV aleatoria ponderado: ", round(ablacion$mae_pond[3] / 1e6, 1), " M.")
