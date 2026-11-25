# ==============================================================================
# 15_ensamble_v2.R
# Objetivo: superar el mejor puntaje público de la competencia (148,2 M). El
#           mejor envío del equipo es el 27 (150,45 M). 
#
# Un solo envío: 28_ensamble_v2_nivel
#   Es la combinación de las tres palancas y la apuesta con mejor puntaje
#   público esperado:
#     a) ensamble de 20, 23 y dos modelos nuevos que juntan las variables del
#        20 con las palabras del anuncio y el objetivo precio por m2
#        (profundidades 6 y 8, promedio de 3 semillas cada uno);
#     b) promedio de duplicados, calibración estirada y re-anuncios (como el
#        27, que tiene 150,45 M);
#     c) factor de nivel estimado con una regla que, aplicada al modelo 16,
#        reproduce el factor que Kaggle premió (sección 7). El factor no toca
#        los inmuebles con re-anuncio.
#
# Entradas: 02_Data/Processed/{db_train,db_test,base_limpia,features_mejorado,
#           pesos_adversarial_texto}.rds; 02_Data/Raw/Kaggle/{train,test}.csv
#           03_Output/Predictions/oof_{16,20}_{aleatoria,localidad}.rds,
#           oof_23_xgb_palabras_m2_{aleatoria,localidad}.rds
#           03_Output/Models/{20_xgb_mejorado,23_xgb_palabras_m2}.rds
# Salidas:  03_Output/Submissions/28_ensamble_v2_nivel.csv (el único envío)
#           03_Output/Models/{xgb_completo_palabras_m2_prof6,
#             xgb_completo_palabras_m2_prof8}.rds (miembros, en caché)
#           03_Output/Predictions/oof_28_ensamble_v2_nivel_*.rds
#           fila nueva en resultados_modelos.csv
#           03_Output/Tables/ensamble_v2_{pesos,comparacion,factor}.csv
#
# ==============================================================================

source(here::here("01_Scripts", "00_utils.R"))
pacman::p_load(xgboost, tidytext, SnowballC, stopwords, stringi, scales)

ruta_pred <- here("03_Output", "Predictions")
ruta_tab  <- here("03_Output", "Tables")
ruta_mod  <- here("03_Output", "Models")

nucleos <- max(1, parallel::detectCores() - 1)
limites_precio <- c(3e8, 1.65e9)
recortar <- function(p) pmin(pmax(p, limites_precio[1]), limites_precio[2])

n_semillas <- 3                       # semillas por modelo (1 = prueba rápida)
semillas   <- semilla + seq_len(n_semillas) - 1

##### 1. Datos: base del equipo + variables nuevas del 20 (13_xgb_mejorado.R) #####
datos   <- fun_datos_modelo()
folds   <- fun_todos_folds(datos$train)
f_ale   <- folds$aleatoria
f_loc   <- folds$localidad
n_train <- nrow(datos$train)

# Faltantes nativos (igual que 10/12/13): donde 05_merge.R imputó la mediana
# se vuelve a poner NA y XGBoost aprende hacia qué rama mandarlos.
fun_restaurar_na <- function(df) {
  df <- df %>% mutate(num_vref_m2_200m = na_if(num_vref_m2_200m, 0))
  for (b in str_subset(names(df), "^bin_.+_na$")) {
    v <- paste0("num_", str_remove(str_remove(b, "^bin_"), "_na$"))
    if (v %in% names(df)) df[[v]][df[[b]] == 1] <- NA
  }
  df
}

# Variables nuevas del 20, ya calculadas y guardadas por 13_xgb_mejorado.R:
# precisión de coordenadas, entorno suavizado, 28 barrios mencionados en el
# texto, área rescatada, comparable exacto.
features <- readRDS(here("02_Data", "Processed", "features_mejorado.rds"))
vars_nuevas <- setdiff(names(features), c("property_id", "area_rescate"))
fun_agregar <- function(df) {
  df %>%
    left_join(features, by = "property_id") %>%
    mutate(num_area = coalesce(num_area, area_rescate),
           bin_area_na = if_else(bin_area_rescatada == 1, 0L, as.integer(bin_area_na)),
           num_area_por_habitac = coalesce(num_area_por_habitac, num_area / pmax(num_habitac, 1))) %>%
    select(-area_rescate)
}
train_x <- fun_agregar(fun_restaurar_na(datos$train))
test_x  <- fun_agregar(fun_restaurar_na(datos$test))
predictores <- c(datos$predictores, vars_nuevas)

fml  <- reformulate(predictores, intercept = FALSE)
todo <- bind_rows(train_x[predictores], test_x[predictores])
x_todo  <- model.matrix(fml, model.frame(fml, data = todo, na.action = na.pass))
x_train <- x_todo[seq_len(n_train), , drop = FALSE]
x_test  <- x_todo[-seq_len(n_train), , drop = FALSE]
rm(todo, x_todo)

y_train <- datos$train$obj_log_price
precio  <- datos$train$obj_price
es_chap <- datos$train$cat_localidad == "CHAPINERO"

# Pesos: parecido con test (w) y casas con peso 0,3 (lo que escogió el modelo 20 que hicimos).
w <- datos$train %>% select(property_id) %>%
  left_join(readRDS(here("02_Data", "Processed", "pesos_adversarial_texto.rds")),
            by = "property_id") %>%
  pull(peso)
stopifnot(!anyNA(w))
peso_entreno <- w^0.5 * if_else(datos$train$cat_tipo == "Casa", 0.3, 1)
peso_entreno <- peso_entreno / mean(peso_entreno)

# Offset del precio por m2 (como el 22/23 que se tiene en el script 14). El área es la
# imputada por 05_merge.R, salvo donde el 20 la rescató del texto; se acota a
# los percentiles 1 y 99 de train para que un área absurda no dispare el
# precio. 
fun_area_offset <- function(base, conx) {
  if_else(coalesce(conx$bin_area_rescatada, 0L) == 1, conx$num_area, base$num_area)
}
area_tr <- fun_area_offset(datos$train, train_x)
area_te <- fun_area_offset(datos$test,  test_x)
lim_area <- quantile(area_tr, c(0.01, 0.99), na.rm = TRUE)
off_train <- log(pmin(pmax(area_tr, lim_area[1]), lim_area[2]))
off_test  <- log(pmin(pmax(area_te, lim_area[1]), lim_area[2]))
stopifnot(!anyNA(off_train), !anyNA(off_test))

# Métricas en pesos (recortadas como los envíos).
fun_metricas <- function(pred_log) {
  ok  <- !is.na(pred_log)
  err <- recortar(exp(pred_log)) - precio
  tibble(mae        = mean(abs(err[ok])),
         mae_pond   = weighted.mean(abs(err[ok]), w[ok]),
         mae_chap   = mean(abs(err[ok & es_chap])),
         sesgo_chap = mean(err[ok & es_chap]))
}

##### 2. Palabras del anuncio como columnas (mismo procedimiento del 23) #####

# Minúsculas, sin tildes, sin palabras vacías, raíces de Snowball, palabras y
# bigramas
fun_dtm <- function(min_prop = 0.02) {
  textos <- readRDS(here("02_Data", "Processed", "base_limpia.rds")) %>%
    distinct(property_id, .keep_all = TRUE) %>%
    transmute(property_id, texto = paste(replace_na(title, ""), replace_na(description, "")))
  docs <- bind_rows(datos$train %>% select(property_id), datos$test %>% select(property_id)) %>%
    left_join(textos, by = "property_id") %>%
    mutate(doc = row_number(),
           texto = replace_na(texto, "") %>% str_to_lower() %>%
             stri_trans_general("Latin-ASCII") %>% str_replace_all("[^a-z ]", " ") %>% str_squish())
  vacias <- stri_trans_general(stopwords::stopwords("es", source = "snowball"), "Latin-ASCII")
  palabras <- docs %>% select(doc, texto) %>%
    unnest_tokens(palabra, texto, token = "words") %>%
    filter(!palabra %in% vacias, nchar(palabra) > 2) %>%
    mutate(raiz = wordStem(palabra, language = "spanish"))
  bigramas <- palabras %>% group_by(doc) %>% mutate(siguiente = lead(raiz)) %>% ungroup() %>%
    filter(!is.na(siguiente)) %>% transmute(doc, termino = paste(raiz, siguiente))
  terminos <- bind_rows(palabras %>% transmute(doc, termino = raiz), bigramas) %>%
    distinct(doc, termino)
  vocabulario <- terminos %>% count(termino) %>%
    filter(n >= ceiling(min_prop * nrow(docs))) %>% pull(termino)
  terminos <- terminos %>% filter(termino %in% vocabulario)
  m <- matrix(0, nrow = nrow(docs), ncol = length(vocabulario),
              dimnames = list(NULL, paste0("pal_", make.names(vocabulario))))
  m[cbind(terminos$doc, match(terminos$termino, vocabulario))] <- 1
  message("Palabras como columnas: ", ncol(m), " términos")
  list(train = m[seq_len(n_train), , drop = FALSE], test = m[-seq_len(n_train), , drop = FALSE])
}
pal <- fun_dtm(0.02)
xw_train <- cbind(x_train, pal$train)
xw_test  <- cbind(x_test,  pal$test)
rm(pal); invisible(gc())
message("Matriz completa: ", ncol(xw_train), " columnas (", ncol(x_train),
        " del 20 + palabras)")

##### 3. XGBoost con offset, varias semillas #####

fun_mejor_ronda <- function(modelo) {
  ronda <- attributes(modelo)$early_stop$best_iteration
  if (is.null(ronda)) ronda <- as.integer(xgb.attr(modelo, "best_iteration")) + 1
  as.integer(ronda)
}

# Una CV con offset y una semilla. Con rondas = NULL usa parada temprana con
# el MAE ponderado por w en validación (el criterio del 13/14).
fun_cv <- function(parametros, folds, sem, rondas = NULL, max_rondas = 3000) {
  pred <- rep(NA_real_, n_train)
  rondas_fold <- integer(length(folds))
  par <- modifyList(parametros, list(seed = sem))
  for (i in seq_along(folds)) {
    f <- folds[[i]]
    d_tr <- xgb.DMatrix(xw_train[f$train, , drop = FALSE],
                        label = y_train[f$train] - off_train[f$train],
                        weight = peso_entreno[f$train])
    d_va <- xgb.DMatrix(xw_train[f$val, , drop = FALSE],
                        label = y_train[f$val] - off_train[f$val], weight = w[f$val])
    set.seed(sem)
    if (is.null(rondas)) {
      m <- xgb.train(params = par, data = d_tr, nrounds = max_rondas,
                     evals = list(val = d_va), early_stopping_rounds = 50, verbose = 0)
      rondas_fold[i] <- fun_mejor_ronda(m)
    } else {
      m <- xgb.train(params = par, data = d_tr, nrounds = rondas, verbose = 0)
      rondas_fold[i] <- rondas
    }
    pred[f$val] <- predict(m, d_va, iterationrange = c(1, rondas_fold[i])) + off_train[f$val]
  }
  list(pred = pred, rondas = rondas_fold)
}

# Modelo completo con promedio de semillas.
#   - Rondas: parada temprana con la primera semilla (CV aleatoria); las
#     demás semillas usan esas mismas rondas.
#   - CV aleatoria: promedio de las n semillas (en log). Es la que se usa
#     para los pesos del ensamble y la calibración.
#   - CV por localidades: UNA semilla (solo se usa para la regla del factor
#     de la sección 7).
#   - Test: promedio de n modelos entrenados con el conjunto de train.

fun_modelo <- function(nombre, parametros) {
  ruta <- file.path(ruta_mod, paste0(nombre, ".rds"))
  if (file.exists(ruta)) { message("Cargando ", nombre); return(readRDS(ruta)) }
  message("\n== ", nombre, " ==")
  cv1 <- fun_cv(parametros, f_ale, semillas[1])
  rondas <- round(mean(cv1$rondas))
  message("  rondas: ", rondas)
  oof_ale <- cv1$pred
  if (n_semillas > 1) {
    otras <- map(semillas[-1], ~ fun_cv(parametros, f_ale, .x, rondas = rondas)$pred)
    oof_ale <- rowMeans(cbind(oof_ale, do.call(cbind, otras)))
  }
  oof_loc <- fun_cv(parametros, f_loc, semillas[1], rondas = rondas)$pred
  d_todo <- xgb.DMatrix(xw_train, label = y_train - off_train, weight = peso_entreno)
  d_test <- xgb.DMatrix(xw_test)
  preds <- map(semillas, function(s) {
    set.seed(s)
    m <- xgb.train(params = modifyList(parametros, list(seed = s)), data = d_todo,
                   nrounds = rondas, verbose = 0)
    list(pred = predict(m, d_test) + off_test, imp = xgb.importance(model = m))
  })
  res <- list(nombre = nombre,
              oof = list(aleatoria = oof_ale, localidad = oof_loc),
              pred_test = rowMeans(sapply(preds, `[[`, "pred")),
              importancia = preds[[1]]$imp,
              hiperparam = data.frame(objetivo = "log(precio/m2)", gamma_peso = 0.5,
                                      peso_casa = 0.3, max_depth = parametros$max_depth,
                                      min_child_weight = parametros$min_child_weight,
                                      colsample = parametros$colsample_bytree,
                                      eta = parametros$eta, rondas = rondas,
                                      semillas = n_semillas, columnas = ncol(xw_train)))
  saveRDS(res, ruta)
  m_ale <- fun_metricas(oof_ale)
  message("  CV aleatoria: MAE ", round(m_ale$mae / 1e6, 1), " M, ponderado ",
          round(m_ale$mae_pond / 1e6, 1), " M | CV localidades (1 semilla): ",
          round(fun_metricas(oof_loc)$mae / 1e6, 1), " M")
  res
}

par_base <- list(objective = "reg:squarederror", eval_metric = "mae",
                 eta = 0.05, subsample = 0.8, colsample_bytree = 0.5,
                 max_depth = 6, min_child_weight = 1, nthread = nucleos)

# 3.1 Miembro principal: profundidad 6 y min_child_weight 1 (los
# del 20); colsample 0,5 (como el 23) porque hay un gran número de columnas de
# palabras y conviene que cada árbol vea una muestra de ellas.
res_prof6 <- fun_modelo("xgb_completo_palabras_m2_prof6", par_base)

# 3.2 Variante más profunda.
# Árboles de profundidad 8 capturan interacciones de más alto orden (p. ej.
# barrio x estrato x área x palabra). 
res_prof8 <- fun_modelo("xgb_completo_palabras_m2_prof8",
                        modifyList(par_base, list(max_depth = 8, min_child_weight = 5)))

##### 4. Promedio de duplicados en test (idea de nuestro modelo 20) #####

# Anuncios con la misma descripción (de al menos 40 caracteres) reciben el
# promedio de sus predicciones: si son el mismo inmueble, deberían tener el
# mismo precio y el promedio reduce ruido. Se aplica también a la CV, para
# que la calibración se estime sobre predicciones tratadas igual que test.
fun_normalizar <- function(x) {
  x %>% replace_na("") %>% str_to_lower() %>% stri_trans_general("Latin-ASCII") %>%
    str_replace_all("[^a-z0-9 ]", " ") %>% str_squish()
}
crudo_tr <- read_csv(here("02_Data", "Raw", "Kaggle", "train.csv"), show_col_types = FALSE)
crudo_te <- read_csv(here("02_Data", "Raw", "Kaggle", "test.csv"),  show_col_types = FALSE)
desc_train <- fun_normalizar(crudo_tr$description[match(datos$train$property_id, crudo_tr$property_id)])
desc_test  <- fun_normalizar(crudo_te$description[match(datos$test$property_id,  crudo_te$property_id)])
fun_promediar_duplicados <- function(pred_log, desc) {
  grupo <- if_else(nchar(desc) >= 40, desc, paste0("unico_", seq_along(desc)))
  ave(pred_log, grupo, FUN = function(z) mean(z, na.rm = TRUE))
}

##### 5. Ensamble v2 #####
fun_leer <- function(archivo) {
  r <- readRDS(file.path(ruta_pred, archivo))
  p <- if ("pred_log" %in% names(r)) r$pred_log else log(r$pred)
  p[match(datos$train$property_id, r$property_id)]
}
miembros <- list(
  `20_xgb_mejorado` = list(oof = list(aleatoria = fun_leer("oof_20_aleatoria.rds"),
                                      localidad = fun_leer("oof_20_localidad.rds")),
                           pred_test = readRDS(file.path(ruta_mod, "20_xgb_mejorado.rds"))$pred_test),
  `23_xgb_palabras_m2` = list(oof = list(aleatoria = fun_leer("oof_23_xgb_palabras_m2_aleatoria.rds"),
                                         localidad = fun_leer("oof_23_xgb_palabras_m2_localidad.rds")),
                              pred_test = readRDS(file.path(ruta_mod, "23_xgb_palabras_m2.rds"))$pred_test),
  xgb_completo_palabras_m2_prof6 = res_prof6,
  xgb_completo_palabras_m2_prof8 = res_prof8)

fun_mat <- function(esq) sapply(miembros, function(r) recortar(exp(r$oof[[esq]])))
oof_ale  <- fun_mat("aleatoria")
oof_loc  <- fun_mat("localidad")
test_mat <- sapply(miembros, function(r) recortar(exp(r$pred_test)))

fold_ale <- integer(n_train)
for (i in seq_along(f_ale)) fold_ale[f_ale[[i]]$val] <- i

# Pesos convexos vía softmax (siempre >= 0 y suman 1).
fun_pesos <- function(P, obs, pond) {
  objetivo <- function(t) { a <- exp(t) / sum(exp(t)); weighted.mean(abs(P %*% a - obs), pond) }
  t <- optim(rep(0, ncol(P)), objetivo, method = "Nelder-Mead", control = list(maxit = 5000))$par
  setNames(exp(t) / sum(exp(t)), colnames(P))
}
ok <- complete.cases(oof_ale)
ens_cf <- rep(NA_real_, n_train)
for (k in seq_along(f_ale)) {
  entrena <- ok & fold_ale != k
  a_k <- fun_pesos(oof_ale[entrena, ], precio[entrena], w[entrena])
  ens_cf[ok & fold_ale == k] <- oof_ale[ok & fold_ale == k, ] %*% a_k
}
pesos <- fun_pesos(oof_ale[ok, ], precio[ok], w[ok])
print(round(pesos, 3))
write_csv(enframe(pesos, "modelo", "peso"), file.path(ruta_tab, "ensamble_v2_pesos.csv"))

# Ensamble (en log) + promedio de duplicados.
ens <- list(aleatoria = fun_promediar_duplicados(log(ens_cf), desc_train),
            localidad = fun_promediar_duplicados(log(as.numeric(oof_loc %*% pesos)), desc_train),
            test      = fun_promediar_duplicados(log(as.numeric(test_mat %*% pesos)), desc_test))

# 6. Calibración estirada y re-anuncios (como los modelos 26 y 27)

# Calibración en log: p_cal = a + m + b (p - m). b > 1 corrige la regresión a
# la media; a corrige el nivel. Se estima con la CV aleatoria ponderada.
fun_calibrar <- function(p, obs, pond) {
  m <- mean(p)
  objetivo <- function(th) weighted.mean(abs(recortar(exp(th[1] + m + th[2] * (p - m))) - obs), pond)
  th <- optim(c(0, 1), objetivo, method = "Nelder-Mead")$par
  list(a = th[1], b = th[2], m = m)
}
aplicar_cal <- function(p, cal) cal$a + cal$m + cal$b * (p - cal$m)

okc <- !is.na(ens$aleatoria)
cal_cf <- rep(NA_real_, n_train)
for (k in seq_along(f_ale)) {
  entrena <- okc & fold_ale != k
  cal_k <- fun_calibrar(ens$aleatoria[entrena], precio[entrena], w[entrena])
  cal_cf[okc & fold_ale == k] <- aplicar_cal(ens$aleatoria[okc & fold_ale == k], cal_k)
}
cal <- fun_calibrar(ens$aleatoria[okc], precio[okc], w[okc])
message("Calibración: a = ", round(cal$a, 4), ", b = ", round(cal$b, 4))
ens_cal <- list(aleatoria = cal_cf,
                localidad = aplicar_cal(ens$localidad, cal),
                test      = aplicar_cal(ens$test, cal))

# Re-anuncios: mismo procedimiento del modelo 27 
fun_llaves <- function(crudo, base) {
  crudo %>% filter(property_id %in% base$property_id) %>%
    transmute(property_id, desc = fun_normalizar(description),
              llave = paste(desc, bedrooms, replace_na(as.character(bathrooms), "na"),
                            property_type, sep = "|")) %>%
    filter(nchar(desc) >= 40) %>%
    left_join(base %>% select(property_id, num_x, num_y), by = "property_id")
}
ll_train <- fun_llaves(crudo_tr, datos$train) %>%
  left_join(datos$train %>% select(property_id, obj_price), by = "property_id")
ll_test <- fun_llaves(crudo_te, datos$test)
re_test <- ll_test %>%
  inner_join(ll_train %>% select(llave, id_ref = property_id, x_ref = num_x, y_ref = num_y,
                                 precio_ref = obj_price),
             by = "llave", relationship = "many-to-many") %>%
  filter(sqrt((num_x - x_ref)^2 + (num_y - y_ref)^2) <= 3000) %>%
  group_by(property_id) %>%
  summarise(precio_reanuncio = median(precio_ref), .groups = "drop")
message("Inmuebles de test con re-anuncio: ", nrow(re_test))
precio_re <- re_test$precio_reanuncio[match(datos$test$property_id, re_test$property_id)]
tiene_re  <- !is.na(precio_re)

# Predicción sin factor (solo para la tabla de comparación; no se envía).
pred_sin_factor <- if_else(tiene_re, log(precio_re), ens_cal$test)

##### 7. Factor de nivel #####
fun_factor_chap <- function(pred_log, grilla = seq(0.95, 1.20, by = 0.0025)) {
  p <- exp(pred_log)
  mae <- map_dbl(grilla, ~ mean(abs(recortar(.x * p[es_chap]) - precio[es_chap])))
  grilla[which.min(mae)]
}
fun_regla <- function(oof_ale, oof_loc) {
  f_a <- fun_factor_chap(oof_ale); f_l <- fun_factor_chap(oof_loc)
  tibble(factor_aleatoria = f_a, factor_localidad = f_l, factor_regla = sqrt(f_a * f_l))
}
tabla_factor <- bind_rows(
  fun_regla(fun_leer("oof_16_aleatoria.rds"), fun_leer("oof_16_localidad.rds")) %>%
    mutate(modelo = "16 (validacion: Kaggle premio 1,054)", .before = 1),
  fun_regla(ens_cal$aleatoria, ens_cal$localidad) %>%
    mutate(modelo = "Ensamble v2 calibrado", .before = 1))
print(tabla_factor)
write_csv(tabla_factor, file.path(ruta_tab, "ensamble_v2_factor.csv"))
factor_nivel <- tabla_factor$factor_regla[2]
message("Factor de nivel: ", round(factor_nivel, 4))

# Para envío
pred_final <- if_else(tiene_re, log(precio_re), ens_cal$test + log(factor_nivel))
fun_submission(datos$test$property_id, pred_final, "28_ensamble_v2_nivel",
               limites = limites_precio)

##### 8. Registro y comparación #####
oof_final <- map(ens_cal[c("aleatoria", "localidad")], ~ .x + log(factor_nivel))
iwalk(oof_final, ~ saveRDS(tibble(property_id = datos$train$property_id, pred_log = .x),
                           file.path(ruta_pred, paste0("oof_28_ensamble_v2_nivel_", .y, ".rds"))))
fun_registrar_resultado(
  "28_ensamble_v2_nivel", "Ensamble",
  paste0(paste(names(pesos), round(pesos, 3), sep = " = ", collapse = ", "),
         "; cal a = ", round(cal$a, 4), ", b = ", round(cal$b, 4),
         "; factor = ", round(factor_nivel, 4), "; re-anuncios = ", sum(tiene_re)),
  imap_dfr(oof_final, ~ tibble(esquema = .y, mae_cop = fun_metricas(.x)$mae)),
  "28_ensamble_v2_nivel")

comparacion <- bind_rows(
  imap_dfr(miembros, ~ bind_cols(tibble(modelo = .y), fun_metricas(.x$oof$aleatoria),
                                 tibble(mae_loc = fun_metricas(.x$oof$localidad)$mae))),
  bind_cols(tibble(modelo = "ensamble v2 (sin calibrar)"), fun_metricas(ens$aleatoria),
            tibble(mae_loc = fun_metricas(ens$localidad)$mae)),
  bind_cols(tibble(modelo = "ensamble calibrado (sin factor)"), fun_metricas(ens_cal$aleatoria),
            tibble(mae_loc = fun_metricas(ens_cal$localidad)$mae)),
  bind_cols(tibble(modelo = "28_ensamble_v2_nivel (envio)"), fun_metricas(oof_final$aleatoria),
            tibble(mae_loc = fun_metricas(oof_final$localidad)$mae))) %>%
  mutate(mediana_test = c(map_dbl(miembros, ~ median(recortar(exp(.x$pred_test)))),
                          median(recortar(exp(ens$test))), median(recortar(exp(pred_sin_factor))),
                          median(recortar(exp(pred_final)))))
write_csv(comparacion, file.path(ruta_tab, "ensamble_v2_comparacion.csv"))
print(comparacion %>% mutate(across(where(is.double), ~ round(.x / 1e6, 1))), width = Inf)