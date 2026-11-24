# ==============================================================================
# 14_ensamble_reanuncios.R
# Objetivo: complementar el mejor modelo del equipo (16_xgb_pesos_texto y sus
#           ajustes de sesgo 18/19) con cuatro ideas que todavía no se han
#           probado en el repositorio:
#
#   A. Tres modelos que se equivocan distinto que el 16:
#      
#        22_xgb_precio_m2     XGBoost del log del precio POR M2 (el área entra
#                             como multiplicador exacto);
#        23_xgb_palabras_m2   el mismo, más las palabras del anuncio como
#                             columnas del árbol (no solo el resumen Lasso);
#        24_nnet_robusta      red neuronal con entradas acotadas, que corrige
#                             las predicciones explosivas del 17_nnet_texto.
#   B. Ensamble (25). Promedio ponderado del 16 con 22, 23 y 24, 
#     con pesos estimados fuera de muestra.
#   C. Calibración "estirada" (26). Los ajustes 18 y 19 multiplican todo por
#      un factor constante (corrigen el NIVEL). Aquí también se corrige la
#      PENDIENTE: el modelo encoge los precios hacia la media (sobreestima lo
#      barato y subestima lo caro), y Chapinero es caro.
#   D. Re-anuncios (27). Hay inmuebles de test que ya se anunciaron en train:
#      misma descripción, mismas habitaciones, mismos baños y mismo tipo. En
#      train, el precio de esos "re-anuncios" predice el precio con un MAE de
#      ~12-14 M, contra ~100 M del mejor modelo. Para esos inmuebles de test
#      se usa el precio observado del anuncio de train.
#
#
# Validación: la del equipo. Los hiperparámetros, los pesos del ensamble y la
#   calibración se escogen con la CV ALEATORIA por grupos y el MAE ponderado
#   por el parecido con test (w), que según 03_diagnostico_errores.R es la
#   que ordena los modelos igual que Kaggle. Se reporta además la CV por
#   localidades, con los pesos y la calibración ya fijos (sin re-estimar).
#
# Entradas: 02_Data/Processed/db_{train,test}.rds, base_limpia.rds,
#           pesos_adversarial_texto.rds
#           02_Data/Raw/Kaggle/{train,test}.csv
#           03_Output/Models/16_xgb_pesos_texto.rds
#           03_Output/Predictions/oof_16_{aleatoria,localidad}.rds
#             (los crea 02_EDA/03_diagnostico_errores.R)
#           Opcional: 03_Output/Models/20_xgb_mejorado.rds y
#             03_Output/Predictions/oof_20_{aleatoria,localidad}.rds
# Salidas:  03_Output/Submissions/22_*.csv ... 27_*.csv
#           03_Output/Models/22_*.rds ... y filas en resultados_modelos.csv
#           03_Output/Predictions/oof_{22,23,24}_{aleatoria,localidad}.rds
#           03_Output/Tables/ensamble_{pesos,comparacion,reanuncios}.csv
#           03_Output/Figures/ensamble_comparacion.{pdf,png}
#
# ==============================================================================

source(here::here("01_Scripts", "00_utils.R"))
pacman::p_load(xgboost, nnet, tidytext, SnowballC, stopwords, stringi, scales)

ruta_pred <- here("03_Output", "Predictions")
ruta_tab  <- here("03_Output", "Tables")
ruta_fig  <- here("03_Output", "Figures")
ruta_mod  <- here("03_Output", "Models")
walk(c(ruta_pred, ruta_tab, ruta_fig), dir.create, recursive = TRUE, showWarnings = FALSE)

nucleos <- max(1, parallel::detectCores() - 1)
limites_precio <- c(3e8, 1.65e9)
recortar <- function(p) pmin(pmax(p, limites_precio[1]), limites_precio[2])

correr_red <- TRUE    # FALSE por si queremos saltarnos la red neuronal
incluir_20 <- TRUE    # incluir el 20_xgb_mejorado en el ensamble


##### 1. Datos, folds, pesos y matrices #####
datos   <- fun_datos_modelo()
folds   <- fun_todos_folds(datos$train)
f_ale   <- folds$aleatoria        # escoger (como 13_xgb_mejorado.R)
f_loc   <- folds$localidad        # reportar
n_train <- nrow(datos$train)

# Faltantes nativos, igual que 10_xgb_pesos_texto.R / 12_ajuste_sesgo.R: donde
# 05_merge.R imputó la mediana (bin_<x>_na = 1) se vuelve a poner NA y XGBoost
# aprende hacia qué lado mandar los faltantes
fun_restaurar_na <- function(df) {
  df <- df %>% mutate(num_vref_m2_200m = na_if(num_vref_m2_200m, 0))
  banderas <- str_subset(names(df), "^bin_.+_na$")
  for (b in banderas) {
    v <- paste0("num_", str_remove(str_remove(b, "^bin_"), "_na$"))
    if (v %in% names(df)) df[[v]][df[[b]] == 1] <- NA
  }
  df
}
fun_matriz <- function(train, test) {
  fml  <- reformulate(datos$predictores, intercept = FALSE)
  todo <- bind_rows(train[datos$predictores], test[datos$predictores])
  x <- model.matrix(fml, model.frame(fml, data = todo, na.action = na.pass))
  list(train = x[seq_len(n_train), , drop = FALSE], test = x[-seq_len(n_train), , drop = FALSE])
}
x_na  <- fun_matriz(fun_restaurar_na(datos$train), fun_restaurar_na(datos$test))  # XGBoost
x_imp <- fun_matriz(datos$train, datos$test)                                       # red

y_train <- datos$train$obj_log_price
precio  <- datos$train$obj_price
es_chap <- datos$train$cat_localidad == "CHAPINERO"

# Pesos de parecido con test (w) y peso de entrenamiento del modelo 16
# (w^0,5 normalizado a media 1), los mismos de 10_xgb_pesos_texto.R.
modelo_16 <- readRDS(file.path(ruta_mod, "16_xgb_pesos_texto.rds"))
hp16 <- modelo_16$hiperparam
w <- datos$train %>%
  select(property_id) %>%
  left_join(readRDS(here("02_Data", "Processed", "pesos_adversarial_texto.rds")),
            by = "property_id") %>%
  pull(peso)
stopifnot(!anyNA(w))
peso_16 <- w^hp16$gamma_peso
peso_16 <- peso_16 / mean(peso_16)

# El área se acota a los percentiles
# 1 y 99 de train (44 y ~386 m2) porque un área absurda (p. ej. 1.000 m2 en
# un apartamento) multiplicaría la predicción. 
lim_area  <- quantile(datos$train$num_area, c(0.01, 0.99), na.rm = TRUE)
off_train <- log(pmin(pmax(datos$train$num_area, lim_area[1]), lim_area[2]))
off_test  <- log(pmin(pmax(datos$test$num_area,  lim_area[1]), lim_area[2]))

# Métricas en pesos de un vector de predicciones (en log), siempre recortadas
# al rango de train como en los envíos.
fun_metricas <- function(pred_log) {
  ok  <- !is.na(pred_log)
  err <- recortar(exp(pred_log)) - precio
  tibble(mae            = mean(abs(err[ok])),
         mae_pond       = weighted.mean(abs(err[ok]), w[ok]),
         mae_chap       = mean(abs(err[ok & es_chap])),
         sesgo_chap     = mean(err[ok & es_chap]),
         pct_sobre_chap = 100 * mean(err[ok & es_chap] > 0))
}

# ------------------------------------------------------------------------------
# 2. Modelo 16 del equipo: predicciones fuera de muestra y de test
# ------------------------------------------------------------------------------
# No se re-entrena: se leen las predicciones fuera de muestra que guardó
# 03_diagnostico_errores.R (mismos folds, porque fun_folds_cv fija la semilla).
fun_leer_oof_equipo <- function(modelo, esquema) {
  ruta <- file.path(ruta_pred, paste0("oof_", modelo, "_", esquema, ".rds"))
  if (!file.exists(ruta)) return(NULL)
  r <- readRDS(ruta)
  col <- if ("pred_log" %in% names(r)) r$pred_log else log(r$pred)
  col[match(datos$train$property_id, r$property_id)]
}
oof16 <- list(aleatoria = fun_leer_oof_equipo("16", "aleatoria"),
              localidad = fun_leer_oof_equipo("16", "localidad"))
if (any(map_lgl(oof16, is.null))) {
  stop("Faltan las predicciones fuera de muestra del 16: correr primero ",
       "01_Scripts/02_EDA/03_diagnostico_errores.R")
}
res_16 <- list(nombre = "16_xgb_pesos_texto", oof = oof16, pred_test = modelo_16$pred_test)

# ------------------------------------------------------------------------------
# 3. Palabras del anuncio como columnas (para el modelo 23)
# ------------------------------------------------------------------------------
# El equipo resumió las palabras en UNA variable (la predicción del Lasso) y
# en 15 tópicos. Eso captura el efecto PROMEDIO de cada palabra. Con las
# palabras como columnas, el árbol puede encontrar interacciones: por ejemplo,
# que "remodel" (remodelado) vale más en un edificio viejo de estrato 6 que en
# uno nuevo, o que "vista" vale más en un piso alto.
# Se construyen igual que en 06_texto_como_datos.R (minúsculas, sin tildes,
# sin palabras vacías, raíces de Snowball, palabras y bigramas), pero con un
# vocabulario más corto (términos en al menos 2% de los anuncios) para que la
# matriz quepa en memoria como matriz densa.
# ¿Por qué densa y no dispersa? XGBoost trata las celdas vacías de una matriz
# dispersa como FALTANTES, no como ceros. Al mezclar las variables numéricas
# (que tienen ceros de verdad y NA de verdad) con las palabras en una matriz
# dispersa, se confundirían ceros y faltantes.
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
  palabras <- docs %>%
    select(doc, texto) %>%
    unnest_tokens(palabra, texto, token = "words") %>%
    filter(!palabra %in% vacias, nchar(palabra) > 2) %>%
    mutate(raiz = wordStem(palabra, language = "spanish"))
  bigramas <- palabras %>%
    group_by(doc) %>% mutate(siguiente = lead(raiz)) %>% ungroup() %>%
    filter(!is.na(siguiente)) %>% transmute(doc, termino = paste(raiz, siguiente))
  terminos <- bind_rows(palabras %>% transmute(doc, termino = raiz), bigramas) %>%
    distinct(doc, termino)                                  # uso (0/1), no conteo
  vocabulario <- terminos %>% count(termino) %>%
    filter(n >= ceiling(min_prop * nrow(docs))) %>% pull(termino)
  terminos <- terminos %>% filter(termino %in% vocabulario)
  m <- matrix(0, nrow = nrow(docs), ncol = length(vocabulario),
              dimnames = list(NULL, paste0("pal_", make.names(vocabulario))))
  m[cbind(terminos$doc, match(terminos$termino, vocabulario))] <- 1
  message("Palabras como columnas: ", ncol(m), " términos")
  list(train = m[seq_len(n_train), , drop = FALSE], test = m[-seq_len(n_train), , drop = FALSE])
}

# ------------------------------------------------------------------------------
# 4. XGBoost del precio por m2 (con y sin palabras)
# ------------------------------------------------------------------------------
# Parámetros del 16 (profundidad 6, min_child_weight 1, eta 0,05, submuestreo
# 0,8), con sus mismos pesos y faltantes nativos. Lo único que cambia es la
# variable objetivo (y, en el 23, las columnas).
parametros_16 <- list(objective = "reg:squarederror", eval_metric = "mae",
                      eta = hp16$eta, subsample = 0.8, colsample_bytree = 0.7,
                      max_depth = hp16$max_depth, min_child_weight = hp16$min_child_weight,
                      nthread = nucleos)

fun_mejor_ronda <- function(modelo) {
  ronda <- attributes(modelo)$early_stop$best_iteration
  if (is.null(ronda)) ronda <- as.integer(xgb.attr(modelo, "best_iteration")) + 1
  as.integer(ronda)
}

# CV con offset. Con rondas = NULL usa parada temprana con el MAE ponderado
# por w en el fold de validación (el criterio de 13_xgb_mejorado.R).
fun_cv_xgb_m2 <- function(x, folds, parametros, rondas = NULL, max_rondas = 3000) {
  pred <- rep(NA_real_, n_train)
  rondas_fold <- integer(length(folds))
  for (i in seq_along(folds)) {
    f <- folds[[i]]
    d_tr <- xgb.DMatrix(x[f$train, , drop = FALSE],
                        label = y_train[f$train] - off_train[f$train], weight = peso_16[f$train])
    d_va <- xgb.DMatrix(x[f$val, , drop = FALSE],
                        label = y_train[f$val] - off_train[f$val], weight = w[f$val])
    set.seed(semilla)
    if (is.null(rondas)) {
      m <- xgb.train(params = parametros, data = d_tr, nrounds = max_rondas,
                     evals = list(val = d_va), early_stopping_rounds = 50, verbose = 0)
      rondas_fold[i] <- fun_mejor_ronda(m)
    } else {
      m <- xgb.train(params = parametros, data = d_tr, nrounds = rondas, verbose = 0)
      rondas_fold[i] <- rondas
    }
    pred[f$val] <- predict(m, d_va, iterationrange = c(1, rondas_fold[i])) + off_train[f$val]
  }
  list(pred = pred, rondas = rondas_fold)
}

# Corre un modelo completo: CV aleatoria (escoge rondas), CV por localidades
# (rondas fijas), ajuste final, envío y registro. Guarda todo en caché.
fun_modelo_m2 <- function(envio, x_tr, x_te, parametros, descripcion) {
  ruta_cache <- file.path(ruta_mod, paste0(envio, ".rds"))
  if (file.exists(ruta_cache)) {
    message("Cargando ", envio, " (ya entrenado)")
    return(readRDS(ruta_cache))
  }
  message("\n== ", envio, " ==")
  cv_ale <- fun_cv_xgb_m2(x_tr, f_ale, parametros)
  rondas <- round(mean(cv_ale$rondas))
  cv_loc <- fun_cv_xgb_m2(x_tr, f_loc, parametros, rondas = rondas)
  set.seed(semilla)
  final <- xgb.train(params = parametros,
                     data = xgb.DMatrix(x_tr, label = y_train - off_train, weight = peso_16),
                     nrounds = rondas, verbose = 0)
  pred_test <- predict(final, xgb.DMatrix(x_te)) + off_test
  hp <- data.frame(objetivo = "log(precio/m2)", gamma_peso = hp16$gamma_peso,
                   max_depth = parametros$max_depth,
                   min_child_weight = parametros$min_child_weight,
                   colsample = parametros$colsample_bytree, eta = parametros$eta,
                   rondas = rondas, columnas = ncol(x_tr))
  res <- list(nombre = envio, oof = list(aleatoria = cv_ale$pred, localidad = cv_loc$pred),
              pred_test = pred_test, hiperparam = hp,
              importancia = xgb.importance(model = final))
  fun_cerrar_modelo(res, "XGBoost", descripcion)
  saveRDS(res, ruta_cache)
  res
}

# Envío, registro y predicciones fuera de muestra de un modelo.
fun_cerrar_modelo <- function(res, familia, descripcion) {
  fun_submission(datos$test$property_id, res$pred_test, res$nombre, limites = limites_precio)
  iwalk(res$oof, ~ saveRDS(tibble(property_id = datos$train$property_id, pred_log = .x),
                           file.path(ruta_pred, paste0("oof_", res$nombre, "_", .y, ".rds"))))
  mae <- imap_dfr(res$oof, ~ tibble(esquema = .y, mae_cop = fun_metricas(.x)$mae))
  fun_registrar_resultado(res$nombre, familia, fun_texto_hiperparam(res$hiperparam), mae, res$nombre)
  m_ale <- fun_metricas(res$oof$aleatoria)
  message("  ", res$nombre, " | CV aleatoria: MAE ", round(m_ale$mae / 1e6, 1),
          " M, ponderado ", round(m_ale$mae_pond / 1e6, 1), " M | CV localidades: MAE ",
          round(fun_metricas(res$oof$localidad)$mae / 1e6, 1), " M")
  message("  Descripción para Kaggle: ", descripcion)
}

# 4.1 (22) Precio por m2.
# Hedónicamente, precio = (precio por m2) x área. El precio por m2 depende de
# la ubicación y la calidad; el área ya la conocemos. Con el objetivo en
# log(precio) - log(área), el árbol no gasta particiones en aprender que lo
# grande vale más: eso entra exacto (es una elasticidad de 1 del precio al
# área, que el modelo puede corregir porque log(área) sigue siendo predictor).
# Por sí solo rinde parecido al 16; su valor es que se equivoca DISTINTO
# (piensa en "cuánto vale el m2 aquí"), lo que lo hace útil en el ensamble.
res_22 <- fun_modelo_m2("22_xgb_precio_m2", x_na$train, x_na$test, parametros_16,
                        "XGBoost (config. del 16) sobre log(precio/m2), offset log(área), pesos w^0,5.")

# 4.2 (23) Precio por m2 + palabras como columnas.
# colsample_bytree baja a 0,5: con cientos de columnas de palabras, cada árbol
# ve una muestra más chica de columnas, lo que lo diversifica y evita que unas
# pocas palabras dominen.
pal <- fun_dtm(min_prop = 0.02)
res_23 <- fun_modelo_m2("23_xgb_palabras_m2", cbind(x_na$train, pal$train),
                        cbind(x_na$test, pal$test),
                        modifyList(parametros_16, list(colsample_bytree = 0.5)),
                        "XGBoost log(precio/m2) con las palabras del anuncio como columnas.")
rm(pal); invisible(gc())

# ------------------------------------------------------------------------------
# 5. (24) Red neuronal robusta
# ------------------------------------------------------------------------------
# El 17_nnet_texto tuvo MAE de 348 M en localidades y de más de 40.000 M en
# bloques: algunas predicciones "explotan". La causa: la red se estandariza
# con media y desviación, pero en una zona NO vista algunas variables toman
# valores muy por fuera del rango de entrenamiento (10 o 20 desviaciones); la
# red, que es una función no acotada de sus entradas, extrapola sin control.
# Los árboles no tienen ese problema porque su predicción siempre es el
# promedio de alguna hoja.
# Corrección: transformar cada variable continua a su CUANTIL en train y
# luego a la normal estándar (qnorm). Así toda entrada queda entre -3,1 y
# +3,1, también en test: un valor más alto que todos los de train recibe el
# cuantil 99,9%, no "20 desviaciones". Es la misma idea de
# QuantileTransformer en scikit-learn. Las dummies solo se centran y escalan.
# Además: objetivo por m2 con offset (como el 22), una capa oculta de 8
# neuronas, penalización L2 (decay) de 0,5 y salida lineal.
# En la réplica, la CV por localidades pasó de 348 M (el 17) a ~150 M.
fun_cuantil_normal <- function(a_mat, b_mat) {
  for (j in seq_len(ncol(a_mat))) {
    a <- a_mat[, j]
    if (length(unique(a)) <= 2) {                      # dummies
      m <- mean(a); s <- sd(a); if (!is.finite(s) || s == 0) s <- 1
      a_mat[, j] <- (a - m) / s; b_mat[, j] <- (b_mat[, j] - m) / s
      next
    }
    p <- seq(0.001, 0.999, length.out = 200)
    q <- quantile(a, p, names = FALSE)
    f <- function(v) approx(q, qnorm(p), xout = v, rule = 2, ties = mean)$y
    a_mat[, j] <- f(a); b_mat[, j] <- f(b_mat[, j])
  }
  list(a = a_mat, b = b_mat)
}
fun_red <- function(filas, x_nuevo, off_nuevo) {
  z <- fun_cuantil_normal(x_imp$train[filas, , drop = FALSE], x_nuevo)
  set.seed(semilla)
  m <- nnet(z$a, y_train[filas] - off_train[filas], size = 8, decay = 0.5, linout = TRUE,
            maxit = 300, MaxNWts = 1e5, trace = FALSE)
  as.numeric(predict(m, z$b)) + off_nuevo
}

res_24 <- NULL
if (correr_red) {
  ruta_24 <- file.path(ruta_mod, "24_nnet_robusta.rds")
  if (file.exists(ruta_24)) {
    res_24 <- readRDS(ruta_24)
  } else {
    message("\n== 24_nnet_robusta == (unos 20 minutos)")
    fun_cv_red <- function(folds) {
      pred <- rep(NA_real_, n_train)
      for (i in seq_along(folds)) {
        f <- folds[[i]]
        pred[f$val] <- fun_red(f$train, x_imp$train[f$val, , drop = FALSE], off_train[f$val])
        message("  fold ", i, " listo")
      }
      pred
    }
    res_24 <- list(nombre = "24_nnet_robusta",
                   oof = list(aleatoria = fun_cv_red(f_ale), localidad = fun_cv_red(f_loc)),
                   pred_test = fun_red(seq_len(n_train), x_imp$test, off_test),
                   hiperparam = data.frame(size = 8, decay = 0.5, entradas = "cuantil normal",
                                           objetivo = "log(precio/m2)"))
    fun_cerrar_modelo(res_24, "Red neuronal",
                      "nnet 1 capa (8), decay 0,5, entradas transformadas a cuantil normal, log(precio/m2).")
    saveRDS(res_24, ruta_24)
  }
}

# Opcional: el 20_xgb_mejorado del equipo, si ya se corrió.
res_20 <- NULL
if (incluir_20 && file.exists(file.path(ruta_mod, "20_xgb_mejorado.rds"))) {
  oof20 <- list(aleatoria = fun_leer_oof_equipo("20", "aleatoria"),
                localidad = fun_leer_oof_equipo("20", "localidad"))
  if (!any(map_lgl(oof20, is.null))) {
    res_20 <- list(nombre = "20_xgb_mejorado", oof = oof20,
                   pred_test = readRDS(file.path(ruta_mod, "20_xgb_mejorado.rds"))$pred_test)
    message("Se incluye el 20_xgb_mejorado en el ensamble")
  }
}

# ==============================================================================
# 6. (25) ENSAMBLE
# ==============================================================================
# Si dos modelos tienen errores e1 y e2 con la misma varianza s2 y correlación
# rho, el promedio tiene varianza s2 (1 + rho) / 2: mientras rho < 1, promediar
# reduce el error. Es la lógica de bagging, pero entre modelos distintos.
# Pesos: no negativos y que suman 1 (combinación convexa), escogidos para
# minimizar el MAE en pesos PONDERADO por w con las predicciones fuera de
# muestra de la CV aleatoria. Usar predicciones dentro de muestra premiaría al
# modelo más sobreajustado.
# Para no engañarnos, el MAE del ensamble se calcula con "cross-fitting": los
# pesos que se aplican a un fold se estiman solo con los otros cuatro. Los
# pesos finales (para test) se estiman con todos los folds. Por último, esos
# pesos fijos se aplican a las predicciones de la CV por localidades, que no
# se usaron para estimarlos.
miembros <- compact(list(res_16, res_20, res_22, res_23, res_24))
names(miembros) <- map_chr(miembros, "nombre")
fun_mat <- function(esquema) sapply(miembros, function(r) recortar(exp(r$oof[[esquema]])))
oof_ale  <- fun_mat("aleatoria")
oof_loc  <- fun_mat("localidad")
test_mat <- sapply(miembros, function(r) recortar(exp(r$pred_test)))

fold_ale <- integer(n_train)
for (i in seq_along(f_ale)) fold_ale[f_ale[[i]]$val] <- i

# Pesos convexos vía softmax: w_j = exp(t_j) / sum(exp(t)). Así optim() busca
# t sin restricciones y los pesos siempre son >= 0 y suman 1.
fun_pesos <- function(P, obs, pond) {
  objetivo <- function(t) {
    a <- exp(t) / sum(exp(t))
    weighted.mean(abs(P %*% a - obs), pond)
  }
  t <- optim(rep(0, ncol(P)), objetivo, method = "Nelder-Mead",
             control = list(maxit = 5000))$par
  setNames(exp(t) / sum(exp(t)), colnames(P))
}

ok <- complete.cases(oof_ale)
ens_ale_cf <- rep(NA_real_, n_train)
for (k in seq_along(f_ale)) {
  entrena <- ok & fold_ale != k
  a_k <- fun_pesos(oof_ale[entrena, ], precio[entrena], w[entrena])
  ens_ale_cf[ok & fold_ale == k] <- oof_ale[ok & fold_ale == k, ] %*% a_k
}
pesos_ens <- fun_pesos(oof_ale[ok, ], precio[ok], w[ok])
print(round(pesos_ens, 3))
write_csv(enframe(pesos_ens, "modelo", "peso"), file.path(ruta_tab, "ensamble_pesos.csv"))

res_25 <- list(nombre = "25_ensamble",
               oof = list(aleatoria = log(ens_ale_cf),
                          localidad = log(as.numeric(oof_loc %*% pesos_ens))),
               pred_test = log(as.numeric(test_mat %*% pesos_ens)),
               hiperparam = as.data.frame(as.list(round(pesos_ens, 3)), check.names = FALSE))
fun_cerrar_modelo(res_25, "Ensamble",
                  paste("Promedio ponderado (pesos fuera de muestra):",
                        paste(names(pesos_ens), round(pesos_ens, 2), sep = " = ", collapse = ", ")))

# ==============================================================================
# 7. (26) CALIBRACIÓN ESTIRADA
# ==============================================================================
# El diagnóstico del equipo (diag_sesgo_deciles.csv) muestra regresión a la
# media: el modelo sobreestima los inmuebles baratos y subestima los caros.
# Es lo esperable: los árboles predicen promedios de hojas y la
# regularización encoge todo hacia el centro. Como test (Chapinero) es caro,
# ese encogimiento se vuelve un sesgo negativo.
# Los envíos 18 y 19 corrigen solo el nivel: precio x factor (en log,
# sumar una constante a). Aquí se corrige nivel Y pendiente, en log:
#     p_cal = a + m + b (p - m),
# donde m es la media de las predicciones fuera de muestra. Con b > 1 las
# predicciones se "estiran" alejándose de m: los caros suben, los baratos
# bajan. a y b se escogen minimizando el MAE ponderado por w, con
# cross-fitting para evaluar, y se aplican tal cual a la CV por localidades.
fun_calibrar <- function(p, obs, pond) {
  m <- mean(p)
  objetivo <- function(th) weighted.mean(abs(recortar(exp(th[1] + m + th[2] * (p - m))) - obs), pond)
  th <- optim(c(0, 1), objetivo, method = "Nelder-Mead")$par
  list(a = th[1], b = th[2], m = m)
}
aplicar_cal <- function(p, cal) cal$a + cal$m + cal$b * (p - cal$m)

p25 <- res_25$oof$aleatoria
ok_c <- !is.na(p25)
cal_cf <- rep(NA_real_, n_train)
for (k in seq_along(f_ale)) {
  entrena <- ok_c & fold_ale != k
  cal_k <- fun_calibrar(p25[entrena], precio[entrena], w[entrena])
  cal_cf[ok_c & fold_ale == k] <- aplicar_cal(p25[ok_c & fold_ale == k], cal_k)
  message("  Calibración sin el fold ", k, ": a = ", round(cal_k$a, 3), ", b = ", round(cal_k$b, 3))
}
cal <- fun_calibrar(p25[ok_c], precio[ok_c], w[ok_c])
res_26 <- list(nombre = "26_ensamble_calibrado",
               oof = list(aleatoria = cal_cf, localidad = aplicar_cal(res_25$oof$localidad, cal)),
               pred_test = aplicar_cal(res_25$pred_test, cal),
               hiperparam = data.frame(a = round(cal$a, 4), b = round(cal$b, 4)))
fun_cerrar_modelo(res_26, "Ensamble",
                  paste0("Ensamble 25 con calibración log: a = ", round(cal$a, 3),
                         ", b = ", round(cal$b, 3), " (estira la regresión a la media)."))

# ==============================================================================
# 8. (27) RE-ANUNCIOS
# ==============================================================================
# Un mismo apartamento suele publicarse varias veces (otra inmobiliaria, otra
# fecha). En train, entre anuncios distintos (sin contar los duplicados
# exactos que 05_merge.R ya quitó) con la MISMA descripción, las mismas
# habitaciones, los mismos baños y el mismo tipo, el precio es idéntico en
# ~77% de los casos y el MAE de usar el precio del otro anuncio es ~12-14 M.
# El mejor modelo tiene, en esos mismos anuncios, un MAE de ~95 M. Para esos casos, el precio
# observado es mucho mejor predictor que cualquier modelo.
# Diferencia con el "comparable exacto" de 13_xgb_mejorado.R: ese usa
# anuncios con las mismas coordenadas pero con OTRA descripción (vecinos en el
# mismo edificio); aquí es la MISMA descripción (el mismo anuncio publicado
# otra vez), sin exigir las mismas coordenadas (los re-anuncios suelen tener
# coordenadas distintas porque cada portal geocodifica a su manera).
# Reglas: descripción de al menos 40 caracteres (las cortas son genéricas,
# p. ej. "apartamento en venta"), y distancia entre los dos anuncios de hasta
# 3 km (una descripción idéntica a 10 km es más probablemente un texto modelo
# de una inmobiliaria que el mismo inmueble).
# La mezcla final es lambda x precio_reanuncio + (1 - lambda) x modelo, con
# lambda escogido en train comparando contra las predicciones fuera de
# muestra del 26.
fun_normalizar <- function(x) {
  x %>% replace_na("") %>% str_to_lower() %>% stri_trans_general("Latin-ASCII") %>%
    str_replace_all("[^a-z0-9 ]", " ") %>% str_squish()
}
fun_llaves <- function(archivo, base) {
  read_csv(here("02_Data", "Raw", "Kaggle", archivo), show_col_types = FALSE) %>%
    filter(property_id %in% base$property_id) %>%
    transmute(property_id,
              desc = fun_normalizar(description),
              llave = paste(desc, bedrooms, replace_na(as.character(bathrooms), "na"),
                            property_type, sep = "|")) %>%
    filter(nchar(desc) >= 40) %>%
    left_join(base %>% select(property_id, num_x, num_y), by = "property_id")
}
ll_train <- fun_llaves("train.csv", datos$train) %>%
  left_join(datos$train %>% select(property_id, obj_price), by = "property_id")
ll_test  <- fun_llaves("test.csv", datos$test)

# Para cada anuncio de `consulta`, mediana del precio de los anuncios de train
# con la misma llave, a menos de `dist_max` metros y con otro property_id.
fun_reanuncio <- function(consulta, dist_max = 3000) {
  consulta %>%
    inner_join(ll_train %>% select(llave, id_ref = property_id, x_ref = num_x,
                                   y_ref = num_y, precio_ref = obj_price),
               by = "llave", relationship = "many-to-many") %>%
    filter(property_id != id_ref,
           sqrt((num_x - x_ref)^2 + (num_y - y_ref)^2) <= dist_max) %>%
    group_by(property_id) %>%
    summarise(precio_reanuncio = median(precio_ref), n_reanuncios = n(), .groups = "drop")
}

# 8.1 Evidencia en train: ¿qué tan bueno es el re-anuncio frente al modelo?
ev <- fun_reanuncio(ll_train) %>%
  left_join(tibble(property_id = datos$train$property_id, obs = precio,
                   modelo = recortar(exp(res_26$oof$aleatoria))), by = "property_id")
tabla_lambda <- map_dfr(seq(0, 1, by = 0.1), function(l) {
  tibble(lambda = l, n = nrow(ev),
         mae = mean(abs(l * ev$precio_reanuncio + (1 - l) * ev$modelo - ev$obs)))
})
print(tabla_lambda %>% mutate(mae_millones = round(mae / 1e6, 1)), n = Inf)
lambda <- tabla_lambda$lambda[which.min(tabla_lambda$mae)]
message("Re-anuncios en train: ", nrow(ev), " | MAE modelo ",
        round(mean(abs(ev$modelo - ev$obs)) / 1e6, 1), " M; MAE re-anuncio ",
        round(mean(abs(ev$precio_reanuncio - ev$obs)) / 1e6, 1), " M; lambda = ", lambda)

# 8.2 Aplicación a test.
re_test <- fun_reanuncio(ll_test)
message("Inmuebles de test con re-anuncio en train: ", nrow(re_test))

# OJO al leer la CV del 27: en train ~1 de cada 4 anuncios tiene un
# re-anuncio (por eso su MAE de CV cae tanto), pero en test solo ~1,3%. La
# ganancia esperada en Kaggle es mucho menor: aproximadamente
#   (inmuebles de test con re-anuncio / total de test) x
#   (MAE del modelo en re-anuncios - MAE del re-anuncio).
# Como el modelo se equivoca más en Chapinero que en train, esta cuenta con
# el MAE de train es más bien conservadora.
ganancia_esperada <- nrow(re_test) / nrow(datos$test) *
  (mean(abs(ev$modelo - ev$obs)) - min(tabla_lambda$mae))
message("Ganancia esperada en el MAE de test por los re-anuncios: ~",
        round(ganancia_esperada / 1e6, 1), " M")
pred_27 <- tibble(property_id = datos$test$property_id,
                  modelo = recortar(exp(res_26$pred_test))) %>%
  left_join(re_test, by = "property_id") %>%
  mutate(final = if_else(is.na(precio_reanuncio), modelo,
                         lambda * precio_reanuncio + (1 - lambda) * modelo))
write_csv(bind_rows(tabla_lambda %>% mutate(tabla = "lambda en train"),
                    tibble(tabla = "test", n = nrow(re_test),
                           mae = NA_real_, lambda = lambda)),
          file.path(ruta_tab, "ensamble_reanuncios.csv"))

# El MAE de la CV del 27 es el del 26 con los re-anuncios de train corregidos.
oof27 <- res_26$oof$aleatoria
idx_ev <- match(ev$property_id, datos$train$property_id)
oof27[idx_ev] <- log(lambda * ev$precio_reanuncio + (1 - lambda) * ev$modelo)
res_27 <- list(nombre = "27_ensamble_calibrado_reanuncios",
               oof = list(aleatoria = oof27, localidad = res_26$oof$localidad),
               pred_test = log(pred_27$final),
               hiperparam = data.frame(lambda = lambda, dist_max_m = 3000,
                                       n_test = nrow(re_test)))
fun_cerrar_modelo(res_27, "Ensamble",
                  paste0("Ensamble calibrado 26 + precio de re-anuncios de train (", nrow(re_test),
                         " inmuebles de test, lambda = ", lambda, ")."))

# ==============================================================================
# 9. COMPARACIÓN
# ==============================================================================
todos <- compact(list(res_16, res_20, res_22, res_23, res_24, res_25, res_26, res_27))
comparacion <- map_dfr(todos, function(r) {
  bind_cols(tibble(modelo = r$nombre),
            fun_metricas(r$oof$aleatoria) %>% rename_with(~ paste0(.x, "_ale")),
            tibble(mae_loc = fun_metricas(r$oof$localidad)$mae,
                   mae_chap_loc = fun_metricas(r$oof$localidad)$mae_chap,
                   sesgo_chap_loc = fun_metricas(r$oof$localidad)$sesgo_chap,
                   mediana_test = median(recortar(exp(r$pred_test)))))
}) %>%
  mutate(dif_vs_16_pond = mae_pond_ale - mae_pond_ale[modelo == "16_xgb_pesos_texto"])
write_csv(comparacion, file.path(ruta_tab, "ensamble_comparacion.csv"))
print(comparacion %>% mutate(across(where(is.double) & !starts_with("pct"), ~ round(.x / 1e6, 1))),
      width = Inf)

p_comp <- comparacion %>%
  select(modelo, `CV aleatoria, MAE ponderado por w` = mae_pond_ale,
         `CV por localidades, MAE` = mae_loc) %>%
  pivot_longer(-modelo, names_to = "metrica", values_to = "mae") %>%
  mutate(modelo = factor(modelo, levels = rev(map_chr(todos, "nombre")))) %>%
  ggplot(aes(mae / 1e6, modelo)) +
  geom_col(fill = "#2a78d6", width = 0.65) +
  geom_text(aes(label = number(mae / 1e6, accuracy = 0.1, decimal.mark = ",")),
            hjust = -0.15, size = 3.2) +
  facet_wrap(~ metrica, scales = "free_x") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(title = "Ensamble, calibración y re-anuncios frente al modelo 16",
       subtitle = "Predicciones fuera de muestra, millones de COP",
       x = NULL, y = NULL) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank(), plot.title.position = "plot")
ggsave(file.path(ruta_fig, "ensamble_comparacion.png"), p_comp,
       width = 10, height = 4.8, dpi = 300, bg = "white")
ggsave(file.path(ruta_fig, "ensamble_comparacion.pdf"), p_comp, width = 10, height = 4.8)

# ------------------------------------------------------------------------------
# QUÉ SUBIR A KAGGLE Y CÓMO LEERLO
# ------------------------------------------------------------------------------
# Orden sugerido: 27 (todo junto), 25 (solo ensamble), 26 (ensamble +
# calibración). Así cada paso se mide por separado: 25 vs 16 mide el ensamble,
# 26 vs 25 la calibración y 27 vs 26 los re-anuncios. 22, 23 y 24 solos no
# deberían ganarle al 16 (no es su función), solo vale la pena subirlos si
# sobran envíos.
# * Sobre el factor de los envíos 18/19: el 26 ya corrige el nivel (a). No
#   combinarlo además con el factor 1,054 de diag_factor_optimo.csv sin
#   probar, porque se corregiría dos veces.
# * dif_vs_16_pond < 0: el modelo mejora al 16 con el criterio que mejor
#   ha anticipado a Kaggle. EXCEPCIÓN: el 27, cuya CV exagera la ganancia
#   porque en train hay muchos más re-anuncios que en test (ver 8.2); para él
#   la cifra relevante es la "ganancia esperada" que imprime la sección 8.
# ==============================================================================
