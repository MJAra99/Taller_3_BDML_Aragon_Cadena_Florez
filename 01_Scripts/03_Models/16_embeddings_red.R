# ==============================================================================
# 16_embeddings_red.R
# Objetivo: último ensayo. Agregar al ensamble del 28 la información del texto
#           vista con embeddings de frases preentrenados (transfer learning,
#           Lecture 15), en vez de bolsa de palabras.
#
#   Hasta ahora todo el texto entra como bolsa de palabras: Lasso sobre
#   palabras, 15 tópicos LDA y ~700 palabras como columnas. Eso ignora el
#   contexto y el orden ("vista a los cerros" vs. "cerca de los cerros"). Un
#   embedding de frases resume cada anuncio en un vector denso (384 o 768
#   dimensiones) donde anuncios que dicen lo mismo con palabras distintas
#   quedan cerca.
#
#   Tres usos de los embeddings (calculados por 01_Data/07_embeddings_texto.py):
#   1. PCA a 30 componentes (la misma idea de LSA de clase: comprimir el texto
#      en las direcciones de mayor varianza). Entran como columnas al XGBoost.
#   2. Un modelo solo-texto: ridge sobre los embeddings completos que predice
#      el log del precio por m2, fuera de muestra (stacking, como el Lasso de
#      06_texto_como_datos.R). Su predicción entra como una columna más.
#   3. Una red neuronal (nnet) con las variables tabulares y los embeddings,
#      con las entradas transformadas a cuantil normal (como la 24). Esta red
#      SÍ ve algo que los árboles no ven bien (variables densas y continuas
#      del texto), así que puede ganar peso en el ensamble.
#
#   Luego se repite el ensamble del 28 (mismos miembros + los dos nuevos), con
#   promedio de duplicados, calibración estirada, re-anuncios y la regla del
#   factor de nivel. Para comparar con justicia, también se recalcula el
#   ensamble SIN los miembros nuevos con el mismo procedimiento.
#
# Envíos:
#   29_ensamble_embeddings   el ensamble completo (la apuesta para Kaggle)
#   30_nnet_embeddings       la red sola (puntaje de la familia de redes)
#
# Entradas: 02_Data/Processed/{db_train,db_test,base_limpia,features_mejorado,
#             pesos_adversarial_texto}.rds
#           02_Data/Processed/embeddings_texto_<modelo>.csv.gz (del .py)
#           02_Data/Raw/Kaggle/{train,test}.csv
#           03_Output/Models/{20_xgb_mejorado,23_xgb_palabras_m2,
#             xgb_completo_palabras_m2_prof6,xgb_completo_palabras_m2_prof8}.rds
#           03_Output/Predictions/oof_{16,20}_{aleatoria,localidad}.rds,
#             oof_23_xgb_palabras_m2_{aleatoria,localidad}.rds
# Salidas:  03_Output/Submissions/29_ensamble_embeddings.csv, 30_nnet_embeddings.csv
#           03_Output/Models/{xgb_emb_prof6,nnet_emb,emb_ridge}.rds (en caché)
#           03_Output/Predictions/oof_29_ensamble_embeddings_{aleatoria,localidad}.rds
#           03_Output/Predictions/oof_30_nnet_embeddings_{aleatoria,localidad}.rds
#           03_Output/Tables/emb_{pca_varianza,pesos,comparacion,factor}.csv
#           03_Output/Figures/emb_pca_varianza.{pdf,png}
#           filas nuevas en resultados_modelos.csv
#
# Tiempo aproximado: 40-70 min (XGBoost x semillas + red). Cada miembro se
# guarda en caché: si el .rds existe se carga (borrarlo para re-entrenar).
# ==============================================================================

source(here::here("01_Scripts", "00_utils.R"))
pacman::p_load(xgboost, nnet, glmnet, tidytext, SnowballC, stopwords, stringi, scales)

modelo_emb   <- "minilm"   # "minilm" o "beto": el que se haya corrido en el .py
n_pca        <- 30         # componentes principales del embedding
n_semillas   <- 3          # semillas del XGBoost (1 = prueba rápida)
correr_red   <- TRUE

ruta_pred <- here("03_Output", "Predictions")
ruta_tab  <- here("03_Output", "Tables")
ruta_mod  <- here("03_Output", "Models")
ruta_fig  <- here("03_Output", "Figures")
walk(c(ruta_pred, ruta_tab, ruta_fig), dir.create, recursive = TRUE, showWarnings = FALSE)

nucleos <- max(1, parallel::detectCores() - 1)
limites_precio <- c(3e8, 1.65e9)
recortar <- function(p) pmin(pmax(p, limites_precio[1]), limites_precio[2])
semillas <- semilla + seq_len(n_semillas) - 1

# Los miembros del 28 deben existir (los crea 15_ensamble_v2.R).
requeridos <- file.path(ruta_mod, c("20_xgb_mejorado.rds", "23_xgb_palabras_m2.rds",
                                    "xgb_completo_palabras_m2_prof6.rds",
                                    "xgb_completo_palabras_m2_prof8.rds"))
if (!all(file.exists(requeridos))) {
  stop("Faltan modelos del 28 en 03_Output/Models: ",
       paste(basename(requeridos[!file.exists(requeridos)]), collapse = ", "),
       ". Corre primero 15_ensamble_v2.R.")
}
ruta_emb <- here("02_Data", "Processed", paste0("embeddings_texto_", modelo_emb, ".csv.gz"))
if (!file.exists(ruta_emb)) stop("No encuentro ", ruta_emb, ". Corre primero 07_embeddings_texto.py.")

# ------------------------------------------------------------------------------
# 1. Datos (igual que 15_ensamble_v2.R)
# ------------------------------------------------------------------------------
datos   <- fun_datos_modelo()
folds   <- fun_todos_folds(datos$train)
f_ale   <- folds$aleatoria
f_loc   <- folds$localidad
n_train <- nrow(datos$train)
n_test  <- nrow(datos$test)

fun_restaurar_na <- function(df) {
  df <- df %>% mutate(num_vref_m2_200m = na_if(num_vref_m2_200m, 0))
  for (b in str_subset(names(df), "^bin_.+_na$")) {
    v <- paste0("num_", str_remove(str_remove(b, "^bin_"), "_na$"))
    if (v %in% names(df)) df[[v]][df[[b]] == 1] <- NA
  }
  df
}

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

fun_matriz <- function(tr, te, pred) {
  fml  <- reformulate(pred, intercept = FALSE)
  todo <- bind_rows(tr[pred], te[pred])
  x    <- model.matrix(fml, model.frame(fml, data = todo, na.action = na.pass))
  list(train = x[seq_len(n_train), , drop = FALSE], test = x[-seq_len(n_train), , drop = FALSE])
}
x_na <- fun_matriz(train_x, test_x, predictores)   # con NA nativos, para XGBoost

y_train <- datos$train$obj_log_price
precio  <- datos$train$obj_price
es_chap <- datos$train$cat_localidad == "CHAPINERO"

w <- datos$train %>% select(property_id) %>%
  left_join(readRDS(here("02_Data", "Processed", "pesos_adversarial_texto.rds")),
            by = "property_id") %>%
  pull(peso)
stopifnot(!anyNA(w))
peso_entreno <- w^0.5 * if_else(datos$train$cat_tipo == "Casa", 0.3, 1)
peso_entreno <- peso_entreno / mean(peso_entreno)

fun_area_offset <- function(base, conx) {
  if_else(coalesce(conx$bin_area_rescatada, 0L) == 1, conx$num_area, base$num_area)
}
area_tr <- fun_area_offset(datos$train, train_x)
area_te <- fun_area_offset(datos$test,  test_x)
lim_area <- quantile(area_tr, c(0.01, 0.99), na.rm = TRUE)
off_train <- log(pmin(pmax(area_tr, lim_area[1]), lim_area[2]))
off_test  <- log(pmin(pmax(area_te, lim_area[1]), lim_area[2]))
stopifnot(!anyNA(off_train), !anyNA(off_test))

fun_metricas <- function(pred_log) {
  ok  <- !is.na(pred_log)
  err <- recortar(exp(pred_log)) - precio
  tibble(mae        = mean(abs(err[ok])),
         mae_pond   = weighted.mean(abs(err[ok]), w[ok]),
         mae_chap   = mean(abs(err[ok & es_chap])),
         sesgo_chap = mean(err[ok & es_chap]))
}

fold_ale <- integer(n_train)
for (i in seq_along(f_ale)) fold_ale[f_ale[[i]]$val] <- i

# ------------------------------------------------------------------------------
# 2. Embeddings: lectura y PCA
# ------------------------------------------------------------------------------
message("Leyendo embeddings (", modelo_emb, ")")
emb <- read_csv(ruta_emb, show_col_types = FALSE)
cols_emb <- str_subset(names(emb), "^emb_")
fila_tr <- match(datos$train$property_id, emb$property_id)
fila_te <- match(datos$test$property_id,  emb$property_id)
if (anyNA(fila_tr) || anyNA(fila_te)) {
  stop("Hay inmuebles sin embedding: train ", sum(is.na(fila_tr)), ", test ", sum(is.na(fila_te)))
}
e_train <- as.matrix(emb[fila_tr, cols_emb])
e_test  <- as.matrix(emb[fila_te, cols_emb])
rm(emb); invisible(gc())
message("Embeddings: ", ncol(e_train), " dimensiones")

# PCA con train y test juntos (no usa el precio, así que no hay fuga).
pca <- prcomp(rbind(e_train, e_test), center = TRUE, scale. = FALSE, rank. = n_pca)
var_exp <- tibble(componente = seq_along(pca$sdev),
                  prop = pca$sdev^2 / sum(pca$sdev^2)) %>%
  mutate(acumulada = cumsum(prop))
write_csv(var_exp %>% filter(componente <= 100), file.path(ruta_tab, "emb_pca_varianza.csv"))
message("Varianza explicada por ", n_pca, " componentes: ",
        percent(var_exp$acumulada[n_pca], 0.1))

pc_train <- pca$x[seq_len(n_train), , drop = FALSE]
pc_test  <- pca$x[-seq_len(n_train), , drop = FALSE]
colnames(pc_train) <- colnames(pc_test) <- sprintf("num_emb_pc%02d", seq_len(n_pca))

p_var <- var_exp %>% filter(componente <= 60) %>%
  ggplot(aes(componente, prop)) +
  geom_col(aes(fill = componente <= n_pca), width = 0.8, show.legend = FALSE) +
  geom_line(aes(y = acumulada / 10), color = "#52514e") +
  scale_fill_manual(values = c(`TRUE` = "#2a78d6", `FALSE` = "#c3c2b7")) +
  scale_y_continuous(labels = label_percent(),
                     sec.axis = sec_axis(~ . * 10, labels = label_percent(), name = "Acumulada")) +
  labs(title = "PCA de los embeddings del texto",
       subtitle = paste0("Proporción de varianza por componente (barras azules: las ", n_pca,
                         " que entran al modelo); línea: acumulada"),
       x = "Componente", y = "Proporción de varianza") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"), panel.grid.minor = element_blank())
ggsave(file.path(ruta_fig, "emb_pca_varianza.png"), p_var, width = 8, height = 4.5, dpi = 300, bg = "white")
ggsave(file.path(ruta_fig, "emb_pca_varianza.pdf"), p_var, width = 8, height = 4.5)

# ------------------------------------------------------------------------------
# 3. Modelo solo-texto: ridge sobre el embedding completo (stacking)
# ------------------------------------------------------------------------------
# Objetivo: log(precio/m2), ponderado como los demás. Para train la predicción
# es fuera de muestra (folds aleatorios); para test, un ridge con todo train.
# La validación por localidades usa esta misma columna: es una aproximación
# (igual que el Lasso de 06_texto_como_datos.R).
ruta_ridge <- file.path(ruta_mod, "emb_ridge.rds")
if (file.exists(ruta_ridge)) {
  ridge <- readRDS(ruta_ridge)
} else {
  message("\n== Ridge solo-texto ==")
  y_m2 <- y_train - off_train
  oof_ridge <- rep(NA_real_, n_train)
  for (k in seq_along(f_ale)) {
    f <- f_ale[[k]]
    set.seed(semilla)
    m <- cv.glmnet(e_train[f$train, ], y_m2[f$train], weights = peso_entreno[f$train],
                   alpha = 0, nfolds = 5)
    oof_ridge[f$val] <- as.numeric(predict(m, e_train[f$val, ], s = "lambda.min"))
    message("  fold ", k, " listo")
  }
  set.seed(semilla)
  m_todo <- cv.glmnet(e_train, y_m2, weights = peso_entreno, alpha = 0, nfolds = 5)
  ridge <- list(oof = oof_ridge,
                test = as.numeric(predict(m_todo, e_test, s = "lambda.min")),
                lambda = m_todo$lambda.min)
  saveRDS(ridge, ruta_ridge)
}
message("Ridge solo-texto (solo el texto + el área): MAE CV aleatoria ",
        round(fun_metricas(ridge$oof + off_train)$mae / 1e6, 1), " M")

# ------------------------------------------------------------------------------
# 4. Miembro A: XGBoost con palabras + componentes + ridge (precio por m2)
# ------------------------------------------------------------------------------
# Mismo procedimiento del 23 / 28 para las palabras como columnas.
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
  list(train = m[seq_len(n_train), , drop = FALSE], test = m[-seq_len(n_train), , drop = FALSE])
}

fun_mejor_ronda <- function(modelo) {
  ronda <- attributes(modelo)$early_stop$best_iteration
  if (is.null(ronda)) ronda <- as.integer(xgb.attr(modelo, "best_iteration")) + 1
  as.integer(ronda)
}

fun_cv_xgb <- function(parametros, folds, sem, xtr, rondas = NULL, max_rondas = 3000) {
  pred <- rep(NA_real_, n_train)
  rondas_fold <- integer(length(folds))
  par <- modifyList(parametros, list(seed = sem))
  for (i in seq_along(folds)) {
    f <- folds[[i]]
    d_tr <- xgb.DMatrix(xtr[f$train, , drop = FALSE], label = y_train[f$train] - off_train[f$train],
                        weight = peso_entreno[f$train])
    d_va <- xgb.DMatrix(xtr[f$val, , drop = FALSE], label = y_train[f$val] - off_train[f$val],
                        weight = w[f$val])
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

fun_modelo_xgb <- function(nombre, parametros, xtr, xte) {
  ruta <- file.path(ruta_mod, paste0(nombre, ".rds"))
  if (file.exists(ruta)) { message("Cargando ", nombre); return(readRDS(ruta)) }
  message("\n== ", nombre, " (", ncol(xtr), " columnas) ==")
  cv1 <- fun_cv_xgb(parametros, f_ale, semillas[1], xtr)
  rondas <- round(mean(cv1$rondas))
  message("  rondas: ", rondas)
  oof_ale <- cv1$pred
  if (n_semillas > 1) {
    otras <- map(semillas[-1], ~ fun_cv_xgb(parametros, f_ale, .x, xtr, rondas = rondas)$pred)
    oof_ale <- rowMeans(cbind(oof_ale, do.call(cbind, otras)))
  }
  oof_loc <- fun_cv_xgb(parametros, f_loc, semillas[1], xtr, rondas = rondas)$pred
  d_todo <- xgb.DMatrix(xtr, label = y_train - off_train, weight = peso_entreno)
  d_test <- xgb.DMatrix(xte)
  preds <- map(semillas, function(s) {
    set.seed(s)
    m <- xgb.train(params = modifyList(parametros, list(seed = s)), data = d_todo,
                   nrounds = rondas, verbose = 0)
    list(pred = predict(m, d_test) + off_test, imp = xgb.importance(model = m))
  })
  res <- list(nombre = nombre, oof = list(aleatoria = oof_ale, localidad = oof_loc),
              pred_test = rowMeans(sapply(preds, `[[`, "pred")),
              importancia = preds[[1]]$imp, rondas = rondas)
  saveRDS(res, ruta)
  message("  CV aleatoria: MAE ", round(fun_metricas(oof_ale)$mae / 1e6, 1), " M, ponderado ",
          round(fun_metricas(oof_ale)$mae_pond / 1e6, 1), " M")
  res
}

pal <- fun_dtm(0.02)
xa_train <- cbind(x_na$train, pal$train, pc_train, num_emb_ridge = ridge$oof)
xa_test  <- cbind(x_na$test,  pal$test,  pc_test,  num_emb_ridge = ridge$test)
rm(pal); invisible(gc())

par_base <- list(objective = "reg:squarederror", eval_metric = "mae",
                 eta = 0.05, subsample = 0.8, colsample_bytree = 0.5,
                 max_depth = 6, min_child_weight = 1, nthread = nucleos)
res_xgb_emb <- fun_modelo_xgb("xgb_emb_prof6", par_base, xa_train, xa_test)
rm(xa_train, xa_test); invisible(gc())

imp_emb <- res_xgb_emb$importancia %>%
  mutate(grupo = case_when(str_detect(Feature, "^num_emb_pc") ~ "Componentes del embedding",
                           Feature == "num_emb_ridge" ~ "Ridge solo-texto",
                           str_detect(Feature, "^pal_") ~ "Palabras (columnas)",
                           TRUE ~ "Resto")) %>%
  group_by(grupo) %>% summarise(ganancia = sum(Gain), .groups = "drop")
print(imp_emb)

# ------------------------------------------------------------------------------
# 5. Miembro B: red neuronal con tabulares + embeddings
# ------------------------------------------------------------------------------
# Entradas: las variables del 20 con la imputación de 05_merge.R (la red no
# acepta NA), los 30 componentes y el ridge solo-texto, todo transformado a
# cuantil normal (ver 14_ensamble_reanuncios.R, sección 5). Objetivo:
# log(precio/m2) con offset. Una capa oculta de 16 neuronas y decay 1 (más
# entradas que la 24, así que más regularización).
fun_cuantil_normal <- function(a_mat, b_mat) {
  for (j in seq_len(ncol(a_mat))) {
    a <- a_mat[, j]
    if (length(unique(a)) <= 2) {
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

res_red <- NULL
if (correr_red) {
  ruta_red <- file.path(ruta_mod, "nnet_emb.rds")
  if (file.exists(ruta_red)) {
    res_red <- readRDS(ruta_red)
  } else {
    message("\n== nnet_emb ==")
    # Imputación para la red: la de 05_merge.R; lo que siga en NA (variables
    # nuevas del 20) se imputa con la mediana de train.
    x_imp <- fun_matriz(fun_agregar(datos$train), fun_agregar(datos$test), predictores)
    medianas <- apply(x_imp$train, 2, median, na.rm = TRUE)
    medianas[!is.finite(medianas)] <- 0
    for (j in seq_len(ncol(x_imp$train))) {
      x_imp$train[is.na(x_imp$train[, j]), j] <- medianas[j]
      x_imp$test[is.na(x_imp$test[, j]), j]   <- medianas[j]
    }
    constantes <- apply(x_imp$train, 2, function(v) length(unique(v)) <= 1)
    xr_train <- cbind(x_imp$train[, !constantes], pc_train, num_emb_ridge = ridge$oof)
    xr_test  <- cbind(x_imp$test[, !constantes],  pc_test,  num_emb_ridge = ridge$test)
    message("  entradas de la red: ", ncol(xr_train))

    fun_red <- function(filas, x_nuevo, off_nuevo) {
      z <- fun_cuantil_normal(xr_train[filas, , drop = FALSE], x_nuevo)
      set.seed(semilla)
      m <- nnet(z$a, y_train[filas] - off_train[filas], weights = peso_entreno[filas],
                size = 16, decay = 1, linout = TRUE, maxit = 400, MaxNWts = 1e5, trace = FALSE)
      as.numeric(predict(m, z$b)) + off_nuevo
    }
    fun_cv_red <- function(fl, etiqueta) {
      pred <- rep(NA_real_, n_train)
      for (i in seq_along(fl)) {
        f <- fl[[i]]
        pred[f$val] <- fun_red(f$train, xr_train[f$val, , drop = FALSE], off_train[f$val])
        message("  ", etiqueta, ": fold ", i, " listo")
      }
      pred
    }
    res_red <- list(nombre = "nnet_emb",
                    oof = list(aleatoria = fun_cv_red(f_ale, "aleatoria"),
                               localidad = fun_cv_red(f_loc, "localidad")),
                    pred_test = fun_red(seq_len(n_train), xr_test, off_test))
    saveRDS(res_red, ruta_red)
    rm(x_imp, xr_train, xr_test); invisible(gc())
  }
  message("Red con embeddings: MAE CV aleatoria ",
          round(fun_metricas(res_red$oof$aleatoria)$mae / 1e6, 1), " M, ponderado ",
          round(fun_metricas(res_red$oof$aleatoria)$mae_pond / 1e6, 1), " M; localidades ",
          round(fun_metricas(res_red$oof$localidad)$mae / 1e6, 1), " M")
}

# ------------------------------------------------------------------------------
# 6. Ensamble (mismo procedimiento del 28)
# ------------------------------------------------------------------------------
fun_leer <- function(archivo) {
  r <- readRDS(file.path(ruta_pred, archivo))
  p <- if ("pred_log" %in% names(r)) r$pred_log else log(r$pred)
  p[match(datos$train$property_id, r$property_id)]
}
miembros_28 <- list(
  `20_xgb_mejorado` = list(oof = list(aleatoria = fun_leer("oof_20_aleatoria.rds"),
                                      localidad = fun_leer("oof_20_localidad.rds")),
                           pred_test = readRDS(file.path(ruta_mod, "20_xgb_mejorado.rds"))$pred_test),
  `23_xgb_palabras_m2` = list(oof = list(aleatoria = fun_leer("oof_23_xgb_palabras_m2_aleatoria.rds"),
                                         localidad = fun_leer("oof_23_xgb_palabras_m2_localidad.rds")),
                              pred_test = readRDS(file.path(ruta_mod, "23_xgb_palabras_m2.rds"))$pred_test),
  xgb_completo_palabras_m2_prof6 = readRDS(file.path(ruta_mod, "xgb_completo_palabras_m2_prof6.rds")),
  xgb_completo_palabras_m2_prof8 = readRDS(file.path(ruta_mod, "xgb_completo_palabras_m2_prof8.rds")))
miembros_emb <- c(miembros_28, list(xgb_emb_prof6 = res_xgb_emb))
if (!is.null(res_red)) miembros_emb <- c(miembros_emb, list(nnet_emb = res_red))

# Duplicados, re-anuncios y factor: igual que el 28.
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
re_test <- fun_llaves(crudo_te, datos$test) %>%
  inner_join(ll_train %>% select(llave, x_ref = num_x, y_ref = num_y, precio_ref = obj_price),
             by = "llave", relationship = "many-to-many") %>%
  filter(sqrt((num_x - x_ref)^2 + (num_y - y_ref)^2) <= 3000) %>%
  group_by(property_id) %>%
  summarise(precio_reanuncio = median(precio_ref), .groups = "drop")
precio_re <- re_test$precio_reanuncio[match(datos$test$property_id, re_test$property_id)]
tiene_re  <- !is.na(precio_re)
message("Inmuebles de test con re-anuncio: ", sum(tiene_re))

fun_pesos <- function(P, obs, pond) {
  objetivo <- function(t) { a <- exp(t) / sum(exp(t)); weighted.mean(abs(P %*% a - obs), pond) }
  t <- optim(rep(0, ncol(P)), objetivo, method = "Nelder-Mead", control = list(maxit = 5000))$par
  setNames(exp(t) / sum(exp(t)), colnames(P))
}
fun_calibrar <- function(p, obs, pond) {
  m <- mean(p)
  objetivo <- function(th) weighted.mean(abs(recortar(exp(th[1] + m + th[2] * (p - m))) - obs), pond)
  th <- optim(c(0, 1), objetivo, method = "Nelder-Mead")$par
  list(a = th[1], b = th[2], m = m)
}
aplicar_cal <- function(p, cal) cal$a + cal$m + cal$b * (p - cal$m)
fun_factor_chap <- function(pred_log, grilla = seq(0.95, 1.20, by = 0.0025)) {
  p <- exp(pred_log)
  mae <- map_dbl(grilla, ~ mean(abs(recortar(.x * p[es_chap]) - precio[es_chap])))
  grilla[which.min(mae)]
}

# Todo el procedimiento del 28 para una lista de miembros.
fun_ensamblar <- function(miembros) {
  fun_mat <- function(esq) sapply(miembros, function(r) recortar(exp(r$oof[[esq]])))
  oof_ale <- fun_mat("aleatoria"); oof_loc <- fun_mat("localidad")
  test_mat <- sapply(miembros, function(r) recortar(exp(r$pred_test)))
  ok <- complete.cases(oof_ale)
  ens_cf <- rep(NA_real_, n_train)
  for (k in seq_along(f_ale)) {
    entrena <- ok & fold_ale != k
    a_k <- fun_pesos(oof_ale[entrena, ], precio[entrena], w[entrena])
    ens_cf[ok & fold_ale == k] <- oof_ale[ok & fold_ale == k, ] %*% a_k
  }
  pesos <- fun_pesos(oof_ale[ok, ], precio[ok], w[ok])
  ens <- list(aleatoria = fun_promediar_duplicados(log(ens_cf), desc_train),
              localidad = fun_promediar_duplicados(log(as.numeric(oof_loc %*% pesos)), desc_train),
              test      = fun_promediar_duplicados(log(as.numeric(test_mat %*% pesos)), desc_test))
  okc <- !is.na(ens$aleatoria)
  cal_cf <- rep(NA_real_, n_train)
  for (k in seq_along(f_ale)) {
    entrena <- okc & fold_ale != k
    cal_k <- fun_calibrar(ens$aleatoria[entrena], precio[entrena], w[entrena])
    cal_cf[okc & fold_ale == k] <- aplicar_cal(ens$aleatoria[okc & fold_ale == k], cal_k)
  }
  cal <- fun_calibrar(ens$aleatoria[okc], precio[okc], w[okc])
  ens_cal <- list(aleatoria = cal_cf, localidad = aplicar_cal(ens$localidad, cal),
                  test = aplicar_cal(ens$test, cal))
  f_a <- fun_factor_chap(ens_cal$aleatoria); f_l <- fun_factor_chap(ens_cal$localidad)
  factor_nivel <- sqrt(f_a * f_l)
  list(pesos = pesos, cal = cal, ens = ens, ens_cal = ens_cal,
       factor = tibble(factor_aleatoria = f_a, factor_localidad = f_l, factor_regla = factor_nivel),
       pred_final = if_else(tiene_re, log(precio_re), ens_cal$test + log(factor_nivel)),
       oof_final = map(ens_cal[c("aleatoria", "localidad")], ~ .x + log(factor_nivel)))
}

message("\nEnsamble sin embeddings (recalcula el 28 como control)")
e28 <- fun_ensamblar(miembros_28)
message("Ensamble con embeddings")
e29 <- fun_ensamblar(miembros_emb)
print(round(e29$pesos, 3))
write_csv(enframe(e29$pesos, "modelo", "peso"), file.path(ruta_tab, "emb_pesos.csv"))
write_csv(bind_rows(e28$factor %>% mutate(ensamble = "28 (control)", .before = 1),
                    e29$factor %>% mutate(ensamble = "29 (con embeddings)", .before = 1)),
          file.path(ruta_tab, "emb_factor.csv"))

# ------------------------------------------------------------------------------
# 7. Envíos, registro y comparación
# ------------------------------------------------------------------------------
fun_submission(datos$test$property_id, e29$pred_final, "29_ensamble_embeddings",
               limites = limites_precio)
iwalk(e29$oof_final, ~ saveRDS(tibble(property_id = datos$train$property_id, pred_log = .x),
                               file.path(ruta_pred, paste0("oof_29_ensamble_embeddings_", .y, ".rds"))))
fun_registrar_resultado(
  "29_ensamble_embeddings", "Ensamble",
  paste0(paste(names(e29$pesos), round(e29$pesos, 3), sep = " = ", collapse = ", "),
         "; embeddings = ", modelo_emb, ", PCA = ", n_pca,
         "; cal b = ", round(e29$cal$b, 4), "; factor = ", round(e29$factor$factor_regla, 4)),
  imap_dfr(e29$oof_final, ~ tibble(esquema = .y, mae_cop = fun_metricas(.x)$mae)),
  "29_ensamble_embeddings")

if (!is.null(res_red)) {
  fun_submission(datos$test$property_id, res_red$pred_test, "30_nnet_embeddings",
                 limites = limites_precio)
  iwalk(res_red$oof, ~ saveRDS(tibble(property_id = datos$train$property_id, pred_log = .x),
                               file.path(ruta_pred, paste0("oof_30_nnet_embeddings_", .y, ".rds"))))
  fun_registrar_resultado(
    "30_nnet_embeddings", "Red neuronal",
    paste0("nnet 1 capa (16), decay 1, entradas cuantil normal: variables del 20 + ",
           n_pca, " PC del embedding (", modelo_emb, ") + ridge solo-texto; log(precio/m2)"),
    imap_dfr(res_red$oof, ~ tibble(esquema = .y, mae_cop = fun_metricas(.x)$mae)),
    "30_nnet_embeddings")
}

fila <- function(nombre, oof_a, oof_l) {
  bind_cols(tibble(modelo = nombre), fun_metricas(oof_a), tibble(mae_loc = fun_metricas(oof_l)$mae))
}
comparacion <- bind_rows(
  imap_dfr(miembros_emb, ~ fila(.y, .x$oof$aleatoria, .x$oof$localidad)),
  fila("ridge solo-texto", ridge$oof + off_train, ridge$oof + off_train),
  fila("ensamble 28 calibrado (control, sin factor)", e28$ens_cal$aleatoria, e28$ens_cal$localidad),
  fila("ensamble 29 calibrado (sin factor)", e29$ens_cal$aleatoria, e29$ens_cal$localidad)
)
write_csv(comparacion, file.path(ruta_tab, "emb_comparacion.csv"))
print(comparacion %>% mutate(across(where(is.double), ~ round(.x / 1e6, 1))), width = Inf)

message("\nCómo leerlo: compara mae_pond del 'ensamble 29' contra el 'ensamble 28 (control)'.",
        "\nSi el 29 no baja al menos ~0,5 M, no vale la pena gastar un envío en él;",
        " el 30 sí vale la pena subirlo como puntaje de la familia de redes.")
