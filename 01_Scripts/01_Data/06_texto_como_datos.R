# ==============================================================================
# 06_texto_como_datos.R
# Objetivo: agregar a la base final predictores construidos con el texto como
#           datos (bolsa de palabras) a partir del título y la descripción de
#           cada anuncio:
#           1. Matriz documento-término (DTM): texto en minúsculas, sin números
#              ni signos, sin palabras vacías del español y con raíces del
#              stemmer Snowball en español. Los términos son palabras y bigramas
#              de raíces que aparecen en al menos 0,5% de los anuncios.
#           2. Regresión Lasso sobre las palabras (como en Gentzkow y Shapiro,
#              2010): Lasso del log del precio y del log del precio por m2
#              sobre los indicadores de uso de cada término, con lambda escogido
#              por validación cruzada (gamlr::cv.gamlr). Su predicción entra
#              como predictor: el precio que sugiere el texto. En train la
#              predicción es fuera de muestra: cada anuncio recibe la predicción
#              de un Lasso estimado sin su grupo de localidades (los mismos 5
#              folds de localidades completas por fuera que se usan para
#              escoger los modelos), para que la variable no contenga el precio
#              del propio anuncio. En test se usa el Lasso estimado con todo
#              train.
#           3. Tópicos con Latent Dirichlet Allocation (maptpx::topics), con el
#              número de tópicos escogido por el Bayes Factor. El modelo se
#              estima con una muestra aleatoria de 5.000 anuncios de train y
#              test, porque con todos los anuncios cada valor de K tarda más de
#              10 minutos; después se calculan las proporciones de cada tópico
#              para todos los anuncios con el modelo estimado. Esas
#              proporciones entran como predictores. Los tópicos no usan el
#              precio.
#           Si el script se corre de nuevo, reemplaza las columnas num_txt_*
#           que ya existan en la base.
# Entradas: 02_Data/Processed/db_{train,test}.rds
#           02_Data/Processed/base_limpia.rds (título y descripción)
# Salidas:  02_Data/Processed/db_{train,test}.rds con las columnas num_txt_*
#           03_Output/Tables/texto_lasso_terminos.csv, texto_topicos_terminos.csv,
#           texto_topicos_bayes_factor.csv
#           03_Output/Figures/texto_lasso_terminos.png
# ==============================================================================

source(here::here("01_Scripts", "00_utils.R"))
pacman::p_load(stringi, tidytext, SnowballC, stopwords, Matrix, gamlr, maptpx, slam)

ruta_proc    <- here("02_Data", "Processed")
ruta_tablas  <- here("03_Output", "Tables")
ruta_figuras <- here("03_Output", "Figures")
walk(c(ruta_tablas, ruta_figuras), dir.create, showWarnings = FALSE, recursive = TRUE)

db_train <- readRDS(file.path(ruta_proc, "db_train.rds")) %>% select(-starts_with("num_txt_"))
db_test  <- readRDS(file.path(ruta_proc, "db_test.rds"))  %>% select(-starts_with("num_txt_"))

textos <- readRDS(file.path(ruta_proc, "base_limpia.rds")) %>%
  distinct(property_id, .keep_all = TRUE) %>%
  transmute(property_id, texto = paste(replace_na(title, ""), replace_na(description, "")))

# Documentos en el orden: primero train, luego test.
docs <- bind_rows(db_train %>% select(property_id), db_test %>% select(property_id)) %>%
  left_join(textos, by = "property_id") %>%
  mutate(doc = row_number(), texto = replace_na(texto, ""))
n_train <- nrow(db_train)

# ------------------------------------------------------------------------------
# 1. Matriz documento-término
# ------------------------------------------------------------------------------
# Limpieza: minúsculas, sin tildes, sin números ni signos.
docs <- docs %>%
  mutate(texto = texto %>%
           str_to_lower() %>%
           stri_trans_general("Latin-ASCII") %>%
           str_replace_all("[^a-z ]", " ") %>%
           str_squish())

# Palabras vacías del español sin tildes, más palabras de una o dos letras.
vacias <- stri_trans_general(stopwords::stopwords("es", source = "snowball"), "Latin-ASCII")

# Palabras en orden dentro de cada anuncio, sin palabras vacías y con raíces.
palabras <- docs %>%
  select(doc, texto) %>%
  unnest_tokens(palabra, texto, token = "words") %>%
  filter(!palabra %in% vacias, nchar(palabra) > 2) %>%
  mutate(raiz = wordStem(palabra, language = "spanish"))

# Bigramas: pares de raíces consecutivas dentro del mismo anuncio, después de
# quitar las palabras vacías (por ejemplo "zon verd" por "zonas verdes").
bigramas <- palabras %>%
  group_by(doc) %>%
  mutate(siguiente = lead(raiz)) %>%
  ungroup() %>%
  filter(!is.na(siguiente)) %>%
  transmute(doc, termino = paste(raiz, siguiente))

terminos <- bind_rows(palabras %>% transmute(doc, termino = raiz), bigramas) %>%
  count(doc, termino, name = "conteo")

# Vocabulario: términos presentes en al menos 0,5% de los anuncios.
minimo_docs <- ceiling(0.005 * nrow(docs))
vocabulario <- terminos %>%
  count(termino, name = "n_docs") %>%
  filter(n_docs >= minimo_docs) %>%
  arrange(desc(n_docs)) %>%
  pull(termino)

terminos <- terminos %>% filter(termino %in% vocabulario)

dtm <- sparseMatrix(i = terminos$doc,
                    j = match(terminos$termino, vocabulario),
                    x = terminos$conteo,
                    dims = c(nrow(docs), length(vocabulario)),
                    dimnames = list(docs$property_id, vocabulario))

message("DTM: ", nrow(dtm), " anuncios x ", ncol(dtm), " términos; ",
        sum(rowSums(dtm) == 0), " anuncios sin ningún término del vocabulario")

# Indicadores de uso de cada término (1 si el anuncio lo usa).
dtm_uso <- (dtm > 0) * 1

x_train <- dtm_uso[seq_len(n_train), , drop = FALSE]
x_test  <- dtm_uso[-seq_len(n_train), , drop = FALSE]

# ------------------------------------------------------------------------------
# 2. Lasso del precio sobre las palabras
# ------------------------------------------------------------------------------
folds_localidad <- fun_folds_cv(db_train, "localidad")

# Predicción fuera de muestra en train (por folds) y predicción en test con el
# Lasso estimado en las filas `usar` de train. `y` tiene una entrada por fila
# de train; solo se usan las filas `usar` para estimar.
fun_lasso_texto <- function(y, usar = rep(TRUE, n_train)) {
  pred_train <- rep(NA_real_, n_train)
  for (f in folds_localidad) {
    entrena <- intersect(f$train, which(usar))
    set.seed(semilla)
    modelo <- cv.gamlr(x_train[entrena, , drop = FALSE], y[entrena])
    pred_train[f$val] <- as.numeric(predict(modelo, x_train[f$val, , drop = FALSE],
                                            select = "min"))
  }
  set.seed(semilla)
  completo <- cv.gamlr(x_train[usar, , drop = FALSE], y[usar])
  list(pred_train = pred_train,
       pred_test  = as.numeric(predict(completo, x_test, select = "min")),
       coef       = coef(completo, select = "min")[-1, ])
}

message("Lasso del log del precio sobre las palabras")
lasso_precio <- fun_lasso_texto(db_train$obj_log_price)

# Para el precio por m2 se usan solo los anuncios con área observada.
message("Lasso del log del precio por m2 sobre las palabras")
lasso_precio_m2 <- fun_lasso_texto(log(db_train$obj_price / db_train$num_area),
                                   usar = db_train$bin_area_na == 0)

db_train <- db_train %>%
  mutate(num_txt_lasso_log_precio    = lasso_precio$pred_train,
         num_txt_lasso_log_precio_m2 = lasso_precio_m2$pred_train)
db_test <- db_test %>%
  mutate(num_txt_lasso_log_precio    = lasso_precio$pred_test,
         num_txt_lasso_log_precio_m2 = lasso_precio_m2$pred_test)

# Calidad de las predicciones fuera de muestra en train.
db_train %>%
  summarise(cor_log_precio = cor(num_txt_lasso_log_precio, obj_log_price),
            mae_texto_millones = mean(abs(exp(num_txt_lasso_log_precio) - obj_price)) / 1e6,
            cor_log_precio_m2 = cor(num_txt_lasso_log_precio_m2[bin_area_na == 0],
                                    log(obj_price / num_area)[bin_area_na == 0])) %>%
  print()

# Términos con mayor coeficiente positivo y negativo en el Lasso del precio
# por m2 (estimado con todo train).
coeficientes <- tibble(termino = names(lasso_precio_m2$coef),
                       coef_log_precio_m2 = as.numeric(lasso_precio_m2$coef)) %>%
  left_join(tibble(termino = names(lasso_precio$coef),
                   coef_log_precio = as.numeric(lasso_precio$coef)), by = "termino") %>%
  filter(coef_log_precio_m2 != 0 | coef_log_precio != 0) %>%
  arrange(desc(coef_log_precio_m2))
write_csv(coeficientes, file.path(ruta_tablas, "texto_lasso_terminos.csv"))

g_terminos <- bind_rows(slice_max(coeficientes, coef_log_precio_m2, n = 15),
                        slice_min(coeficientes, coef_log_precio_m2, n = 15)) %>%
  mutate(termino = fct_reorder(termino, coef_log_precio_m2),
         signo = if_else(coef_log_precio_m2 > 0, "Sube el precio por m2", "Baja el precio por m2")) %>%
  ggplot(aes(x = coef_log_precio_m2, y = termino, fill = signo)) +
  geom_col() +
  scale_fill_manual(values = c("Sube el precio por m2" = "#0072B2",
                               "Baja el precio por m2" = "#D55E00")) +
  labs(title = "Términos con mayor efecto en el Lasso del log del precio por m2",
       x = "Coeficiente", y = NULL, fill = NULL) +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave(file.path(ruta_figuras, "texto_lasso_terminos.png"), g_terminos,
       width = 8, height = 7, dpi = 300, bg = "white")

# ------------------------------------------------------------------------------
# 3. Tópicos con LDA
# Se estiman con los conteos de una muestra de 5.000 anuncios que tienen al
# menos un término del vocabulario. Las proporciones de los tópicos de todos
# los anuncios se calculan con el modelo estimado (predict de maptpx). Los
# anuncios sin términos reciben la proporción promedio de cada tópico.
# ------------------------------------------------------------------------------
con_texto <- which(rowSums(dtm) > 0)
set.seed(semilla)
muestra_lda <- sample(con_texto, min(5000, length(con_texto)))

topicos <- topics(as.simple_triplet_matrix(dtm[muestra_lda, , drop = FALSE]),
                  K = c(5, 10, 15, 20), verb = 1)
k_topicos <- ncol(topicos$omega)
message("Número de tópicos escogido por el Bayes Factor: ", k_topicos)

write_csv(tibble(K = as.integer(names(topicos$BF)), log_bayes_factor = as.numeric(topicos$BF)),
          file.path(ruta_tablas, "texto_topicos_bayes_factor.csv"))

# Términos de cada tópico ordenados por lift: probabilidad del término en el
# tópico dividida por su frecuencia promedio en el corpus.
frecuencia <- colSums(dtm[muestra_lda, , drop = FALSE]) / sum(dtm[muestra_lda, , drop = FALSE])
lift <- topicos$theta / frecuencia
terminos_topicos <- map_dfr(seq_len(k_topicos), function(k) {
  orden <- order(lift[, k], decreasing = TRUE)[1:15]
  tibble(topico = k, termino = rownames(lift)[orden], lift = lift[orden, k],
         prob_en_topico = topicos$theta[orden, k])
})
write_csv(terminos_topicos, file.path(ruta_tablas, "texto_topicos_terminos.csv"))
terminos_topicos %>%
  group_by(topico) %>%
  summarise(terminos = paste(head(termino, 8), collapse = ", ")) %>%
  print(n = Inf, width = Inf)

proporciones <- matrix(rep(colMeans(topicos$omega), each = nrow(docs)), nrow = nrow(docs))
proporciones[con_texto, ] <- predict(topicos, as.simple_triplet_matrix(dtm[con_texto, , drop = FALSE]))
colnames(proporciones) <- sprintf("num_txt_topico_%02d", seq_len(k_topicos))

db_train <- bind_cols(db_train, as_tibble(proporciones[seq_len(n_train), , drop = FALSE]))
db_test  <- bind_cols(db_test,  as_tibble(proporciones[-seq_len(n_train), , drop = FALSE]))

# ------------------------------------------------------------------------------
# 4. Guardar la base
# ------------------------------------------------------------------------------
stopifnot(!anyNA(select(db_train, starts_with("num_txt_"))),
          !anyNA(select(db_test, starts_with("num_txt_"))))

saveRDS(db_train, file.path(ruta_proc, "db_train.rds"))
saveRDS(db_test,  file.path(ruta_proc, "db_test.rds"))

message("Base guardada con ", sum(str_detect(names(db_train), "^num_txt_")),
        " columnas de texto como datos: ", ncol(db_train), " columnas en train y ",
        ncol(db_test), " en test")
