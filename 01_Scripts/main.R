# ==============================================================================
# 01_Scripts/main.R
# Archivo principal del Problem Set 3 (precios de vivienda en Chapinero).
# Corre todos los scripts del repositorio en el orden en que dependen unos de
# otros y regenera las bases procesadas, los modelos, los envíos a Kaggle y
# todas las tablas y figuras de las presentaciones.
#
# Uso: abrir Taller_3_BDML_Aragon_Cadena_Florez.Rproj y correr
#        source("01_Scripts/main.R")
#
# Cada script corre en un proceso de R nuevo (Rscript), para que la memoria se
# libere entre pasos y ningún script dependa de objetos que dejó otro. Si un
# script falla, la ejecución se detiene e indica cuál fue.
#
# Tiempo aproximado de la ejecución completa: entre 7 y 9 horas en un
# computador personal. Los pasos más largos son las variables espaciales
# (~1 h), el Random Forest (~1,5 h), los XGBoost con pesos (~40 min cada uno),
# la red neuronal (~1 h) y los ensambles (~1-2 h).
# ==============================================================================

# ------------------------------------------------------------------------------
# Opciones de ejecución
# ------------------------------------------------------------------------------
# Descargas de internet (Mapas Bogotá, datos abiertos y elevación). Las capas
# geográficas de Mapas Bogotá no se guardan en el repositorio por su tamaño:
# si no están en 02_Data/Raw/MapasBogota/capas se descargan aunque esta opción
# sea FALSE. Con TRUE se vuelven a descargar todas las fuentes externas.
correr_descargas <- FALSE

# Embeddings de texto en Python (01_Data/07_embeddings_texto.py, requiere
# Python 3 con sentence-transformers; ver requirements.txt). Con FALSE se usa
# 02_Data/Processed/embeddings_texto_minilm.csv.gz, que está en el repositorio.
correr_python <- FALSE

# ------------------------------------------------------------------------------
# Preparación
# ------------------------------------------------------------------------------
if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages("pacman", repos = "https://cloud.r-project.org")
}
pacman::p_load(here)

raiz <- here::here()
rscript <- file.path(R.home("bin"), "Rscript")
inicio_total <- Sys.time()
registro <- data.frame(script = character(), minutos = numeric())

# Corre un script de R en un proceso nuevo desde la raíz del repositorio.
correr <- function(script) {
  ruta <- file.path(raiz, script)
  if (!file.exists(ruta)) stop("No existe el script ", script)
  message("\n", strrep("=", 78), "\n", format(Sys.time(), "%H:%M"), "  ", script,
          "\n", strrep("=", 78))
  inicio <- Sys.time()
  estado <- system2(rscript, shQuote(ruta), wait = TRUE)
  if (!identical(as.integer(estado), 0L)) {
    stop("El script ", script, " terminó con error (código ", estado, "). ",
         "Revise la salida de arriba y vuelva a correr 01_Scripts/main.R.")
  }
  minutos <- round(as.numeric(difftime(Sys.time(), inicio, units = "mins")), 1)
  registro[nrow(registro) + 1, ] <<- list(script, minutos)
  message("Terminó en ", minutos, " minutos")
}

# Corre el script de Python de los embeddings con el intérprete disponible.
correr_python_script <- function(script) {
  python <- Sys.which(c("python", "python3", "py"))
  python <- python[python != ""][1]
  if (is.na(python)) stop("No se encontró Python. Instálelo o use correr_python <- FALSE.")
  message("\n", strrep("=", 78), "\n", format(Sys.time(), "%H:%M"), "  ", script,
          "\n", strrep("=", 78))
  inicio <- Sys.time()
  estado <- system2(python, shQuote(file.path(raiz, script)), wait = TRUE)
  if (!identical(as.integer(estado), 0L)) stop("El script ", script, " terminó con error.")
  registro[nrow(registro) + 1, ] <<- list(script,
    round(as.numeric(difftime(Sys.time(), inicio, units = "mins")), 1))
}

# Carpetas de datos procesados y de resultados. Se crean si no existen, para
# que el proyecto corra desde cero aunque se borren 02_Data/Processed y
# 03_Output: todo lo que hay en ellas lo generan los scripts.
for (carpeta in c("02_Data/Processed", "02_Data/Raw/MapasBogota/capas",
                  "03_Output/Figures", "03_Output/Tables", "03_Output/Models",
                  "03_Output/Predictions", "03_Output/Submissions")) {
  dir.create(file.path(raiz, carpeta), recursive = TRUE, showWarnings = FALSE)
}

capas_necesarias <- file.path(raiz, "02_Data", "Raw", "MapasBogota", "capas",
                              c("01_valor_referencia.geojson", "44_localidades.geojson",
                                "45_upz.geojson"))
faltan_capas <- !all(file.exists(capas_necesarias))
faltan_embeddings <- !file.exists(file.path(raiz, "02_Data", "Processed",
                                            "embeddings_texto_minilm.csv.gz"))

# ------------------------------------------------------------------------------
# 0. Fuentes externas
# ------------------------------------------------------------------------------
# Inventario de las capas de Mapas Bogotá y de datos abiertos (de donde salió
# la selección de 02_Data/Raw/MapasBogota/capas_seleccionadas.csv).
if (correr_descargas) correr("01_Scripts/00_Fuentes_externas/01_inventario_mapas_bogota.R")
# Descarga de las capas seleccionadas.
if (correr_descargas || faltan_capas) correr("01_Scripts/00_Fuentes_externas/02_descarga_capas.R")

# ------------------------------------------------------------------------------
# 1. Construcción de la base
# ------------------------------------------------------------------------------
correr("01_Scripts/01_Data/01_clean_base.R")          # limpieza de train y test
correr("01_Scripts/01_Data/02_text_features.R")       # variables extraídas del texto
correr("01_Scripts/01_Data/03_spatial_features.R")    # variables de Mapas Bogotá
if (correr_descargas || !file.exists(file.path(raiz, "02_Data", "Processed", "elevacion.rds"))) {
  correr("01_Scripts/01_Data/04_elevacion.R")         # altitud (descarga de AWS)
}
correr("01_Scripts/01_Data/05_merge.R")               # base final e imputación
correr("01_Scripts/01_Data/06_texto_como_datos.R")    # Lasso sobre palabras y LDA
if (correr_python || faltan_embeddings) {
  correr_python_script("01_Scripts/01_Data/07_embeddings_texto.py")  # embeddings
}

# ------------------------------------------------------------------------------
# 2. Modelos, exploración y diagnóstico
# Los scripts de exploración y diagnóstico usan modelos ya estimados, por eso
# se intercalan con los de modelos en el orden en que se necesitan.
# ------------------------------------------------------------------------------
correr("01_Scripts/02_EDA/01_chapinero_vs_bogota.R")  # Chapinero frente a Bogotá
correr("01_Scripts/03_Models/01_baseline.R")          # envíos 01 a 04
correr("01_Scripts/03_Models/02_lm.R")                # envío 05
correr("01_Scripts/03_Models/03_enet.R")              # envío 06
correr("01_Scripts/03_Models/04_cart.R")              # envío 07
correr("01_Scripts/03_Models/05_rf.R")                # envío 08
correr("01_Scripts/03_Models/06_xgboost.R")           # envío 09
correr("01_Scripts/02_EDA/02_errores_xgb.R")          # errores del envío 09
correr("01_Scripts/03_Models/07_xgb_pesos.R")         # envíos 10 y 11
correr("01_Scripts/03_Models/08_xgb_base_final.R")    # envío 12
correr("01_Scripts/03_Models/09_xgb_pesos_base_final.R")  # envíos 13 y 14
correr("01_Scripts/03_Models/10_xgb_pesos_texto.R")   # envíos 15 y 16
correr("01_Scripts/03_Models/11_nnet_texto.R")        # envío 17
correr("01_Scripts/03_Models/12_ajuste_sesgo.R")      # envíos 18 y 19
correr("01_Scripts/02_EDA/03_diagnostico_errores.R")  # diagnóstico del envío 16
correr("01_Scripts/03_Models/13_xgb_mejorado.R")      # envíos 20 y 21
correr("01_Scripts/03_Models/14_ensamble_reanuncios.R")  # envíos 22 a 27
correr("01_Scripts/03_Models/15_ensamble_v2.R")       # envío 28
correr("01_Scripts/03_Models/16_embeddings_red.R")    # envíos 29 y 30

# ------------------------------------------------------------------------------
# 3. Resultados para las presentaciones
# ------------------------------------------------------------------------------
correr("01_Scripts/04_Results/01_mapas.R")            # mapas de Bogotá
correr("01_Scripts/04_Results/02_negocio.R")          # decisión de compra

# ------------------------------------------------------------------------------
# Resumen
# ------------------------------------------------------------------------------
message("\nEjecución completa en ",
        round(as.numeric(difftime(Sys.time(), inicio_total, units = "hours")), 2), " horas")
print(registro, row.names = FALSE)
