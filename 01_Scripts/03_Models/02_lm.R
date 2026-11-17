# ==============================================================================
# 02_lm.R
# Objetivo: regresión lineal (MCO) del log del precio con todos los
#           predictores de la base completa. No tiene hiperparámetros; se
#           reporta su MAE en pesos con los cuatro esquemas de validación.
# Entradas: 02_Data/Processed/db_{train,test}.rds
# Salidas:  03_Output/Submissions/05_lm_completa.csv
#           03_Output/Models/05_lm_completa.rds
#           03_Output/Models/resultados_modelos.csv (fila del modelo)
# ==============================================================================

source(here::here("01_Scripts", "00_utils.R"))

resultado <- fun_correr_caret(envio = "05_lm_completa",
                              familia = "Regresión lineal",
                              metodo = "lm")
