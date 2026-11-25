# ==============================================================================
# 02_lm.R
# Objetivo: regresión lineal (MCO) del log del precio con todos los
#           predictores de la base completa. No tiene hiperparámetros; se
#           reporta su MAE en pesos con los cuatro esquemas de validación.
#           Usa la base completa con la que se subió el envío: deja por
#           fuera las variables que se agregaron después (fun_vars_excluidas
#           en 00_utils.R, versión "completa").
# Entradas: 02_Data/Processed/db_{train,test}.rds
# Salidas:  03_Output/Submissions/05_lm_completa.csv
#           03_Output/Models/05_lm_completa.rds
#           03_Output/Models/resultados_modelos.csv (fila del modelo)
# ==============================================================================

source(here::here("01_Scripts", "00_utils.R"))

resultado <- fun_correr_caret(envio = "05_lm_completa",
                              version = "completa",
                              familia = "Regresión lineal",
                              metodo = "lm")
