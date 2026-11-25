# ==============================================================================
# 05_rf.R
# Objetivo: Random Forest del log del precio con todos los predictores de la
#           base completa (ranger, 200 árboles). Se afinan el número de
#           variables candidatas en cada división (mtry) y el tamaño mínimo
#           de las hojas (min.node.size).
#           Usa la base completa con la que se subió el envío: deja por
#           fuera las variables que se agregaron después (fun_vars_excluidas
#           en 00_utils.R, versión "completa").
# Entradas: 02_Data/Processed/db_{train,test}.rds
# Salidas:  03_Output/Submissions/08_rf_completa.csv
#           03_Output/Models/08_rf_completa.rds
#           03_Output/Models/resultados_modelos.csv (fila del modelo)
# ==============================================================================

source(here::here("01_Scripts", "00_utils.R"))
pacman::p_load(ranger)

grid <- expand.grid(mtry = c(20, 40),
                    splitrule = "variance",
                    min.node.size = c(5, 20))

resultado <- fun_correr_caret(envio = "08_rf_completa",
                              version = "completa",
                              familia = "Random Forest",
                              metodo = "ranger",
                              grid = grid,
                              num.trees = 200)
