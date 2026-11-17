# ==============================================================================
# 03_enet.R
# Objetivo: Elastic Net del log del precio con todos los predictores de la
#           base completa, estandarizados. Se afinan la mezcla entre Ridge y
#           Lasso (alpha) y la penalización (lambda).
# Entradas: 02_Data/Processed/db_{train,test}.rds
# Salidas:  03_Output/Submissions/06_enet_completa.csv
#           03_Output/Models/06_enet_completa.rds
#           03_Output/Models/resultados_modelos.csv (fila del modelo)
# ==============================================================================

source(here::here("01_Scripts", "00_utils.R"))
pacman::p_load(glmnet)

grid <- expand.grid(alpha  = c(0, 0.25, 0.5, 0.75, 1),
                    lambda = 10^seq(-5, -1, length.out = 12))

resultado <- fun_correr_caret(envio = "06_enet_completa",
                              familia = "Elastic Net",
                              metodo = "glmnet",
                              grid = grid,
                              preProcess = c("center", "scale"))
