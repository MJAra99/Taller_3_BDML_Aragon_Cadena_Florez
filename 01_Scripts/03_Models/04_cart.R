# ==============================================================================
# 04_cart.R
# Objetivo: árbol de regresión (CART) del log del precio con todos los
#           predictores de la base completa. Se afina el parámetro de
#           complejidad (cp), que controla el tamaño del árbol.
# Entradas: 02_Data/Processed/db_{train,test}.rds
# Salidas:  03_Output/Submissions/07_cart_completa.csv
#           03_Output/Models/07_cart_completa.rds
#           03_Output/Models/resultados_modelos.csv (fila del modelo)
# ==============================================================================

source(here::here("01_Scripts", "00_utils.R"))
pacman::p_load(rpart)

grid <- expand.grid(cp = 10^seq(-6, -2, length.out = 12))

resultado <- fun_correr_caret(envio = "07_cart_completa",
                              familia = "CART",
                              metodo = "rpart",
                              grid = grid)
