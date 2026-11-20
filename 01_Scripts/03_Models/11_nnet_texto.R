# ==============================================================================
# 11_nnet_texto.R
# Objetivo: red neuronal de una capa oculta (perceptrón multicapa, paquete nnet)
#           del log del precio con todos los predictores de la base final,
#           incluidas las variables de texto como datos (predicciones del Lasso
#           sobre las palabras y tópicos LDA).
#           - Los predictores se estandarizan con la media y la desviación
#             estándar de train (las redes se estiman por gradiente y son
#             sensibles a la escala) y las variables categóricas entran como
#             dummies.
#           - La capa oculta usa activación logística y la salida es lineal
#             (linout = TRUE), porque el objetivo es continuo.
#           - Se afinan el número de neuronas de la capa oculta (size) y la
#             penalización de los pesos (decay), que regulariza la red como una
#             penalización L2. Con penalizaciones bajas (decay de 0,1 o menos)
#             algunas predicciones de localidades no vistas en el
#             entrenamiento resultaron explosivas, por eso la grilla usa
#             penalizaciones de 1 y 3.
#           - Los hiperparámetros se escogen con el MAE en pesos de la
#             validación cruzada con localidades completas por fuera, igual que
#             en los demás modelos.
# Entradas: 02_Data/Processed/db_{train,test}.rds
# Salidas:  03_Output/Submissions/17_nnet_texto.csv
#           03_Output/Models/17_nnet_texto.rds
#           03_Output/Models/resultados_modelos.csv (fila del modelo)
# ==============================================================================

source(here::here("01_Scripts", "00_utils.R"))
pacman::p_load(nnet)

grid <- expand.grid(size  = c(3, 5, 8),
                    decay = c(1, 3))

# maxit: máximo de iteraciones del optimizador (BFGS). MaxNWts: máximo de
# pesos permitido; con unas 150 variables y 8 neuronas la red tiene cerca de
# 1.200 pesos, más que el límite por defecto de nnet (1.000).
resultado <- fun_correr_caret(envio = "17_nnet_texto",
                              familia = "Red neuronal",
                              metodo = "nnet",
                              grid = grid,
                              preProcess = c("center", "scale"),
                              linout = TRUE,
                              trace = FALSE,
                              maxit = 300,
                              MaxNWts = 1e5)
