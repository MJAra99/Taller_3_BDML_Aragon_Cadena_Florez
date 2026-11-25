# Problem Set 3: Predicción de precios de vivienda en Chapinero

**Big Data y Machine Learning para Economía Aplicada (MECA 4107), Universidad de los Andes, 2026-20**
**Equipo 1:** Mauricio Aragón, Jonathan Cadena y Esteban Flórez

## Objetivo

Una empresa inmobiliaria quiere comprar vivienda en la localidad de Chapinero (Bogotá) usando un modelo que prediga el precio de venta. El reto es que los datos de entrenamiento (anuncios de Properati entre 2019 y 2021) cubren casi toda Bogotá, pero solo 305 de los 36.407 anuncios de train están en Chapinero, mientras que el 94 % de los 10.286 anuncios de test están allí.

El repositorio:

1. construye una base con variables de la vivienda, del texto del anuncio y del entorno (capas de Mapas Bogotá, elevación y el precio de los vecinos);
2. estima seis familias de modelos (regresión lineal, Elastic Net, CART, Random Forest, XGBoost y redes neuronales), validadas con validación cruzada estándar y espacial;
3. genera los envíos a la competencia de Kaggle [uniandes-bdml-202620-ps-3](https://www.kaggle.com/competitions/uniandes-bdml-202620-ps-3), que se evalúa con el error absoluto medio (MAE, Mean Absolute Error) en pesos. Los scripts generan 30 archivos de envío; a Kaggle se subieron 25 (del 01 al 21 y del 27 al 30). Los archivos 22 a 26 son modelos intermedios que se combinan en los ensambles;
4. traduce las predicciones en una regla de compra para la empresa, considerando que sobreestimar el precio es más costoso que subestimarlo.

El mejor envío es el **29_ensamble_embeddings**: MAE de 151,5 millones de pesos en el puntaje privado de Kaggle.

## Cómo reproducir los resultados

1. Clonar el repositorio y abrir `Taller_3_BDML_Aragon_Cadena_Florez.Rproj` en RStudio.
2. Correr en la consola:

   ```r
   source("01_Scripts/main.R")
   ```

`01_Scripts/main.R` corre todos los scripts en el orden en que dependen unos de otros, cada uno en un proceso de R nuevo, y regenera las bases procesadas, los modelos, los envíos, todas las tablas y figuras, y los PDF de las dos presentaciones. Si un script falla, se detiene e indica cuál fue. Los paquetes de R se instalan solos con `pacman::p_load` la primera vez.

**Tiempo:** entre 7 y 9 horas en un computador personal. Al final, `main.R` imprime cuánto tardó cada script.

**Opciones al comienzo de `main.R`:**

| Opción | Valor por defecto | Qué hace |
|---|---|---|
| `correr_descargas` | `FALSE` | Con `TRUE` vuelve a descargar todas las fuentes externas (Mapas Bogotá, datos abiertos y elevación). Las capas geográficas de Mapas Bogotá no se guardan en el repositorio por su tamaño, así que se descargan automáticamente si no están, aunque la opción sea `FALSE`. |
| `correr_python` | `FALSE` | Con `TRUE` vuelve a calcular los embeddings de texto en Python. Con `FALSE` usa `02_Data/Processed/embeddings_texto_minilm.csv.gz`, que está en el repositorio. |

### Requisitos

- **R 4.3 o superior** y RStudio. Los paquetes principales son `tidyverse`, `here`, `caret`, `sf`, `dbscan`, `spatialsample`, `glmnet`, `rpart`, `ranger`, `xgboost` (versión 3), `nnet`, `gamlr`, `maptpx`, `tidytext`, `SnowballC`, `elevatr`, `scales` y `patchwork`.
- **Conexión a internet**, para descargar las capas de Mapas Bogotá y la elevación la primera vez.
- **Python 3.10 o superior**, solo si `correr_python <- TRUE`: `pip install -r requirements.txt`.
- **LaTeX** (MiKTeX, TeX Live o TinyTeX) con `beamer` y `tikz`, para compilar las presentaciones. Si no está instalado, `main.R` genera todo lo demás y avisa que no pudo compilarlas.

### Nota sobre la reproducibilidad

Los modelos se estimaron a medida que la base crecía. Los envíos 05 a 11 se estimaron con versiones intermedias de la base, antes de agregar el precio de los vecinos, el Catastro por manzana y el texto como datos. `main.R` reconstruye siempre la base final, así que al correrlo desde cero esos modelos se estiman con la base final, y su MAE de validación cruzada puede diferir un poco del registrado en las presentaciones. Los puntajes público y privado que obtuvo cada envío en Kaggle están en `02_Data/Raw/Kaggle/kaggle_puntajes.csv`, copiados de la página de envíos de la competencia; es un insumo, porque ningún script lo genera. Los envíos del 12 en adelante usan la base final, y se reproducen igual.

## Estructura del repositorio

```
├── README.md
├── requirements.txt             Librerías de Python (solo para los embeddings)
├── 01_Scripts
│   ├── main.R                   Archivo principal: corre todo el proyecto
│   ├── 00_utils.R               Funciones comunes (carga, validación cruzada, MAE, envíos)
│   ├── 00_Fuentes_externas      Inventario y descarga de capas de Mapas Bogotá
│   ├── 01_Data                  Limpieza y construcción de variables
│   ├── 02_EDA                   Exploración y diagnóstico de errores
│   ├── 03_Models                Modelos y envíos a Kaggle
│   └── 04_Results               Mapas y análisis de negocio para las presentaciones
├── 02_Data
│   ├── Raw
│   │   ├── Kaggle               train.csv, test.csv, plantilla de envío y puntajes de los envíos
│   │   └── MapasBogota          Inventario, selección y capas descargadas
│   └── Processed                Bases intermedias y finales (.rds) y embeddings
├── 03_Output
│   ├── Figures                  Figuras en PDF y PNG
│   ├── Tables                   Tablas en CSV y TeX
│   ├── Models                   Modelos estimados (.rds) y resultados_modelos.csv
│   ├── Predictions              Predicciones fuera de muestra de la validación cruzada
│   └── Submissions              Los archivos de envío a Kaggle
└── 04_Slides                    Presentaciones en Beamer y sus PDF
    ├── best_equipo_01.tex       Best Model Deep Dive
    ├── compare_equipo_01.tex    Algorithm Comparison
    └── comunes                  Preámbulo y las 10 diapositivas que comparten las dos
```

**Prefijos de las variables:** `num_` numéricas, `bin_` dummies (`bin_<x>_na` marca si `num_<x>` fue imputada), `cat_` categóricas, `obj_` variable objetivo (precio y log del precio). Las funciones propias empiezan por `fun_`.

## Scripts, en el orden en que los corre `main.R`

### 0. Fuentes externas (`01_Scripts/00_Fuentes_externas`)

| Script | Qué hace | Salidas |
|---|---|---|
| `01_inventario_mapas_bogota.R` | Lista las capas disponibles en los servicios ArcGIS de Catastro Bogotá y en Datos Abiertos Bogotá | `02_Data/Raw/MapasBogota/inventario_*.csv` |
| `02_descarga_capas.R` | Descarga las capas de `capas_seleccionadas.csv`, recortadas a los años y la zona de los anuncios | `02_Data/Raw/MapasBogota/capas/*` y `registro_descarga.csv` |

### 1. Construcción de la base (`01_Scripts/01_Data`)

| Script | Qué hace | Salidas |
|---|---|---|
| `01_clean_base.R` | Limpia train y test de Kaggle | `Processed/base_limpia.rds` |
| `02_text_features.R` | Extrae del título y la descripción el área, piso, parqueaderos, habitaciones, baños, estrato y dummies de características y barrios | `Processed/text_features.rds` |
| `03_spatial_features.R` | Calcula las variables de entorno con las capas de Mapas Bogotá: Catastro por manzana, comercio, transporte, educación, parques, riesgo, etc. | `Processed/spatial_features.rds` |
| `04_elevacion.R` | Descarga la altitud de cada anuncio | `Processed/elevacion.rds` |
| `05_merge.R` | Une todo, quita anuncios repetidos, imputa faltantes y calcula el precio de los vecinos | `Processed/db_train.rds`, `db_test.rds` |
| `06_texto_como_datos.R` | Lasso del precio sobre las palabras (fuera de muestra) y tópicos LDA (Latent Dirichlet Allocation) | columnas `num_txt_*` en `db_*.rds`; tablas y figura `texto_*` |
| `07_embeddings_texto.py` | Embeddings de frases (MiniLM multilingüe) de cada anuncio | `Processed/embeddings_texto_minilm.csv.gz` |

### 2. Modelos, exploración y diagnóstico (`01_Scripts/03_Models` y `01_Scripts/02_EDA`)

Cada modelo guarda su envío en `03_Output/Submissions/<envío>.csv`, el modelo en `03_Output/Models/<envío>.rds` y una fila con el MAE de los cuatro esquemas de validación cruzada (aleatoria, bloques de 2 km, bloques con buffer de 300 m y localidades completas por fuera) en `03_Output/Models/resultados_modelos.csv`.

| Script | Qué hace | Envíos y otras salidas |
|---|---|---|
| `02_EDA/01_chapinero_vs_bogota.R` | Compara Chapinero con el resto de Bogotá | figuras y tablas `eda_*` |
| `03_Models/01_baseline.R` | Cuatro modelos simples con las variables originales | 01_lm_base, 02_enet_base, 03_cart_base, 04_rf_base |
| `03_Models/02_lm.R` | Regresión lineal con la base completa | 05_lm_completa |
| `03_Models/03_enet.R` | Elastic Net | 06_enet_completa |
| `03_Models/04_cart.R` | Árbol de regresión | 07_cart_completa |
| `03_Models/05_rf.R` | Random Forest | 08_rf_completa |
| `03_Models/06_xgboost.R` | XGBoost | 09_xgb_completa |
| `02_EDA/02_errores_xgb.R` | Cobertura de test y errores del XGBoost 09 | `eda_cobertura_test`, `eda_errores_*`, `eda_importancia_xgb`, `eda_observado_predicho`, `eda_error_decil_precio` |
| `03_Models/07_xgb_pesos.R` | XGBoost con faltantes nativos y con pesos por parecido con test | 10_xgb_na_nativo, 11_xgb_pesos; `pesos_*`, `xgb_pesos_*` |
| `03_Models/08_xgb_base_final.R` | XGBoost con la base final | 12_xgb_base_final |
| `03_Models/09_xgb_pesos_base_final.R` | XGBoost con pesos, base final | 13_xgb_na_nativo_base_final, 14_xgb_pesos_base_final; `*_base_final` |
| `03_Models/10_xgb_pesos_texto.R` | XGBoost con pesos y texto como datos | 15_xgb_na_nativo_texto, 16_xgb_pesos_texto; `*_texto` |
| `03_Models/11_nnet_texto.R` | Red neuronal de una capa oculta | 17_nnet_texto |
| `03_Models/12_ajuste_sesgo.R` | Corrige la subestimación del 16 con un factor | 18_xgb_pesos_texto_sesgo_ponderado, 19_xgb_pesos_texto_sesgo_chapinero; `ajuste_sesgo_factores` |
| `02_EDA/03_diagnostico_errores.R` | Diagnóstico de errores del 16 por segmento y su traslado a test | `diag_*`, `Predictions/oof_16_*` |
| `03_Models/13_xgb_mejorado.R` | XGBoost con precisión de coordenadas, barrios en el texto y otras mejoras | 20_xgb_mejorado, 21_xgb_mejorado_sesgo; `xgb_mejorado_*` |
| `03_Models/14_ensamble_reanuncios.R` | Modelos del precio por m2, red robusta, ensamble, calibración y anuncios repetidos | 22_xgb_precio_m2 a 27_ensamble_calibrado_reanuncios; `ensamble_*` |
| `03_Models/15_ensamble_v2.R` | Ensamble con varias semillas y ajuste de nivel | 28_ensamble_v2_nivel; `ensamble_v2_*` |
| `03_Models/16_embeddings_red.R` | Ensamble con embeddings y red neuronal con embeddings | 29_ensamble_embeddings, 30_nnet_embeddings; `emb_*` |

### 3. Resultados para las presentaciones (`01_Scripts/04_Results`)

| Script | Qué hace | Salidas |
|---|---|---|
| `01_mapas.R` | Mapas de Bogotá: train y test, precio por m2, predicción, error, cobertura, folds, MAE por localidad, pesos y contexto urbano | figuras `mapa_01` a `mapa_09`; `mapa_mae_localidad_modelos.csv` |
| `02_negocio.R` | Riesgo de sobreestimación, margen de seguridad, semáforo de compra, curva de decisión y Kaggle frente a negocio | figuras `negocio_01` a `negocio_05`; tablas `negocio_*` |
| `03_figuras_presentaciones.R` | Escalera del error, importancia por grupo, familias, validaciones frente a Kaggle, tablas comparativas y las cifras que citan las diapositivas | figuras y tablas `pres_*`; `pres_cifras.tex` |

### 4. Presentaciones (`04_Slides`)

`main.R` compila al final `best_equipo_01.tex` y `compare_equipo_01.tex` con `pdflatex`. Las dos toman las figuras de `03_Output/Figures`, las tablas de `03_Output/Tables` y las cifras del texto de `03_Output/Tables/pres_cifras.tex`, así que todo lo que muestran sale del código. Las 10 diapositivas que comparten están en `04_Slides/comunes`. Cada diapositiva tiene notas para quien expone (`\note`).

## De dónde sale cada figura y tabla

Todas quedan en `03_Output/Figures` (PDF y PNG) y `03_Output/Tables` (CSV y TeX). El prefijo del nombre indica el script que la genera:

| Prefijo | Script |
|---|---|
| `eda_balance_*`, `eda_caracteristicas_*`, `eda_correlaciones_*`, `eda_entorno_*`, `eda_mapa_*`, `eda_menciones_*`, `eda_precio_*`, `eda_soporte_*`, `eda_tabla_*` | `02_EDA/01_chapinero_vs_bogota.R` |
| `eda_cobertura_*`, `eda_errores_*`, `eda_error_decil_*`, `eda_importancia_xgb`, `eda_observado_predicho` | `02_EDA/02_errores_xgb.R` |
| `diag_*` | `02_EDA/03_diagnostico_errores.R` |
| `texto_*` | `01_Data/06_texto_como_datos.R` |
| `pesos_*`, `xgb_pesos_*` (sin sufijo, `_base_final`, `_texto`) | `03_Models/07`, `09` y `10` |
| `ajuste_sesgo_*` | `03_Models/12_ajuste_sesgo.R` |
| `xgb_mejorado_*` | `03_Models/13_xgb_mejorado.R` |
| `ensamble_*` | `03_Models/14_ensamble_reanuncios.R` |
| `ensamble_v2_*` | `03_Models/15_ensamble_v2.R` |
| `emb_*` | `03_Models/16_embeddings_red.R` |
| `mapa_*` | `04_Results/01_mapas.R` |
| `negocio_*` | `04_Results/02_negocio.R` |
| `pres_*` | `04_Results/03_figuras_presentaciones.R` |
| `resultados_modelos.csv` | todos los scripts de `03_Models` |

## De dónde sale cada diapositiva

Las diapositivas C1 a C10 están en las dos presentaciones (`04_Slides/comunes/c01_*.tex` a `c10_*.tex`). Las figuras están en `03_Output/Figures` y las tablas en `03_Output/Tables`.

| Diapositiva | Mensaje | Figura o tabla | Script |
|---|---|---|---|
| C1 | El mejor modelo se equivoca en 151,5 millones por vivienda | `pres_cifras.tex` y diagrama en TikZ | `04_Results/03_figuras_presentaciones.R` |
| C2 | El reto: predecir donde casi no hay datos | `mapa_01_train_test` | `04_Results/01_mapas.R` |
| C3 | Chapinero es más caro, casi todo apartamentos y con ubicación más imprecisa | `mapa_02_precio_m2_upz`; cifras de `pres_cifras.tex` | `01_mapas.R`; `03_figuras_presentaciones.R` |
| C4 | Qué le agregamos a los datos | `mapa_09_contexto_urbano`; cifras de `pres_cifras.tex` | `01_mapas.R`; `03_figuras_presentaciones.R` |
| C5 | Qué revela el modelo sobre los precios | `pres_importancia_grupos` | `03_figuras_presentaciones.R` |
| C6 | Cada idea, cuánto bajó el error | `pres_escalera` | `03_figuras_presentaciones.R` |
| C7 | Cómo validamos | `mapa_06_folds_espaciales` | `01_mapas.R` |
| C8 | Le enseñamos al modelo a mirar lo que se parece a Chapinero | `mapa_08_pesos_upz` | `01_mapas.R` |
| C9 | Lo que gana en Kaggle no es lo que gana la empresa | `negocio_05_kaggle_vs_negocio` | `04_Results/02_negocio.R` |
| C10 | ¿Compraríamos? | `negocio_03_semaforo_test`; cifras de `pres_cifras.tex` | `02_negocio.R`; `03_figuras_presentaciones.R` |

**`best_equipo_01` (Best Model Deep Dive).** Orden: portada, C1 a C8, B1, B2, B3, C9, B4, B5, B6 y C10.

| Diapositiva | Mensaje | Figura o tabla | Script |
|---|---|---|---|
| B1 | El modelo distingue zonas dentro de Chapinero | `mapa_03_chapinero_prediccion_test` | `01_mapas.R` |
| B2 | Dónde se equivoca: subestima las viviendas caras | `mapa_04_chapinero_error_oof`, `diag_sesgo_deciles` | `01_mapas.R`; `02_EDA/03_diagnostico_errores.R` |
| B3 | Dónde el modelo tiene poca información para predecir | `mapa_05_chapinero_cobertura` | `01_mapas.R` |
| B4 | Dónde pagaríamos de más | `negocio_01_riesgo_sobreestimacion_upz` | `02_negocio.R` |
| B5 | Cuánto ofertar por debajo de la predicción en cada zona | `negocio_02_margen_seguridad_upz` | `02_negocio.R` |
| B6 | La trampa de Zillow | `negocio_04_curva_margen` | `02_negocio.R` |

**`compare_equipo_01` (Algorithm Comparison).** Orden: portada, C1, C2, C3, C4, C7, P1 a P5, C8, C6, P6, C5, P7, C9 y C10.

| Diapositiva | Mensaje | Figura o tabla | Script |
|---|---|---|---|
| P1 | Seis familias, un ganador claro | `pres_familias` | `03_figuras_presentaciones.R` |
| P2 | Por qué pierden los modelos simples | `pres_modelos_simples` | `03_figuras_presentaciones.R` |
| P3 | Con los mismos datos, el algoritmo lo cambia todo | `pres_misma_base.tex` | `03_figuras_presentaciones.R` |
| P4 | La validación aleatoria escoge el modelo equivocado | `pres_puestos_validacion` | `03_figuras_presentaciones.R` |
| P5 | Dónde gana el boosting: el error por localidad | `mapa_07_mae_localidad_modelos` | `01_mapas.R` |
| P6 | ¿Qué validación anticipa Kaggle? | `pres_cv_vs_kaggle`, `pres_publico_privado` | `03_figuras_presentaciones.R` |
| P7 | Ventajas y desventajas de cada familia | `pres_ventajas.tex` | `03_figuras_presentaciones.R` |

