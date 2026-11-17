# ==============================================================================
# 03_spatial_features.R
# Objetivo: construir los predictores espaciales de cada propiedad a partir de
#           las capas descargadas de Mapas Bogotá, Datos Abiertos Bogotá, el
#           Jardín Botánico y el repositorio del curso. Hay cuatro tipos de
#           cálculo:
#           - distancia en metros al elemento más cercano de una capa;
#           - número de elementos a menos de un radio;
#           - resumen (mediana, promedio, moda o proporción) de los valores de
#             las manzanas cercanas, con el año catastral igual al año del
#             anuncio;
#           - atributos del polígono donde cae la propiedad (UPZ, UPL,
#             localidad, zona de amenaza, etc.).
#           Todos los cálculos se hacen en EPSG:9377, en metros.
#           Las capas de delitos de alto impacto (34_delitos) y del índice de
#           seguridad nocturna (36a_seguridad_nocturna) no se usan: vienen por
#           localidad y casi todo test está en una sola localidad (Chapinero),
#           de modo que en test serían prácticamente constantes. Los puntos de
#           evaluación de seguridad nocturna (36b_seguridad_nocturna) no traen
#           columnas numéricas que se puedan resumir.
# Entradas: 02_Data/Processed/base_limpia.rds
#           02_Data/Raw/MapasBogota/capas/*
# Salidas:  02_Data/Processed/spatial_features.rds (una fila por property_id)
# ==============================================================================

source(here::here("01_Scripts", "00_utils.R"))

base <- readRDS(here("02_Data", "Processed", "base_limpia.rds"))

puntos <- base %>%
  select(property_id, num_anio, num_x, num_y) %>%
  st_as_sf(coords = c("num_x", "num_y"), crs = crs_metros, remove = FALSE)

# Cada bloque agrega columnas a `sp`, que tiene una fila por propiedad en el
# mismo orden de `puntos`.
sp <- tibble(property_id = puntos$property_id)

fun_mensaje <- function(texto) message(format(Sys.time(), "%H:%M:%S"), "  ", texto)

# ------------------------------------------------------------------------------
# 1. Catastro por manzana, con el año catastral igual al año del anuncio
# ------------------------------------------------------------------------------
anios_catastro <- 2019:2021

# Año catastral que corresponde a cada anuncio: el mismo año si está entre
# 2019 y 2021; si no, el más cercano.
anio_ref <- pmin(pmax(puntos$num_anio, min(anios_catastro)), max(anios_catastro))

# Aplica un resumen por año: para los anuncios de cada año usa solo las
# manzanas de ese año. `tabla` tiene columnas x, y, anio y valor.
fun_por_anio <- function(tabla, radio, fun = median, tipo = "resumen", condicion = NULL) {
  resultado <- rep(NA_real_, nrow(puntos))
  for (a in anios_catastro) {
    filas <- which(anio_ref == a)
    ref <- tabla %>% filter(anio == a, !is.na(valor))
    if (length(filas) == 0 || nrow(ref) == 0) next
    coords <- as.matrix(ref[, c("x", "y")])
    resultado[filas] <- if (tipo == "resumen") {
      fun_resumen_radio(puntos[filas, ], coords, ref$valor, radio, fun)
    } else {
      fun_proporcion_radio(puntos[filas, ], coords, ref$valor, radio)
    }
  }
  resultado
}

# 1.1 Valor de referencia del m2 de terreno. Esta capa también da la ubicación
# de cada manzana (MANCODIGO) por año, que usan las capas que se descargaron
# sin geometría.
fun_mensaje("Catastro: valor de referencia")
v_ref <- fun_leer_capa("01_valor_referencia") %>%
  mutate(MANCODIGO = str_pad(as.character(MANCODIGO), 9, pad = "0"),
         anio = fun_anio(ANO),
         valor = as.numeric(V_REF))
coords_vref <- fun_coords(v_ref)
manzanas <- v_ref %>%
  st_drop_geometry() %>%
  transmute(MANCODIGO, anio, x = coords_vref[, 1], y = coords_vref[, 2], valor)

sp$num_vref_m2_200m <- fun_por_anio(manzanas, radio = 200)

# Ubica una tabla sin geometría (MANCODIGO, año, valor) con las coordenadas de
# las manzanas. Si una manzana no aparece en ese año, se usa su ubicación en
# otro año. Imprime el porcentaje de registros ubicados.
ubicacion_codigo <- manzanas %>% distinct(MANCODIGO, .keep_all = TRUE) %>%
  select(MANCODIGO, x_otro = x, y_otro = y)

fun_ubicar_manzanas <- function(tabla, nombre) {
  ubicada <- tabla %>%
    left_join(manzanas %>% select(MANCODIGO, anio, x, y), by = c("MANCODIGO", "anio")) %>%
    left_join(ubicacion_codigo, by = "MANCODIGO") %>%
    mutate(x = coalesce(x, x_otro), y = coalesce(y, y_otro)) %>%
    select(-x_otro, -y_otro)
  message("  ", nombre, ": ", round(100 * mean(!is.na(ubicada$x)), 1),
          " % de los registros ubicados con MANCODIGO")
  ubicada %>% filter(!is.na(x))
}

fun_tabla_csv <- function(id, campo_anio, campo_valor, numerico = TRUE) {
  fun_leer_capa(id) %>%
    transmute(MANCODIGO = str_pad(MANCODIGO, 9, pad = "0"),
              anio = fun_anio(.data[[campo_anio]]),
              valor = if (numerico) as.numeric(.data[[campo_valor]]) else .data[[campo_valor]])
}

# 1.2 Avalúo catastral por m2 en propiedad horizontal (con geometría propia).
fun_mensaje("Catastro: avalúo en propiedad horizontal")
av_ph <- fun_leer_capa("02_avaluo_ph")
coords_ph <- fun_coords(av_ph)
tabla_ph <- tibble(x = coords_ph[, 1], y = coords_ph[, 2],
                   anio = fun_anio(av_ph$VIGENCIA), valor = as.numeric(av_ph$AV_CAT_PH))
sp$num_avaluo_ph_m2_200m <- fun_por_anio(tabla_ph, radio = 200)

# 1.3 a 1.6 Capas sin geometría, ubicadas con MANCODIGO.
fun_mensaje("Catastro: avalúo, altura, uso y densidad predial por manzana")
av_mz  <- fun_ubicar_manzanas(fun_tabla_csv("03_avaluo_manzana", "VIGENCIA", "MED_VALOR_CATAS"),
                              "03_avaluo_manzana")
altura <- fun_ubicar_manzanas(fun_tabla_csv("05_altura", "ANO", "ALTURA"), "05_altura")
uso    <- fun_ubicar_manzanas(fun_tabla_csv("06_uso", "ANO", "GRUPOUSOECON", numerico = FALSE),
                              "06_uso")
dens   <- fun_ubicar_manzanas(fun_tabla_csv("07_densidad_predial", "ANO", "N_PREDIOS"),
                              "07_densidad_predial")

sp$num_avaluo_mz_200m  <- fun_por_anio(av_mz, radio = 200)
sp$num_pisos_200m      <- fun_por_anio(altura, radio = 200, fun = mean)
sp$num_predios_mz_200m <- fun_por_anio(dens, radio = 200, fun = mean)
sp$num_prop_resid_300m <- fun_por_anio(uso %>% mutate(valor = valor == "RESIDENCIAL"),
                                       radio = 300, tipo = "proporcion")
sp$num_prop_comer_300m <- fun_por_anio(uso %>% mutate(valor = str_detect(valor, "COMERCIO")),
                                       radio = 300, tipo = "proporcion")

# 1.7 Estrato: el más frecuente entre las manzanas residenciales a menos de
# 200 m (el estrato 0 corresponde a manzanas no residenciales).
fun_mensaje("Estrato")
estrato <- fun_leer_capa("04_estrato") %>% mutate(ESTRATO = as.numeric(ESTRATO))
estrato_res <- estrato %>% filter(ESTRATO %in% 1:6)
sp$num_estrato_200m <- fun_resumen_radio(puntos, fun_coords(estrato_res), estrato_res$ESTRATO,
                                         radio = 200, fun = function(v) as.numeric(fun_moda(v)))

# ------------------------------------------------------------------------------
# 2. Comodidades y comercio
# ------------------------------------------------------------------------------
fun_mensaje("Cultura")
for (capa_id in c("08_museos", "09_teatros", "10_cines", "11_galerias", "12_bibliotecas")) {
  nombre <- str_remove(capa_id, "^\\d+_")
  capa <- fun_leer_capa(capa_id)
  sp[[paste0("num_", nombre, "_1km")]]  <- fun_conteo_radio(puntos, capa, 1000)
  sp[[paste0("num_dist_", nombre)]]     <- fun_dist_min(puntos, capa)
}
sp$num_cultura_1km <- sp$num_museos_1km + sp$num_teatros_1km + sp$num_cines_1km +
  sp$num_galerias_1km + sp$num_bibliotecas_1km

fun_mensaje("Gastronomía y bar")
gastro <- fun_leer_capa("13_gastronomia_bar")
sp$num_gastro_300m <- fun_conteo_radio(puntos, gastro, 300)
sp$num_gastro_1km  <- fun_conteo_radio(puntos, gastro, 1000)

# Establecimientos de la Cámara de Comercio. Los códigos CIIU 5611
# (restaurantes) y 5630 (bares) se buscan en los dos campos CIIU de la capa.
fun_mensaje("Establecimientos de comercio")
estab <- fun_leer_capa("14_establecimientos") %>%
  mutate(ciiu = paste(as.character(ESTCIIU), if ("ESTCCII" %in% names(.)) as.character(ESTCCII) else ""))
restaurantes <- estab %>% filter(str_detect(ciiu, "5611"))
bares        <- estab %>% filter(str_detect(ciiu, "5630"))
sp$num_estab_500m        <- fun_conteo_radio(puntos, estab, 500)
sp$num_restaurantes_500m <- fun_conteo_radio(puntos, restaurantes, 500)
sp$num_bares_500m        <- fun_conteo_radio(puntos, bares, 500)

fun_mensaje("Centros comerciales, plazas de mercado e IPS")
sp$num_dist_cc     <- fun_dist_min(puntos, fun_leer_capa("15_centros_comerciales"))
sp$num_dist_plaza  <- fun_dist_min(puntos, fun_leer_capa("16_plazas_mercado"))
sp$num_ips_1km     <- fun_conteo_radio(puntos, fun_leer_capa("19_ips"), 1000)

# Zonas de interés turístico y distritos creativos: distancia (cero si la
# propiedad está adentro) e indicador de estar a menos de 300 m o adentro.
fun_mensaje("Zonas turísticas y distritos creativos")
sp$num_dist_zona_turistica <- fun_dist_min(puntos, fun_leer_capa("17_zonas_turisticas"))
sp$bin_zona_turistica_300m <- as.integer(sp$num_dist_zona_turistica <= 300)
sp$num_dist_distrito_creativo <- fun_dist_min(puntos, fun_leer_capa("18_distritos_creativos"))
sp$bin_distrito_creativo      <- as.integer(sp$num_dist_distrito_creativo == 0)

# Empresas por UPL (Unidad de Planeamiento Local), con el año del anuncio.
fun_mensaje("Empresas por UPL")
fun_empresas <- function(prefijo, ids) {
  valores <- map2(ids, anios_catastro, function(id, a) {
    capa <- fun_leer_capa(id)
    campo <- names(capa)[str_detect(names(capa), paste0("^", prefijo, a, "$"))][1]
    as.numeric(fun_poligono(puntos, capa, campo)[[1]])
  })
  case_when(anio_ref == 2019 ~ valores[[1]],
            anio_ref == 2020 ~ valores[[2]],
            TRUE             ~ valores[[3]])
}
sp$num_empresas_servicio_upl <- fun_empresas("SERV", c("20a_empresas_servicio_2019",
                                                      "20b_empresas_servicio_2020",
                                                      "20c_empresas_servicio_2021"))
sp$num_empresas_comercio_upl <- fun_empresas("COM", c("20d_empresas_comercial_2019",
                                                     "20e_empresas_comercial_2020",
                                                     "20f_empresas_comercial_2021"))

# ------------------------------------------------------------------------------
# 3. Educación
# ------------------------------------------------------------------------------
fun_mensaje("Educación")
sp$num_ies_1km      <- fun_conteo_radio(puntos, fun_leer_capa("21_educacion_superior"), 1000)
sp$num_colegios_1km <- fun_conteo_radio(puntos, fun_leer_capa("22_colegios"), 1000)

# Puntaje global promedio de Saber 11 de los colegios a menos de 1 km (si no
# hay ninguno, el del colegio más cercano).
saber <- fun_leer_capa("23_saber11") %>% mutate(puntaje = as.numeric(P_PUNTAJE_))
sp$num_saber11_1km <- fun_resumen_radio(puntos, fun_coords(saber), saber$puntaje,
                                        radio = 1000, fun = mean)

# ------------------------------------------------------------------------------
# 4. Verde y espacio público
# ------------------------------------------------------------------------------
fun_mensaje("Árboles")
arboles <- fun_leer_capa("24_arboles")
sp$num_arboles_100m <- fun_conteo_radio(puntos, arboles, 100)
sp$num_arboles_300m <- fun_conteo_radio(puntos, arboles, 300)
rm(arboles)

fun_mensaje("Parques y agua")
parques <- fun_leer_capa("25_parques") %>% mutate(area = as.numeric(st_area(geometry)))
parque_cercano <- st_nearest_feature(puntos, parques)
sp$num_dist_parque <- as.numeric(st_distance(puntos, parques[parque_cercano, ], by_element = TRUE))
sp$num_area_parque <- parques$area[parque_cercano]
sp$num_dist_cuerpo_agua    <- fun_dist_min(puntos, fun_leer_capa("26a_cuerpos_agua"))
sp$num_dist_corriente_agua <- fun_dist_min(puntos, fun_leer_capa("26b_corrientes_agua"))

fun_mensaje("Espacio público efectivo por UPZ")
sp$num_espacio_publico_upz <- as.numeric(
  fun_poligono(puntos, fun_leer_capa("27_espacio_publico"), "EPE_M2_HA")[[1]]
)

# ------------------------------------------------------------------------------
# 5. Movilidad
# ------------------------------------------------------------------------------
fun_mensaje("Movilidad")
sp$num_dist_estacion_tm <- fun_dist_min(puntos, fun_leer_capa("28a_estaciones_tm"))
sp$num_dist_portal_tm   <- fun_dist_min(puntos, fun_leer_capa("28b_portales_tm"))
sp$num_dist_tm          <- pmin(sp$num_dist_estacion_tm, sp$num_dist_portal_tm)
sp$num_paraderos_300m   <- fun_conteo_radio(puntos, fun_leer_capa("29_paraderos_sitp"), 300)
sp$num_dist_ciclorruta  <- fun_dist_min(puntos, fun_leer_capa("30_ciclorrutas"))
sp$num_cicloparq_500m   <- fun_conteo_radio(puntos, fun_leer_capa("31_cicloparqueaderos"), 500)
sp$num_dist_metro       <- fun_dist_min(puntos, fun_leer_capa("32_metro"))
sp$num_dist_parqueo_pago <- fun_dist_min(puntos, fun_leer_capa("33_parqueo_pago"))

# ------------------------------------------------------------------------------
# 6. Seguridad
# ------------------------------------------------------------------------------

# Atracos: número de atracos de la celda de 300 m donde cae la propiedad. Las
# propiedades fuera de toda celda tienen cero atracos registrados.
fun_mensaje("Atracos")
atracos <- fun_leer_capa("35_atracos")
celda <- map_int(st_intersects(puntos, atracos), ~ if (length(.x) == 0) NA_integer_ else .x[1])
sp$num_atracos_celda <- replace_na(as.numeric(atracos$COUNT[celda]), 0)

fun_mensaje("Policía y alumbrado")
sp$num_dist_policia <- fun_dist_min(puntos, fun_leer_capa("37_estaciones_policia"))

# Cuadrante de policía: identificador del polígono donde cae la propiedad.
cuadrantes <- fun_leer_capa("38_cuadrantes") %>% mutate(id_cuadrante = row_number())
sp$cat_cuadrante <- as.character(fun_poligono(puntos, cuadrantes, "id_cuadrante")[[1]])

# Luminarias por hectárea de la UPZ y proporción de luminarias LED.
alumbrado <- fun_leer_capa("39_alumbrado") %>%
  mutate(hectareas = as.numeric(st_area(geometry)) / 10000,
         num_luminarias_ha = as.numeric(TOTAL) / hectareas,
         num_prop_led = as.numeric(LED) / as.numeric(TOTAL))
sp <- bind_cols(sp, fun_poligono(puntos, alumbrado, c("num_luminarias_ha", "num_prop_led")))

# ------------------------------------------------------------------------------
# 7. Riesgo
# ------------------------------------------------------------------------------
fun_mensaje("Amenaza por movimientos en masa y respuesta sísmica")
amenaza <- fun_leer_capa("42_amenaza_remocion")
zona <- map_int(st_intersects(puntos, amenaza), ~ if (length(.x) == 0) NA_integer_ else .x[1])
sp$cat_amenaza_remocion <- replace_na(as.character(amenaza$AMENAZA[zona]), "Sin zonificación")
sp$cat_respuesta_sismica <- as.character(
  fun_poligono(puntos, fun_leer_capa("43_respuesta_sismica"), "RESSIS")[[1]]
)

# ------------------------------------------------------------------------------
# 8. Unidades territoriales (para la validación cruzada espacial)
# ------------------------------------------------------------------------------
fun_mensaje("Localidad, UPZ y sector catastral")
sp$cat_localidad <- as.character(fun_poligono(puntos, fun_leer_capa("44_localidades"),
                                              "LOCNOMBRE")[[1]])
upz <- fun_poligono(puntos, fun_leer_capa("45_upz"), c("CODIGO_UPZ", "NOMBRE"))
sp$cat_upz_codigo <- as.character(upz$CODIGO_UPZ)
sp$cat_upz        <- as.character(upz$NOMBRE)
sector <- fun_poligono(puntos, fun_leer_capa("46_sectores_catastrales"),
                       c("SCACODIGO", "SCANOMBRE"))
sp$cat_sector_codigo <- as.character(sector$SCACODIGO)
sp$cat_sector        <- as.character(sector$SCANOMBRE)

# ------------------------------------------------------------------------------
# 9. Base de salida y resumen
# ------------------------------------------------------------------------------
saveRDS(sp, here("02_Data", "Processed", "spatial_features.rds"))
fun_mensaje(paste("Guardada spatial_features.rds con", ncol(sp) - 1, "variables"))

# Porcentaje de faltantes y mediana de cada variable numérica, en train y test.
sp %>%
  left_join(base %>% select(property_id, muestra), by = "property_id") %>%
  select(muestra, where(is.numeric)) %>%
  pivot_longer(-muestra, names_to = "variable") %>%
  group_by(variable, muestra) %>%
  summarise(pct_na = round(100 * mean(is.na(value)), 1),
            mediana = median(value, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = muestra, values_from = c(pct_na, mediana)) %>%
  print(n = Inf)

# Localidades con propiedades en test.
sp %>%
  left_join(base %>% select(property_id, muestra), by = "property_id") %>%
  count(muestra, cat_localidad) %>%
  pivot_wider(names_from = muestra, values_from = n, values_fill = 0) %>%
  arrange(desc(test)) %>%
  print(n = Inf)
