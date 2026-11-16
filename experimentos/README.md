# Experimentos

Código de desarrollo y de las pruebas que **no** forman parte del modelo final. Se incluye para documentar cómo se llegó a él.

Estos archivos corrían desde la carpeta de desarrollo original, con otra estructura de carpetas: `features.py` y `cache/` en la raíz, las salidas de Kaggle en `kaggle_*/`. Para correrlos hay que adaptar las rutas; los scripts de `pruebas/` toman esa carpeta de la variable `TP3_DESARROLLO`. El modelo final se replica solo con los notebooks de la raíz del repositorio.

**Criterio para adoptar un cambio:** comparar contra la misma base, con los mismos folds y la misma semilla. Se miran las filas netas ganadas sobre 70.000, cuántos de los 5 folds mejoran y un bootstrap pareado (P > 0). Solo se adopta si mejora de forma consistente.

## `desarrollo/` — notebooks del camino principal

| Archivo | Qué hace | Resultado |
|---|---|---|
| `00_features.ipynb` | features compartidas | — |
| `01_baseline.ipynb` | **baseline**: regresión logística con 14 features crudas (sin vendedor ni texto) | 0.8234 (clase mayoritaria: 0.537) |
| `02_lightgbm`, `03_xgboost`, `04_catboost` | boostings con 3 semillas | 0.9483 / 0.9476 / 0.9481 |
| `05_ensamble.ipynb` | peso de TabPFN frente a los boostings (grilla sobre OOF) | curva plana entre 0,4 y 0,7: se usa 0,5 |
| `blend_claude.py` | combinación con Claude por tramos (versión de desarrollo, con variantes Opus/Fable) | — |
| `llm_claude.py` | versión completa del script del LLM, con la variante v2 descartada | — |

## Modelos que se probaron

| Modelo | Resultado | Decisión |
|---|---|---|
| Regresión logística (baseline) | 0.8234 | referencia |
| LightGBM, XGBoost, CatBoost | ~0.948 cada uno, correlación 0.995 entre sí | se promedian |
| TabPFN 3.5 (`gpu_kaggle/tabpfn.ipynb`) | 0.9474; el ensamble con los boostings sube a 0.9491 | **en el modelo final** |
| TabPFN con más estimadores o 2.ª semilla | sin mejora | descartado |
| TabM (`kaggle/tabm.ipynb`) | 0.9456, con las features de ese momento | no sumó al ensamble |
| TabICL (`kaggle/tabicl.ipynb`) | 0.9403, con las features de ese momento | descartado |
| LimiX-2 (`kaggle/limix2.ipynb`) | 0.9474 en el fold 0 (TabPFN: 0.9494); +6 filas al ensamble sobre 14.000; 4 h de GPU por fold | descartado |
| Optuna sobre LightGBM (`pruebas/tune_boosting.py`, `validar_lgb_opt.py`) | mejor log-loss pero accuracy 0.9469 contra 0.9474 | descartado |
| Stacking (modelo de nivel 2) en lugar de promedio | sin mejora | descartado |
| Pseudo-etiquetado de test | sin mejora | descartado |

## Features que se probaron (`pruebas/`, banco de pruebas `harness.py`: LightGBM sobre los 5 folds)

| Prueba | Resultado (filas netas sobre 70.000) | Decisión |
|---|---|---|
| Forma y resolución de las fotos (`e1_fotos.py`, `e2.py`, `fotos.py`): las cuadradas son 86% nuevas, las 4:3 alrededor del 40% | +41 (mejora en 5 de 5 folds) | **adoptada** |
| Atributos y variaciones (`e3.py`, `attrs.py`): repuestos, autos, inmuebles, ropa, kilómetros, talles | +22 más | **adoptada** |
| Transformer sobre la "ficha" completa (`kaggle/transformer_ficha.ipynb`, `eval_ficha.py`) | 0.9138 solo; no suma | descartado |
| Vendedor × forma de foto y categoría × forma de foto (`e4.py`) | −25 / −15 | descartado |
| Títulos duplicados y carga masiva (`e5.py`) | +13, poco consistente | descartado |
| Modelo especialista para vendedores nuevos (`e6.py`) | sin mejora | descartado |
| Vendedor × combinación de varios atributos (`e7.py`) | sin mejora | descartado |
| Vendedor × tanda de carga (minuto, hora, día, semana) (`e8.py`) | sin mejora | descartado |
| Vecinos dentro del vendedor usando título + atributos (`e9.py`) | sin mejora | descartado |
| Promedio de las predicciones de las otras publicaciones del vendedor (`vecinos_vendedor.py`) | sin mejora | descartado |
| Análisis de errores del ensamble por tramo de vendedor (`errores.py`) | los errores se concentran en vendedores sin historial y en vendedores mixtos | motivó usar el LLM por tramos |

## LLM (Claude)

| Prueba | Resultado | Decisión |
|---|---|---|
| Claude Opus 5.5 viendo solo la publicación, vendedores nuevos | 0.9029 → 0.9111 en el tramo | **adoptado** |
| Ídem, vendedores con 1 a 3 publicaciones | 0.9525 → 0.9561 en el tramo | **adoptado** |
| Opus con historial etiquetado del vendedor, vendedores mixtos | 0.9040 → 0.9095 en el tramo (2.000 filas) | adoptado en v8 |
| Fable 5.2 en lugar de Opus en los mixtos (`fable_mixtos2.py`) | mismo accuracy, mejor log-loss (0.2248 contra 0.2282); en el leaderboard público, +1 fila | **adoptado (v9, la elegida)** |
| Fable en vendedores nuevos (`fable_combo.py`) | solo es mejor (0.879 contra 0.858), pero ya combinado con el ensamble es peor (0.9072 contra 0.9096) | descartado |
| Promediar varios LLM (Opus, Sonnet, Haiku, Fable) | opinan casi igual (Opus y Fable: correlación 0.96-0.97); no suma | descartado |
| Variante v2: ejemplos resueltos en la instrucción + títulos del vendedor | sin mejora | descartado |

## Versiones enviadas a Kaggle

| Versión | Cambio | Público | Privado |
|---|---|---|---|
| v7 | ensamble + Claude Opus en vendedores nuevos y con 1-3 publicaciones | 0.95095 | — |
| v8 | + features de fotos, atributos y variaciones; + Claude con historial en los mixtos | 0.95071 | — |
| **v9** | Fable en lugar de Opus en los mixtos | 0.95076 | **0.95188** (3.er puesto) |

Promediar v7 con v8 empeoraba la validación cruzada: −29 filas con pesos 50/50.
