# MercadoLibre: ¿nuevo o usado? — Examen 3, Análisis Predictivo (ITBA)

Competencia de Kaggle `ap-2026-q-2`: predecir si una publicación de MercadoLibre Argentina (2015) es de un producto
**nuevo** o **usado**. La métrica es accuracy.

**Resultado:** 3.er puesto. El envío elegido, `ensamble_fable_v9.csv`, obtuvo **0.95188** en el leaderboard privado
y 0.95076 en el público.

## Contenido

| | |
|---|---|
| `entrenar_modelo_final.ipynb` | **(b)** entrena y guarda el modelo final en `modelo_final/` |
| `aplicar_modelo_final.ipynb` | **(c)** aplica el modelo final a test y genera `submissions/ensamble_fable_v9.csv`; además lo compara con el archivo enviado |
| `features.py` | feature engineering (lo usan los dos notebooks) |
| `gpu_kaggle/` | notebooks que necesitan GPU y corrieron en Kaggle: 3 transformers sobre el texto y TabPFN. En `salidas/` están sus predicciones |
| `llm/` | `llm_claude.py` (consulta a Claude) y `puntajes/`: las respuestas que usa el modelo final |
| `datos/` | `categories.json` (árbol de categorías de la API pública de MercadoLibre) y el script que lo descargó; `embeddings_e5/`, los embeddings de los títulos. Acá van también los datos de la competencia |
| `submissions/referencia_ensamble_fable_v9.csv` | el archivo que se envió a Kaggle |
| `experimentos/` | todo lo que se probó y no forma parte del modelo final (ver su README) |

## Modelo final

```
publicación ──► features (200) ──┬─► LightGBM  ┐
 (train+test)    · de la fila    ├─► XGBoost   ├─ promedio ─┐
                 · del vendedor  ├─► CatBoost  ┘  (3 semillas c/u)
                   y la categoría│                           ├─ 0,5 / 0,5 ─► ensamble ─┐
                 · nivel 1 (OOF) └─► TabPFN 3.5 ─────────────┘                          │
                   TE, TF-IDF, kNN,                                                      ▼
                   3 transformers          Claude (LLM) ──► regresión logística por tramo ──► predicción
                                                            (solo vendedores sin historial,
                                                             con 1-3 publicaciones o mixtos)
```

1. **Features** (`features.py`): 200 columnas.
   - **De la publicación:** precio y centavos, cantidades, tipo de publicación, garantía, envío, medios de pago, fechas, fotos (cantidad, antigüedad, forma y resolución) y atributos y variaciones.
   - **Agregados no supervisados del vendedor y la categoría:** cantidad de publicaciones, % con stock 1, % gratuitas, precio mediano, etc.
   - **Nivel 1, que usan la etiqueta y siempre se calculan out-of-fold:**
     - target encoding suavizado de vendedor, categoría, ciudad, garantía y combinaciones vendedor × atributo;
     - regresiones logísticas sobre TF-IDF y sobre embeddings `multilingual-e5-small` del título y la garantía;
     - kNN de títulos;
     - vecinos de título dentro del mismo vendedor;
     - la predicción de tres transformers afinados (BETO, XLM-RoBERTa y BERTIN).
2. **Ensamble**: 0,5 × TabPFN + 0,5 × promedio de LightGBM, XGBoost y CatBoost.
3. **Combinación con un LLM** donde el ensamble sabe poco del vendedor. Claude estima la probabilidad de que la publicación sea nueva y en cada tramo una regresión logística combina `logit(ensamble)` con `logit(Claude)`:
   - **vendedores sin publicaciones etiquetadas** (28% de test): Claude Opus 5.5 ve solo la publicación;
   - **vendedores con 1 a 3 publicaciones** (20% de test): Claude Opus 5.5, igual;
   - **vendedores mixtos** (≥ 4 publicaciones, entre 20% y 80% nuevas; 6% de test): Claude Fable 5.2 ve además las publicaciones más parecidas del vendedor con su condición real.

**Validación:** 5 folds estratificados fijos (`features.cv_folds`), los mismos para todos los modelos. Todo lo que usa la etiqueta se recalcula dentro de cada fold. En train, el historial del vendedor que ven el ensamble y Claude sale solo de la parte de entrenamiento del fold.

| | accuracy (validación cruzada) |
|---|---|
| Baseline: regresión logística con 14 features crudas | 0.8234 |
| LightGBM / XGBoost / CatBoost | 0.9483 / 0.9476 / 0.9481 |
| TabPFN 3.5 | 0.9474 |
| Ensamble | 0.9491 |
| + Claude, vendedores nuevos (6.457 filas evaluadas) | 0.9029 → 0.9111 en ese tramo |
| + Claude, vendedores con 1-3 publicaciones (1.664 filas) | 0.9525 → 0.9561 en ese tramo |
| + Claude con historial, vendedores mixtos (2.000 filas) | 0.9040 → 0.9095 en ese tramo |

## Cómo replicar

Requisitos: Python 3.13 y `pip install -r requirements.txt`. Copiar `train_data.jsonlines` y `test_data.jsonlines`
de la competencia en `datos/`, o indicar otra carpeta con la variable de entorno `TP3_DATOS`.

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 entrenar_modelo_final.ipynb
jupyter nbconvert --to notebook --execute --inplace aplicar_modelo_final.ipynb
```

El entrenamiento tarda unas 3 horas en CPU (2 h son de los boostings; CatBoost es el más lento). La aplicación tarda unos minutos y termina comparando el CSV generado con
`submissions/referencia_ensamble_fable_v9.csv`.

**Verificado:** con la caché vacía, los dos notebooks generan un archivo idéntico al enviado (0 predicciones distintas). Los notebooks de este repositorio quedaron guardados con las salidas de esa corrida.

### Partes que no se recalculan en local

Hay cuatro insumos que el entrenamiento lee ya calculados:

- **Transformers y TabPFN** (`gpu_kaggle/`): necesitan GPU y corrieron en notebooks de Kaggle.
  - El fine-tuning en GPU no es determinista bit a bit, por eso sus salidas quedan guardadas en `gpu_kaggle/salidas/`.
  - Para regenerarlas:
    1. correr `transformer_beto/xlmr/bertin.ipynb` con los datos de la competencia;
    2. correr el paso 1 del entrenamiento y `python gpu_kaggle/exportar_datos_tabpfn.py`;
    3. subir esa carpeta como dataset;
    4. correr `tabpfn.ipynb`, que necesita el secret `TABPFN_TOKEN` de Prior Labs.
- **Puntajes de Claude** (`llm/puntajes/`): las respuestas de un LLM no son deterministas, así que quedan guardadas. El script que las generó es `llm/llm_claude.py` (se corre con Claude Code, `claude -p`). Selecciona exactamente las mismas filas que los puntajes guardados; eso también está verificado.
- **Embeddings de los títulos** (`datos/embeddings_e5/`, `multilingual-e5-small`):
  - Se pueden recalcular en CPU, pero no salen idénticos bit a bit en otra máquina o con otra versión de las librerías: difieren en ~1e-7.
  - Esa diferencia pasa a `txt_emb`, una regresión logística con C=30 que no llega a converger del todo y la amplifica.
  - Al final cambian 21 predicciones de 30.000.
  - Por eso se guardan los que usó el modelo enviado. Si se borra la carpeta, el notebook los recalcula.
- **`datos/categories.json`**: descargado una vez de la API pública de MercadoLibre con `datos/fetch_categories.py`. La API cambia con el tiempo.

### ¿Hay fuga de información?

No. Ningún modelo ve la etiqueta de la fila que predice:

- El target encoding y los vecinos se calculan out-of-fold.
- El historial etiquetado que recibe Claude en los vendedores mixtos sale, en train, solo de la parte de entrenamiento del fold de cada fila. En test sale de todo train, que es información disponible al momento de predecir.
- No se usó ningún dato externo con etiquetas.
