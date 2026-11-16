"""Exporta las features del modelo (las mismas que reciben los boostings, con las columnas de los transformers) para
subirlas como dataset de Kaggle y correr gpu_kaggle/tabpfn.ipynb. Requiere haber corrido el paso 1 de
entrenar_modelo_final.ipynb (features en cache/) y tener las salidas de los transformers en gpu_kaggle/salidas/.

Uso (desde la raiz del repositorio):
    python gpu_kaggle/exportar_datos_tabpfn.py
    kaggle datasets create -p gpu_kaggle/datos_tabpfn      (o "kaggle datasets version" si ya existe)
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import features as F  # noqa: E402

out = os.path.join(ROOT, "gpu_kaggle", "datos_tabpfn")
os.makedirs(out, exist_ok=True)
for f in range(F.N_FOLDS):
    A, B = F.load_fold(f)
    A.to_parquet(f"{out}/fold{f}_tr.parquet"); B.to_parquet(f"{out}/fold{f}_va.parquet")
A, B = F.load_full()
A.to_parquet(f"{out}/full_tr.parquet"); B.to_parquet(f"{out}/full_te.parquet")
shutil.copy(f"{F.CACHE}/y.npy", f"{out}/y.npy")
print("listo:", sorted(os.listdir(out)))
