"""Banco de pruebas: LightGBM (config notebook, 1 o 3 semillas) sobre los folds actuales + columnas extra.
Uso desde otro script: base = run(None); new = run(extra_df); compare(base, new)."""
import sys, os, time, numpy as np, pandas as pd, lightgbm as lgb
DESARROLLO = os.environ.get("TP3_DESARROLLO", ".")   # carpeta del proyecto de desarrollo (features.py, cache/)
sys.path.insert(0, DESARROLLO)
os.chdir(DESARROLLO)
import features as F
y = F.load_y(); folds = F.cv_folds(y)
P = dict(objective="binary", learning_rate=0.02, n_estimators=1200, num_leaves=63, min_child_samples=10,
         colsample_bytree=0.3, subsample=0.8, subsample_freq=1, reg_lambda=5, n_jobs=6, verbose=-1)
DATA = [F.load_fold(f) for f in range(5)]
TR = None
def raw():
    global TR
    if TR is None:
        tr, te = F.load("train_data.jsonlines"), F.load("test_data.jsonlines")
        TR = pd.concat([tr.drop(columns=["condition"]), te], ignore_index=True)
    return TR
def run(extra=None, seeds=(42,), fold_extra=None, tag=""):
    """extra: DataFrame alineado con train (70k filas), sin etiqueta. fold_extra(f, a, b) -> (EA, EB) para features con etiqueta."""
    oof = np.zeros(len(y)); t0 = time.time()
    for f, ((a, b), (A, B)) in enumerate(zip(folds, DATA)):
        A, B = A.copy(), B.copy()
        if extra is not None:
            for c in extra.columns: A[c], B[c] = extra[c].values[a], extra[c].values[b]
        if fold_extra is not None:
            EA, EB = fold_extra(f, a, b)
            for c in EA.columns: A[c], B[c] = EA[c].values, EB[c].values
        oof[b] = np.mean([lgb.LGBMClassifier(**P, random_state=s).fit(A, y[a]).predict_proba(B)[:, 1] for s in seeds], 0)
    print(f"{tag} acc {((oof > .5) == y).mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
    return oof
def compare(p0, p1, tag=""):
    c0, c1 = (p0 > .5) == y, (p1 > .5) == y; d = c1.astype(int) - c0
    rng = np.random.default_rng(0); bs = np.array([d[rng.integers(0, len(d), len(d))].sum() for _ in range(2000)])
    pf = [d[b].sum() for a, b in folds]
    print(f"{tag}: {c0.mean():.4f} -> {c1.mean():.4f} | netas {d.sum():+d} | por fold {pf} | P>0 {(bs > 0).mean():.2f}", flush=True)
def base(seeds=(42,)):
    p = f"{F.CACHE}/exp_base_{len(seeds)}.npy"
    if os.path.exists(p): return np.load(p)
    o = run(None, seeds, tag="base"); np.save(p, o); return o
