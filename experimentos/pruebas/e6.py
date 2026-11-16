"""Especialista para vendedores sin historial: LightGBM entrenado solo con filas cuyo vendedor no tiene etiquetas
en la parte de ajuste (ten_seller_id == 0), sin las columnas de target encoding del vendedor. Se combina con el global."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from harness import *
b = base()
spec = np.full(len(y), np.nan); t0 = time.time()
for f, ((a, bb), (A, B)) in enumerate(zip(folds, DATA)):
    ma, mb = A["ten_seller_id"].values == 0, B["ten_seller_id"].values == 0
    drop = [c for c in A.columns if c.startswith(("te_sx", "ten_sx", "te_seller", "ten_seller", "snn"))]
    for params in [P]:
        m = lgb.LGBMClassifier(**{**P, "n_estimators": 800}, random_state=42).fit(A.loc[ma].drop(columns=drop), y[a][ma])
        spec[bb[mb]] = m.predict_proba(B.loc[mb].drop(columns=drop))[:, 1]
    print(f"fold {f}: train esp {ma.sum()} | valid esp {mb.sum()} ({time.time()-t0:.0f}s)", flush=True)
m = ~np.isnan(spec)
print(f"vendedores nuevos ({m.sum()}): global {((b[m] > .5) == y[m]).mean():.4f} | especialista {((spec[m] > .5) == y[m]).mean():.4f}")
for w in [.3, .5, .7, 1.0]:
    p = b.copy(); p[m] = (1 - w) * b[m] + w * spec[m]
    compare(b, p, f"global + especialista peso {w}")
np.save(f"{F.CACHE}/exp_spec.npy", spec)
