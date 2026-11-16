"""Idea: las otras publicaciones SIN etiqueta del mismo vendedor (en el fold de validacion / en test) tambien tienen
prediccion. Como los vendedores suelen ser puros, el promedio de esas predicciones es informacion extra.
Nivel 2: LR sobre [logit(ens), promedio leave-one-out del logit de los pares, log(1+k)], CV agrupada por vendedor."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
import features as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
y = F.load_y(); folds = F.cv_folds(y)
z = np.load("kaggle_tabpfn/output_v4/tabpfn.npz")["oof"]
boost = np.mean([np.load(f"cache/oof_{n}.npy") for n in ["lightgbm", "xgboost", "catboost"]], 0)
ens = .5 * z + .5 * boost
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
s = F.load("train_data.jsonlines")["seller_id"].values
L = lg(ens); peer = np.zeros(len(y)); k = np.zeros(len(y)); hist = np.zeros(len(y))
for f, (a, b) in enumerate(folds):
    _, B = F.load_fold(f); hist[b] = np.round(B["ten_seller_id"].values / .8)
    d = pd.DataFrame({"s": s[b], "L": L[b]})
    g = d.groupby("s")["L"]; tot, n = g.transform("sum").values, g.transform("size").values
    k[b] = n - 1; peer[b] = np.where(n > 1, (tot - L[b]) / np.maximum(n - 1, 1), 0)
acc = lambda p: ((p > .5) == y).mean()
print(f"ensamble OOF {acc(ens):.4f}")
for nom, m in [("nuevos (hist 0)", hist == 0), ("hist 1-3", (hist >= 1) & (hist <= 3)), ("hist >3", hist > 3)]:
    print(f"  {nom}: n {m.sum()} | con pares {(m & (k > 0)).mean() / m.mean():.2f} | k medio {k[m].mean():.1f}")
X = np.c_[L, peer, np.log1p(k), peer * (hist == 0), L * (hist == 0), (hist == 0), np.log1p(hist)]
p2 = np.zeros(len(y))
for a, b in GroupKFold(10).split(X, y, s):
    p2[b] = LogisticRegression(C=1.0, max_iter=2000).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
c1, c2 = (ens > .5) == y, (p2 > .5) == y; d = c2.astype(int) - c1
rng = np.random.default_rng(0); bs = np.array([d[rng.integers(0, len(d), len(d))].sum() for _ in range(2000)])
print(f"nivel 2 con pares: {c2.mean():.4f} | filas netas {d.sum()} | IC90 [{np.percentile(bs,5):.0f}, {np.percentile(bs,95):.0f}] | P>0 {(bs>0).mean():.2f}")
for nom, m in [("nuevos", hist == 0), ("1-3", (hist >= 1) & (hist <= 3)), (">3", hist > 3)]:
    print(f"  {nom}: {c1[m].mean():.4f} -> {c2[m].mean():.4f} ({d[m].sum():+d})")
