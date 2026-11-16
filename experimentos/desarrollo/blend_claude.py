"""Combina el ensamble (TabPFN + boostings) con la opinion de Claude Opus donde el ensamble sabe poco del vendedor o
donde el vendedor es mixto, que es donde Claude aporta. Para el resto queda el ensamble.

Tramos (en train, "historial" = publicaciones del vendedor en la parte de entrenamiento del fold; en test, en todo train):
  - vendedor nuevo (0 publicaciones): Claude ve solo la publicacion (puntajes v1)
  - vendedor con 1 a 3 publicaciones: idem
  - vendedor mixto (>= 4 publicaciones, entre 20% y 80% nuevas): Claude ve ademas el historial etiquetado del vendedor
    (sus publicaciones mas parecidas con la condicion real, puntajes v3)
En cada tramo la combinacion es una regresion logistica sobre [logit(ensamble), logit(Claude)], ajustada con las
publicaciones de train del tramo que evaluo Claude. Si faltan puntajes de test de un tramo, ese tramo queda con el ensamble.

Uso: python blend_claude.py <carpeta_oof_boostings> <npz_tabpfn> <nombre_csv> [llm_mixtos]
  llm_mixtos (tramo mixto): opus (por defecto, v8), fable, o opus+fable (las dos opiniones como entradas de la regresion)
"""
import glob
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import features as F

BOOST_DIR, TABPFN, NAME = sys.argv[1], sys.argv[2], sys.argv[3]
LLM_MIXTOS = sys.argv[4] if len(sys.argv) > 4 else "opus"
VERS_MIXTOS = {"opus": ["claude-opus-5-5-v3"], "fable": ["claude-fable-5-2-v3"],
               "opus+fable": ["claude-opus-5-5-v3", "claude-fable-5-2-v3"]}[LLM_MIXTOS]
N = ["lightgbm", "xgboost", "catboost"]
lg = lambda q: np.log(np.clip(q, .01, .99) / (1 - np.clip(q, .01, .99)))
acc = lambda p, t: ((p > .5) == t).mean()


def claude(version, parts):
    P = {}
    for part in parts:
        for f in glob.glob(f"{F.CACHE}/llm/{version}/{part}/*.json"):
            P.update({int(k): v / 100 for k, v in json.load(open(f, encoding="utf8"))["p"].items()})
    return P


y = F.load_y()
z = np.load(TABPFN)
ens_oof = 0.5 * z["oof"] + 0.5 * np.mean([np.load(f"{BOOST_DIR}/oof_{n}.npy") for n in N], 0)
ens_tst = 0.5 * z["test"] + 0.5 * np.mean([np.load(f"{BOOST_DIR}/test_{n}.npy") for n in N], 0)

# historial del vendedor y fraccion de nuevas en el: en train con la parte de entrenamiento de cada fold, en test con todo train
seller = F.load("train_data.jsonlines")["seller_id"].values
seller_te = F.load("test_data.jsonlines")["seller_id"].values
hist, pur = np.zeros(len(y)), np.full(len(y), np.nan)
for a, b in F.cv_folds(y):
    g = pd.Series(y[a]).groupby(seller[a]).agg(["mean", "size"])
    hist[b] = pd.Series(seller[b]).map(g["size"]).fillna(0).values
    pur[b] = pd.Series(seller[b]).map(g["mean"]).values
g = pd.Series(y).groupby(seller).agg(["mean", "size"])
hist_te = pd.Series(seller_te).map(g["size"]).fillna(0).values
pur_te = pd.Series(seller_te).map(g["mean"]).values

mixto = lambda h, p: (h >= 4) & (p >= .2) & (p <= .8)
TRAMOS = [  # nombre, filtro (historial, pureza), versiones de Claude, partes de train, parte de test
    ("vendedores nuevos", lambda h, p: h == 0, ["claude-opus-5-5"], ["piloto", "etapa1", "nuevos_extra"], "test_nuevos"),
    ("vendedores con 1-3 publicaciones", lambda h, p: (h >= 1) & (h <= 3), ["claude-opus-5-5"], ["piloto", "etapa1"], "test_pocos"),
    ("vendedores mixtos (con historial)", mixto, VERS_MIXTOS, ["mixtos_train", "mixtos_train2"], "test_mixtos"),
]

proba = ens_tst.copy()
for nombre, filtro, versiones, partes, parte_test in TRAMOS:
    Pts = [claude(v, [parte_test]) for v in versiones]
    rows = np.flatnonzero(filtro(hist_te, pur_te))
    if not all(len(y) + r in Pt for Pt in Pts for r in rows):
        print(f"{nombre}: faltan puntajes de Claude en test ({parte_test}), el tramo queda con el ensamble")
        continue
    # --- ajuste y validacion en train
    Ps = [claude(v, partes) for v in versiones]
    idx = np.array(sorted(set.intersection(*[set(P) for P in Ps])), dtype=int)
    tr = idx[filtro(hist[idx], pur[idx])]
    Z = np.c_[lg(ens_oof[tr]), *[lg(np.array([P[i] for i in tr])) for P in Ps]]
    yy = y[tr]
    q = np.mean([cross_val_predict(LogisticRegression(), Z, yy, cv=StratifiedKFold(10, shuffle=True, random_state=s),
                                   method="predict_proba")[:, 1] for s in range(10)], 0)
    m = LogisticRegression().fit(Z, yy)
    print(f"{nombre}: {len(tr)} en train | ensamble {acc(ens_oof[tr], yy):.4f} -> con Claude {acc(q, yy):.4f} | "
          f"pesos (logit) ensamble {m.coef_[0][0]:.3f}, Claude {np.round(m.coef_[0][1:], 3)}, intercepto {m.intercept_[0]:.3f}")
    # --- aplicacion a test
    proba[rows] = m.predict_proba(np.c_[lg(ens_tst[rows]), *[lg(np.array([Pt[len(y) + r] for r in rows])) for Pt in Pts]])[:, 1]
    print(f"  test: {len(rows)} publicaciones | cambian {((proba[rows] > .5) != (ens_tst[rows] > .5)).sum()} predicciones")
print(f"accuracy OOF del ensamble (sin Claude): {acc(ens_oof, y):.4f}")
F.save_submission(proba, NAME)
np.save(f"{F.CACHE}/test_{NAME}.npy", proba)
