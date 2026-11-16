"""Donde se equivoca el ensamble (OOF) y cuanto se podria ganar en cada lugar."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import features as F

y = F.load_y()
z = np.load("kaggle_tabpfn/output_v4/tabpfn.npz")
boost = np.mean([np.load(f"cache/oof_{n}.npy") for n in ["lightgbm", "xgboost", "catboost"]], 0)
ens = 0.5 * z["oof"] + 0.5 * boost
err = (ens > .5) != y
print(f"accuracy OOF {1 - err.mean():.4f} | errores {err.sum()}")
hist = np.zeros(len(y))
for f, (a, b) in enumerate(F.cv_folds(y)):
    _, B = F.load_fold(f); hist[b] = np.round(B["ten_seller_id"].values / 0.8)
tr = F.load("train_data.jsonlines")
seg = pd.cut(hist, [-1, 0, 3, 10, 30, 100, 1e9], labels=["0", "1-3", "4-10", "11-30", "31-100", ">100"])
# pureza del vendedor en el resto del train: fraccion nueva entre sus otras publicaciones (OOF del fold)
df = pd.DataFrame({"seg": seg, "err": err, "conf": np.abs(ens - .5)})
t = df.groupby("seg", observed=True).agg(n=("err", "size"), errores=("err", "sum"), acc=("err", lambda e: 1 - e.mean()))
t["% de errores"] = (100 * t.errores / t.errores.sum()).round(1)
print(t.round(4))
print("\nerrores segun confianza del ensamble |p-0.5|:")
print(df.groupby(pd.cut(df.conf, [0, .1, .2, .3, .4, .5]), observed=True).agg(n=("err", "size"), errores=("err", "sum")).assign(
    acc=lambda d: (1 - d.errores / d.n).round(4)))
# umbral optimo
th = np.arange(.4, .61, .01)
print("\nmejor umbral:", max(((ens > t_) == y).mean() for t_ in th).round(4), "en", th[np.argmax([((ens > t_) == y).mean() for t_ in th])].round(2))
# vendedores con historial: errores en vendedores mixtos vs puros
s = tr["seller_id"].values
mix = pd.Series(y).groupby(s).transform("mean").values
pure = (mix == 0) | (mix == 1)
m = hist > 10
print(f"\nvendedores con >10 en train: errores en vendedores puros {err[m & pure].sum()} | mixtos {err[m & ~pure].sum()} (de {err[m].sum()})")
mixm = m & ~pure
print("  mixtos: n", mixm.sum(), "acc", (1 - err[mixm].mean()).round(4))
