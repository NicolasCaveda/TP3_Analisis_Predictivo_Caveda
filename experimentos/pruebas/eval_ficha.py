"""LightGBM (config de la notebook, 3 semillas) con la feature txt_ficha vs el actual, y efecto en el ensamble."""
import sys, time, numpy as np, lightgbm as lgb
sys.path.insert(0, ".")
import features as F
y = F.load_y(); folds = F.cv_folds(y); acc = lambda p: ((p > .5) == y).mean()
for c, d in F.TRANSFORMERS.items():
    z = np.load(f"{d}/output/bert_level1.npz"); print(f"{c} solo: {acc(1 / (1 + np.exp(-z['oof'])) if z['oof'].min() < 0 else z['oof']):.4f}")
P = dict(objective="binary", learning_rate=0.02, n_estimators=1200, num_leaves=63, min_child_samples=10,
         colsample_bytree=0.3, subsample=0.8, subsample_freq=1, reg_lambda=5, n_jobs=6, verbose=-1)
fp = lambda A, ya, B: np.mean([lgb.LGBMClassifier(**P, random_state=s).fit(A, ya).predict_proba(B)[:, 1] for s in [42, 43, 44]], 0)
oof = np.zeros(len(y)); t0 = time.time()
for f, (a, b) in enumerate(folds):
    A, B = F.load_fold(f); assert "txt_ficha" in A
    oof[b] = fp(A, y[a], B); print(f"fold {f}: {((oof[b] > .5) == y[b]).mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
np.save(f"{F.CACHE}/oof_lightgbm_ficha.npy", oof)
old = np.load(f"{F.CACHE}/oof_lightgbm.npy")
rng = np.random.default_rng(0)
def comp(nom, p_old, p_new):
    c1, c2 = (p_old > .5) == y, (p_new > .5) == y; d = c2.astype(int) - c1
    bs = np.array([d[rng.integers(0, len(d), len(d))].sum() for _ in range(2000)])
    print(f"{nom}: {c1.mean():.4f} -> {c2.mean():.4f} | netas {d.sum()} | IC90 [{np.percentile(bs,5):.0f}, {np.percentile(bs,95):.0f}] | P>0 {(bs>0).mean():.2f}")
comp("LightGBM", old, oof)
z = np.load("kaggle_tabpfn/output_v4/tabpfn.npz")["oof"]
xc = [np.load(f"{F.CACHE}/oof_{n}.npy") for n in ["xgboost", "catboost"]]
e_old = .5 * z + .5 * np.mean([old] + xc, 0)
comp("ensamble (solo LightGBM con ficha)", e_old, .5 * z + .5 * np.mean([oof] + xc, 0))
for w in [.33, .5, .67]:
    comp(f"ensamble con LGB-ficha {w:.0%} de la parte boosting", e_old, .5 * z + .5 * ((1 - w) * np.mean([old] + xc, 0) + w * oof))
hist = np.zeros(len(y))
for f, (a, b) in enumerate(folds):
    _, B = F.load_fold(f); hist[b] = np.round(B["ten_seller_id"].values / .8)
for nom, m in [("nuevos", hist == 0), ("1-3", (hist >= 1) & (hist <= 3)), (">3", hist > 3)]:
    print(f"  LightGBM {nom}: {acc(old)*0 + ((old[m] > .5) == y[m]).mean():.4f} -> {((oof[m] > .5) == y[m]).mean():.4f}")
