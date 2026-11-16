"""Valida la mejor config de Optuna con 3 semillas y la compara dentro del ensamble (sin pisar los archivos actuales)."""
import sys, time, numpy as np, lightgbm as lgb, optuna
sys.path.insert(0, ".")
import features as F
optuna.logging.set_verbosity(optuna.logging.WARNING)
best = optuna.load_study(study_name="lightgbm", storage=f"sqlite:///{F.CACHE}/optuna.db").best_trial.params
P = dict(objective="binary", subsample_freq=1, n_jobs=6, verbose=-1, **best)
y = F.load_y(); folds = F.cv_folds(y); t0 = time.time()
fp = lambda A, ya, B: np.mean([lgb.LGBMClassifier(**P, random_state=s).fit(A, ya).predict_proba(B)[:, 1] for s in [42, 43, 44]], 0)
oof = np.zeros(len(y))
for f, (a, b) in enumerate(folds):
    A, B = F.load_fold(f); oof[b] = fp(A, y[a], B); print(f"fold {f} {time.time()-t0:.0f}s", flush=True)
np.save(f"{F.CACHE}/oof_lightgbm_opt.npy", oof)
old = np.load(f"{F.CACHE}/oof_lightgbm.npy")
acc = lambda p: ((p > .5) == y).mean()
print(f"LightGBM solo: actual {acc(old):.4f} | optuna {acc(oof):.4f}")
z = np.load("kaggle_tabpfn/output_v4/tabpfn.npz")["oof"]
oth = [np.load(f"{F.CACHE}/oof_{n}.npy") for n in ["xgboost", "catboost"]]
e_old = .5 * z + .5 * np.mean([old] + oth, 0); e_new = .5 * z + .5 * np.mean([oof] + oth, 0)
c_old, c_new = (e_old > .5) == y, (e_new > .5) == y
rng = np.random.default_rng(0); d = c_new.astype(int) - c_old
bs = np.array([d[rng.integers(0, len(d), len(d))].sum() for _ in range(2000)])
print(f"ensamble: actual {c_old.mean():.4f} | optuna {c_new.mean():.4f} | filas netas {d.sum()} | P(>0) {(bs > 0).mean():.2f}")
A, B = F.load_full(); np.save(f"{F.CACHE}/test_lightgbm_opt.npy", fp(A, y, B)); print(f"test listo {time.time()-t0:.0f}s")
