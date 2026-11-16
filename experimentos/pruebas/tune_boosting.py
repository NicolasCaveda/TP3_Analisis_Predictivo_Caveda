"""Busqueda de hiperparametros con Optuna (TPE) sobre los mismos 5 folds del resto del trabajo.

Optimiza la log-loss out-of-fold (mas estable que el accuracy para comparar configuraciones) con una sola semilla por
prueba; la mejor configuracion despues se valida con las 3 semillas de las notebooks. El estudio se guarda en
cache/optuna.db, asi que se puede cortar y retomar.

Uso: python tune_boosting.py <lightgbm|xgboost> <n_pruebas>
"""
import sys
import time

import numpy as np
import optuna
from sklearn.metrics import log_loss

import features as F

MODEL, N_TRIALS = sys.argv[1], int(sys.argv[2])
y = F.load_y()
folds = F.cv_folds(y)
DATA = [F.load_fold(f) for f in range(len(folds))]

# configuracion actual de las notebooks, como punto de partida
BASE = {"lightgbm": dict(learning_rate=0.02, n_estimators=1200, num_leaves=63, min_child_samples=10, colsample_bytree=0.3,
                         subsample=0.8, reg_lambda=5.0, reg_alpha=1e-3, min_split_gain=1e-3),
        "xgboost": dict(learning_rate=0.03, n_estimators=1200, max_depth=9, min_child_weight=2.0, colsample_bytree=0.3,
                        subsample=0.8, reg_lambda=3.0, reg_alpha=1e-3, gamma=1e-3)}[MODEL]


def make(p, seed=42):
    if MODEL == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(objective="binary", subsample_freq=1, n_jobs=6, verbose=-1, random_state=seed, **p)
    import xgboost as xgb
    return xgb.XGBClassifier(tree_method="hist", enable_categorical=True, max_cat_to_onehot=64, n_jobs=6, random_state=seed, **p)


def suggest(t):
    p = dict(learning_rate=t.suggest_float("learning_rate", 0.01, 0.06, log=True),
             n_estimators=t.suggest_int("n_estimators", 500, 3000, step=100),
             colsample_bytree=t.suggest_float("colsample_bytree", 0.1, 0.7),
             subsample=t.suggest_float("subsample", 0.5, 1.0),
             reg_lambda=t.suggest_float("reg_lambda", 0.1, 50, log=True),
             reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True))
    if MODEL == "lightgbm":
        p.update(num_leaves=t.suggest_int("num_leaves", 15, 255, log=True),
                 min_child_samples=t.suggest_int("min_child_samples", 5, 200, log=True),
                 min_split_gain=t.suggest_float("min_split_gain", 1e-3, 1.0, log=True))
    else:
        p.update(max_depth=t.suggest_int("max_depth", 4, 12),
                 min_child_weight=t.suggest_float("min_child_weight", 0.5, 50, log=True),
                 gamma=t.suggest_float("gamma", 1e-3, 2.0, log=True))
    return p


def objective(t):
    p, oof, t0 = suggest(t), np.zeros(len(y)), time.time()
    for (a, b), (A, B) in zip(folds, DATA):
        oof[b] = make(p).fit(A, y[a]).predict_proba(B)[:, 1]
    acc = ((oof > .5) == y).mean()
    t.set_user_attr("accuracy", acc)
    print(f"prueba {t.number}: logloss {log_loss(y, oof):.5f} | accuracy {acc:.4f} | {(time.time() - t0) / 60:.1f} min", flush=True)
    return log_loss(y, oof)


study = optuna.create_study(study_name=MODEL, storage=f"sqlite:///{F.CACHE}/optuna.db", load_if_exists=True,
                            direction="minimize", sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=10))
if not study.trials:
    study.enqueue_trial(BASE)
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.optimize(objective, n_trials=N_TRIALS - len(study.trials))
b = study.best_trial
print(f"\nmejor prueba {b.number}: logloss {b.value:.5f} | accuracy {b.user_attrs['accuracy']:.4f}\n{b.params}")
