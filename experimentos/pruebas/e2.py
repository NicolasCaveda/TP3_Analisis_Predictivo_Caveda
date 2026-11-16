import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from harness import *
import fotos
R = raw(); n = len(y)
which = sys.argv[1]
if which == "conf3":
    X = fotos.basic(R).iloc[:n]
    b3 = np.load(f"{F.CACHE}/oof_lightgbm.npy")   # mismo modelo, 3 semillas, sin fotos
    p = run(X, seeds=(42, 43, 44), tag="fotos 3 semillas"); np.save(f"{F.CACHE}/exp_fotos3.npy", p)
    compare(b3, p, "LightGBM 3 semillas: fotos basicas")
    z = np.load("kaggle_tabpfn/output_v4/tabpfn.npz")["oof"]; xc = [np.load(f"{F.CACHE}/oof_{m}.npy") for m in ["xgboost", "catboost"]]
    compare(.5 * z + .5 * np.mean([b3] + xc, 0), .5 * z + .5 * np.mean([p] + xc, 0), "ensamble (solo LGB con fotos)")
else:
    X1 = fotos.basic(R).iloc[:n]; X2 = fotos.extended(R).iloc[:n]
    p1 = run(X1, tag="basic"); p2 = run(X2, tag="extended")
    compare(p1, p2, "extendidas vs basicas"); compare(base(), p2, "extendidas vs base")
