"""Combinaciones vendedor x forma de foto y categoria x forma de foto (target encoding anidado, como SELLER_COMBOS)."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from harness import *
import fotos
R = raw(); n = len(y)
X1 = fotos.basic(R)
ar = X1["pic_ar0"].values
shape = np.select([np.isnan(ar), np.abs(ar - 1) < .02, np.abs(ar - 4 / 3) < .02, np.abs(ar - .75) < .02, ar < 1], ["na", "sq", "43", "34", "vert"], "horiz")
K = pd.DataFrame({"sx_pic": R["seller_id"].astype(str).values + "|" + shape,
                  "cx_pic": R["category_id"].astype(str).values + "|" + shape}).iloc[:n].reset_index(drop=True)
X1 = X1.iloc[:n].reset_index(drop=True)
def fe(cols):
    def f(fold, a, b):
        ta, (tb,) = F.target_encode(K.iloc[a].reset_index(drop=True), y[a], [K.iloc[b].reset_index(drop=True)], cols=cols)
        return ta, tb
    return f
np.save(f"{F.CACHE}/exp_fotos1.npy", p1 := run(X1, tag="fotos"))
for cols in [["sx_pic"], ["cx_pic"], ["sx_pic", "cx_pic"]]:
    compare(p1, run(X1, fold_extra=fe(cols), tag=str(cols)), f"fotos + TE {cols}")
