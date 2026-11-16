"""Target encoding de vendedor x combinacion de varios atributos (anidado como SELLER_COMBOS)."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from harness import *
X = pd.read_parquet(f"{F.CACHE}/Xtr.parquet")
ar = X["pic_ar0"].values
shape = np.select([np.isnan(ar), np.abs(ar - 1) < .02, (np.abs(ar - 4 / 3) < .02) | (np.abs(ar - .75) < .02)], ["na", "sq", "43"], "otra")
s = X["seller_id"].astype(str)
lt, iq1, wn, sh = X["listing_type_id"].astype(str), (X["initial_quantity"] == 1).astype(int).astype(str), X["warranty_null"].astype(str), X["shipping_mode"].astype(str) + X["free_shipping"].astype(str)
K = pd.DataFrame({"sx_c3": s + "|" + lt + "|" + iq1 + "|" + wn,
                  "sx_c5": s + "|" + lt + "|" + iq1 + "|" + wn + "|" + sh + "|" + shape,
                  "sx_shape": s + "|" + shape + "|" + iq1})
def fe(cols):
    def f(fold, a, b):
        ta, (tb,) = F.target_encode(K.iloc[a].reset_index(drop=True), y[a], [K.iloc[b].reset_index(drop=True)], cols=cols)
        return ta, tb
    return f
b = base()
for cols in [["sx_c3"], ["sx_c5"], ["sx_c3", "sx_c5", "sx_shape"]]:
    compare(b, run(None, fold_extra=fe(cols), tag=str(cols)), f"+ TE {cols}")
