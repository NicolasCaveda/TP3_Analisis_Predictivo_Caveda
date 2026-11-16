"""Vecinos dentro del vendedor en un espacio titulo + atributos tabulares (no solo titulo)."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from harness import *
n = len(y)
X = pd.read_parquet(f"{F.CACHE}/Xtr.parquet")
E = np.load(f"{F.CACHE}/emb_e5small.npy")[:n]
ar = X["pic_ar0"].fillna(0).values
Z = np.c_[(X["log_price"] - X["log_price"].mean()) / X["log_price"].std(), (X["initial_quantity"] == 1), X["warranty_null"],
          X["free_shipping"], X["listing_type_id"].astype(str) == "free", X["listing_type_id"].astype(str).str.startswith("gold"),
          np.abs(ar - 1) < .02, (np.abs(ar - 4 / 3) < .02) | (np.abs(ar - .75) < .02), X["shipping_mode"].astype(str) == "me2"].astype("float32")
s = X["seller_id"].astype(str).values
def vecs(wt):
    V = np.c_[E * wt, Z * (1 - wt) / np.sqrt(Z.shape[1]) * 3]
    return (V / np.linalg.norm(V, axis=1, keepdims=True)).astype("float32")
def fe(wt):
    V = vecs(wt)
    def f(fold, a, b):
        A = F.seller_knn(s[a], V[a], s[a], V[a], y[a], loo=True); B = F.seller_knn(s[b], V[b], s[a], V[a], y[a])
        A.columns = B.columns = [c.replace("snn", "snx") for c in A.columns]
        return A, B
    return f
b = base()
for wt in [.5, .3]:
    compare(b, run(None, fold_extra=fe(wt), tag=f"snx w={wt}"), f"+ vecinos titulo+atributos (peso titulo {wt})")
