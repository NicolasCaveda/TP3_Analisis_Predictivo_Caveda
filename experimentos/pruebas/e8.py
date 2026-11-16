"""Vendedor x tanda de carga: TE anidado de seller|minuto, seller|hora, seller|dia de creacion."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from harness import *
R = raw(); n = len(y)
dc = pd.to_datetime(R["date_created"], errors="coerce").iloc[:n].reset_index(drop=True)
s = R["seller_id"].astype(str).iloc[:n].reset_index(drop=True)
K = pd.DataFrame({"sx_min": s + "|" + dc.dt.floor("min").astype(str), "sx_hour": s + "|" + dc.dt.floor("h").astype(str),
                  "sx_day": s + "|" + dc.dt.floor("D").astype(str), "sx_week": s + "|" + dc.dt.to_period("W").astype(str)})
yy = y
for c in K.columns:   # diagnostico: pureza dentro de la tanda vs dentro del vendedor
    g = pd.DataFrame({"k": K[c], "s": s, "y": yy})
    big = g.groupby("k")["y"].transform("size") > 1
    print(c, "filas en tandas >1:", big.mean().round(3), "| tandas puras:", (g[big].groupby("k")["y"].agg(lambda v: v.min() == v.max())).mean().round(3), flush=True)
def fe(cols):
    def f(fold, a, b):
        ta, (tb,) = F.target_encode(K.iloc[a].reset_index(drop=True), y[a], [K.iloc[b].reset_index(drop=True)], cols=cols)
        return ta, tb
    return f
b = base()
for cols in [["sx_min"], ["sx_hour", "sx_day"], ["sx_min", "sx_hour", "sx_day", "sx_week"]]:
    compare(b, run(None, fold_extra=fe(cols), tag=str(cols)), f"+ TE {cols}")
