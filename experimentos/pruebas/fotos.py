"""Variables de forma/resolucion de fotos (sin etiqueta), para las 100k filas (train+test)."""
import numpy as np, pandas as pd
def dims(d, k):
    try: a, b = d[k].split("x"); return int(a), int(b)
    except Exception: return (np.nan, np.nan)
def _f(l):
    if not l: return [np.nan] * 10
    s = [dims(d, "size") for d in l]; m = [dims(d, "max_size") for d in l]
    ar = np.array([a / b if b else np.nan for a, b in s])
    w0, h0 = m[0]
    sq = np.mean(np.abs(ar - 1) < .02); f43 = np.mean((np.abs(ar - 4 / 3) < .02) | (np.abs(ar - .75) < .02))
    long = [max(a, b) for a, b in m if a == a]
    return [ar[0], sq, f43, w0, h0, max(long) if long else np.nan, min(long) if long else np.nan,
            len(set(s)), np.mean([q.get("quality") == "" for q in l]), np.nanstd(ar)]
COLS = ["pic_ar0", "pic_sq", "pic_43", "pic_w0", "pic_h0", "pic_long_max", "pic_long_min", "pic_nsizes", "pic_qempty", "pic_ar_std"]
def basic(R):
    X = pd.DataFrame(R["pictures"].map(_f).tolist(), columns=COLS)
    g = X.assign(s=R["seller_id"].values).groupby("s")
    X["s_pic_sq"] = g["pic_sq"].transform("mean"); X["s_pic_43"] = g["pic_43"].transform("mean")
    return X
def _g(l):
    if not l: return [np.nan] * 6
    s = [dims(d, "size") for d in l]; m = [dims(d, "max_size") for d in l]
    ar = np.array([a / b if b else np.nan for a, b in s])
    long = np.array([max(a, b) for a, b in m if a == a] or [np.nan])
    return [np.mean(np.abs(ar - .75) < .02), np.mean(np.abs(ar - 4 / 3) < .02), np.mean(long >= 1200),
            np.mean(long < 500), float(s[0] == (500, 500)), np.mean([a != c or b != d for (a, b), (c, d) in zip(s, m)])]
def extended(R):
    X = basic(R)
    E = pd.DataFrame(R["pictures"].map(_g).tolist(), columns=["pic_34", "pic_43h", "pic_hires", "pic_lowres", "pic_500sq", "pic_resized"])
    X = pd.concat([X, E], axis=1)
    k = X.assign(s=R["seller_id"].values, c=R["category_id"].values)
    for c in ["pic_hires", "pic_lowres", "pic_long_max"]: X["s_" + c] = k.groupby("s")[c].transform("mean")
    for c in ["pic_sq", "pic_43", "pic_hires"]: X["c_" + c] = k.groupby("c")[c].transform("mean")
    X["pic_sq_vs_cat"] = X["pic_sq"] - X["c_pic_sq"]; X["pic_sq_vs_seller"] = X["pic_sq"] - X["s_pic_sq"]
    return X
