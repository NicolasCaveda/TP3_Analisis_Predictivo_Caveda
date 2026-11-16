import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from harness import *
R = raw(); n = len(y)
assert np.allclose(DATA[0][1]["log_price"].values, np.log1p(R["price"].clip(lower=0).values[folds[0][1]]))  # alineacion
def dims(d, k):
    try: a, b = d[k].split("x"); return int(a), int(b)
    except Exception: return (np.nan, np.nan)
def feats(l):
    if not l: return [np.nan] * 10
    s = [dims(d, "size") for d in l]; m = [dims(d, "max_size") for d in l]
    ar = np.array([a / b if b else np.nan for a, b in s])
    w0, h0 = m[0]
    sq = np.mean(np.abs(ar - 1) < .02); f43 = np.mean((np.abs(ar - 4 / 3) < .02) | (np.abs(ar - .75) < .02))
    long = [max(a, b) for a, b in m if a == a]
    return [ar[0], sq, f43, w0, h0, max(long) if long else np.nan, min(long) if long else np.nan,
            len(set(s)), np.mean([q.get("quality") == "" for q in l]), np.nanstd(ar)]
cols = ["pic_ar0", "pic_sq", "pic_43", "pic_w0", "pic_h0", "pic_long_max", "pic_long_min", "pic_nsizes", "pic_qempty", "pic_ar_std"]
X = pd.DataFrame(R["pictures"].map(feats).tolist(), columns=cols)
g = X.assign(s=R["seller_id"].values).groupby("s")
X["s_pic_sq"] = g["pic_sq"].transform("mean"); X["s_pic_43"] = g["pic_43"].transform("mean")
X = X.iloc[:n].reset_index(drop=True)
b0 = base()
compare(b0, run(X, tag="fotos"), "fotos (forma y resolucion)")
