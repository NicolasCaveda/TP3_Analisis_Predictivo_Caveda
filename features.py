"""Feature engineering del modelo final del TP3 (MercadoLibre: nuevo vs usado)."""
import json
import os
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

SEED = 42
TARGET = "condition"
ROOT = os.path.dirname(os.path.abspath(__file__))
# train_data.jsonlines y test_data.jsonlines (de la competencia) van en datos/; se puede cambiar con TP3_DATOS
DATOS = os.environ.get("TP3_DATOS", os.path.join(ROOT, "datos"))
CATEGORIES_PATH = os.path.join(ROOT, "datos", "categories.json")

CAT_LOW = ["listing_type_id", "buying_mode", "status", "shipping_mode", "state", "currency_id", "cat_l1", "cat_l2"]
# combinaciones vendedor x atributo: un vendedor que publica nuevos y usados suele diferenciarlos por tipo de
# publicacion, stock, garantia, envio o precio; el target encoding de cada combinacion captura eso
SELLER_COMBOS = ["sx_lt", "sx_iq1", "sx_wnull", "sx_ship", "sx_price", "sx_tusado"]
CAT_HIGH = ["seller_id", "category_id", "city", "warranty_norm", "seller_cat"] + SELLER_COMBOS

TITLE_PATTERNS = {
    "t_nuevo": r"\bnuev[oa]s?\b",
    "t_usado": r"\busad[oa]s?\b",
    "t_antiguo": r"\bantigu|\bvintage\b|\bretro\b|\bcoleccion",
    "t_estado": r"\bimpecable|\bexcelente|\bestado\b|\bjoya\b|\bpoco uso\b|\bsin uso\b",
    "t_original": r"\boriginal",
    "t_oferta": r"\boferta|\bpromo|\bliquid|\bmayor",
    "t_envio": r"\benvio|\bgratis\b|\bcuotas\b",
    "t_pack": r"\bx ?\d+\b|\bpack\b|\bkit\b|\bset\b|\bcombo\b|\blote\b",
    "t_libro": r"\blibro|\bedit|\bedicion|\btomo\b|\brevista",
    "t_garantia": r"\bgarantia",
    "t_caja": r"\ben caja\b|\bcerrad|\bsellad|\bblister",
}


def load(path):
    return pd.read_json(os.path.join(DATOS, path), lines=True, convert_dates=False)


def norm_text(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip()


def _days(s):
    """Serie de fechas (ISO o epoch ms) -> dias desde epoch (float)."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("float64") / 86_400_000.0
    d = pd.to_datetime(s, utc=True, errors="coerce")
    return (d - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(days=1)


def _pic_month(pid):
    """'19391-MLA8021791613_092014' -> meses desde el anio 0 (2014*12+9). NaN si el formato es viejo."""
    m = re.search(r"_(\d{2})(\d{4})(?:-|$)", pid or "")
    if not m:
        return np.nan
    mm, yy = int(m.group(1)), int(m.group(2))
    if not (1 <= mm <= 12 and 2005 <= yy <= 2016):
        return np.nan
    return yy * 12 + mm


def _dims(d, key):
    try:
        a, b = d[key].split("x")
        return int(a), int(b)
    except Exception:
        return np.nan, np.nan


PIC_SHAPE_COLS = ["pic_ar0", "pic_sq", "pic_43", "pic_w0", "pic_h0", "pic_long_max", "pic_long_min", "pic_nsizes",
                  "pic_qempty", "pic_ar_std"]


def _pic_shape(l):
    """Relacion de aspecto de la primera foto, fraccion de fotos cuadradas y 4:3/3:4, resolucion original, etc."""
    if not l:
        return [np.nan] * len(PIC_SHAPE_COLS)
    s = [_dims(d, "size") for d in l]
    m = [_dims(d, "max_size") for d in l]
    ar = np.array([a / b if b else np.nan for a, b in s])
    long = [max(a, b) for a, b in m if a == a]
    return [ar[0], np.mean(np.abs(ar - 1) < .02), np.mean((np.abs(ar - 4 / 3) < .02) | (np.abs(ar - .75) < .02)),
            m[0][0], m[0][1], max(long) if long else np.nan, min(long) if long else np.nan, len(set(s)),
            np.mean([d.get("quality") == "" for d in l]), np.nanstd(ar)]


def _kms(l):
    """Kilometros declarados (atributo de autos)."""
    for a in l:
        if (a.get("id") or "").endswith("KMTS"):
            try:
                return float(re.sub(r"[^\d.]", "", a.get("value_name") or "") or "nan")
            except Exception:
                return np.nan
    return np.nan


def row_features(df):
    """Features que dependen solo de la fila (sin target ni agregados)."""
    X = pd.DataFrame(index=df.index)

    # --- precio y cantidades
    p = df["price"].astype(float)
    X["price"] = p
    X["log_price"] = np.log1p(p.clip(lower=0))
    X["base_price_diff"] = df["base_price"].astype(float) - p
    X["price_cents"] = (p * 100).round() % 100
    X["price_has_cents"] = (X["price_cents"] != 0).astype(int)
    X["price_mod10"] = p.round() % 10
    X["price_mod100"] = p.round() % 100
    X["price_round100"] = (p % 100 == 0).astype(int)
    X["has_original_price"] = df["original_price"].notna().astype(int)
    for c in ["initial_quantity", "sold_quantity", "available_quantity"]:
        X[c] = df[c].astype(float)
    X["sold_ratio"] = X["sold_quantity"] / X["initial_quantity"].clip(lower=1)
    X["avail_ratio"] = X["available_quantity"] / X["initial_quantity"].clip(lower=1)
    X["qty_gap"] = X["initial_quantity"] - X["available_quantity"] - X["sold_quantity"]

    # --- categoricas simples
    for c in ["listing_type_id", "buying_mode", "status", "currency_id"]:
        X[c] = df[c].astype(str)
    X["accepts_mercadopago"] = df["accepts_mercadopago"].astype(int)
    X["automatic_relist"] = df["automatic_relist"].astype(int)
    X["has_video"] = df["video_id"].notna().astype(int)
    X["official_store"] = df["official_store_id"].notna().astype(int)
    X["has_catalog"] = df["catalog_product_id"].notna().astype(int)
    X["has_parent"] = df["parent_item_id"].notna().astype(int)
    X["n_sub_status"] = df["sub_status"].str.len()
    X["n_deals"] = df["deal_ids"].str.len()
    X["n_descriptions"] = df["descriptions"].str.len()

    # --- vendedor / ubicacion / categoria
    X["seller_id"] = df["seller_id"].astype(str)
    X["category_id"] = df["category_id"].astype(str)
    addr = df["seller_address"]
    X["state"] = addr.map(lambda a: a["state"]["id"] or "NA")
    X["city"] = addr.map(lambda a: norm_text(a["city"]["name"]) or "NA")
    X["seller_cat"] = X["seller_id"] + "_" + X["category_id"]
    X["item_num"] = df["id"].str[3:].astype(float)
    X["cat_num"] = df["category_id"].str[3:].astype(float)
    # arbol de categorias (dato externo: API publica de MercadoLibre, ver datos/fetch_categories.py).
    # Solo ~25% de las filas tienen una categoria que sigue existiendo hoy; el resto queda en "NA".
    cats = json.load(open(CATEGORIES_PATH, encoding="utf8")) if os.path.exists(CATEGORIES_PATH) else {}
    path = df["category_id"].map(lambda c: (cats.get(c) or {}).get("path") or [])
    X["cat_l1"] = path.map(lambda p: p[0][0] if len(p) > 0 else "NA")
    X["cat_l2"] = path.map(lambda p: p[1][0] if len(p) > 1 else "NA")
    X["cat_depth"] = path.str.len()

    # --- garantia
    w = df["warranty"].map(norm_text)
    X["warranty_norm"] = w.where(w != "", "NA")
    X["warranty_null"] = (w == "").astype(int)
    X["warranty_len"] = w.str.len()
    X["w_sin"] = w.str.contains(r"\bsin\b|\bno\b", regex=True).astype(int)
    X["w_tiempo"] = w.str.contains(r"\bmes|\bano|\bdia", regex=True).astype(int)
    X["w_fabrica"] = w.str.contains(r"fabrica|oficial|fabricante", regex=True).astype(int)
    X["w_reputacion"] = w.str.contains(r"reputacion|calificacion", regex=True).astype(int)
    X["w_digits"] = w.str.contains(r"\d", regex=True).astype(int)

    # --- envio
    sh = df["shipping"]
    X["local_pick_up"] = sh.map(lambda s: int(bool(s.get("local_pick_up"))))
    X["free_shipping"] = sh.map(lambda s: int(bool(s.get("free_shipping"))))
    X["shipping_mode"] = sh.map(lambda s: s.get("mode") or "NA")
    X["ship_methods_null"] = sh.map(lambda s: int(s.get("methods") is None))
    X["ship_n_methods"] = sh.map(lambda s: len(s.get("free_methods") or []))
    X["ship_n_tags"] = sh.map(lambda s: len(s.get("tags") or []))
    X["ship_dimensions"] = sh.map(lambda s: int(s.get("dimensions") is not None))

    # --- medios de pago fuera de MercadoPago y tags
    pm = df["non_mercado_pago_payment_methods"].map(lambda l: {d["id"] for d in l})
    X["n_pay_methods"] = pm.str.len()
    for pid in ["MLATB", "MLAWC", "MLAMO", "MLAOT", "MLAVS", "MLAMC", "MLAAM", "MLAVE", "MLADC", "MLACD", "MLABC", "MLAMS", "MLAWT", "MLAMP"]:
        X["pay_" + pid] = pm.map(lambda s, pid=pid: int(pid in s))
    tg = df["tags"].map(set)
    for t in ["dragged_bids_and_visits", "good_quality_thumbnail", "dragged_visits", "free_relist", "poor_quality_thumbnail"]:
        X["tag_" + t] = tg.map(lambda s, t=t: int(t in s))

    # --- variaciones / atributos
    X["n_variations"] = df["variations"].str.len()
    X["n_attributes"] = df["attributes"].str.len()

    # --- fechas
    created, updated = _days(df["date_created"]), _days(df["last_updated"])
    start, stop = _days(df["start_time"]), _days(df["stop_time"])
    X["created_day"] = created
    X["start_day"] = start
    X["duration_days"] = stop - start
    X["updated_minus_created"] = updated - created
    X["created_minus_start"] = created - start
    X["updated_minus_start"] = updated - start
    X["created_hour"] = (created % 1) * 24
    X["start_hour"] = (start % 1) * 24
    X["start_dow"] = np.floor(start) % 7

    # --- fotos: cantidad, tamanio y antiguedad (el id de la foto trae mes/anio de subida)
    pics = df["pictures"]
    X["n_pictures"] = pics.str.len()

    def _area(l, key):
        out = []
        for d in l:
            try:
                a, b = d[key].split("x")
                out.append(int(a) * int(b))
            except Exception:
                pass
        return max(out) if out else np.nan

    X["pic_max_area"] = pics.map(lambda l: _area(l, "max_size"))
    X["pic_area"] = pics.map(lambda l: _area(l, "size"))
    pm_ = pics.map(lambda l: [_pic_month(d.get("id")) for d in l])
    X["pic_month_min"] = pm_.map(lambda l: np.nanmin(l) if len(l) and not np.all(np.isnan(l)) else np.nan)
    X["pic_month_max"] = pm_.map(lambda l: np.nanmax(l) if len(l) and not np.all(np.isnan(l)) else np.nan)
    X["pic_old_format"] = pm_.map(lambda l: int(len(l) > 0 and np.isnan(l[0])))
    start_month = pd.to_datetime(start, unit="D")
    start_month = start_month.dt.year * 12 + start_month.dt.month
    X["pic_age_months"] = start_month - X["pic_month_min"]
    X["pic_span_months"] = X["pic_month_max"] - X["pic_month_min"]
    item_num = df["id"].str[3:]
    X["thumb_other_item"] = [int(bool(t) and n not in t) for t, n in zip(df["thumbnail"], item_num)]
    X["pic_n_other_item"] = [sum(n not in (d.get("id") or "") for d in l) for l, n in zip(pics, item_num)]

    # --- forma y resolucion de las fotos: las de catalogo suelen ser cuadradas (86% nuevos) y las de camara o
    # celular 4:3 o 3:4 (~40% nuevos); el area sola no distingue la forma
    X = pd.concat([X, pd.DataFrame(pics.map(_pic_shape).tolist(), columns=PIC_SHAPE_COLS, index=X.index)], axis=1)
    X["pic_pref_len"] = pics.map(lambda l: len(l[0]["id"].split("-")[0]) if l and "-" in (l[0].get("id") or "") else -1)
    X["pic_pref_len_mean"] = pics.map(lambda l: np.mean([len(d["id"].split("-")[0]) for d in l if "-" in (d.get("id") or "")]) if l else np.nan)

    # --- contenido de atributos y variaciones (antes solo se contaban)
    at = df["attributes"].map(lambda l: {a.get("id") or "" for a in l})
    X["at_part"] = at.map(lambda s: int(bool(s & {"PART_NUMBER", "MLA-PART_NUMBER"})))     # repuestos: 97% nuevos
    X["at_auto"] = at.map(lambda s: int(any(i.startswith(("MLA1743", "MLA1744")) for i in s)))  # autos: 7% nuevos
    X["at_inmueble"] = at.map(lambda s: int(any(i.startswith("MLA1459") for i in s)))
    X["at_ropa"] = at.map(lambda s: int(bool(s & {"GENDER", "Season"})))
    X["at_kms"] = df["attributes"].map(_kms)
    va = df["variations"]
    X["var_scf"] = va.map(lambda l: int(any(v.get("seller_custom_field") for v in l)))
    X["var_ncolor2"] = va.map(lambda l: int(any(a.get("name") == "Color Secundario" for v in l for a in v["attribute_combinations"])))
    X["var_talle"] = va.map(lambda l: int(any(a.get("name") == "Talle" for v in l for a in v["attribute_combinations"])))
    X["var_avail"] = va.map(lambda l: sum(v.get("available_quantity") or 0 for v in l) if l else np.nan)
    X["var_npics"] = va.map(lambda l: np.mean([len(v.get("picture_ids") or []) for v in l]) if l else np.nan)
    X["var_price_spread"] = va.map(lambda l: (max(v["price"] for v in l) - min(v["price"] for v in l)) if l else np.nan)

    # --- titulo
    t_raw = df["title"].fillna("")
    t = t_raw.map(norm_text)
    X["title_len"] = t.str.len()
    X["title_words"] = t.str.split().str.len()
    X["title_upper_ratio"] = t_raw.map(lambda s: sum(c.isupper() for c in s) / max(len(s), 1))
    X["title_digit_ratio"] = t_raw.map(lambda s: sum(c.isdigit() for c in s) / max(len(s), 1))
    X["title_punct"] = t_raw.map(lambda s: sum(not c.isalnum() and not c.isspace() for c in s))
    for name, pat in TITLE_PATTERNS.items():
        X[name] = t.str.contains(pat, regex=True).astype(int)
    return X


def group_features(X):
    """Agregados NO supervisados por vendedor / categoria. Se calculan sobre train+test juntos
    (no usan la etiqueta), lo que da informacion tambien para vendedores que no estan en train."""
    X = X.copy()
    X["iq1"] = (X["initial_quantity"] == 1).astype(float)
    X["is_free"] = (X["listing_type_id"] == "free").astype(float)
    g = X.groupby("seller_id")
    X["s_n"] = g["price"].transform("size")
    X["s_ncat"] = g["category_id"].transform("nunique")
    X["s_cat_ratio"] = X["s_ncat"] / X["s_n"]
    X["s_price_med"] = g["log_price"].transform("median")
    X["s_price_std"] = g["log_price"].transform("std")
    X["s_iq1"] = g["iq1"].transform("mean")
    X["s_iq_mean"] = g["initial_quantity"].transform("mean")
    X["s_sold_mean"] = g["sold_quantity"].transform("mean")
    X["s_free"] = g["is_free"].transform("mean")
    X["s_wnull"] = g["warranty_null"].transform("mean")
    X["s_relist"] = g["automatic_relist"].transform("mean")
    X["s_npic"] = g["n_pictures"].transform("mean")
    X["s_cents"] = g["price_has_cents"].transform("mean")
    X["s_title_len"] = g["title_len"].transform("mean")
    X["s_pic_age"] = g["pic_age_months"].transform("mean")
    X["s_duration"] = g["duration_days"].transform("mean")
    X["s_tnuevo"] = g["t_nuevo"].transform("mean")
    X["s_tusado"] = g["t_usado"].transform("mean")
    X["s_start_span"] = g["start_day"].transform("max") - g["start_day"].transform("min")
    X["s_pic_sq"] = g["pic_sq"].transform("mean")
    X["s_pic_43"] = g["pic_43"].transform("mean")
    X["s_pic_pref_len"] = g["pic_pref_len_mean"].transform("mean")
    X["price_vs_seller"] = X["log_price"] - X["s_price_med"]
    c = X.groupby("category_id")
    X["c_n"] = c["price"].transform("size")
    X["c_nseller"] = c["seller_id"].transform("nunique")
    X["c_price_med"] = c["log_price"].transform("median")
    X["c_iq1"] = c["iq1"].transform("mean")
    X["c_free"] = c["is_free"].transform("mean")
    X["c_sold_mean"] = c["sold_quantity"].transform("mean")
    X["c_wnull"] = c["warranty_null"].transform("mean")
    X["price_vs_cat"] = X["log_price"] - X["c_price_med"]
    X["sc_n"] = X.groupby("seller_cat")["price"].transform("size")
    X["city_n"] = X.groupby("city")["price"].transform("size")
    X["w_n"] = X.groupby("warranty_norm")["price"].transform("size")
    s = X["seller_id"]
    X["sx_lt"] = s + "|" + X["listing_type_id"].astype(str)
    X["sx_iq1"] = s + "|" + X["iq1"].astype(int).astype(str)
    X["sx_wnull"] = s + "|" + X["warranty_null"].astype(str)
    X["sx_ship"] = s + "|" + X["shipping_mode"].astype(str) + X["free_shipping"].astype(str)
    X["sx_price"] = s + "|" + pd.qcut(X["log_price"], 10, labels=False, duplicates="drop").astype(str)
    X["sx_tusado"] = s + "|" + X["t_usado"].astype(str) + X["t_nuevo"].astype(str)
    return X.drop(columns=["iq1", "is_free"])


def build(train, test):
    """Devuelve (X_train, X_test) con features de fila + agregados no supervisados."""
    full = pd.concat([train.drop(columns=[TARGET], errors="ignore"), test], ignore_index=True)
    X = group_features(row_features(full))
    for c in CAT_LOW:
        X[c] = X[c].astype("category")
    n = len(train)
    return X.iloc[:n].reset_index(drop=True), X.iloc[n:].reset_index(drop=True)


def text_field(df):
    """Texto para los modelos de texto: titulo + garantia."""
    return (df["title"].fillna("").map(norm_text) + " | " + df["warranty"].map(norm_text)).values


# ---------------------------------------------------------------- target encoding
def _te_fit(keys, y, prior, m):
    s = pd.DataFrame({"k": keys.values, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (s["sum"] + prior * m) / (s["count"] + m), s["count"]


def target_encode(X_fit, y_fit, X_apply_list, cols=CAT_HIGH, m=2.0, n_splits=5, seed=SEED):
    """Target encoding suavizado. Para X_fit se calcula out-of-fold (cada fila se codifica sin ver
    su propia etiqueta); para los X_apply se usa el encoding ajustado con todo X_fit.
    Devuelve (te_fit, [te_apply...]) como DataFrames con columnas te_<col> y ten_<col> (n visto)."""
    y_fit = np.asarray(y_fit, dtype=float)
    prior = y_fit.mean()
    te_fit = pd.DataFrame(index=X_fit.index)
    te_apply = [pd.DataFrame(index=Xa.index) for Xa in X_apply_list]
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for c in cols:
        oof = np.full(len(X_fit), np.nan)
        oofn = np.zeros(len(X_fit))
        for a, b in skf.split(X_fit, y_fit):
            enc, cnt = _te_fit(X_fit[c].iloc[a], y_fit[a], prior, m)
            oof[b] = X_fit[c].iloc[b].map(enc).values
            oofn[b] = X_fit[c].iloc[b].map(cnt).fillna(0).values
        te_fit["te_" + c], te_fit["ten_" + c] = oof, oofn
        enc, cnt = _te_fit(X_fit[c], y_fit, prior, m)
        for Xa, ta in zip(X_apply_list, te_apply):
            ta["te_" + c] = Xa[c].map(enc).values
            # se escala a 4/5 para que el conteo sea comparable al visto en entrenamiento (OOF)
            ta["ten_" + c] = Xa[c].map(cnt).fillna(0).values * (n_splits - 1) / n_splits
    return te_fit, te_apply


# ---------------------------------------------------------------- features de nivel 1 (texto)
N_FOLDS = 5
CACHE = os.path.join(ROOT, "cache")


def cv_folds(y):
    """Particion fija que usan TODAS las notebooks, para que los modelos sean comparables."""
    return list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=0).split(y, y))


def linear_oof(Ma, ya, Mb, C, seed=1):
    """Regresion logistica sobre una matriz de texto (TF-IDF o embeddings).
    Devuelve el score out-of-fold para las filas de ajuste y el score del modelo completo para Mb."""
    from sklearn.linear_model import LogisticRegression

    solver = "liblinear" if hasattr(Ma, "tocsr") else "lbfgs"
    oof = np.zeros(Ma.shape[0])
    for i, j in StratifiedKFold(5, shuffle=True, random_state=seed).split(ya, ya):
        oof[j] = LogisticRegression(C=C, max_iter=500, solver=solver).fit(Ma[i], ya[i]).decision_function(Ma[j])
    full = LogisticRegression(C=C, max_iter=500, solver=solver).fit(Ma, ya)
    return oof, full.decision_function(Mb)


def knn_features(Eq, Ep, yp, ks=(5, 20, 100), loo=False):
    """Para cada fila de Eq busca sus vecinos mas cercanos (coseno) en Ep y devuelve, para cada k,
    la proporcion ponderada de vecinos 'new' y la similitud media. loo=True excluye a la propia fila
    (se usa cuando Eq y Ep son el mismo conjunto, para no ver la etiqueta propia)."""
    out = np.zeros((len(Eq), len(ks) * 2), dtype="float32")
    kmax = max(ks)
    for s in range(0, len(Eq), 4000):
        S = Eq[s:s + 4000] @ Ep.T
        if loo:
            S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -1
        idx = np.argpartition(-S, kmax, axis=1)[:, :kmax]
        sim = np.take_along_axis(S, idx, 1)
        o = np.argsort(-sim, 1)
        idx, sim = np.take_along_axis(idx, o, 1), np.take_along_axis(sim, o, 1)
        lab = yp[idx]
        for i, k in enumerate(ks):
            w = sim[:, :k] ** 4
            out[s:s + 4000, 2 * i] = (lab[:, :k] * w).sum(1) / w.sum(1)
            out[s:s + 4000, 2 * i + 1] = sim[:, :k].mean(1)
    cols = [f"knn_{'p' if j == 0 else 'sim'}{k}" for k in ks for j in (0, 1)]
    return pd.DataFrame(out, columns=cols)


def seller_knn(sq, Eq, sp, Ep, yp, loo=False):
    """Vecinos de titulo dentro del MISMO vendedor: para cada fila de (sq, Eq) busca, entre las filas
    etiquetadas (sp, Ep, yp) de ese vendedor, las de titulo mas parecido. loo=True cuando Q y P son el mismo
    conjunto (la fila no puede verse a si misma). NaN si el vendedor no tiene otras filas etiquetadas."""
    out = np.full((len(sq), 5), np.nan, dtype="float32")
    gp = pd.Series(np.arange(len(sp))).groupby(np.asarray(sp)).apply(np.array).to_dict()
    gq = pd.Series(np.arange(len(sq))).groupby(np.asarray(sq)).apply(np.array).to_dict()
    for s, rows in gq.items():
        cand = gp.get(s)
        if cand is None:
            continue
        S = Eq[rows] @ Ep[cand].T
        if loo:
            S[rows[:, None] == cand[None, :]] = -2
        order = np.argsort(-S, axis=1)
        for r in range(len(rows)):
            o = order[r][S[r, order[r]] > -1.5]
            if len(o) == 0:
                continue
            sims, labs = S[r, o], yp[cand[o]]
            w = np.clip(sims[:3], 0, None) ** 4 + 1e-9
            close = sims > 0.92
            out[rows[r]] = [labs[0], sims[0], (labs[:3] * w).sum() / w.sum(), close.sum(),
                            labs[close].mean() if close.any() else np.nan]
    return pd.DataFrame(out, columns=["snn1_y", "snn1_sim", "snn3_y", "snn_close_n", "snn_close_y"])


def level1(Xa, ya, Xb, Ta, Tb, Ea, Eb):
    """Arma la matriz final para los modelos de arboles: features numericas + target encoding
    + score del modelo TF-IDF + score del modelo de embeddings + kNN sobre embeddings.
    Todo lo que usa la etiqueta se calcula out-of-fold para (Xa, ya) y con el ajuste completo para Xb."""
    Xa, Xb = Xa.reset_index(drop=True), Xb.reset_index(drop=True)
    num = [c for c in Xa.columns if c not in CAT_HIGH]
    tea, (teb,) = target_encode(Xa, ya, [Xb])
    A, B = pd.concat([Xa[num], tea], axis=1), pd.concat([Xb[num], teb], axis=1)
    A["txt_tfidf"], B["txt_tfidf"] = linear_oof(Ta, ya, Tb, C=10)
    A["txt_emb"], B["txt_emb"] = linear_oof(Ea, ya, Eb, C=30)
    A = pd.concat([A, knn_features(Ea, Ea, ya, loo=True)], axis=1)
    B = pd.concat([B, knn_features(Eb, Ea, ya)], axis=1)
    A = pd.concat([A, seller_knn(Xa["seller_id"].values, Ea, Xa["seller_id"].values, Ea, ya, loo=True)], axis=1)
    B = pd.concat([B, seller_knn(Xb["seller_id"].values, Eb, Xa["seller_id"].values, Ea, ya)], axis=1)
    return A, B


# Transformers afinados en Kaggle con GPU (gpu_kaggle/transformer_*.ipynb). Cada uno aporta una feature.
TRANSFORMERS = {"txt_bert": "beto", "txt_xlmr": "xlmr", "txt_bertin": "bertin"}


def _transformers():
    """Predicciones de cada transformer (salidas de los notebooks de Kaggle en gpu_kaggle/salidas/)."""
    return {c: np.load(os.path.join(ROOT, "gpu_kaggle", "salidas", f"{d}.npz")) for c, d in TRANSFORMERS.items()}


def load_fold(f):
    A, B = pd.read_parquet(f"{CACHE}/fold{f}_tr.parquet"), pd.read_parquet(f"{CACHE}/fold{f}_va.parquet")
    for c, z in _transformers().items():  # inner: out-of-fold dentro del train del fold; valid: modelo entrenado con ese train
        A[c], B[c] = z[f"inner{f}"], z[f"valid{f}"]
    return A, B


def load_full():
    A, B = pd.read_parquet(f"{CACHE}/full_tr.parquet"), pd.read_parquet(f"{CACHE}/full_te.parquet")
    for c, z in _transformers().items():  # oof: cada fila predicha por un modelo que no la vio; test: promedio de los 5 modelos
        A[c], B[c] = z["oof"], z["test"]
    return A, B


def load_y():
    return np.load(f"{CACHE}/y.npy")


def save_submission(proba, name, threshold=0.5):
    """Guarda el CSV para Kaggle: ID = numero de fila del test empezando en 1, condition = new/used.
    Ademas deja una copia con el id original de la publicacion (MLA...) en submissions/id_original/."""
    out = os.path.join(ROOT, "submissions")
    os.makedirs(f"{out}/id_original", exist_ok=True)
    cond = np.where(proba > threshold, "new", "used")
    ids = pd.read_csv(f"{CACHE}/test_ids.csv")["id"]
    pd.DataFrame({"id": ids, "condition": cond}).to_csv(f"{out}/id_original/{name}.csv", index=False)
    sub = pd.DataFrame({"ID": np.arange(1, len(cond) + 1), "condition": cond})
    sub.to_csv(f"{out}/{name}.csv", index=False)
    return sub
