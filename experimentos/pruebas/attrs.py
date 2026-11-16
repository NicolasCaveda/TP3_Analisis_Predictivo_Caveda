"""Variables de atributos, variaciones y formato del id de foto (sin etiqueta)."""
import re, numpy as np, pandas as pd
def build(R):
    X = pd.DataFrame(index=R.index)
    at = R["attributes"].map(lambda l: {a.get("id") or "" for a in l})
    X["at_part"] = at.map(lambda s: int(bool(s & {"PART_NUMBER", "MLA-PART_NUMBER"})))
    X["at_auto"] = at.map(lambda s: int(any(i.startswith("MLA1744") or i.startswith("MLA1743") for i in s)))
    X["at_inmueble"] = at.map(lambda s: int(any(i.startswith("MLA1459") for i in s)))
    X["at_ropa"] = at.map(lambda s: int(bool(s & {"GENDER", "Season"})))
    def kms(l):
        for a in l:
            if (a.get("id") or "").endswith("KMTS"):
                try: return float(re.sub(r"[^\d.]", "", a.get("value_name") or "") or "nan")
                except Exception: return np.nan
        return np.nan
    X["at_kms"] = R["attributes"].map(kms)
    va = R["variations"]
    X["var_scf"] = va.map(lambda l: int(any(v.get("seller_custom_field") for v in l)))
    X["var_ncolor2"] = va.map(lambda l: int(any(a.get("name") == "Color Secundario" for v in l for a in v["attribute_combinations"])))
    X["var_talle"] = va.map(lambda l: int(any(a.get("name") == "Talle" for v in l for a in v["attribute_combinations"])))
    X["var_avail"] = va.map(lambda l: sum(v.get("available_quantity") or 0 for v in l) if l else np.nan)
    X["var_npics"] = va.map(lambda l: np.mean([len(v.get("picture_ids") or []) for v in l]) if l else np.nan)
    X["var_price_spread"] = va.map(lambda l: (max(v["price"] for v in l) - min(v["price"] for v in l)) if l else np.nan)
    X["pic_pref_len"] = R["pictures"].map(lambda l: len(l[0]["id"].split("-")[0]) if l and "-" in (l[0].get("id") or "") else -1)
    X["pic_pref_len_mean"] = R["pictures"].map(lambda l: np.mean([len(d["id"].split("-")[0]) for d in l if "-" in (d.get("id") or "")]) if l else np.nan)
    X["s_pic_pref_len"] = X.assign(s=R["seller_id"].values).groupby("s")["pic_pref_len_mean"].transform("mean")
    return X.reset_index(drop=True)
