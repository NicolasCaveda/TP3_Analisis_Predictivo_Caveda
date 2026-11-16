"""Opinion de un LLM (Claude) como feature: probabilidad de 0 a 100 de que cada publicacion sea NUEVA.

Se ejecuta con Claude Code en modo no interactivo (`claude -p`), con el plan del usuario (no la API).
Las respuestas de un LLM no son deterministas: por eso los puntajes que usa el modelo final quedan guardados en
llm/puntajes/ y el entrenamiento y la aplicacion los leen de ahi. Este script es el que los genero.

Uso:
    python llm/llm_claude.py <modelo> <filas> [v1|v3]
Corridas que usa el modelo final (ver README):
    claude-opus-5-5   piloto | etapa1 | nuevos_extra | test_nuevos | test_pocos      (v1: Claude ve solo la publicacion)
    claude-fable-5-2  mixtos_train | mixtos_train2 | test_mixtos  v3                   (v3: ademas, historial etiquetado del vendedor)
Los resultados se guardan por lote en llm/puntajes/<modelo>[-v3]/<filas>/ y la corrida se puede reanudar.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import features as F  # noqa: E402

MODEL, ROWS = sys.argv[1], sys.argv[2]
# v1: cada publicacion sola. v3: ademas, el historial etiquetado del vendedor (para vendedores mixtos)
VERSION = sys.argv[3] if len(sys.argv) > 3 else "v1"
assert VERSION in ("v1", "v3")
BATCH = 25 if VERSION == "v3" else 50   # v3 lleva historial: lotes mas chicos
PUNTAJES = os.path.join(F.ROOT, "llm", "puntajes")
OUT = os.path.join(PUNTAJES, MODEL + ("" if VERSION == "v1" else f"-{VERSION}"), ROWS)

SYSTEM = (
    "Sos un experto en el mercado de MercadoLibre Argentina en 2015. Vas a recibir publicaciones numeradas, "
    "cada una con su título, garantía, precio en pesos argentinos de 2015, stock, tipo de publicación, envío, "
    "medios de pago y datos del vendedor. Para cada una estimá la probabilidad, de 0 a 100, de que el producto "
    "publicado sea NUEVO (y no usado). Tené en cuenta qué producto es y si suele venderse nuevo o usado, si el "
    "precio es razonable para un producto nuevo, si el vendedor parece un comercio (muchas publicaciones, stock "
    "mayor a 1, publicaciones pagas) o un particular, y lo que digan el título y la garantía. "
    "Respondé solo con el JSON pedido, con un elemento por publicación y el mismo id."
)
SCHEMA = json.dumps({
    "type": "object", "additionalProperties": False, "required": ["items"],
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["id", "p_nuevo"],
        "properties": {"id": {"type": "integer"}, "p_nuevo": {"type": "integer"}}}}},
})
LT = {"free": "gratuita", "bronze": "bronce", "silver": "plata", "gold": "oro", "gold_special": "oro especial",
      "gold_premium": "oro premium", "gold_pro": "oro pro"}


def describe(raw, X):
    """Texto compacto de cada publicacion (solo datos de la publicacion y agregados sin etiqueta)."""
    cats = json.load(open(F.CATEGORIES_PATH, encoding="utf8")) if os.path.exists(F.CATEGORIES_PATH) else {}
    out = []
    for i in range(len(raw)):
        r, x = raw.iloc[i], X.iloc[i]
        path = (cats.get(r["category_id"]) or {}).get("path") or []
        cat = " > ".join(p[1] for p in path) if path else "desconocida"
        sh = r["shipping"] or {}
        pays = ", ".join(d["description"] for d in r["non_mercado_pago_payment_methods"]) or "solo MercadoPago"
        w = r["warranty"] if isinstance(r["warranty"], str) and r["warranty"].strip() else "no informa"
        out.append(
            f"Título: {r['title']} | Categoría: {cat} | Garantía: {w[:120]} | Precio: {r['price']:.0f} {r['currency_id']} | "
            f"Stock inicial {int(r['initial_quantity'])}, vendidos {int(r['sold_quantity'])}, disponibles {int(r['available_quantity'])} | "
            f"Publicación {LT.get(r['listing_type_id'], r['listing_type_id'])}, modo {r['buying_mode']} | "
            f"Envío: {sh.get('mode')}, gratis {'sí' if sh.get('free_shipping') else 'no'}, retiro en persona {'sí' if sh.get('local_pick_up') else 'no'} | "
            f"Fotos: {len(r['pictures'])} | Pagos: {pays} | Vendedor: {int(x['s_n'])} publicaciones, "
            f"{100 * x['s_iq1']:.0f}% con stock 1, {100 * x['s_free']:.0f}% gratuitas"
        )
    return out


# Uso del plan: el mismo plan alimenta la conversacion con Claude, asi que el script se limita a un gasto
# (equivalente en USD de API) por ventana de 5 horas y, si aparece un aviso de limite, espera en vez de insistir.
WINDOW_USD = float(os.environ.get("LLM_WINDOW_USD", "4"))
WINDOW_S = 5 * 3600
STATE = os.path.join(F.CACHE, "llm_ventana.json")


def _state():
    try:
        st = json.load(open(STATE, encoding="utf8"))
    except Exception:
        st = {"start": time.time(), "spent": 0.0}
    if time.time() - st["start"] > WINDOW_S:
        st = {"start": time.time(), "spent": 0.0}
    return st


def _wait_window(st, reason):
    resume = st["start"] + WINDOW_S
    print(f"{time.strftime('%d/%m %H:%M')} pausa ({reason}); sigue a las {time.strftime('%d/%m %H:%M', time.localtime(resume))}", flush=True)
    time.sleep(max(60, resume - time.time()))


def call(batch_id, ids, texts):
    path = os.path.join(OUT, f"{batch_id:05d}.json")
    if os.path.exists(path):
        return
    got, cost = ask(ids, texts)
    missing = [k for k in ids if k not in got]
    if got and 0 < len(missing) <= 5:          # a veces omite alguna publicacion del lote: se piden aparte
        sub = [t for k, t in zip(ids, texts) if k in missing]
        more, c2 = ask(missing, sub)
        got.update(more); cost += c2
    if set(got) >= set(ids):
        json.dump({"p": {str(k): got[k] for k in ids}, "cost": cost}, open(path, "w", encoding="utf8"))
    else:
        print(f"lote {batch_id}: faltan {len(set(ids) - set(got))} publicaciones", flush=True)


def ask(ids, texts):
    """Una consulta a Claude; devuelve ({id: p_nuevo}, costo). Reintenta si la respuesta no se puede leer."""
    prompt = "\n".join(f"[{k}] {t}" for k, t in zip(ids, texts))   # va por stdin (la linea de comandos tiene tope de largo)
    cmd = ["claude", "-p", "--model", MODEL, "--effort", "low", "--system-prompt", SYSTEM,
           "--tools", "", "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence",
           "--exclude-dynamic-system-prompt-sections", "--output-format", "json", "--json-schema", SCHEMA]
    fails, best, cost = 0, {}, 0.0
    while fails < 3:
        st = _state()
        if st["spent"] >= WINDOW_USD:
            _wait_window(st, f"tope de {WINDOW_USD} USD equivalentes en esta ventana")
            continue
        res = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf8", cwd=tempfile.gettempdir(), timeout=900)
        try:
            d = json.loads(res.stdout)
        except Exception:
            d = {}
        msg = str(d.get("result", "")) + res.stderr
        if d.get("api_error_status") == 429 or ((d.get("is_error") or not d) and "limit" in msg.lower()):
            st["spent"] = WINDOW_USD                     # limite del plan: esperar a la proxima ventana
            json.dump(st, open(STATE, "w"))
            _wait_window(st, "aviso de límite del plan")
            continue
        st["spent"] += d.get("total_cost_usd") or 0
        cost += d.get("total_cost_usd") or 0
        json.dump(st, open(STATE, "w"))
        try:
            items = {it["id"]: it["p_nuevo"] for it in d["structured_output"]["items"] if it["id"] in set(ids)}
            if len(items) > len(best):
                best = items
            if len(best) >= len(ids) - 5:          # completo, o casi (lo que falta se pide aparte en call)
                return best, cost
        except Exception:
            pass
        fails += 1
    print(f"consulta de {len(ids)}: falló tras 3 intentos ({msg[:200]})", flush=True)
    return best, cost


def pilot_rows(n_train):
    return np.sort(np.random.default_rng(0).choice(n_train, 2000, replace=False))


def stage1_rows(n_train):
    """6.000 publicaciones de train que no estan en la piloto (semilla fija)."""
    rest = np.setdiff1d(np.arange(n_train), pilot_rows(n_train))
    return np.sort(np.random.default_rng(1).choice(rest, 6000, replace=False))


def new_in_fold_extra(train, n=4000):
    """n publicaciones de train mas, fuera de piloto y etapa 1, cuyo vendedor no tiene publicaciones en la parte de
    entrenamiento de su fold (vendedor nuevo). Sirven para ajustar y validar mejor la combinacion en ese tramo."""
    usadas = np.r_[pilot_rows(len(train)), stage1_rows(len(train))]
    s, nuevo = train["seller_id"].values, np.zeros(len(train), bool)
    for a, b in F.cv_folds(F.load_y()):
        nuevo[b] = ~pd.Series(s[b]).isin(set(s[a])).values
    cand = np.setdiff1d(np.flatnonzero(nuevo), usadas)
    return np.sort(np.random.default_rng(3).choice(cand, n, replace=False))


HIST_SYSTEM = (
    "\n\nCada publicación trae además el historial del vendedor: algunas de sus otras publicaciones (las de título más "
    "parecido) con su condición REAL, y cuántas nuevas y usadas tiene en total. Este vendedor publica productos nuevos y "
    "usados, así que usá el historial para entender qué tipo de productos vende nuevos y cuáles usados (por ejemplo, "
    "accesorios nuevos y equipos usados) y en qué se diferencian sus publicaciones de cada tipo (precio, stock, tipo de "
    "publicación, garantía), y decidí para esta publicación en particular."
)


def _fit_parts(n_train):
    """Para cada fila de train, la parte de entrenamiento de su fold (sus etiquetas son las que se pueden mostrar)."""
    part = np.zeros(n_train, int)
    for f, (a, b) in enumerate(F.cv_folds(F.load_y())):
        part[b] = f
    return part, [a for a, b in F.cv_folds(F.load_y())]


def mixed_purity(train):
    """Para cada fila de train: publicaciones del vendedor y fraccion de nuevas en la parte de entrenamiento de su fold."""
    s, yv = train["seller_id"].values, F.load_y()
    cnt, pur = np.zeros(len(train)), np.full(len(train), np.nan)
    for a, b in F.cv_folds(yv):
        g = pd.Series(yv[a]).groupby(s[a]).agg(["mean", "size"])
        cnt[b] = pd.Series(s[b]).map(g["size"]).fillna(0).values
        pur[b] = pd.Series(s[b]).map(g["mean"]).values
    return cnt, pur


def mixed_train_rows(train, n=1000):
    """n filas de train de vendedores mixtos (>= 4 publicaciones en la parte de entrenamiento, 20%-80% nuevas)."""
    cnt, pur = mixed_purity(train)
    cand = np.flatnonzero((cnt >= 4) & (pur >= .2) & (pur <= .8))
    return np.sort(np.random.default_rng(4).choice(cand, n, replace=False))


def mixed_train_rows2(train, n=1000):
    """Otras n filas de vendedores mixtos de train, distintas de mixed_train_rows."""
    cnt, pur = mixed_purity(train)
    cand = np.setdiff1d(np.flatnonzero((cnt >= 4) & (pur >= .2) & (pur <= .8)), mixed_train_rows(train))
    return np.sort(np.random.default_rng(5).choice(cand, n, replace=False))


def mixed_test_rows(train, test):
    g = pd.Series((train[F.TARGET] == "new").values).groupby(train["seller_id"].values).agg(["mean", "size"])
    cnt, pur = test["seller_id"].map(g["size"]).fillna(0).values, test["seller_id"].map(g["mean"]).values
    return len(train) + np.flatnonzero((cnt >= 4) & (pur >= .2) & (pur <= .8))


def seller_history(train, raw, rows, k=4):
    """Para cada fila, hasta k publicaciones nuevas y k usadas del mismo vendedor (las de titulo mas parecido) con su
    condicion real. En train solo se usan etiquetas de la parte de entrenamiento del fold de la fila; en test, todo train."""
    E = np.load(os.path.join(F.CACHE, "emb_e5small.npy"))
    yv, n = F.load_y(), len(train)
    fold_of, fits = _fit_parts(n)
    fit_sets = [np.zeros(n, bool) for _ in fits]
    for m, a in zip(fit_sets, fits):
        m[a] = True
    by_seller = train.groupby("seller_id").indices
    LT_ = lambda v: LT.get(v, v)
    out = []
    for i in rows:
        pool = by_seller.get(raw["seller_id"].iat[i], np.array([], int))
        pool = pool[pool != i]
        if i < n:
            pool = pool[fit_sets[fold_of[i]][pool]]
        sims = E[pool] @ E[i]
        lines = []
        for lab, nom in [(1, "NUEVO"), (0, "USADO")]:
            p = pool[yv[pool] == lab]
            for j in p[np.argsort(-sims[yv[pool] == lab])][:k]:
                r = raw.iloc[j]
                lines.append(f"{str(r['title'])[:70]} (${r['price']:.0f}, stock {int(r['initial_quantity'])}, "
                             f"{LT_(r['listing_type_id'])}) => {nom}")
        nn = int(yv[pool].sum())
        out.append(f" | Historial del vendedor: {nn} nuevas y {len(pool) - nn} usadas; las más parecidas: " + "; ".join(lines))
    return out


if __name__ == "__main__":
    train, test = F.load("train_data.jsonlines"), F.load("test_data.jsonlines")
    raw = pd.concat([train.drop(columns=[F.TARGET]), test], ignore_index=True)
    # agregados del vendedor sobre train+test (no usan la etiqueta)
    g = raw.groupby("seller_id")
    X = pd.DataFrame({"s_n": g["id"].transform("size"),
                      "s_iq1": g["initial_quantity"].transform(lambda v: (v == 1).mean()),
                      "s_free": g["listing_type_id"].transform(lambda v: (v == "free").mean())})
    rows = {"piloto": lambda: pilot_rows(len(train)),     # 2.000 filas de train al azar
            "etapa1": lambda: stage1_rows(len(train)),     # 6.000 filas de train mas
            # test solo de vendedores sin historial en train: es donde Claude aporta (etapa 1)
            "test_nuevos": lambda: len(train) + np.flatnonzero(~test["seller_id"].isin(train["seller_id"]).values),
            # test de vendedores con 1 a 3 publicaciones en train: segundo tramo donde Claude aporta algo
            "test_pocos": lambda: len(train) + np.flatnonzero(
                test["seller_id"].map(train["seller_id"].value_counts()).fillna(0).between(1, 3).values),
            # 4.000 vendedores nuevos mas de train (fuera de piloto/etapa 1)
            "nuevos_extra": lambda: new_in_fold_extra(train),
            # vendedores mixtos (20%-80% nuevas, >= 4 publicaciones con etiqueta): con historial etiquetado (v3)
            "mixtos_train": lambda: mixed_train_rows(train),
            "mixtos_train2": lambda: mixed_train_rows2(train),
            "test_mixtos": lambda: mixed_test_rows(train, test)}[ROWS]()
    os.makedirs(OUT, exist_ok=True); os.makedirs(F.CACHE, exist_ok=True)
    texts = describe(raw.iloc[rows], X.iloc[rows])
    if VERSION == "v3":
        SYSTEM = SYSTEM + HIST_SYSTEM
        texts = [t + h for t, h in zip(texts, seller_history(train, raw, rows))]
    jobs = [(s // BATCH, list(map(int, rows[s:s + BATCH])), texts[s:s + BATCH]) for s in range(0, len(rows), BATCH)]
    print(f"{time.strftime('%d/%m %H:%M')} {MODEL} {ROWS}: {len(jobs)} lotes, tope {WINDOW_USD} USD eq. por ventana de 5 h", flush=True)
    for k, j in enumerate(jobs):             # de a uno, para no consumir el plan de golpe
        if os.path.exists(os.path.join(OUT, f"{j[0]:05d}.json")):
            continue                          # lote ya hecho en una corrida anterior
        call(*j)
        if k % 10 == 9:
            print(f"{time.strftime('%d/%m %H:%M')} {k + 1}/{len(jobs)} lotes", flush=True)
        time.sleep(10)
    print("listo:", len(os.listdir(OUT)), "lotes en", OUT, flush=True)
