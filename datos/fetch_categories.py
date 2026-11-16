"""Descarga el arbol de categorias (path_from_root) de la API publica de MercadoLibre
para todas las category_id presentes en train y test. Genera datos/categories.json (se corrio una vez, en octubre de 2026;
la API cambia con el tiempo, por eso el archivo resultante se incluye en el repositorio)."""
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import os

import pandas as pd

DATOS = os.environ.get("TP3_DATOS", os.path.dirname(os.path.abspath(__file__)))

ids = set()
for f in ["train_data.jsonlines", "test_data.jsonlines"]:
    ids |= set(pd.read_json(os.path.join(DATOS, f), lines=True, convert_dates=False)["category_id"])


def get(cid):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"https://api.mercadolibre.com/categories/{cid}", timeout=20) as r:
                d = json.load(r)
            return cid, {"name": d["name"], "path": [[p["id"], p["name"]] for p in d["path_from_root"]]}
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return cid, None
            time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2 * (attempt + 1))
    return cid, None


with ThreadPoolExecutor(8) as ex:
    out = dict(ex.map(get, sorted(ids)))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "categories.json"), "w", encoding="utf8"), ensure_ascii=False)
print(len(out), "categorias;", sum(v is None for v in out.values()), "sin datos")
