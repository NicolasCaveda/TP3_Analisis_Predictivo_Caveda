import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from harness import *
assert "pic_sq" in DATA[0][0].columns   # base nueva (con fotos y atributos)
X = pd.read_parquet(f"{F.CACHE}/exp_dup.parquet")
b = base()
compare(b, run(X, tag="duplicados+masivo"), "duplicados + carga masiva")
