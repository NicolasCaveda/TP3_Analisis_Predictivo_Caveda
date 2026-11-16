import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from harness import *
import fotos, attrs
R = raw(); n = len(y)
X1 = fotos.basic(R).iloc[:n].reset_index(drop=True)
X3 = pd.concat([X1, attrs.build(R).iloc[:n].reset_index(drop=True)], axis=1)
p1 = run(X1, tag="fotos"); p3 = run(X3, tag="fotos+atributos")
compare(p1, p3, "atributos encima de fotos")
