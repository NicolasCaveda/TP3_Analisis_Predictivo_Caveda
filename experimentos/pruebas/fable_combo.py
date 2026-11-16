import glob, json, numpy as np, pandas as pd, sys
sys.path.insert(0, ".")
import features as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
lg = lambda q: np.log(np.clip(q,.01,.99)/(1-np.clip(q,.01,.99)))
def cl(v, parts):
    P={}; c=0
    for p in parts:
        for f in glob.glob(f"{F.CACHE}/llm/{v}/{p}/*.json"):
            d=json.load(open(f,encoding="utf8")); P.update({int(k):x/100 for k,x in d["p"].items()}); c+=d.get("cost",0)
    return P,c
y=F.load_y(); z=np.load("kaggle_tabpfn/output_v6/tabpfn.npz")
ens=.5*z["oof"]+.5*np.mean([np.load(f"cache/oof_{n}.npy") for n in ["lightgbm","xgboost","catboost"]],0)
O,_=cl("claude-opus-5-5",["piloto","etapa1","nuevos_extra"]); Fb,cf=cl("claude-fable-5-2",["nuevos_train"])
idx=np.array(sorted(set(O)&set(Fb)),dtype=int); yy=y[idx]
print("filas comunes",len(idx),"costo fable cache USD eq",round(cf,2))
def cv(Z):
    q=np.mean([cross_val_predict(LogisticRegression(),Z,yy,cv=StratifiedKFold(10,shuffle=True,random_state=s),method="predict_proba")[:,1] for s in range(10)],0)
    return q
e=lg(ens[idx]); o=lg(np.array([O[i] for i in idx])); f=lg(np.array([Fb[i] for i in idx]))
acc=lambda q:((q>.5)==yy).mean()
print("ens",acc(ens[idx]),"opus solo",acc(np.array([O[i] for i in idx])),"fable solo",acc(np.array([Fb[i] for i in idx])))
qo=cv(np.c_[e,o]); qf=cv(np.c_[e,f]); qb=cv(np.c_[e,o,f]); qa=cv(np.c_[e,(o+f)/2])
for n,q in [("ens+opus",qo),("ens+fable",qf),("ens+opus+fable",qb),("ens+prom(opus,fable)",qa)]: print(n,round(acc(q),4))
rng=np.random.default_rng(0)
for n,q in [("ens+opus+fable",qb),("ens+prom",qa)]:
    d=((q>.5)==yy).astype(int)-((qo>.5)==yy).astype(int)
    print(n,"vs ens+opus netas",d.sum(),"P>0",np.mean([d[rng.integers(0,len(d),len(d))].sum()>0 for _ in range(2000)]))
print("corr opus-fable",np.corrcoef(o,f)[0,1])
