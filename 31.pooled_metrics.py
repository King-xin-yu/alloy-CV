# -*- coding: utf-8 -*-
"""pooled metrics and baselines (Table S6)."""

import sys, os, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "alloycv_common.py")) else os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import alloycv_common as ac
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
RS=getattr(ac,'RS',42)
W = (os.path.join(ROOT, "results") + os.sep) if os.path.isdir(os.path.join(ROOT, "results")) else ROOT + os.sep
def oof(df, feats, target, protocol, repeats=10, seed0=1000):
    sub=df[df[target].notna()]
    X=sub[feats]; y=sub[target].to_numpy(dtype=float)
    g=sub.groupby(['Ni','Si','Ce','La']).ngroup().to_numpy()
    res=[]
    rngs=[0] if protocol=='random' else list(range(repeats))
    for r in rngs:
        pred=np.full(y.shape,np.nan); folds=np.zeros(len(y),int)
        if protocol=='random':
            splitter=KFold(10,shuffle=True,random_state=RS)
            it=splitter.split(X)
        else:
            groups=np.unique(g); rng=np.random.default_rng(seed0+r)
            perm=rng.permutation(groups); fold_of={gg:i%10 for i,gg in enumerate(perm)}
            f=np.array([fold_of[gg] for gg in g]); it=[(np.where(f!=k)[0],np.where(f==k)[0]) for k in range(10)]
        for k,(tr,te) in enumerate(it):
            if len(te)==0: continue
            pipe=ac.make_pipeline(feats, ac.RandomForestRegressor(n_estimators=400,random_state=RS,n_jobs=-1))
            pipe.fit(X.iloc[tr],y[tr]); pred[te]=pipe.predict(X.iloc[te]); folds[te]=k+1
        res.append((r,pred,folds))
    return y,g,res
def metrics(y,pred):
    return dict(R2=r2_score(y,pred), MAE=mean_absolute_error(y,pred), RMSE=float(np.sqrt(mean_squared_error(y,pred))))
rows=[]; folds_rows=[]
for name,df,feats in [('A',ac.dataset_A(),ac.FEATURES_A),('B',ac.dataset_B(),ac.FEATURES_B)]:
    for t in ac.TARGETS:
        for proto in ['random','grouped']:
            t0=time.time()
            y,g,res=oof(df,feats,t,proto)
            ms=[metrics(y,p) for _,p,_ in res]
            # mean baseline
            mb=[]
            import numpy as _np
            for r,pred,folds in res:
                pass
            rows.append(dict(dataset=name,target=t,protocol=proto,n=len(y),
                R2_mean=float(np.mean([m['R2'] for m in ms])),R2_sd=float(np.std([m['R2'] for m in ms])),
                MAE_mean=float(np.mean([m['MAE'] for m in ms])),MAE_sd=float(np.std([m['MAE'] for m in ms])),
                RMSE_mean=float(np.mean([m['RMSE'] for m in ms])),RMSE_sd=float(np.std([m['RMSE'] for m in ms])),
                R2_min=float(np.min([m['R2'] for m in ms])),R2_max=float(np.max([m['R2'] for m in ms]))))
            r,folds=res[0][0],res[0][2]
            for k in range(1,11):
                m=folds==k
                if m.sum(): folds_rows.append(dict(dataset=name,target=t,protocol=proto,rep=r,fold=k,n=int(m.sum()),y_std=float(np.std(y[m])),y_mean=float(np.mean(y[m])),
                    r2=float(r2_score(y[m],res[0][1][m]))))
            print('%s %s %s done %.0fs'%(name,t,proto,time.time()-t0),flush=True)
pd.DataFrame(rows).to_csv(os.path.join(W,'r12_pooled.csv'),index=False)
pd.DataFrame(folds_rows).to_csv(os.path.join(W,'r12_folds.csv'),index=False)
print(pd.DataFrame(rows).to_string())
