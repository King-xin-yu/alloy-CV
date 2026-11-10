# -*- coding: utf-8 -*-
"""publication-grouped validation (Table S5)."""

import sys, os, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import alloycv_common as ac
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
RS=getattr(ac,'RS',42); HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); W=(os.path.join(ROOT,'results')+os.sep) if os.path.isdir(os.path.join(ROOT,'results')) else ROOT+os.sep
if not os.path.isdir(W): W=HERE+os.sep
D=os.path.join(ROOT,'data') if os.path.isdir(os.path.join(ROOT,'data')) else ROOT
if not os.path.isdir(D): D=HERE
def load(name):
    df=ac.dataset_A() if name=='A' else ac.dataset_B()
    m=pd.read_excel(D+'/dataset_%s_with_source.xlsx'%name)
    assert len(m)==len(df)
    df=df.copy(); df['Source']=m['Source'].values
    return df
def cvpred(df,feats,target,repeats=10,seed0=2000):
    sub=df[df[target].notna()].reset_index(drop=True)
    X=sub[feats]; y=sub[target].to_numpy(float)
    g=sub['Source'].astype(str).to_numpy(); groups=np.unique(g); out=[]
    for r in range(repeats):
        rng=np.random.default_rng(seed0+r); perm=rng.permutation(groups)
        fo={gg:i%10 for i,gg in enumerate(perm)}; f=np.array([fo[gg] for gg in g])
        pred=np.full(len(y),np.nan)
        for k in range(10):
            te=f==k; tr=~te
            if not te.any(): continue
            pipe=ac.make_pipeline(feats, ac.RandomForestRegressor(n_estimators=400,random_state=RS,n_jobs=-1))
            pipe.fit(X[tr],y[tr]); pred[te]=pipe.predict(X[te])
        out.append(pred)
    return sub,y,out
prow=[]; fsizes=[]
for name in ('A','B'):
    df=load(name); feats=ac.FEATURES_A if name=='A' else ac.FEATURES_B
    for t in ac.TARGETS:
        t0=time.time(); sub,y,preds=cvpred(df,feats,t)
        ms=[dict(R2=r2_score(y,p),MAE=mean_absolute_error(y,p),RMSE=float(np.sqrt(mean_squared_error(y,p)))) for p in preds]
        prow.append(dict(dataset=name,target=t,n=len(y),R2=float(np.mean([m['R2'] for m in ms])),R2_sd=float(np.std([m['R2'] for m in ms])),MAE=float(np.mean([m['MAE'] for m in ms])),RMSE=float(np.mean([m['RMSE'] for m in ms])),R2_min=float(np.min([m['R2'] for m in ms])),R2_max=float(np.max([m['R2'] for m in ms])),R2_median=float(np.median([m['R2'] for m in ms]))))
        print('%s %s publication-grouped %.0fs'%(name,t,time.time()-t0),flush=True)
pd.DataFrame(prow).to_csv(W+'_pubgroup.csv',index=False)
print(pd.DataFrame(prow).round(3).to_string())
