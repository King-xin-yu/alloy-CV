# -*- coding: utf-8 -*-
"""composition-level out-of-fold errors (Table S7)."""

import sys, os, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import alloycv_common as ac
from sklearn.metrics import r2_score, mean_absolute_error
RS=getattr(ac,'RS',42); HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); W=(os.path.join(ROOT,'results')+os.sep) if os.path.isdir(os.path.join(ROOT,'results')) else ROOT+os.sep
if not os.path.isdir(W): W=HERE+os.sep
D=os.path.join(ROOT,'data') if os.path.isdir(os.path.join(ROOT,'data')) else ROOT
if not os.path.isdir(D): D=HERE
def load(n):
    df=ac.dataset_A() if n=='A' else ac.dataset_B()
    m=pd.read_excel(D+'/dataset_%s_with_source.xlsx'%n)
    df=df.copy(); df['Source']=m['Source'].values; return df
def grouped_pred(df,feats,target,repeats=5,seed0=1000):
    sub=df[df[target].notna()].reset_index(drop=True)
    X=sub[feats]; y=sub[target].to_numpy(float)
    g=sub.groupby(['Ni','Si','Ce','La']).ngroup().to_numpy()
    groups=np.unique(g); preds=[]
    for r in range(repeats):
        rng=np.random.default_rng(seed0+r); perm=rng.permutation(groups)
        fo={gg:i%10 for i,gg in enumerate(perm)}; f=np.array([fo[gg] for gg in g])
        pred=np.full(len(y),np.nan)
        for k in range(10):
            te=f==k; tr=~te
            if not te.any(): continue
            pipe=ac.make_pipeline(feats, ac.RandomForestRegressor(n_estimators=400,random_state=RS,n_jobs=-1))
            pipe.fit(X[tr],y[tr]); pred[te]=pipe.predict(X[te])
        preds.append(pred)
    return sub,y,np.mean(preds,axis=0)
rows=[]
for n in ('A','B'):
    df=load(n); feats=ac.FEATURES_A if n=='A' else ac.FEATURES_B
    for t in ac.TARGETS:
        t0=time.time(); sub,y,p=grouped_pred(df,feats,t)
        sub=sub.copy(); sub['err']=p-y
        gb=sub.groupby(['Ni','Si','Ce','La']).agg(k=('err','size'),bias=('err','mean'),mae=('err',lambda s: float(np.mean(np.abs(s)))),ybar=(t,'mean')).reset_index()
        gb['dataset']=n; gb['target']=t
        rows.append(gb)
        print('%s %s compositions %d %.0fs'%(n,t,len(gb),time.time()-t0),flush=True)
allc=pd.concat(rows,ignore_index=True)
allc.to_csv(W+'_comp_errors.csv',index=False)
print(allc.groupby(['dataset','target']).agg(comp=('k','size'),multi=('k',lambda s:(s>=3).sum()),mae_med=('mae','median'),bias_abs_med=('bias',lambda s:s.abs().median())).round(3).to_string())
