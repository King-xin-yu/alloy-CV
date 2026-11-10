# -*- coding: utf-8 -*-
"""mean-predictor and linear-regression baselines."""

import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import alloycv_common as ac
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); W=(os.path.join(ROOT,'results')+os.sep) if os.path.isdir(os.path.join(ROOT,'results')) else ROOT+os.sep
RS=getattr(ac,'RS',42)
def base(df,feats,target,protocol,repeats=10,seed0=1000):
    sub=df[df[target].notna()]; X=sub[feats]; y=sub[target].to_numpy(float)
    g=sub.groupby(['Ni','Si','Ce','La']).ngroup().to_numpy()
    out={'mean':[], 'lin':[]}
    for r in (range(repeats) if protocol=='grouped' else [0]):
        if protocol=='random':
            it=list(KFold(10,shuffle=True,random_state=RS).split(X))
        else:
            groups=np.unique(g); rng=np.random.default_rng(seed0+r); perm=rng.permutation(groups)
            fo={gg:i%10 for i,gg in enumerate(perm)}; f=np.array([fo[gg] for gg in g])
            it=[(np.where(f!=k)[0],np.where(f==k)[0]) for k in range(10)]
        for key in ('mean','lin'):
            pred=np.full(len(y),np.nan)
            for tr,te in it:
                if len(te)==0: continue
                if key=='mean': pred[te]=y[tr].mean()
                else:
                    m=LinearRegression().fit(X.iloc[tr],y[tr]); pred[te]=m.predict(X.iloc[te])
            out[key].append(dict(R2=r2_score(y,pred),MAE=mean_absolute_error(y,pred),RMSE=float(np.sqrt(mean_squared_error(y,pred)))))
    return {k:{m:(float(np.mean([z[m] for z in v])),float(np.std([z[m] for z in v]))) for m in ('R2','MAE','RMSE')} for k,v in out.items()}
rows=[]
for name,df,feats in [('A',ac.dataset_A(),ac.FEATURES_A),('B',ac.dataset_B(),ac.FEATURES_B)]:
    for t in ac.TARGETS:
        for proto in ['random','grouped']:
            b=base(df,feats,t,proto)
            rows.append(dict(dataset=name,target=t,protocol=proto,
                mean_R2=b['mean']['R2'][0], lin_R2=b['lin']['R2'][0],
                mean_RMSE=b['mean']['RMSE'][0], lin_RMSE=b['lin']['RMSE'][0]))
            print(name,t,proto,'meanR2 %.3f linR2 %.3f (linRMSE %.1f)'%(b['mean']['R2'][0],b['lin']['R2'][0],b['lin']['RMSE'][0]),flush=True)
pd.DataFrame(rows).to_csv(os.path.join(W,'r12_base.csv'),index=False)
print('saved')
