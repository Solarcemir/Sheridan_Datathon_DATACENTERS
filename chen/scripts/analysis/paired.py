# Analysis script (run from chen/ with IBM/data linked; see README). Produced numbers in context_Chen.md.
import sys, pandas as pd, numpy as np
from sklearn.metrics import average_precision_score as ap
h=pd.read_parquet('processed/historical_2024_2025.parquet',columns=['cve_id','published','snapshot_month','assigner','kev_date_added'])
o=pd.read_parquet('results/cv_oof_lr.parquet').merge(pd.read_parquet('results/cv_oof_lora.parquet')[['cve_id','lora_control']],on='cve_id').merge(h,on='cve_id')
o['nov4']=(o.kev_date_added.notna()&(o.kev_date_added<=pd.to_datetime(o.published.dt.year.astype(str)+'-11-04'))).astype(int)
r=lambda c:o[c].rank(pct=True)
for c in ['lr_full_llmx_noemb','lr_best_fresh']:
    o['avg_'+c]=(r(c)+r('lora_control'))/2
K=517; rng=np.random.default_rng(0)
def topk(s,y,idx):
    ss,yy=s[idx],y[idx]; return yy[np.argpartition(-ss,K)[:K]].sum()
pairs=[('lr_best_fresh','lr_full_llmx_noemb'),('avg_lr_best_fresh','avg_lr_full_llmx_noemb')]
for lab in ['label','nov4']:
    y=o[lab].to_numpy(); print(f'\nlabels: {lab} ({y.sum()} positives), hits@{K} and AP; paired bootstrap of the difference (A - B)')
    for a,b in pairs:
        A,B=o[a].to_numpy(),o[b].to_numpy()
        d_h,d_ap=[],[]
        for _ in range(500):
            i=rng.integers(0,len(y),len(y)); d_h.append(topk(A,y,i)-topk(B,y,i)); d_ap.append(ap(y[i],A[i])-ap(y[i],B[i]))
        full=lambda s:int(y[np.argsort(-s)[:K]].sum())
        print(f'  {a[:44]:44} {full(A):3d} AP {ap(y,A):.4f} | {b[:30]:30} {full(B):3d} AP {ap(y,B):.4f} | '
              f'diff hits 95% CI {np.percentile(d_h,2.5):+.0f}..{np.percentile(d_h,97.5):+.0f}, AP {np.percentile(d_ap,2.5):+.4f}..{np.percentile(d_ap,97.5):+.4f}')
