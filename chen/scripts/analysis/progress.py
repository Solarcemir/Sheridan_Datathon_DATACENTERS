# Analysis script (run from chen/ with IBM/data linked; see README). Produced numbers in context_Chen.md.
import pandas as pd, numpy as np
from sklearn.metrics import average_precision_score as ap
h=pd.read_parquet('processed/historical_2024_2025.parquet',columns=['cve_id','published','snapshot_month','assigner','kev_date_added'])
o=pd.read_parquet('results/cv_oof_lr.parquet').merge(pd.read_parquet('results/cv_oof_lora.parquet')[['cve_id','lora_control']],on='cve_id').merge(h,on='cve_id')
o['nov4']=(o.kev_date_added.notna()&(o.kev_date_added<=pd.to_datetime(o.published.dt.year.astype(str)+'-11-04'))).astype(int)
o['best']=(o.lr_full_llmx_noemb.rank(pct=True)+o.lora_control.rank(pct=True))/2
o['base']=o.baseline_code_epss
K=517; rng=np.random.default_rng(0); n=len(o)
def capped_rank(s,cap):
    d=o.assign(_s=s).sort_values('_s',ascending=False)
    keep=d.groupby(['assigner','snapshot_month']).cumcount()<cap if cap else np.ones(len(d),bool)
    out=np.full(n,-1e9); out[d.index[keep]]=-np.arange(int(keep.sum()))
    return out
rows=[]
for lab in ['label','nov4']:
    y=o[lab].to_numpy()
    for cap in [0,3]:
        sb,sB=capped_rank(o.base.to_numpy(),cap),capped_rank(o.best.to_numpy(),cap)
        top=lambda s,i: y[i][np.argpartition(-s[i],K)[:K]].sum()
        hb,hB=int(y[np.argsort(-sb)[:K]].sum()),int(y[np.argsort(-sB)[:K]].sum())
        bs=[];bB=[];d=[];da=[]
        for _ in range(1000):
            i=rng.integers(0,n,n); a,b=top(sb,i),top(sB,i); bs.append(a); bB.append(b); d.append(b-a)
            if cap==0: da.append(ap(y[i],o.best.to_numpy()[i])-ap(y[i],o.base.to_numpy()[i]))
        q=lambda x:np.percentile(x,[2.5,97.5])
        rows.append({'labels':'KEV by Dec 31 2025' if lab=='label' else 'KEV by Nov 4 (judges)','positives':int(y.sum()),'cap':cap or 'none',
            'baseline hits':f'{hb} ({q(bs)[0]:.0f}-{q(bs)[1]:.0f})','best hits':f'{hB} ({q(bB)[0]:.0f}-{q(bB)[1]:.0f})',
            'difference':f'+{hB-hb} ({q(d)[0]:+.0f} to {q(d)[1]:+.0f})',
            'baseline AP':round(ap(y,o.base),3) if cap==0 else '','best AP':round(ap(y,o.best),3) if cap==0 else '',
            'AP diff CI':f'{q(da)[0]:+.3f} to {q(da)[1]:+.3f}' if cap==0 else ''})
print(pd.DataFrame(rows).to_string(index=False))
