# Analysis script (run from chen/ with IBM/data linked; see README). Produced numbers in context_Chen.md.
import pandas as pd, numpy as np
from sklearn.metrics import average_precision_score as ap
h=pd.read_parquet('processed/historical_2024_2025.parquet',columns=['cve_id','published','snapshot_month','assigner','kev_date_added'])
def prep(f):
    o=pd.read_parquet(f).merge(h,on='cve_id')
    o['nov4']=(o.kev_date_added.notna()&(o.kev_date_added<=pd.to_datetime(o.published.dt.year.astype(str)+'-11-04'))).astype(int)
    return o
def table(o,cols,title,lora=True):
    n=len(o); K=round(500*n/72670); rng=np.random.default_rng(0)
    print(f'\n{title}: {n:,} CVEs, {int(o.label.sum())} / {int(o.nov4.sum())} positives (Dec31 / Nov4), K={K}')
    base=o.baseline_code_epss.to_numpy()
    rows=[]
    for c in cols:
        s=o[c].to_numpy()
        def hits(s,y,cap=0):
            d=o.assign(_s=s).sort_values('_s',ascending=False)
            if cap: d=d[d.groupby(['assigner','snapshot_month']).cumcount()<cap]
            return int(d[y].head(K).sum())
        y=o.nov4.to_numpy(); diffs=[]
        for _ in range(500):
            i=rng.integers(0,n,n); t=lambda v: y[i][np.argpartition(-v[i],K)[:K]].sum(); diffs.append(t(s)-t(base))
        rows.append({'model':c,'hits Dec31':hits(s,'label'),'hits Nov4':hits(s,'nov4'),'Nov4 cap3':hits(s,'nov4',3),
                     'Nov4 vs baseline 95% CI':f'{np.percentile(diffs,2.5):+.0f}..{np.percentile(diffs,97.5):+.0f}',
                     'AP Dec31':round(ap(o.label,s),4),'AP Nov4':round(ap(o.nov4,s),4)})
    print(pd.DataFrame(rows).to_string(index=False))
r=lambda o,c:o[c].rank(pct=True)
# 5-fold
o=prep('results/cv_oof_lr.parquet').merge(pd.read_parquet('results/cv_oof_lora.parquet')[['cve_id','lora_control']],on='cve_id')
o['avg_best_lora']=(r(o,'lr_full_llmx_noemb')+r(o,'lora_control'))/2
o['avg_best_lgbm']=(r(o,'lr_full_llmx_noemb')+r(o,'lgbm_best'))/2
o['avg_best_lora_lgbm']=(r(o,'lr_full_llmx_noemb')+r(o,'lora_control')+r(o,'lgbm_best'))/3
o['avg_best_lora_lgbmluis']=(r(o,'lr_full_llmx_noemb')+r(o,'lora_control')+r(o,'lgbm_luis_only'))/3
table(o,['baseline_code_epss','lr_full_llmx_noemb','lr_best_luis_sib','lr_best_luis_all','lr_luis_only','lgbm_luis_only','lgbm_best','lora_control',
         'avg_best_lora','avg_best_lgbm','avg_best_lora_lgbm','avg_best_lora_lgbmluis'],'5-fold purged CV (all blocks)')
# forward-only
f=prep('results/cv_fwd_lr.parquet'); f=f[f.fold>0].copy()
f['avg_best_lgbm']=(r(f,'lr_full_llmx_noemb')+r(f,'lgbm_best'))/2
f['avg_bestluis_lgbmluis']=(r(f,'lr_best_luis_all')+r(f,'lgbm_luis_only'))/2
table(f,['baseline_code_epss','lr_features','lr_features_meta_embknn_pca64_team','lr_full_llmx_noemb','lr_best_luis_all','lgbm_luis_only','lgbm_best','avg_best_lgbm','avg_bestluis_lgbmluis'],'FORWARD-ONLY (blocks 1-4, each trained only on earlier blocks)')
# same blocks 1-4 from the 5-fold run, for comparison
o4=o[o.fold>0].copy()
table(o4,['baseline_code_epss','lr_full_llmx_noemb','lgbm_best','avg_best_lgbm','avg_best_lora'],'5-fold CV restricted to blocks 1-4 (same rows as forward-only)')
