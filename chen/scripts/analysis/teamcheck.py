# Analysis script (run from chen/ with IBM/data linked; see README). Produced numbers in context_Chen.md.
import pandas as pd, numpy as np
h=pd.read_parquet('processed/historical_2024_2025.parquet',columns=['cve_id','published','snapshot_month','assigner','label','kev_date_added'])
o=pd.read_parquet('results/cv_oof_lr.parquet').merge(pd.read_parquet('results/cv_oof_lora.parquet'),on='cve_id',how='left').merge(h,on=['cve_id'],suffixes=('','_h'))
o['label_nov4']=((o.kev_date_added.notna()) & (o.kev_date_added<=pd.to_datetime(o.published.dt.year.astype(str)+'-11-04'))).astype(int)
print(f"positives: our label (KEV by 2025-12-31) {int(o.label.sum())} | judge-aligned (KEV by Nov 4 of publish year) {int(o.label_nov4.sum())}")
K=517
def capped(df, col, cap):
    d=df.sort_values(col,ascending=False)
    if cap: d=d[d.groupby(['assigner','snapshot_month']).cumcount()<cap]
    return d.head(K)
cols=['baseline_code_epss','lr_features_meta','lr_features_meta_embknn','lr_features_meta_embknn_pca64','lr_features_meta_embknn_pca64_granite','lora_control','avg_lr_lora']
rows=[]
for c in cols:
    if c not in o or o[c].isna().any(): continue
    r={'model':c}
    for cap in [None,3,5,8]:
        t=capped(o,c,cap); r[f'cap{cap or "-"} our']=int(t.label.sum()); r[f'cap{cap or "-"} nov4']=int(t.label_nov4.sum())
    t=capped(o,c,None); r['max per assigner-month']=int(t.groupby(['assigner','snapshot_month']).size().max())
    r['top assigner-month']=t.groupby(['assigner','snapshot_month']).size().idxmax()[0][:22]
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
