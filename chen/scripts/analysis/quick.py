# Analysis script (run from chen/ with IBM/data linked; see README). Produced numbers in context_Chen.md.
import pandas as pd, numpy as np
from sklearn.metrics import average_precision_score as ap
h=pd.read_parquet('processed/historical_2024_2025.parquet',columns=['cve_id','published','snapshot_month','assigner','kev_date_added'])
o=pd.read_parquet('results/cv_oof_lr.parquet').merge(pd.read_parquet('results/cv_oof_lora.parquet')[['cve_id','lora_control']],on='cve_id').merge(h,on='cve_id')
o['nov4']=(o.kev_date_added.notna()&(o.kev_date_added<=pd.to_datetime(o.published.dt.year.astype(str)+'-11-04'))).astype(int)
tf=pd.read_parquet('processed/team_feat_historical.parquet').set_index('cve_id')
o['text_kev_lr_alone']=o.cve_id.map(tf.text_kev_lr)
r=lambda c:o[c].rank(pct=True)
o['avg_lr_lora']=(r('lr_features_meta_embknn_pca64')+r('lora_control'))/2
o['avg_lrteam_lora']=(r('lr_features_meta_embknn_pca64_team')+r('lora_control'))/2
o['avg_lrteam_lora_lgbm']=(r('lr_features_meta_embknn_pca64_team')+r('lora_control')+r('lgbm_features_meta_embknn_team'))/3
o['avg_lrteam_lgbm']=(r('lr_features_meta_embknn_pca64_team')+r('lgbm_features_meta_embknn_team'))/2
o['avg_age_prior_lora']=(r('lr_full_age_prior')+r('lora_control'))/2
o['avg_llmx_lora']=(r('lr_features_meta_embknn_pca64_team_llmx')+r('lora_control'))/2
o['avg_llmx_prior_lora']=(r('lr_full_llmx_prior')+r('lora_control'))/2
K=517
def hits(c,lab,cap):
    d=o.sort_values(c,ascending=False)
    if cap: d=d[d.groupby(['assigner','snapshot_month']).cumcount()<cap]
    return int(d[lab].head(K).sum())
cols=['baseline_code_epss','lr_features','lr_features_llmx','lr_features_meta_embknn_pca64_team','lr_full_prior','lr_features_meta_embknn_pca64_team_llmx','lr_full_llmx_prior','lora_control','avg_lrteam_lora','avg_llmx_lora','avg_llmx_prior_lora']
rows=[]
for c in cols:
    rows.append({'model':c,'hits our':hits(c,'label',0),'hits nov4':hits(c,'nov4',0),'nov4 cap3':hits(c,'nov4',3),'nov4 cap5':hits(c,'nov4',5),
                 'AP our':round(ap(o.label,o[c]),4),'AP nov4':round(ap(o.nov4,o[c]),4)})
print(f"119 positives (our label) / {o.nov4.sum()} (Nov 4); top {K}")
print(pd.DataFrame(rows).to_string(index=False))
o[['cve_id','avg_lrteam_lora','avg_lrteam_lgbm','avg_lrteam_lora_lgbm','text_kev_lr_alone']].to_parquet('results/cv_oof_avg.parquet',index=False)
