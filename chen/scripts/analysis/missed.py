# Run from chen/scripts/ (uses ../processed, ../results, ../IBM/data).
# Analysis script (run from chen/ with IBM/data linked; see README). Produced numbers in context_Chen.md.
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from stack import features, FLAGS
R='../results/'
h=pd.read_parquet('../processed/historical_2024_2025.parquet')
tr=h[h.snapshot_month.between('2024-01','2024-12')].copy(); te=h[h.snapshot_month.between('2025-01','2025-09')].copy()
for n,(a,b) in {'zero':('all2024_zero','holdout_zero'),'many':('all2024_logo','holdout_many')}.items():
    tr[n]=tr.cve_id.map(pd.read_parquet(R+a+'.parquet').set_index('cve_id').score).clip(lower=-30)
    te[n]=te.cve_id.map(pd.read_parquet(R+b+'.parquet').set_index('cve_id').score).clip(lower=-30)
A=pd.concat([features(tr),tr[['zero','many']]],axis=1); B=pd.concat([features(te),te[['zero','many']]],axis=1)
sc=StandardScaler().fit(A); lr=LogisticRegression(class_weight='balanced',max_iter=5000).fit(sc.transform(A),tr.label)
te['stack']=lr.decision_function(sc.transform(B))
te['code']=te[FLAGS].max(axis=1); te['base']=te.code*10+te.epss.fillna(0)
for c in ['base','zero','many','stack']: te['r_'+c]=te[c].rank(ascending=False,method='min').astype(int)
te['best_rank']=te[['r_base','r_zero','r_many','r_stack']].min(axis=1)
k=pd.read_csv('../IBM/data/known_exploited.csv',usecols=['cve_id','vendor','product','name','date_added','known_ransomware_use'])
p=te[te.label==1].merge(k,on='cve_id')
p['epss_pct']=te.groupby('snapshot_month').epss.rank(pct=True).reindex(p.index) if False else None
pct=te.set_index('cve_id').groupby('snapshot_month').epss.rank(pct=True)
p['epss_pct']=(100*p.cve_id.map(pct)).round(0)
miss=p[p.best_rank>1000].sort_values('best_rank',ascending=False); hit=p[p.best_rank<=1000]
print(f'2025 positives: {len(p)}; missed by every method (best rank > 1000): {len(miss)}\n')
pd.set_option('display.width',250)
for r in miss.itertuples():
    pub_lag=(pd.Timestamp(r.date_added)-r.published).days
    print(f'{r.cve_id} | best {r.best_rank} (base {r.r_base} zero {r.r_zero} many {r.r_many} stack {r.r_stack}) | pub {r.published:%Y-%m-%d} -> KEV +{pub_lag}d | ransomware {r.known_ransomware_use}')
    print(f'   {r.vendor} / {r.product} | assigner {r.assigner} | CVSS {r.cvss3_score} {r.attack_vector} PR:{r.privileges_required} UI:{r.user_interaction} | CVSS4 {r.cvss4_score} | CWE {r.cwe_ids} | EPSS pct {r.epss_pct} | code {int(r.code)}')
    print(f'   {" ".join(str(r.description).split())[:300]}\n')
def summ(d):
    return pd.Series({'n':len(d),'days to KEV (median)':(pd.to_datetime(d.date_added)-d.published).dt.days.median(),
      'CVSS3 missing %':100*d.cvss3_score.isna().mean(),'CVSS3 median':d.cvss3_score.median(),'network %':100*(d.attack_vector=='NETWORK').mean(),
      'UI required %':100*(d.user_interaction=='REQUIRED').mean(),'PR none %':100*(d.privileges_required=='NONE').mean(),
      'any code %':100*d.code.mean(),'EPSS pct median':d.epss_pct.median(),'ransomware %':100*(d.known_ransomware_use=='Known').mean(),
      'desc length median':d.description.str.len().median(),'Microsoft %':100*(d.vendor=='Microsoft').mean()})
print(pd.DataFrame({'missed':summ(miss),'found':summ(hit)}).round(1).to_string())
print('\nvendors missed:', miss.vendor.value_counts().to_dict())
print('vendors found:', hit.vendor.value_counts().to_dict())

# --- does the vendor already have KEV history (entries added before this CVE's snapshot month ended)?
import re
kev=pd.read_csv('../IBM/data/known_exploited.csv',usecols=['vendor','date_added']); kev['date_added']=pd.to_datetime(kev.date_added)
vend=sorted({v for v in kev.vendor.dropna().unique() if len(v)>=4 and v.lower() not in {'multiple','unknown','apple, google'}}, key=len, reverse=True)
pat={v: re.compile(r'\b'+re.escape(v.lower())+r'\b') for v in vend}
def prior_kev(desc, snap_end):
    d=str(desc).lower(); hit=[v for v in vend if pat[v].search(d)]
    if not hit: return 0
    return int(((kev.vendor.isin(hit)) & (kev.date_added<=snap_end)).sum())
te['snap_end']=pd.PeriodIndex(te.snapshot_month,freq='M').to_timestamp(how='end')
samp=pd.concat([te[te.label==1], te[te.label==0].sample(3000,random_state=0)])
samp['prior_kev']=[prior_kev(d,s) for d,s in zip(samp.description,samp.snap_end)]
pk=samp.set_index('cve_id').prior_kev
for name,d in [('missed positives',miss),('found positives',hit),('random negatives (3000)',te[te.label==0].sample(3000,random_state=0))]:
    v=d.cve_id.map(pk)
    print(f'{name:26} vendor in description has prior KEV entries: {100*(v>0).mean():4.0f}%   median count {v.median():.0f}')
print('\nmissed positives, prior KEV count of named vendor:')
print(miss.assign(pk=miss.cve_id.map(pk))[['cve_id','vendor','pk']].to_string(index=False))
