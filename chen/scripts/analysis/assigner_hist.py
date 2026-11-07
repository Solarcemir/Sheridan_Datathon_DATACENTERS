# Run from chen/scripts/ (uses ../processed, ../results, ../IBM/data).
# Analysis script (run from chen/ with IBM/data linked; see README). Produced numbers in context_Chen.md.
import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
exec(open(sys.argv[1]).read().split('# --- does the vendor')[0].replace("print(","(lambda *a,**k:None)("))  # reuse te/miss/hit quietly
v=pd.read_csv('../IBM/data/vulnerabilities.csv',usecols=['cve_id','published','assigner']); v['published']=pd.to_datetime(v.published)
kev=pd.read_csv('../IBM/data/known_exploited.csv',usecols=['cve_id','date_added']); kev['date_added']=pd.to_datetime(kev.date_added)
kv=kev.merge(v[['cve_id','assigner']],on='cve_id',how='left')
te['snap_end']=pd.PeriodIndex(te.snapshot_month,freq='M').to_timestamp(how='end')
# per (assigner, snapshot month): KEV entries added by then, CVEs published in the prior 3 years
rows=[]
for (a,se),g in te.groupby(['assigner','snap_end']):
    n_kev=int(((kv.assigner==a)&(kv.date_added<=se)).sum())
    n_cve=int(((v.assigner==a)&(v.published<=se)&(v.published>se-pd.DateOffset(years=3))).sum())
    rows.append((a,se,n_kev,n_cve))
H=pd.DataFrame(rows,columns=['assigner','snap_end','prior_kev','cves_3y'])
te2=te.merge(H,on=['assigner','snap_end'],how='left')
te2['kev_rate']=te2.prior_kev/te2.cves_3y.clip(lower=1)
def show(name,d):
    d=te2[te2.cve_id.isin(d.cve_id)]
    print(f'{name:22} assigner has prior KEV: {100*(d.prior_kev>0).mean():4.0f}% | median prior KEV {d.prior_kev.median():5.0f} | median KEV per CVE {d.kev_rate.median():.4f}')
show('missed positives',miss); show('found positives',hit); show('all negatives',te[te.label==0])
m=te2[te2.cve_id.isin(miss.cve_id)][['cve_id','assigner','prior_kev','cves_3y','kev_rate']]
print(m.assign(kev_rate=m.kev_rate.round(4)).to_string(index=False))
# how strong is it as a ranker on its own, and among CVEs with no exploit code?
for c in ['kev_rate','prior_kev']:
    o=te2.sort_values(c,ascending=False)
    print(f'\nrank by assigner {c}: hits@243 {int(o.label.head(243).sum())}, hits@1000 {int(o.label.head(1000).sum())}')
nc=te2[te2.code==0].sort_values('kev_rate',ascending=False)
print(f'among no-exploit-code CVEs ({len(nc):,}, {int(nc.label.sum())} positives): top 1000 by assigner KEV rate holds {int(nc.label.head(1000).sum())}')
