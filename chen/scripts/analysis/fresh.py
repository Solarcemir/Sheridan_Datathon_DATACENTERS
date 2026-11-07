# Analysis script (run from chen/ with IBM/data linked; see README). Produced numbers in context_Chen.md.
import pandas as pd, numpy as np
D='IBM/data/'
vul=pd.read_csv(D+'vulnerabilities.csv',usecols=['cve_id','published','assigner','description']); vul['published']=pd.to_datetime(vul.published)
sb=pd.read_csv(D+'scoreboard_2026.csv'); kev=pd.read_csv(D+'known_exploited.csv'); kev['date_added']=pd.to_datetime(kev.date_added)
v26=vul[vul.published.between('2026-01-01','2026-09-30')].copy()
miss=v26[~v26.cve_id.isin(sb.cve_id)].copy()
miss['snap']=miss.published.dt.strftime('%Y-%m').replace({'2026-06':'2026-07'})
print('hidden 2026 KEV by snapshot month:', miss.snap.value_counts().sort_index().to_dict())
# how many scoreboard CVEs would get a "same assigner had a fresh KEV at or before my snapshot (last 3 months)" flag
sb2=sb.merge(vul[['cve_id','assigner']],on='cve_id')
def mi(m): return int(m[:4])*12+int(m[5:7])
miss['m']=miss.snap.map(mi); sb2['m']=sb2.snapshot_month.map(mi)
fresh=miss.groupby('assigner').m.apply(list).to_dict()
sb2['fresh3']=[any(0<=M-x<=2 for x in fresh.get(a,[])) for a,M in zip(sb2.assigner,sb2.m)]
print(f'scoreboard CVEs whose assigner had a hidden KEV in the same or previous 2 snapshot months: {int(sb2.fresh3.sum()):,} of {len(sb2):,}')
# mirror on 2024-25: hidden = same-year CVEs added to KEV within their publish month
h=pd.read_parquet('processed/historical_2024_2025.parquet'); h=h[h.snapshot_month<='2025-09'].copy()
k=kev.merge(vul[['cve_id','published','assigner']],on='cve_id')
k=k[k.published.between('2024-01-01','2025-09-30')]
k=k[k.date_added<=k.published+pd.offsets.MonthEnd(0)].copy()
k['m']=k.published.dt.strftime('%Y-%m').map(mi)
fr=k.groupby('assigner').m.apply(list).to_dict()
h['m']=h.snapshot_month.map(mi)
h['fresh3']=[any(0<=M-x<=2 for x in fr.get(a,[])) for a,M in zip(h.assigner,h.m)]
s=h.fresh3
print(f'\n2024-25 mirror: {len(k)} same-year in-month KEV CVEs')
print(f'  CVEs whose assigner had one in the same or previous 2 months: {int(s.sum()):,} ({100*s.mean():.1f}%), positives {int(h.label[s].sum())} of {int(h.label.sum())}')
print(f'  KEV rate {100*h.label[s].mean():.3f}% vs {100*h.label[~s].mean():.3f}%  -> x{h.label[s].mean()/h.label[~s].mean():.1f}')
# same release (same assigner + same publish day) sibling went straight to KEV
k['d']=k.published
sib=set(zip(k.assigner,k.d))
h['sib_fresh']=[(a,p) in sib for a,p in zip(h.assigner,h.published)]
s=h.sib_fresh
print(f'  sibling in the same release went straight to KEV: {int(s.sum()):,} CVEs, positives {int(h.label[s].sum())}, KEV rate {100*h.label[s].mean():.2f}% (x{h.label[s].mean()/h.label[~s].mean():.0f})')
print('  examples:', h[s & (h.label==1)][['cve_id','assigner']].head(8).values.tolist())
