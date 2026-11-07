# Analysis script (run from chen/ with IBM/data linked; see README). Produced numbers in context_Chen.md.
import re, pandas as pd, numpy as np
D='IBM/data/'
vul=pd.read_csv(D+'vulnerabilities.csv',usecols=['cve_id','published','assigner','description','nvd_status']); vul['published']=pd.to_datetime(vul.published)
sb=pd.read_csv(D+'scoreboard_2026.csv'); kev=pd.read_csv(D+'known_exploited.csv'); kev['date_added']=pd.to_datetime(kev.date_added)
# A. which Jan-Sep 2026 CVEs are NOT on the scoreboard?
v26=vul[vul.published.between('2026-01-01','2026-09-30')]
miss=v26[~v26.cve_id.isin(sb.cve_id)]
print(f'A. 2026-01..09 published: {len(v26):,}; on scoreboard {v26.cve_id.isin(sb.cve_id).sum():,}; NOT on scoreboard {len(miss):,}')
print('   by month:', miss.published.dt.strftime('%m').value_counts().sort_index().to_dict())
print('   in KEV file (<=2025):', int(miss.cve_id.isin(kev.cve_id).sum()), '| nvd_status:', miss.nvd_status.value_counts().to_dict())
print('   top assigners:', miss.assigner.value_counts().head(12).to_dict())
for r in miss.sample(min(12,len(miss)),random_state=0).itertuples(): print(f'     {r.cve_id} {r.published:%Y-%m-%d} {r.assigner[:24]:24} {" ".join(str(r.description).split())[:100]}')
# B. how long was the CVE public at snapshot time? (day of month effect)
h=pd.read_parquet('processed/historical_2024_2025.parquet'); h=h[h.snapshot_month<='2025-09']
h['day']=h.published.dt.day; h['code']=h[['exploitdb','metasploit','nuclei','poc_github']].max(axis=1)
b=pd.cut(h.day,[0,10,20,31],labels=['1-10','11-20','21-31'])
print('\nB. by publish day-of-month:'); print(h.groupby(b,observed=True).agg(cves=('cve_id','size'),pos=('label','sum'),kev_rate=('label','mean'),has_code=('code','mean'),epss_med=('epss','median')).assign(kev_rate=lambda x:(100*x.kev_rate).round(3),has_code=lambda x:(100*x.has_code).round(2)).to_string())
# KEV CVEs dropped because already on KEV at snapshot, by day of month
k2=kev.merge(vul[['cve_id','published']],on='cve_id'); k2=k2[k2.published.between('2024-01-01','2025-09-30')]
k2['in_month']=k2.date_added<=(k2.published+pd.offsets.MonthEnd(0))
print('   KEV CVEs published 2024-01..2025-09 added within their publish month (so excluded from data):', int(k2.in_month.sum()), 'of', len(k2))
print('   share excluded by publish day bucket:', k2.groupby(pd.cut(k2.published.dt.day,[0,10,20,31]),observed=True).in_month.mean().round(2).to_dict())
# E. descriptions that mention another CVE ID
rx=re.compile(r'CVE-\d{4}-\d{4,7}')
h['refs']=[ [c for c in set(rx.findall(str(d))) if c!=i] for d,i in zip(h.description,h.cve_id)]
h['cut']=pd.to_datetime((h.published.dt.year-1).astype(str)+'-12-31')
kd=kev.set_index('cve_id').date_added
h['ref_any']=h.refs.str.len()>0
h['ref_kev']=[any(c in kd.index and kd[c]<=cut for c in rs) for rs,cut in zip(h.refs,h.cut)]
for c in ['ref_any','ref_kev']:
    s=h[c]; print(f'\nE. {c}: {int(s.sum())} CVEs, {int(h.label[s].sum())} positives, KEV rate {100*h.label[s].mean():.2f}% vs {100*h.label[~s].mean():.3f}%')
print('   examples:', h[h.ref_kev][['cve_id','label']].head(8).values.tolist())
# H. IBM X-Force references
ref=pd.read_csv(D+'references.csv',usecols=['cve_id','domain'])
xf=ref[ref.domain.fillna('').str.contains('xforce')].cve_id.unique()
vv=vul.set_index('cve_id')
yr=vv.published.dt.year.reindex(xf)
print(f'\nH. CVEs with X-Force links: {len(xf):,}; by publish year (last 6):', yr.value_counts().sort_index().tail(6).to_dict())
print('   top assigners:', vv.assigner.reindex(xf).value_counts().head(5).to_dict())
s=h.cve_id.isin(xf); print(f'   2024-25 table: {int(s.sum())} with X-Force link, {int(h.label[s].sum())} positives')
