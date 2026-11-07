# Run from chen/scripts/ (uses ../processed, ../results, ../IBM/data).
# Analysis script (run from chen/ with IBM/data linked; see README). Produced numbers in context_Chen.md.
import re, pandas as pd, numpy as np
h=pd.read_parquet('../processed/historical_2024_2025.parquet')
h=h[h.snapshot_month<='2025-09'].copy()
d=h.description.fillna('').str.lower()
v4=h.cvss4_vector.fillna(''); v3=h.cvss3_vector.fillna('')
def m(vec,key):  # metric value from a CVSS vector
    return vec.str.extract(r'(?:^|/)'+key+r':([A-Za-z]+)')[0]
h['id_year']=h.cve_id.str[4:8].astype(int); h['pub_year']=h.published.dt.year
same_day=h.groupby(['assigner','published']).cve_id.transform('size')
dup_desc=h.groupby(['assigner','published','description']).cve_id.transform('size')
pt=(h.published.dt.weekday==1)&(h.published.dt.day.between(8,14))
sig={
 'CVSS4 E:A (CNA says attacked)': m(v4,'E')=='A',
 'CVSS4 E:P (PoC exists)': m(v4,'E')=='P',
 'CVSS4 AU:Y (automatable)': m(v4,'AU')=='Y',
 'CVSS4 U:Red/Amber (provider urgency)': m(v4,'U').isin(['Red','Amber']),
 'CVSS3 temporal E:H/F/P present': m(v3,'E').isin(['H','F','P']),
 'desc: "in the wild"/"actively exploited"/"aware of"': d.str.contains(r'in the wild|actively exploited|aware of (?:a )?reports?|limited,? (?:highly )?targeted|has been exploited|may have been exploited'),
 'desc: VulDB "exploit has been disclosed to the public"': d.str.contains('exploit has been disclosed to the public'),
 'desc: EOL / unsupported / end-of-life': d.str.contains(r'end[- ]of[- ]life|\beol\b|no longer supported|unsupported when assigned'),
 'desc: unauthenticated': d.str.contains('unauthenticated'),
 'ID year < publish year (old reserved ID)': h.id_year<h.pub_year,
 'ID year <= publish year - 2': h.id_year<=h.pub_year-2,
 'single CVE from assigner that day (out-of-band)': same_day==1,
 'assigner published >=20 CVEs that day (patch drop)': same_day>=20,
 'identical description, same assigner+day (twins)': dup_desc>=2,
 'published on Patch Tuesday': pt,
 'CVSS 4 present': h.cvss4_score.notna(),
}
base=h.label.mean()
rows=[]
for k,s in sig.items():
    s=s.fillna(False).astype(bool)
    n=int(s.sum()); pos=int(h.label[s].sum())
    rate=h.label[s].mean() if n else np.nan; rest=h.label[~s].mean()
    rows.append((k,n,pos,round(100*rate,3) if n else None, round(rate/rest,1) if n and rest>0 else None))
print(f'2024-01..2025-09: {len(h):,} CVEs, {int(h.label.sum())} positives, base KEV rate {100*base:.2f}%\n')
print(pd.DataFrame(rows,columns=['signal','CVEs','positives','KEV rate %','x vs rest']).to_string(index=False))
# CWE-level and CNA-type
cna=h.assigner.str.lower()
typ=np.select([cna.str.contains('mitre'),cna.str.contains('vuldb'),cna.str.contains('github'),cna.str.contains('wordfence|patchstack|wpscan'),cna.str.contains('zdi'),cna.str.contains('vulncheck'),cna.str.contains('psirt|secure@|security@|secalert|product')],
   ['MITRE','VulDB','GitHub','WordPress CNAs','ZDI','VulnCheck','vendor PSIRT-like'],'other/UUID')
print('\nby assigner type:'); print(h.assign(t=typ).groupby('t').label.agg(['size','sum','mean']).assign(mean=lambda x:(100*x['mean']).round(3)).sort_values('mean',ascending=False).to_string())
