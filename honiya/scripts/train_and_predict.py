import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Build relative path from script location to IBM data root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "../../IBM-20261107T174321Z-1-001/IBM/data/data")

print(f"[+] Loading raw datasets from: {DATA_DIR}")
vulns = pd.read_csv(os.path.join(DATA_DIR, "vulnerabilities.csv"))
signals_24 = pd.read_csv(os.path.join(DATA_DIR, "exploit_signals_2024.csv"))
signals_25 = pd.read_csv(os.path.join(DATA_DIR, "exploit_signals_2025.csv"))
kev = pd.read_csv(os.path.join(DATA_DIR, "known_exploited.csv"))
scoreboard = pd.read_csv(os.path.join(DATA_DIR, "scoreboard_2026.csv"))

# Drop leak columns
LEAK_COLUMNS = ['nvd_status', 'last_modified', 'n_affected_cpes', 'n_references', 'cvss3_source', 'cvss2_score', 'cvss2_severity', 'cvss2_vector']
vulns = vulns.drop(columns=[col for col in LEAK_COLUMNS if col in vulns.columns])

# Dates & Target
vulns['pub_date'] = pd.to_datetime(vulns['published'], errors='coerce')
vulns['pub_year'] = vulns['pub_date'].dt.year
vulns['pub_month'] = vulns['pub_date'].dt.to_period('M').astype(str)

df_hist = vulns[vulns['pub_year'].isin([2024, 2025])].copy().reset_index(drop=True)
kev['date_added'] = pd.to_datetime(kev['date_added'], errors='coerce')
kev_dict = dict(zip(kev['cve_id'], kev['date_added']))
df_hist['kev_date'] = df_hist['cve_id'].map(kev_dict)

df_hist['target_exploited'] = (df_hist['kev_date'].notna() & (df_hist['kev_date'] > df_hist['pub_date'])).astype(int)
df_hist = df_hist[~(df_hist['kev_date'].notna() & (df_hist['kev_date'] <= df_hist['pub_date']))].copy()

# Feature engineering
all_signals = pd.concat([signals_24, signals_25], ignore_index=True)
df_hist = df_hist.merge(all_signals, left_on=['cve_id', 'pub_month'], right_on=['cve_id', 'month'], how='left')

EXPLOIT_FLAGS = ['poc_github', 'nuclei', 'metasploit', 'exploitdb']
for col in EXPLOIT_FLAGS:
    df_hist[col] = df_hist[col].fillna(0).astype(int)

df_hist['has_any_exploit'] = (df_hist[EXPLOIT_FLAGS].sum(axis=1) > 0).astype(int)
df_hist['epss'] = df_hist['epss'].fillna(0.0)
df_hist['epss_missing'] = df_hist['epss'].isna().astype(int)
df_hist['cvss_score'] = df_hist['cvss3_score'].fillna(df_hist.get('cvss4_score', 0)).fillna(0.0)
df_hist['is_network'] = (df_hist['attack_vector'] == 'NETWORK').astype(int)
df_hist['is_low_complexity'] = (df_hist['attack_complexity'] == 'LOW').astype(int)
df_hist['is_unauth'] = (df_hist['privileges_required'] == 'NONE').astype(int)
df_hist['is_zero_click'] = (df_hist['user_interaction'] == 'NONE').astype(int)
df_hist['is_scope_changed'] = (df_hist['scope'] == 'CHANGED').astype(int)
df_hist['deadly_triad'] = ((df_hist['is_network'] == 1) & (df_hist['is_unauth'] == 1) & (df_hist['is_zero_click'] == 1)).astype(int)

df_hist['cwe_ids'] = df_hist['cwe_ids'].fillna("")
df_hist['is_cwe_unknown'] = df_hist['cwe_ids'].str.contains('NVD-CWE-noinfo|NVD-CWE-Other', case=False).astype(int)
df_hist['assigner'] = df_hist['assigner'].fillna("unknown")
daily_batch = df_hist.groupby(['assigner', 'published'])['cve_id'].transform('count')
df_hist['is_out_of_band'] = (daily_batch == 1).astype(int)
df_hist['is_patch_dump'] = (daily_batch >= 20).astype(int)
df_hist['is_wordpress_cna'] = df_hist['assigner'].str.contains('wordfence|patchstack|wpscan', case=False).astype(int)
df_hist['is_vulncheck'] = df_hist['assigner'].str.contains('vulncheck', case=False).astype(int)
cve_id_year = df_hist['cve_id'].str.extract(r'CVE-(\d{4})-')[0].astype(float)
df_hist['is_old_reserved_id'] = (cve_id_year < df_hist['pub_year']).astype(int)

desc = df_hist['description'].fillna("").str.lower()
df_hist['txt_unauth'] = desc.str.contains(r'unauthenticat|pre-auth', regex=True).astype(int)
df_hist['txt_auth_bypass'] = desc.str.contains(r'auth.*bypass|bypass.*auth', regex=True).astype(int)
df_hist['txt_rce'] = desc.str.contains(r'remote code execution|\brce\b', regex=True).astype(int)
df_hist['txt_eol'] = desc.str.contains(r'end of life|end-of-life|\beol\b|unsupported', regex=True).astype(int)

features = [
    'epss', 'epss_missing', 'cvss_score', 'poc_github', 'nuclei', 'metasploit', 'exploitdb', 'has_any_exploit',
    'is_network', 'is_low_complexity', 'is_unauth', 'is_zero_click', 'is_scope_changed',
    'deadly_triad', 'is_cwe_unknown', 'is_out_of_band', 'is_patch_dump', 'is_wordpress_cna',
    'is_vulncheck', 'is_old_reserved_id', 'txt_unauth', 'txt_auth_bypass', 'txt_rce', 'txt_eol'
]

# Train
train_df = df_hist[df_hist['pub_year'] == 2024]
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(penalty='l2', C=0.003, class_weight='balanced', max_iter=2000, random_state=42))
])
pipe.fit(train_df[features], train_df['target_exploited'])
print("[+] Model trained successfully with optimal C=0.003.")