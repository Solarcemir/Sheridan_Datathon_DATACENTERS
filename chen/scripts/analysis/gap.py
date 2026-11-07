# Analysis script (run from chen/ with IBM/data linked; see README). Produced numbers in context_Chen.md.
import sys, pandas as pd
d = sys.argv[1]
v = pd.read_csv(f"{d}/vulnerabilities.csv", usecols=["cve_id","published","nvd_status","cvss3_score","cvss3_source","cvss4_score","cwe_ids","n_affected_cpes","n_references"], dtype=str)
kev = set(pd.read_csv(f"{d}/known_exploited.csv", usecols=["cve_id"]).cve_id)
sb = set(pd.read_csv(f"{d}/scoreboard_2026.csv", usecols=["cve_id"]).cve_id)
v["yr"] = v.published.str[:4]
v["grp"] = v.yr.where(v.yr.isin(["2024","2025"]), None)
v.loc[v.cve_id.isin(sb), "grp"] = "2026sb"
v = v[v.grp.notna()]
v["kev"] = v.cve_id.isin(kev)
nvd = "nvd@nist.gov"
def summ(g):
    return pd.Series({
      "n": len(g), "kev": int(g.kev.sum()),
      "analyzed%": 100*g.nvd_status.isin(["Analyzed","Modified"]).mean(),
      "deferred%": 100*(g.nvd_status=="Deferred").mean(),
      "has_cvss3%": 100*g.cvss3_score.notna().mean(),
      "cvss3_by_nvd%": 100*(g.cvss3_source==nvd).mean(),
      "has_cvss4%": 100*g.cvss4_score.notna().mean(),
      "has_cwe%": 100*g.cwe_ids.notna().mean(),
      "has_cpes%": 100*(g.n_affected_cpes.fillna("0").astype(int)>0).mean(),
    })
pd.set_option("display.width", 200)
print(v.groupby("grp").apply(summ).round(1).T)
print("\nstatus by group:"); print(pd.crosstab(v.nvd_status, v.grp))
print("\nKEV rows only (2024/25), analyzed%:", v[v.kev].groupby("grp").apply(lambda g: round(100*g.nvd_status.isin(['Analyzed','Modified']).mean(),1)).to_dict())
