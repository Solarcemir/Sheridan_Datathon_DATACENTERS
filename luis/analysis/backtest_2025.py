"""Backtest harness that mimics the 2026 benchmark on 2025 data.

Cohort  = CVEs published in month m (2025-01 .. 2025-09) that were NOT on KEV at the end of m.
Signals = EPSS + exploit flags from exploit_signals_2025.csv at the end of m (same as scoreboard_2026).
Label y = 1 if CISA added the CVE to KEV after the end of m and on or before 2025-12-31.

Outputs derived/backtest_cohorts_2025.csv and prints hits in the top 500 for the README baselines
plus a simple untrained heuristic. Columns marked LEAK_ are for leakage analysis only, never as features.

Usage:  python analysis/backtest_2025.py [DATA_DIR]
"""
import os
import sys

import numpy as np
import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "..", "IBM-20261107T174321Z-1-001", "IBM", "data", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "derived")
os.makedirs(OUT, exist_ok=True)

FLAGS = ["exploitdb", "metasploit", "nuclei", "poc_github"]
LABEL_END = pd.Timestamp("2025-12-31")

kev = pd.read_csv(f"{DATA}/known_exploited.csv", dtype=str, usecols=["cve_id", "date_added"])
kev_date = pd.to_datetime(kev.set_index("cve_id").date_added)

v = pd.read_csv(f"{DATA}/vulnerabilities.csv", dtype=str, usecols=[
    "cve_id", "published", "assigner", "description", "cvss3_score", "attack_vector",
    "attack_complexity", "privileges_required", "user_interaction", "scope", "cwe_ids",
    "nvd_status", "n_references", "n_affected_cpes"])
v["published"] = pd.to_datetime(v.published)
v["cvss3"] = pd.to_numeric(v.cvss3_score)

sig = pd.read_csv(f"{DATA}/exploit_signals_2025.csv", dtype={"month": str, "cve_id": str})

cohorts = []
for month in [f"2025-{i:02d}" for i in range(1, 10)]:
    end = pd.Timestamp(month) + pd.offsets.MonthEnd(0)
    c = v[v.published.dt.strftime("%Y-%m") == month].copy()
    c = c.merge(sig[sig.month == month][["cve_id", "epss"] + FLAGS], on="cve_id", how="left")
    added = c.cve_id.map(kev_date)
    c = c[~(added <= end)].copy()            # same rule as the scoreboard: not on KEV at snapshot
    added = c.cve_id.map(kev_date)
    c["y"] = ((added > end) & (added <= LABEL_END)).astype(int)
    c["snapshot_month"] = month
    cohorts.append(c)

c = pd.concat(cohorts, ignore_index=True)
c["epss"] = c.epss.fillna(0.0)
c[FLAGS] = c[FLAGS].fillna(0).astype(int)
c["any_flag"] = (c[FLAGS] > 0).any(axis=1).astype(int)
desc = c.description.fillna("")
c["txt_rce"] = desc.str.contains(r"remote code execution|\bRCE\b|execute arbitrary (?:code|commands)", case=False).astype(int)
c["txt_auth_bypass"] = desc.str.contains(r"authentication bypass|bypass authentication|unauthenticated", case=False).astype(int)
c["txt_in_the_wild"] = desc.str.contains(r"in the wild|actively exploited|may have been exploited", case=False).astype(int)
c["txt_wordpress"] = desc.str.contains(r"wordpress", case=False).astype(int)
c = c.rename(columns={"nvd_status": "LEAK_nvd_status", "n_references": "LEAK_n_references",
                      "n_affected_cpes": "LEAK_n_affected_cpes"})

keep = ["cve_id", "snapshot_month", "published", "y", "epss"] + FLAGS + [
    "any_flag", "cvss3", "attack_vector", "attack_complexity", "privileges_required",
    "user_interaction", "scope", "cwe_ids", "assigner", "txt_rce", "txt_auth_bypass",
    "txt_in_the_wild", "txt_wordpress", "LEAK_nvd_status", "LEAK_n_references", "LEAK_n_affected_cpes"]
c[keep].to_csv(os.path.join(OUT, "backtest_cohorts_2025.csv"), index=False)


def hits(score, k=500):
    order = np.argsort(-score.to_numpy(), kind="stable")
    return int(c.y.to_numpy()[order[:k]].sum())


print(f"rows={len(c):,}  positives={c.y.sum()}  base_rate={c.y.mean():.4%}")
print("top-500 hits")
print("  CVSS3 first          ", hits(c.cvss3.fillna(-1)))
print("  EPSS first           ", hits(c.epss))
print("  exploit flag, EPSS   ", hits(c.any_flag * 10 + c.epss))
heur = (c.epss.rank(pct=True) + 2 * c.any_flag + 3 * c.nuclei + 5 * c.txt_in_the_wild
        + 1 * c.txt_rce + 1 * c.txt_auth_bypass - 1 * c.txt_wordpress)
print("  simple heuristic     ", hits(heur))
print("  LEAK nvd_status=Analyzed share, positives vs negatives:",
      round((c[c.y == 1].LEAK_nvd_status == "Analyzed").mean(), 3),
      round((c[c.y == 0].LEAK_nvd_status == "Analyzed").mean(), 3))
