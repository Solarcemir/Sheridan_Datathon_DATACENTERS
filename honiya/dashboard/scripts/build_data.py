"""Precompute every small file the dashboard reads into dashboard/data/.

Usage: python scripts/build_data.py --repo /path/to/Sheridan_Datathon_DATACENTERS [--nvd nvd_2026.parquet]
The NVD file (description + CVSS, display only) comes from scripts/fetch_nvd.py.
"""
import argparse, json, re
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--nvd", default=None)
args = ap.parse_args()
R = Path(args.repo)
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)

GRANITE = ["vendor", "product", "product_type", "internet_exposed", "install_base", "used_by", "precondition",
           "impact", "end_of_life", "fix_status", "typical_attacker", "attacker_appeal"]
CVSS_COLS = ["cvss3", "cv_attack_vector", "cv_attack_complexity", "cv_privileges_required",
             "cv_user_interaction", "cv_scope", "has_cvss4"]

top = pd.read_csv(R / "top500.csv")
scored = pd.read_csv(R / "luis/derived/scored_2026_final.csv")
# final = chen/frontend/build_radar.py: tree + unified LR + 8B LoRA rank average, Granite demotion, cap 3
uni = pd.read_parquet(R / "chen/submissions/lr_unified_scoreboard_2026.parquet")
scored = scored.drop(columns=["linear"]).merge(uni, on="cve_id", how="left")
scored["linear"] = pd.Series(scored.lr_score.fillna(scored.lr_score.min()).to_numpy()).rank(pct=True).to_numpy()
raw = pd.Series(scored[["tree", "linear", "llm"]].mean(axis=1).to_numpy() - 50 * scored.granite_demoted.to_numpy())
nth = scored.assign(_s=raw).sort_values("_s", ascending=False, kind="stable") \
    .groupby(["assigner", "snapshot_month"]).cumcount()
raw[nth[nth >= 3].index] -= 100
scored["score_final"] = raw.to_numpy()
scored = scored.sort_values("score_final", ascending=False, kind="stable").reset_index(drop=True)
assert set(scored.cve_id.head(500)) == set(top.cve_id), "rebuilt top 500 differs from top500.csv"
order = {c: i for i, c in enumerate(top.cve_id)}
scored["_o"] = scored.cve_id.map(order).fillna(len(order) + scored.index.to_series())
scored = scored.sort_values("_o").drop(columns="_o").reset_index(drop=True)
scored["rank_final"] = range(1, len(scored) + 1)
lf = pd.read_parquet(R / "luis/derived/for_chen/luis_features_scoreboard_2026.parquet",
                     columns=["cve_id", "published", "epss_pct", "cwe_ids", "exploitdb", "metasploit", "nuclei",
                              "poc_github"] + CVSS_COLS)
ext = pd.read_parquet(R / "chen/processed/llm_extract.parquet", columns=["cve_id"] + GRANITE)
meta = pd.read_parquet(R / "chen/processed/meta_scoreboard.parquet",
                       columns=["cve_id", "eol_text", "assigner_kev", "product_name_kev", "vendor_name_kev"])
nvd = pd.read_parquet(args.nvd) if args.nvd and Path(args.nvd).exists() else None

# ---------- full scoreboard view (only for selecting rows; not written whole) ----------
sb = (scored[["cve_id", "rank_final", "score_final", "tree", "linear", "llm", "granite_demoted", "snapshot_month",
              "assigner", "epss", "any_flag"]]
      .merge(lf, on="cve_id", how="left").merge(ext, on="cve_id", how="left").merge(meta, on="cve_id", how="left"))
sb = sb.rename(columns={"vendor": "granite_vendor", "product": "granite_product"})

# ---------- blind spot, looking back (reports/missed.txt) ----------
GROUP = {}
for c in ["CVE-2025-6554", "CVE-2025-55177", "CVE-2025-21043", "CVE-2025-21042", "CVE-2025-38352",
          "CVE-2025-0411", "CVE-2025-24054", "CVE-2025-27915"]:
    GROUP[c] = "Targeted zero-days on consumer platforms"
for c in ["CVE-2024-13161", "CVE-2024-13160", "CVE-2024-13159", "CVE-2025-24472", "CVE-2024-53704",
          "CVE-2025-3935", "CVE-2025-6205", "CVE-2025-6204", "CVE-2024-57968", "CVE-2025-25181",
          "CVE-2025-23209", "CVE-2025-35939"]:
    GROUP[c] = "Chained / modest-looking edge & enterprise bugs"
for c in ["CVE-2025-9377", "CVE-2023-52163"]:
    GROUP[c] = "End-of-life botnet targets"
mt = (R / "chen/reports/missed.txt").read_text()
rows = []
for blk in re.split(r"\n(?=CVE-\d{4}-\d+ \|)", mt):
    m = re.match(r"(CVE-\d{4}-\d+) \| best (\d+).*?pub (\S+) -> KEV \+(\d+)d \| ransomware (\w+)\n\s+(.+?) \| assigner (\S+) \| CVSS (\S+) .*?\| CWE (\S+) \| EPSS pct (\S+) \| code (\d)\n?\s*(.*)",
                 blk.strip(), re.S)
    if m:
        rows.append(dict(cve_id=m[1], best_rank_any_method=int(m[2]), published=m[3], days_to_kev=int(m[4]),
                         ransomware=m[5], vendor_product=m[6], assigner=m[7], cvss=m[8], cwe=m[9],
                         epss_pct=m[10], exploit_code=int(m[11]), description=m[12].split("\n")[0][:300],
                         group=GROUP.get(m[1], "other")))
missed = pd.DataFrame(rows)
missed.to_csv(OUT / "missed_2025.csv", index=False)
summ = pd.DataFrame([
    ("n", 22, 30), ("days to KEV (median)", 37.5, 56.0), ("CVSS3 median", 7.6, 9.0), ("network %", 90.9, 80.0),
    ("UI required %", 18.2, 6.7), ("PR none %", 63.6, 83.3), ("any exploit code %", 0.0, 73.3),
    ("EPSS percentile median", 61.0, 96.0), ("ransomware %", 9.1, 10.0), ("description length median", 196, 315.5),
], columns=["metric", "missed", "found"])
summ.to_csv(OUT / "missed_summary.csv", index=False)

# ---------- per-CVE dossier for the submitted 500 (no fields the team story page already shows) ----------
top_rank = dict(zip(top.cve_id, range(1, len(top) + 1)))
d = sb[sb.cve_id.isin(set(top.cve_id))].copy()
d["submission_rank"] = d.cve_id.map(top_rank)
d["in_submission"] = d.submission_rank.notna()
if nvd is not None:
    d = d.merge(nvd[["cve_id", "description", "cvss_score", "cvss_severity", "cvss_vector", "cvss_source"]]
                .rename(columns={"cvss_score": "nvd_cvss", "cvss_severity": "nvd_severity",
                                 "cvss_vector": "nvd_vector", "cvss_source": "nvd_cvss_source"}),
                on="cve_id", how="left")
d["published"] = pd.to_datetime(d["published"]).dt.strftime("%Y-%m-%d")
# inverse of luis/analysis/build_features.py ORD
DECODE = {"cv_attack_vector": {3: "Network", 2: "Adjacent", 1: "Local", 0: "Physical"},
          "cv_attack_complexity": {1: "Low", 0: "High"}, "cv_privileges_required": {2: "None", 1: "Low", 0: "High"},
          "cv_user_interaction": {1: "None", 0: "Required"}, "cv_scope": {1: "Changed", 0: "Unchanged"}}
for c, m in DECODE.items():
    d[c] = d[c].map(m)
# percentile of each voter over the full scoreboard, so the three are comparable
for c in ["tree", "linear", "llm"]:
    d[c + "_pct"] = d.cve_id.map(dict(zip(sb.cve_id, sb[c].rank(pct=True))))
    d[c + "_rank"] = d.cve_id.map(dict(zip(sb.cve_id, sb[c].rank(ascending=False, method="min").astype(int))))
d.sort_values(["in_submission", "submission_rank", "rank_final"], ascending=[False, True, True]) \
    .to_parquet(OUT / "dossier.parquet", index=False)

info = dict(scoreboard_rows=len(sb), dossier=len(d),
            nvd_matched=int(d["description"].notna().sum()) if "description" in d else 0,
            build_info=(R / "luis/reports/final_build_info.txt").read_text())
json.dump(info, open(OUT / "build_info.json", "w"), indent=1)
print(json.dumps({k: v for k, v in info.items() if k != "build_info"}, indent=1))
