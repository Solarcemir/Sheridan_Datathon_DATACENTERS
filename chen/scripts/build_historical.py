"""Build publish-time ("as of snapshot month") views of the data.

Outputs (in processed/):
  historical_2024_2025.parquet  CVEs published 2024-01..2025-12, built the same way
                                as scoreboard_2026.csv, plus KEV labels.
  scoreboard_2026_view.parquet  The 72,670 scoreboard CVEs with the same feature columns.

Rules, mirroring scoreboard_2026.csv:
  - snapshot_month = publish month.
  - EPSS and exploit flags come from that month's row in exploit_signals_*.csv.
    No row -> blank epss and 0 flags (the scoreboard does the same).
  - CVEs already on KEV by the end of snapshot_month are dropped.

Columns dropped everywhere because NIST fills them in after exploitation
(see README "Keep it fair"): nvd_status, last_modified, n_affected_cpes,
n_references, cvss3_source, and all of affected_products.csv / references.csv.
cvss2_* is dropped too: NIST stopped issuing v2 scores, so it is an NVD artifact.

Label: added to KEV after the end of snapshot_month. known_exploited.csv
stops at 2025-12-31, so later CVEs have a shorter window to be labelled
(see horizon_days).
"""

import sys
from pathlib import Path

import pandas as pd

_R = Path(__file__).resolve().parent.parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else next((p for p in (_R / "IBM" / "data", _R.parent / "IBM-20261107T174321Z-1-001" / "IBM" / "data" / "data") if p.exists()), _R / "IBM" / "data")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else _R / "processed"
KEV_END = pd.Timestamp("2025-12-31")

VULN_COLS = [
    "cve_id", "published", "assigner", "description",
    "cvss3_version", "cvss3_score", "cvss3_severity", "cvss3_vector",
    "attack_vector", "attack_complexity", "privileges_required", "user_interaction",
    "scope", "confidentiality_impact", "integrity_impact", "availability_impact",
    "cvss3_exploitability_score", "cvss3_impact_score",
    "cvss4_score", "cvss4_severity", "cvss4_vector",
    "cwe_ids",
]
FLAGS = ["exploitdb", "metasploit", "nuclei", "poc_github"]
SIGNAL_COLS = ["epss"] + FLAGS


def month_end(month: pd.Series) -> pd.Series:
    ends = pd.PeriodIndex(month, freq="M").to_timestamp(how="end").normalize()
    return pd.Series(ends, index=month.index)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    vuln = pd.read_csv(DATA / "vulnerabilities.csv", usecols=VULN_COLS,
                       dtype={"cvss3_version": str})
    vuln["published"] = pd.to_datetime(vuln["published"])

    kev = pd.read_csv(DATA / "known_exploited.csv", usecols=["cve_id", "date_added"])
    kev["kev_date_added"] = pd.to_datetime(kev["date_added"])
    kev = kev[["cve_id", "kev_date_added"]]

    # ---- historical 2024-2025 -------------------------------------------------
    hist = vuln[vuln["published"].between("2024-01-01", "2025-12-31")].copy()
    hist["snapshot_month"] = hist["published"].dt.strftime("%Y-%m")

    sig = pd.concat(
        [pd.read_csv(DATA / f"exploit_signals_{y}.csv") for y in (2024, 2025)],
        ignore_index=True,
    )
    sig = sig[sig["cve_id"].isin(hist["cve_id"])].rename(columns={"month": "snapshot_month"})

    hist = hist.merge(sig, on=["cve_id", "snapshot_month"], how="left")
    hist[FLAGS] = hist[FLAGS].fillna(0).astype("int8")
    hist["in_kev"] = hist["in_kev"].fillna(0).astype("int8")
    hist["has_signal_row"] = hist["epss"].notna() | (hist[FLAGS].sum(axis=1) > 0)

    hist = hist.merge(kev, on="cve_id", how="left")
    snap_end = month_end(hist["snapshot_month"])
    already_kev = (hist["in_kev"] == 1) | (hist["kev_date_added"] <= snap_end)
    n_dropped = int(already_kev.sum())
    hist = hist[~already_kev].copy()
    snap_end = month_end(hist["snapshot_month"])

    hist["label"] = hist["kev_date_added"].notna().astype("int8")
    hist["days_to_kev"] = (hist["kev_date_added"] - hist["published"]).dt.days
    hist["horizon_days"] = (KEV_END - snap_end).dt.days
    hist = hist.drop(columns=["in_kev"])

    # ---- scoreboard 2026 view -------------------------------------------------
    sb = pd.read_csv(DATA / "scoreboard_2026.csv")
    sb = sb.drop(columns=["published"]).merge(vuln, on="cve_id", how="left")
    sb[FLAGS] = sb[FLAGS].fillna(0).astype("int8")

    front = ["cve_id", "published", "snapshot_month"] + SIGNAL_COLS
    rest = [c for c in VULN_COLS if c not in ("cve_id", "published")]
    hist = hist[front + rest + ["has_signal_row", "kev_date_added", "label",
                                "days_to_kev", "horizon_days"]]
    sb = sb[front + rest]

    hist.to_parquet(OUT / "historical_2024_2025.parquet", index=False)
    sb.to_parquet(OUT / "scoreboard_2026_view.parquet", index=False)

    # ---- report ---------------------------------------------------------------
    print(f"historical rows: {len(hist):,}  (dropped {n_dropped} already on KEV at snapshot)")
    print(f"scoreboard rows: {len(sb):,}  (missing description: {sb['description'].isna().sum()})")
    hist["year"] = hist["published"].dt.year
    print("\nper year:")
    print(hist.groupby("year").agg(
        rows=("cve_id", "size"),
        positives=("label", "sum"),
        no_signal_row=("has_signal_row", lambda s: int((~s).sum())),
    ))
    print("\npositives by publish month:")
    print(hist.groupby("snapshot_month")["label"].sum().to_string())
    print("\ndays from publish to KEV (positives):")
    print(hist.loc[hist.label == 1, "days_to_kev"].describe().round(0).to_string())

    print("\nbaseline sanity check: top 500 by EPSS within each year")
    for y, g in hist.groupby("year"):
        top = g.sort_values("epss", ascending=False, na_position="last").head(500)
        flag = g.assign(f=g[FLAGS].max(axis=1)).sort_values(
            ["f", "epss"], ascending=False, na_position="last").head(500)
        print(f"  {y}: EPSS-first {top.label.sum()}  exploit-code-then-EPSS {flag.label.sum()}"
              f"  (of {g.label.sum()} positives in {len(g):,})")

    print("\ncoverage (historical vs scoreboard):")
    cov = pd.DataFrame({
        "historical": hist[["epss", "cvss3_score", "cvss4_score", "cwe_ids"]].notna().mean(),
        "scoreboard": sb[["epss", "cvss3_score", "cvss4_score", "cwe_ids"]].notna().mean(),
    })
    print((100 * cov).round(1))


if __name__ == "__main__":
    main()
