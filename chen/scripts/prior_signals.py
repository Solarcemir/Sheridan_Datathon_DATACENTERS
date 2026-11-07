"""Borrow live exploitation signals from a vendor's OLDER CVEs.

EPSS on an old CVE reflects current exploitation activity; Nuclei/Metasploit coverage reflects how
much attackers and scanners care about a product. A brand-new CVE has none of that yet, but its
vendor's older CVEs do. Reference month (same staleness as the 2026 scoreboard, which can only see
Dec 2025): 2025 rows use Dec 2024, 2026 rows use Dec 2025, 2024 rows use Jan 2024 (the earliest
signals month: a one-month look-ahead on OLD CVEs only, declared). Only CVEs published before the
end of the previous year count as "older".

Two keys:
  assigner  the CNA (always available)
  vendor    old CVEs: their CPE vendor in affected_products.csv (long established for old CVEs; the
            leak concern is only a NEW CVE's own CPE rows). New CVEs: Granite's normalised vendor
            (processed/llm_extract.parquet), matched to the CPE vendor name. Skipped if missing.

  python scripts/prior_signals.py   # -> processed/prior_sig_{historical,scoreboard}.parquet
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = next((p for p in (ROOT / "IBM" / "data", ROOT.parent / "IBM-20261107T174321Z-1-001" / "IBM" / "data" / "data") if p.exists()), ROOT / "IBM" / "data")
PROC = ROOT / "processed"
FLAGS = ["exploitdb", "metasploit", "nuclei", "poc_github"]
REF = {2024: "2024-01", 2025: "2024-12", 2026: "2025-12"}


def norm(s) -> str:
    s = re.sub(r"\(.*?\)", " ", str(s).lower()).replace("_", " ")
    s = re.sub(r"\b(inc|corp|corporation|ltd|llc|co|software|systems|technologies|networks|the)\b", " ", s)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def stats(g: pd.DataFrame) -> pd.DataFrame:
    e = g.epss.fillna(0)
    return pd.DataFrame({
        "n": g.groupby(level=0).size(),
        "epss_p90": e.groupby(level=0).quantile(0.9),
        "epss_max": e.groupby(level=0).max(),
        "n_epss_gt_0_1": (e > 0.1).groupby(level=0).sum(),
        "share_nuclei": g.nuclei.groupby(level=0).mean(),
        "share_metasploit": g.metasploit.groupby(level=0).mean(),
        "n_any_code": g[FLAGS].max(axis=1).groupby(level=0).sum(),
    })


def main() -> None:
    vul = pd.read_csv(DATA / "vulnerabilities.csv", usecols=["cve_id", "published", "assigner"])
    vul["published"] = pd.to_datetime(vul.published)
    sig = pd.concat([pd.read_csv(DATA / f"exploit_signals_{y}.csv") for y in (2024, 2025)])
    sig = sig[sig.month.isin(REF.values())].merge(vul, on="cve_id", how="inner")

    have_vendor = (PROC / "llm_extract.parquet").exists()
    if have_vendor:
        ap = pd.read_csv(DATA / "affected_products.csv", usecols=["cve_id", "vendor", "vulnerable"], dtype=str)
        ap = ap[ap.vulnerable == "Y"].drop_duplicates(["cve_id", "vendor"])
        ap["v"] = ap.vendor.map(norm)
        ex = pd.read_parquet(PROC / "llm_extract.parquet", columns=["cve_id", "vendor"])
        ex["v"] = ex.vendor.map(norm)

    by_year = {}
    for year, ref in REF.items():
        cut = pd.Timestamp(f"{year - 1}-12-31")
        old = sig[(sig.month == ref) & (sig.published <= cut)]
        A = stats(old.set_index("assigner")).add_prefix("asg_")
        V = None
        if have_vendor:
            ov = old.merge(ap[["cve_id", "v"]], on="cve_id").set_index("v")
            V = stats(ov).add_prefix("ven_")
        by_year[year] = (A, V)
        print(f"{year}: reference month {ref}, {len(old):,} older CVEs with signals"
              + (f", {0 if V is None else len(V):,} CPE vendors" if have_vendor else ""), flush=True)

    for name, file in (("historical", "historical_2024_2025.parquet"), ("scoreboard", "scoreboard_2026_view.parquet")):
        df = pd.read_parquet(PROC / file, columns=["cve_id", "published", "assigner"])
        if have_vendor:
            df = df.merge(ex[["cve_id", "v"]], on="cve_id", how="left")
        parts = []
        for year, g in df.groupby(df.published.dt.year):
            A, V = by_year[int(year)]
            p = g[["cve_id"]].copy()
            p = pd.concat([p.reset_index(drop=True), A.reindex(g.assigner).reset_index(drop=True)], axis=1)
            if V is not None:
                p = pd.concat([p, V.reindex(g.v).reset_index(drop=True)], axis=1)
            parts.append(p)
        out = pd.concat(parts, ignore_index=True).fillna(0)
        out.to_parquet(PROC / f"prior_sig_{name}.parquet", index=False)
        msg = f"{name}: {len(out):,} rows"
        if have_vendor:
            msg += f"; vendor matched for {100 * (out.ven_n > 0).mean():.0f}%"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
