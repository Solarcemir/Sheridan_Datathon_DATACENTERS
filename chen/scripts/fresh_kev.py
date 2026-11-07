"""Same-year "went straight onto KEV" history (declared in the write-up).

Hidden set = CVEs that were on KEV by the end of their own snapshot month:
  2024-25: from known_exploited.csv (date_added <= end of the publish month), the same rule that
           dropped them from historical_2024_2025.parquet.
  2026:    the 124 Jan-Sep 2026 CVEs in vulnerabilities.csv that are missing from the scoreboard
           (the scoreboard excludes CVEs on KEV at the end of their snapshot month; June -> July).
For a CVE with snapshot month M, only hidden CVEs from the SAME YEAR with snapshot month <= M
count: KEV is public, so these were known at the end of M. Nothing else from the same year is used.

Features:
  fresh_asg_ytd / fresh_asg_3m    hidden CVEs from the same assigner (year to date / last 3 months)
  fresh_name_ytd / fresh_name_3m  hidden CVEs whose description mentions this CVE's Granite-normalised
                                  vendor or product name

  python scripts/fresh_kev.py   # -> processed/fresh_{historical,scoreboard}.parquet
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = next((p for p in (ROOT / "IBM" / "data", ROOT.parent / "IBM-20261107T174321Z-1-001" / "IBM" / "data" / "data") if p.exists()), ROOT / "IBM" / "data")
PROC = ROOT / "processed"
GENERIC = {"unknown", "n a", "none", "various", "multiple", "wordpress", "linux", "open source", "other"}


def norm(s) -> str:
    s = re.sub(r"\(.*?\)", " ", str(s).lower())
    s = re.sub(r"\b(inc|corp|corporation|ltd|llc|co|software|systems|technologies|networks|the)\b", " ", s)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def mi(m: pd.Series) -> np.ndarray:
    return (m.str[:4].astype(int) * 12 + m.str[5:7].astype(int)).to_numpy()


def main() -> None:
    vul = pd.read_csv(DATA / "vulnerabilities.csv", usecols=["cve_id", "published", "assigner", "description"])
    vul["published"] = pd.to_datetime(vul.published)
    kev = pd.read_csv(DATA / "known_exploited.csv", usecols=["cve_id", "date_added"])
    kev["date_added"] = pd.to_datetime(kev.date_added)
    sb = pd.read_csv(DATA / "scoreboard_2026.csv", usecols=["cve_id"])

    k = kev.merge(vul, on="cve_id")
    k = k[k.published.between("2024-01-01", "2025-12-31")]
    h_hist = k[k.date_added <= k.published + pd.offsets.MonthEnd(0)].copy()
    h_hist["snap"] = h_hist.published.dt.strftime("%Y-%m")
    v26 = vul[vul.published.between("2026-01-01", "2026-09-30")]
    h_26 = v26[~v26.cve_id.isin(sb.cve_id)].copy()
    h_26["snap"] = h_26.published.dt.strftime("%Y-%m").replace({"2026-06": "2026-07"})
    hidden = pd.concat([h_hist, h_26], ignore_index=True)
    hidden["year"] = hidden.snap.str[:4].astype(int)
    hidden["m"] = mi(hidden.snap)
    hidden["desc"] = hidden.description.fillna("").map(norm)
    print(f"hidden same-year KEV: 2024 {int((hidden.year == 2024).sum())}, 2025 {int((hidden.year == 2025).sum())}, "
          f"2026 {int((hidden.year == 2026).sum())}")

    ex = pd.read_parquet(PROC / "llm_extract.parquet", columns=["cve_id", "vendor", "product"])
    ex["v"], ex["p"] = ex.vendor.map(norm), ex["product"].map(norm)
    ex = ex.set_index("cve_id")

    for name, file in (("historical", "historical_2024_2025.parquet"), ("scoreboard", "scoreboard_2026_view.parquet")):
        df = pd.read_parquet(PROC / file, columns=["cve_id", "snapshot_month", "assigner"])
        df["year"], df["m"] = df.snapshot_month.str[:4].astype(int), mi(df.snapshot_month)
        out = {c: np.zeros(len(df), int) for c in ["fresh_asg_ytd", "fresh_asg_3m", "fresh_name_ytd", "fresh_name_3m"]}
        name_cache: dict = {}
        for (yr, m), idx in df.groupby(["year", "m"]).groups.items():
            hy = hidden[(hidden.year == yr) & (hidden.m <= m)]
            if hy.empty:
                continue
            recent = (hy.m >= m - 2).to_numpy()
            asg_ytd = hy.assigner.value_counts()
            asg_3m = hy.assigner[recent].value_counts()
            rows = df.index.get_indexer(idx)
            g = df.loc[idx]
            out["fresh_asg_ytd"][rows] = g.assigner.map(asg_ytd).fillna(0).astype(int).to_numpy()
            out["fresh_asg_3m"][rows] = g.assigner.map(asg_3m).fillna(0).astype(int).to_numpy()
            descs, rec = hy.desc.to_numpy(), recent
            for j, cid in zip(rows, g.cve_id):
                if cid not in ex.index:
                    continue
                keys = tuple(sorted({x for x in (ex.at[cid, "v"], ex.at[cid, "p"]) if len(x) >= 4 and x not in GENERIC}))
                if not keys:
                    continue
                ck = (yr, m, keys)
                if ck not in name_cache:
                    rx = re.compile(r"\b(?:" + "|".join(re.escape(x) for x in keys) + r")\b")
                    hit = np.array([bool(rx.search(d)) for d in descs])
                    name_cache[ck] = (int(hit.sum()), int((hit & rec).sum()))
                out["fresh_name_ytd"][j], out["fresh_name_3m"][j] = name_cache[ck]
        res = pd.DataFrame({"cve_id": df.cve_id, **out})
        res.to_parquet(PROC / f"fresh_{name}.parquet", index=False)
        print(f"{name}: {len(res):,} rows; assigner-3m > 0 for {100 * (res.fresh_asg_3m > 0).mean():.1f}%, "
              f"name-3m > 0 for {100 * (res.fresh_name_3m > 0).mean():.1f}%", flush=True)


if __name__ == "__main__":
    main()
