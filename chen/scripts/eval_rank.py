"""Compare rankings on the 2025 holdout by how many KEV additions they put near the top.

The holdout is ~half the scoreboard's size, so besides hits@500 it reports hits@K where
K = 500 * holdout_size / 72,670 (the same share of the list the judges look at).

  python scripts/eval_rank.py [holdout|val2024] results/zero.parquet results/many.parquet
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FLAGS = ["exploitdb", "metasploit", "nuclei", "poc_github"]
SCOREBOARD_ROWS = 72_670


def hits(df: pd.DataFrame, cols, ks) -> dict:
    ranked = df.sort_values(cols, ascending=False, na_position="last")
    return {f"hits@{k}": int(ranked.label.head(k).sum()) for k in ks}


TARGETS = {"holdout": ("2025-01", "2025-09"), "val2024": ("2024-07", "2024-12"),
           "all2024": ("2024-01", "2024-12")}


def main() -> None:
    args = sys.argv[1:]
    target = "holdout"
    if args and args[0] in TARGETS:
        target, args = args[0], args[1:]
    lo, hi = TARGETS[target]
    hist = pd.read_parquet(ROOT / "processed" / "historical_2024_2025.parquet")
    hold = hist[hist.snapshot_month.between(lo, hi)].copy()
    hold["has_code"] = hold[FLAGS].max(axis=1)
    k_scaled = round(500 * len(hold) / SCOREBOARD_ROWS)
    ks = [k_scaled, 500, 1000]
    print(f"{target} {lo}..{hi}: {len(hold):,} CVEs, {hold.label.sum()} added to KEV later; "
          f"scaled K = {k_scaled}\n")

    rows = {
        "CVSS 3 first": hits(hold, ["cvss3_score", "epss"], ks),
        "EPSS first": hits(hold, ["epss"], ks),
        "exploit code, then EPSS": hits(hold, ["has_code", "epss"], ks),
    }
    for p in args:
        s = pd.read_parquet(p)
        m = hold.merge(s, on="cve_id", how="inner")
        name = Path(p).stem
        if len(m) != len(hold):
            name += f" (scored {len(m):,}/{len(hold):,})"
        rows[name] = hits(m, ["score"], ks)
    print(pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    main()
