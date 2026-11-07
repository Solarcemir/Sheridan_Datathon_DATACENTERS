"""Time-left adjustment: P(counted) = P(exploited eventually) x P(added to KEV within the time left).

For each CVE, time left = judges' cutoff (Nov 4 of its publish year; Nov 4 2026 on the scoreboard)
minus the end of its snapshot month. F(t) = share of past positives added to KEV within t days of
their snapshot end, estimated on the TRAINING folds only (positives from 2024, whose delays are
observed for 12+ months, to avoid right-censoring). Adjusted score = score + w * log F(time left).
Evaluated on the CV out-of-fold scores with judge-aligned (Nov 4) labels.

  python scripts/horizon.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC, RES = ROOT / "processed", ROOT / "results"
K = 517


def ecdf(delays: np.ndarray, t: np.ndarray, floor: float = 0.02) -> np.ndarray:
    d = np.sort(delays)
    return np.maximum(np.searchsorted(d, t, side="right") / len(d), floor)


def main() -> None:
    h = pd.read_parquet(PROC / "historical_2024_2025.parquet",
                        columns=["cve_id", "published", "snapshot_month", "assigner", "kev_date_added"])
    o = (pd.read_parquet(RES / "cv_oof_lr.parquet")
         .merge(pd.read_parquet(RES / "cv_oof_lora.parquet")[["cve_id", "lora_control"]], on="cve_id")
         .merge(h, on="cve_id"))
    snap_end = pd.PeriodIndex(o.snapshot_month, freq="M").to_timestamp(how="end").normalize()
    cutoff = pd.to_datetime(o.published.dt.year.astype(str) + "-11-04")
    o["time_left"] = (cutoff - snap_end).dt.days.clip(lower=0).to_numpy()
    o["delay"] = (o.kev_date_added - snap_end).dt.days.to_numpy()
    o["nov4"] = (o.kev_date_added.notna() & (o.kev_date_added <= cutoff)).astype(int)

    # F per fold, from training-fold positives published in 2024 (fully observed delays)
    logF = np.zeros(len(o))
    for f in sorted(o.fold.unique()):
        tr = (o.fold != f) & (o.label == 1) & (o.published.dt.year == 2024)
        te = (o.fold == f).to_numpy()
        logF[te] = np.log(ecdf(o.delay[tr].to_numpy(), o.time_left[te].to_numpy()))
    o["logF"] = logF
    d24 = o.delay[(o.label == 1) & (o.published.dt.year == 2024)]
    print(f"delay after snapshot end (2024 positives): median {d24.median():.0f} d, "
          f"75% within {d24.quantile(.75):.0f} d, 90% within {d24.quantile(.9):.0f} d")
    print("time left by publish month (Nov-4 rule):",
          o.groupby(o.published.dt.month).time_left.median().astype(int).to_dict())

    def hits(score, lab="nov4", cap=0):
        d = o.assign(_s=score).sort_values("_s", ascending=False)
        if cap:
            d = d[d.groupby(["assigner", "snapshot_month"]).cumcount() < cap]
        return int(d[lab].head(K).sum())

    r = lambda s: pd.Series(s).rank(pct=True).to_numpy()
    lr, lora = o.lr_features_meta_embknn_pca64_team.to_numpy(), o.lora_control.to_numpy()
    rows = []
    for w in (0, 0.5, 1.0, 1.5):
        a_lr, a_lora = lr + w * logF, lora + w * logF
        avg = (r(a_lr) + r(a_lora)) / 2
        base = o.baseline_code_epss.rank(pct=True).to_numpy()
        rows.append({"w": w,
                     "baseline": hits(np.log(base) + w * logF), "LR": hits(a_lr), "LoRA": hits(a_lora),
                     "avg": hits(avg), "avg cap3": hits(avg, cap=3), "avg cap5": hits(avg, cap=5),
                     "avg (our label)": hits(avg, "label")})
    print(f"\nhits@{K}, Nov-4 labels ({o.nov4.sum()} positives) unless noted; w = weight on log F(time left)")
    print(pd.DataFrame(rows).to_string(index=False))
    pd.DataFrame({"cve_id": o.cve_id, "logF": logF}).to_parquet(RES / "cv_logF.parquet", index=False)


if __name__ == "__main__":
    main()
