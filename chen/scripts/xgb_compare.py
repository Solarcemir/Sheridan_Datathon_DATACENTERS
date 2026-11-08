"""Tuned vs fixed XGBoost vs LightGBM in the purged 5-fold CV, alone and as the tree member of the
ensemble with the best LR + the 8B LoRA (control). Judge-aligned Nov-4 labels, K=517, paired bootstrap
vs exploit-then-EPSS and vs the current ensemble (LR + LoRA + LightGBM on Luis's features).

  python scripts/xgb_compare.py      (after scripts/xgb_tune.py)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

ROOT = Path(__file__).resolve().parent.parent
PROC, RES = ROOT / "processed", ROOT / "results"


def main() -> None:
    h = pd.read_parquet(PROC / "historical_2024_2025.parquet",
                        columns=["cve_id", "published", "snapshot_month", "assigner", "kev_date_added"])
    o = pd.read_parquet(RES / "cv_oof_lr.parquet").merge(h, on="cve_id").merge(
        pd.read_parquet(RES / "cv_xgb.parquet"), on="cve_id")
    o["nov4"] = (o.kev_date_added.notna()
                 & (o.kev_date_added <= pd.to_datetime(o.published.dt.year.astype(str) + "-11-04"))).astype(int)
    lo = pd.concat([pd.read_parquet(f) for f in sorted(RES.glob("cv_control_f*.parquet"))]).drop_duplicates("cve_id")
    o["lora"] = o.cve_id.map(lo.set_index("cve_id").score)
    assert o.lora.notna().all()
    r = lambda c: o[c].rank(pct=True)
    trees = ["lgbm_luis_only"] + [c for c in o.columns if c.startswith("xgb_")]
    cols = ["baseline_code_epss", "lr_full_llmx_noemb", "lora"] + trees
    for t in trees:
        o[f"lr+lora+{t}"] = (r("lr_full_llmx_noemb") + r("lora") + r(t)) / 3
        cols.append(f"lr+lora+{t}")
    n, K = len(o), round(500 * len(o) / 72670)
    y = o.nov4.to_numpy()
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, n, n) for _ in range(1000)]
    top = lambda v, i: y[i][np.argpartition(-v[i], K)[:K]].sum()
    base, cur = o.baseline_code_epss.to_numpy(), o["lr+lora+lgbm_luis_only"].to_numpy()

    def hits(s, lab, cap=0):
        d = o.assign(_s=s).sort_values("_s", ascending=False)
        if cap:
            d = d[d.groupby(["assigner", "snapshot_month"]).cumcount() < cap]
        return int(d[lab].head(K).sum())

    rows = []
    for c in cols:
        s = o[c].to_numpy()
        d_base = [top(s, i) - top(base, i) for i in boots]
        d_cur = [top(s, i) - top(cur, i) for i in boots]
        ci = lambda d: f"{np.percentile(d, 2.5):+.0f}..{np.percentile(d, 97.5):+.0f}"
        rows.append({"model": c, "Dec31": hits(s, "label"), "Nov4": hits(s, "nov4"), "Nov4 cap3": hits(s, "nov4", 3),
                     "vs baseline": ci(d_base), "vs current ens.": ci(d_cur),
                     "AP Dec31": round(ap(o.label, s), 4), "AP Nov4": round(ap(o.nov4, s), 4)})
    print(f"5-fold purged CV: {n:,} CVEs, {int(o.label.sum())} / {int(y.sum())} positives (Dec31 / Nov4), K={K}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
