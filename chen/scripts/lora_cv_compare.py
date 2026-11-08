"""Compare the LoRA CV variants (8B control, 8B + Granite profile, 30B) alone and averaged with the
best LR (and Luis's LightGBM), with judge-aligned labels and a paired bootstrap vs exploit-then-EPSS.

  python scripts/lora_cv_compare.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

ROOT = Path(__file__).resolve().parent.parent
PROC, RES = ROOT / "processed", ROOT / "results"


def lora(cfg: str) -> pd.Series | None:
    files = sorted(RES.glob(f"cv_{cfg}_f*.parquet"))
    if len(files) < 5:
        print(f"({cfg}: only {len(files)}/5 folds, skipped)")
        return None
    return pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates("cve_id").set_index("cve_id").score


def main() -> None:
    h = pd.read_parquet(PROC / "historical_2024_2025.parquet",
                        columns=["cve_id", "published", "snapshot_month", "assigner", "kev_date_added"])
    o = pd.read_parquet(RES / "cv_oof_lr.parquet").merge(h, on="cve_id")
    o["nov4"] = (o.kev_date_added.notna()
                 & (o.kev_date_added <= pd.to_datetime(o.published.dt.year.astype(str) + "-11-04"))).astype(int)
    for cfg in ("control", "llmx8b", "lora30b"):
        s = lora(cfg)
        if s is not None:
            o[f"lora_{cfg}"] = o.cve_id.map(s)
    r = lambda c: o[c].rank(pct=True)
    cols = ["baseline_code_epss", "lr_full_llmx_noemb", "lgbm_luis_only"]
    for c in [c for c in o.columns if c.startswith("lora_")]:
        cols.append(c)
        o[f"avg_lr+{c}"] = (r("lr_full_llmx_noemb") + r(c)) / 2
        o[f"avg_lr+{c}+lgbm"] = (r("lr_full_llmx_noemb") + r(c) + r("lgbm_luis_only")) / 3
        cols += [f"avg_lr+{c}", f"avg_lr+{c}+lgbm"]
    n, K = len(o), round(500 * len(o) / 72670)
    rng = np.random.default_rng(0)
    base = o.baseline_code_epss.to_numpy()
    rows = []
    for c in cols:
        s = o[c].to_numpy()

        def hits(lab, cap=0):
            d = o.assign(_s=s).sort_values("_s", ascending=False)
            if cap:
                d = d[d.groupby(["assigner", "snapshot_month"]).cumcount() < cap]
            return int(d[lab].head(K).sum())

        y = o.nov4.to_numpy()
        diffs = []
        for _ in range(500):
            i = rng.integers(0, n, n)
            t = lambda v: y[i][np.argpartition(-v[i], K)[:K]].sum()
            diffs.append(t(s) - t(base))
        rows.append({"model": c, "hits Dec31": hits("label"), "hits Nov4": hits("nov4"), "Nov4 cap3": hits("nov4", 3),
                     "Nov4 vs baseline 95% CI": f"{np.percentile(diffs, 2.5):+.0f}..{np.percentile(diffs, 97.5):+.0f}",
                     "AP Dec31": round(ap(o.label, s), 4), "AP Nov4": round(ap(o.nov4, s), 4)})
    print(f"5-fold purged CV: {n:,} CVEs, {int(o.label.sum())} / {int(o.nov4.sum())} positives (Dec31 / Nov4), K={K}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
