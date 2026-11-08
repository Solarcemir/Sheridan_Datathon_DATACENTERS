"""Check Luis's final recipe (luis/analysis/final_build.py) on our 5-fold purged CV: Granite 30B demotion
filters, XGBoost as the tree voter, his LR averaged into the linear voter. Proxies: xgb_fixed_luis (results/cv_xgb.parquet)
for his XGBoost, lr_luis_only for his LR. Cap 3, K=517, Nov-4 labels, paired bootstrap vs our ensemble.

  python scripts/luis_recipe_check.py   -> reports/luis_recipe_check.txt
"""
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
R, P = ROOT / "results", ROOT / "processed"
h = pd.read_parquet(P / "historical_2024_2025.parquet", columns=["cve_id", "published", "snapshot_month", "assigner", "kev_date_added"])
o = pd.read_parquet(R / "cv_oof_lr.parquet").merge(h, on="cve_id").merge(pd.read_parquet(R / "cv_xgb.parquet"), on="cve_id")
o["nov4"] = (o.kev_date_added.notna() & (o.kev_date_added <= pd.to_datetime(o.published.dt.year.astype(str) + "-11-04"))).astype(int)
lo = pd.concat([pd.read_parquet(f) for f in sorted(R.glob("cv_control_f*.parquet"))]).drop_duplicates("cve_id")
o["lora"] = o.cve_id.map(lo.set_index("cve_id").score)
e = pd.read_parquet(P / "llm_extract.parquet").set_index("cve_id").reindex(o.cve_id)
filt = ((e.impact == "denial_of_service") | (e.typical_attacker == "unlikely_target") | (e.attacker_appeal == "low")
        | (e.product_type == "wordpress_plugin_or_theme")).fillna(False).to_numpy()
print(f"filters catch {filt.sum():,} of {len(o):,} CVEs ({filt.mean():.0%}); positives caught: Dec31 {o.label[filt].sum()}/{o.label.sum()}, Nov4 {o.nov4[filt].sum()}/{o.nov4.sum()}")
print(o.loc[filt & (o.nov4 == 1), "cve_id"].tolist())
r = lambda c: o[c].rank(pct=True)
V = {
  "ours: LR + LoRA + LightGBM": (r("lr_full_llmx_noemb") + r("lora") + r("lgbm_luis_only")) / 3,
  "Luis-style: LR + LoRA + XGB": (r("lr_full_llmx_noemb") + r("lora") + r("xgb_fixed_luis")) / 3,
  "Luis-style, linear = avg(our LR, LR on his feats)": ((r("lr_full_llmx_noemb") + r("lr_luis_only")).rank(pct=True) + r("lora") + r("xgb_fixed_luis")) / 3,
  "same, linear = his LR only": (r("lr_luis_only") + r("lora") + r("xgb_fixed_luis")) / 3,
  "LR + XGB (no LoRA)": (r("lr_full_llmx_noemb") + r("xgb_fixed_luis")) / 2,
}
n, K = len(o), round(500 * len(o) / 72670); y = o.nov4.to_numpy(); yd = o.label.to_numpy()
def ranked(s, use_filter, cap=3):
    s = s.to_numpy() - (50 * filt if use_filter else 0)
    d = o.assign(_s=s).sort_values("_s", ascending=False)
    d = d[d.groupby(["assigner", "snapshot_month"]).cumcount() < cap].head(K)
    m = np.zeros(n, bool); m[d.index] = True; return m
rng = np.random.default_rng(0); B = [rng.integers(0, n, n) for _ in range(1000)]
ref = ranked(V["ours: LR + LoRA + LightGBM"], False)
rows = []
for name, s in V.items():
    for f in (False, True):
        m = ranked(s, f)
        # paired bootstrap of top-K-set hits (selection fixed, resample CVEs)
        d = [(y[i] * m[i]).sum() - (y[i] * ref[i]).sum() for i in B]
        rows.append({"ensemble (cap 3)": name, "filters": "yes" if f else "no", "Nov4": int(y[m].sum()), "Dec31": int(yd[m].sum()),
                     "vs ours no-filter": f"{np.percentile(d, 2.5):+.0f}..{np.percentile(d, 97.5):+.0f}"})
print(f"K={K}, 75 Nov4 positives"); print(pd.DataFrame(rows).to_string(index=False))
