"""Line up our numbers with Chen's CV: hits at the real scoreboard density (K = 500 x rows / 72,670),
A/B per year and pooled 2024+2025, the txt_in_wild ablation, and vendor- vs product-level KEV name matching.

Usage:  python -E -P analysis/density_check.py > reports/density_check.txt
"""
import os, sys
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
import numpy as np, pandas as pd, lightgbm as lgb
R = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); DER = os.path.join(R, "derived")
fd = pd.read_csv(os.path.join(DER, "feature_dictionary.csv"))
SAFE = [f for f in fd[fd.risk == "safe"].feature] + ["days_to_cutoff"]
P = dict(objective="binary", n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8,
         subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, scale_pos_weight=25, verbose=-1, n_jobs=8,
         deterministic=True, force_row_wise=True)
def load(y, cut):
    d = pd.read_csv(os.path.join(DER, f"features_{y}.csv"), parse_dates=["snapshot_end"])
    d["days_to_cutoff"] = (pd.Timestamp(cut) - d.snapshot_end).dt.days; return d
def fit_score(tr, te, F, seeds=10):
    return np.mean([lgb.LGBMClassifier(random_state=s, **P).fit(tr[F], tr.y).predict_proba(te[F])[:, 1] for s in range(seeds)], axis=0)
def capped(df, s, cap=5):
    s = pd.Series(s, index=df.index)
    n = df.assign(_s=s).sort_values("_s", ascending=False, kind="stable").groupby(["assigner", "snapshot_month"]).cumcount()
    s = s.copy(); s[n[n >= cap].index] -= 100; return s.to_numpy()
def hits(y, s, k): return int(np.asarray(y)[np.argsort(-np.asarray(s), kind="stable")[:k]].sum())
splits = {"A": (load(2024, "2024-12-31"), load(2025, "2025-11-04")), "B": (load(2025, "2025-12-31"), load(2024, "2024-11-04"))}
rows = []
pooled = {}
for name, (tr, te) in splits.items():
    yr = 2025 if name == "A" else 2024
    te = te[te.snapshot_month <= f"{yr}-09"].reset_index(drop=True)
    kd = round(500 * len(te) / 72670)
    base = (te.any_flag * 10 + te.epss).to_numpy()
    full = fit_score(tr, te, SAFE)
    nowild = fit_score(tr, te, [f for f in SAFE if f != "txt_in_wild"])
    for label, s in [("exploit then EPSS", base), ("LightGBM safe", full), ("LightGBM safe, no txt_in_wild", nowild)]:
        for cap in (None, 5):
            sc = capped(te, s, cap) if cap else s
            rows.append({"split": name, "model": label, "cap": cap or "none", "K_density": kd,
                         f"hits@density": hits(te.y_nov4, sc, kd), "hits@500": hits(te.y_nov4, sc, 500)})
    pooled[name] = te.assign(base=base, full=full, nowild=nowild)
res = pd.DataFrame(rows)
piv = res.pivot_table(index=["model", "cap"], columns="split", values=["hits@density", "hits@500"]).astype(int)
piv[("hits@density", "A+B")] = piv[("hits@density", "A")] + piv[("hits@density", "B")]
piv[("hits@500", "A+B")] = piv[("hits@500", "A")] + piv[("hits@500", "B")]
print("K at scoreboard density: A =", res[res.split == "A"].K_density.iloc[0], " B =", res[res.split == "B"].K_density.iloc[0])
print(piv.to_string())
# Chen-style pooled view: both test years together, one top K at scoreboard density
both = pd.concat(pooled.values(), ignore_index=True)
K = round(500 * len(both) / 72670)
for col in ("base", "full", "nowild"):
    within = both.groupby("snapshot_month")[col].rank(pct=True)   # years have different score scales; rank within month
    print(f"pooled 2024+2025 Jan-Sep, top {K} of {len(both):,} ({both.y_nov4.sum()} pos): {col:7s} no cap = {hits(both.y_nov4, within, K)}, "
          f"cap5 = {hits(both.y_nov4, capped(both, within.to_numpy()), K)}")
w = both[both.txt_in_wild == 1]
print(f"txt_in_wild rows in test years: {len(w)}, positives {w.y_nov4.sum()}")
lift = pd.read_csv(os.path.join(R, "reports", "univariate_lift.csv")).set_index("feature")
print(lift.loc[["kev_product_hits", "kev_product_hist", "kev_vendor_hist"], ["univariate_auc", "lift_top5pct"]].to_string())
s26 = pd.read_csv(os.path.join(DER, "features_2026.csv"), usecols=["cve_id", "txt_in_wild"])
t5 = set(pd.read_csv(os.path.join(R, "submissions", "top500_v2.csv")).cve_id)
print("2026 scoreboard txt_in_wild:", int(s26.txt_in_wild.sum()), "| of those in top500_v2:", int(s26[s26.cve_id.isin(t5)].txt_in_wild.sum()))
