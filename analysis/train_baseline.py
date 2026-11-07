"""Train and compare rankers on the replicas, draw feature maps, and score 2026.

Evaluation (both directions, because one year has only ~30-50 positives):
  A  train 2024 replica (all months, y through Dec 31, 2024) -> test 2025 Jan-Sep, y_nov4  (honest: past -> future)
  B  train 2025 replica (all months, y through Dec 31, 2025) -> test 2024 Jan-Sep, y_nov4  (robustness check)
  y_nov4 = joined KEV by Nov 4 of that year, the same cutoff day the judges use for 2026.
Final model: retrain on 2024 + 2025 (all months) and score scoreboard_2026.

Inputs : derived/features_{year}.csv (leak-free), derived/granite/granite_features_{year}.csv,
         derived/leak_only_{year}.csv (only for the leakage demo row)
Outputs: reports/model_results.csv, reports/univariate_lift.csv, reports/feature_importance.csv,
         reports/figures/*.png, derived/scored_2026_v1.csv, submissions/top500_v1.csv

Usage:  python -E -P analysis/train_baseline.py
"""
import os

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DER, REP = os.path.join(ROOT, "derived"), os.path.join(ROOT, "reports")
FIG, SUB = os.path.join(REP, "figures"), os.path.join(ROOT, "submissions")
for d in (FIG, SUB):
    os.makedirs(d, exist_ok=True)

fd = pd.concat([pd.read_csv(os.path.join(DER, "feature_dictionary.csv")),
                pd.read_csv(os.path.join(DER, "granite", "feature_dictionary_granite.csv"))], ignore_index=True)
G = fd.set_index("feature").group.to_dict()
RISK = fd.set_index("feature").risk.to_dict()
G["days_to_cutoff"], RISK["days_to_cutoff"] = "time", "safe"
TFIDF = ["text_kev_lr", "text_kev_sim"]
GRANITE = fd[fd.group == "granite"].feature.tolist()
CORE = [f for f in fd[fd.risk == "safe"].feature if f not in TFIDF + GRANITE] + ["days_to_cutoff"]
KEVNOW = fd[fd.group == "kev_now"].feature.tolist()
CVSS = fd[fd.group == "cvss"].feature.tolist()
LEAK = fd[fd.risk == "leak"].feature.tolist()


def load(year, cutoff, months=None):
    df = pd.read_csv(os.path.join(DER, f"features_{year}.csv"), parse_dates=["snapshot_end", "published"])
    df = df.merge(pd.read_csv(os.path.join(DER, "granite", f"granite_features_{year}.csv")), on="cve_id", how="left")
    df = df.merge(pd.read_csv(os.path.join(DER, f"leak_only_{year}.csv")), on="cve_id", how="left")
    df["days_to_cutoff"] = (pd.Timestamp(cutoff) - df.snapshot_end).dt.days
    if months:
        df = df[df.snapshot_month <= months].reset_index(drop=True)
    return df


SPLITS = {
    "A 2024->2025": (load(2024, "2024-12-31"), load(2025, "2025-11-04", "2025-09")),
    "B 2025->2024": (load(2025, "2025-12-31"), load(2024, "2024-11-04", "2024-09")),
}
for k, (tr, te) in SPLITS.items():
    print(f"{k}: train {len(tr):,} rows / {tr.y.sum()} pos  |  test {len(te):,} rows / {te.y_nov4.sum()} pos by Nov 4")


def hits(score, y, k=500):
    s = np.nan_to_num(np.asarray(score, dtype=float), nan=-1e9)
    return int(np.asarray(y)[np.argsort(-s, kind="stable")[:k]].sum())


def rank01(x):
    return pd.Series(np.asarray(x, dtype=float)).rank(pct=True).to_numpy()


def heuristic(df):          # designed while looking at 2025: only split B is a fair test for it
    return df.epss_pct + 2 * df.any_flag + 3 * df.nuclei + 5 * df.txt_in_wild + df.txt_rce + df.txt_unauth - df.txt_wordpress


def flag_epss(df):
    return df.any_flag * 10 + df.epss


LGB_PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, scale_pos_weight=25, verbose=-1)


def fit_gbm(df, feats, seeds=(0, 1, 2, 3, 4)):
    return [lgb.LGBMClassifier(random_state=s, **LGB_PARAMS).fit(df[feats], df.y) for s in seeds]


def gbm_score(models, df, feats):
    return np.mean([m.predict_proba(df[feats])[:, 1] for m in models], axis=0)


GR_ONLY = [f for f in GRANITE if f != "gr_hybrid_lr"]
HYB = ["gr_hybrid_lr", "text_kev_sim"]
VARIANTS = {
    "safe, TF-IDF text": CORE + TFIDF,
    "safe, Granite text": CORE + GR_ONLY,
    "safe, hybrid text (TF-IDF+Granite)": CORE + HYB,
    "safe, all text": CORE + TFIDF + GRANITE,
    "safe hybrid + kev_now": CORE + HYB + KEVNOW,
    "safe hybrid + kev_now + cvss": CORE + HYB + KEVNOW + CVSS,
    "all + LEAK (demo only)": CORE + HYB + KEVNOW + CVSS + LEAK,
}
rows, cache = [], {}


def record(name, feats_label, fn):
    """fn(train_df, test_df) -> scores for test_df; evaluated on both splits."""
    r = {"model": name, "features": feats_label}
    for k, (tr, te) in SPLITS.items():
        s = fn(tr, te)
        cache[(name, feats_label, k)] = s
        r[f"hits@500 {k}"] = hits(s, te.y_nov4)
        r[f"AP {k}"] = round(average_precision_score(te.y_nov4, np.nan_to_num(s, nan=-1e9)), 4)
    r["hits@500 total"] = r["hits@500 A 2024->2025"] + r["hits@500 B 2025->2024"]
    rows.append(r)
    print(f"  {name:32s} {feats_label:34s} A={r['hits@500 A 2024->2025']:3d}  B={r['hits@500 B 2025->2024']:3d}  total={r['hits@500 total']:3d}")


print("\nhits in the top 500 (A: test 2025, B: test 2024)")
record("CVSS 3 first", "baseline", lambda tr, te: te.cvss3.fillna(-1).to_numpy())
record("EPSS first", "baseline", lambda tr, te: te.epss.to_numpy())
record("Exploit flag, then EPSS", "baseline", lambda tr, te: flag_epss(te).to_numpy())
record("Hand heuristic (fair on B only)", "baseline", lambda tr, te: heuristic(te).to_numpy())
record("Text only", "TF-IDF", lambda tr, te: te.text_kev_lr.to_numpy())
record("Text only", "Granite", lambda tr, te: te.gr_kev_lr.to_numpy())
record("Text only", "hybrid TF-IDF+Granite", lambda tr, te: te.gr_hybrid_lr.to_numpy())
lr_pipe = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),  # noqa: E731
                                LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000))
record("Logistic regression", "safe, TF-IDF + Granite",
       lambda tr, te: lr_pipe().fit(tr[CORE + TFIDF + GRANITE], tr.y).predict_proba(te[CORE + TFIDF + GRANITE])[:, 1])
for name, feats in VARIANTS.items():
    record("LightGBM", name, lambda tr, te, f=feats: gbm_score(fit_gbm(tr, f), te, f))

clean = [k for k in VARIANTS if "LEAK" not in k]
best_gbm = max(clean, key=lambda k: (next(r["hits@500 total"] for r in rows if r["model"] == "LightGBM" and r["features"] == k),
                                     "hybrid" in k))     # on a tie, prefer the variant that uses Granite
record("Ensemble: LightGBM + exploit/EPSS", best_gbm,
       lambda tr, te: rank01(cache[("LightGBM", best_gbm, next(k for k, v in SPLITS.items() if v[1] is te))]) + rank01(flag_epss(te)))

BASE = VARIANTS[best_gbm]
ABL = {"- Granite / hybrid text": GRANITE, "- TF-IDF text model": TFIDF, "- text keyword flags": [f for f in BASE if f.startswith("txt_")],
       "- siblings": [f for f in BASE if G.get(f) == "siblings"], "- history": [f for f in BASE if G.get(f) == "history"],
       "- exploit signals": [f for f in BASE if G.get(f) == "signals"]}
for name, drop in ABL.items():
    feats = [f for f in BASE if f not in drop]
    record("LightGBM ablation", name, lambda tr, te, f=feats: gbm_score(fit_gbm(tr, f), te, f))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(REP, "model_results.csv"), index=False)

# ----------------------------------------------------------------------------- feature maps
tr_a, te_a = SPLITS["A 2024->2025"]
pool = pd.concat([tr_a.assign(lbl=tr_a.y), te_a.assign(lbl=te_a.y_nov4)], ignore_index=True)
base_rate = pool.lbl.mean()
lift_rows = []
for f in CORE + TFIDF + GRANITE + KEVNOW + CVSS + LEAK:
    x = pool[f].astype(float)
    ok = x.notna()
    if x[ok].nunique() < 2:
        continue
    top = pool[ok].assign(x=x[ok]).sort_values("x", ascending=False).head(int(ok.sum() * 0.05))
    lift_rows.append({"feature": f, "group": G.get(f), "risk": RISK.get(f, "safe"),
                      "univariate_auc": round(roc_auc_score(pool.lbl[ok], x[ok]), 3),
                      "kev_rate_top5pct": round(top.lbl.mean(), 4), "lift_top5pct": round(top.lbl.mean() / base_rate, 1),
                      "coverage": round(ok.mean(), 3)})
lift = pd.DataFrame(lift_rows).sort_values("lift_top5pct", ascending=False)
lift.to_csv(os.path.join(REP, "univariate_lift.csv"), index=False)
COLOR = {"safe": "#1d7346", "careful": "#a86a08", "leak": "#b8302a"}
t = lift.head(30).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 9))
ax.barh(t.feature, t.lift_top5pct, color=[("#2149c2" if G.get(f) == "granite" else COLOR.get(r, "#556172")) for f, r in zip(t.feature, t.risk)])
ax.set_xlabel("KEV rate in the top 5% of the feature / base rate")
ax.set_title("Univariate lift (2024 + 2025 replicas)\nblue = Granite, green = safe, amber = careful, red = leak")
ax.grid(axis="x", alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "univariate_lift.png"), dpi=130)
plt.close()

cf = [f for f in CORE + TFIDF + GRANITE + KEVNOW + CVSS if te_a[f].nunique() > 1]
corr = te_a[cf].corr(method="spearman")
fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cf)), cf, rotation=90, fontsize=7)
ax.set_yticks(range(len(cf)), cf, fontsize=7)
plt.colorbar(im, fraction=0.03)
ax.set_title("Spearman correlation between features (2025 replica)")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "feature_correlation.png"), dpi=130)
plt.close()

# ----------------------------------------------------------------------------- final model -> 2026
FEATS = BASE
use_ens = next(r["hits@500 total"] for r in rows if r["model"].startswith("Ensemble")) > \
    next(r["hits@500 total"] for r in rows if r["model"] == "LightGBM" and r["features"] == best_gbm)
train_all = pd.concat([load(2024, "2024-12-31"), load(2025, "2025-12-31")], ignore_index=True)
s26 = load(2026, "2026-11-04")
final = fit_gbm(train_all, FEATS)
gbm26 = gbm_score(final, s26, FEATS)
s26["score"] = rank01(gbm26) + rank01(flag_epss(s26)) if use_ens else gbm26
contrib = np.mean([m.predict(s26[FEATS], pred_contrib=True) for m in final], axis=0)[:, :-1]
imp = pd.DataFrame({"feature": FEATS, "group": [G.get(f) for f in FEATS], "mean_abs_shap_2026": np.abs(contrib).mean(axis=0),
                    "gain": np.mean([m.booster_.feature_importance("gain") for m in final], axis=0)}) \
    .sort_values("mean_abs_shap_2026", ascending=False)
imp.to_csv(os.path.join(REP, "feature_importance.csv"), index=False)
t = imp.head(25).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 8))
ax.barh(t.feature, t.mean_abs_shap_2026, color=["#2149c2" if g == "granite" else "#556172" for g in t.group])
ax.set_xlabel("mean |SHAP| on the 2026 scoreboard")
ax.set_title(f"What the final model leans on (blue = Granite)\nLightGBM [{best_gbm}]")
ax.grid(axis="x", alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "feature_importance.png"), dpi=130)
plt.close()

order = np.argsort(-s26.score.to_numpy(), kind="stable")
s26, contrib = s26.iloc[order].reset_index(drop=True), contrib[order]


def reasons(i):
    idx = np.argsort(-contrib[i])[:3]
    return "; ".join(f"{FEATS[j]}={s26[FEATS[j]].iat[i]:.3g} (+{contrib[i, j]:.2f})" for j in idx if contrib[i, j] > 0)


s26["rank"] = np.arange(1, len(s26) + 1)
s26["top_reasons"] = [reasons(i) if i < 500 else "" for i in range(len(s26))]
s26[["rank", "cve_id", "score", "snapshot_month", "assigner", "epss", "any_flag", "gr_kev_lr", "top_reasons"]] \
    .to_csv(os.path.join(DER, "scored_2026_v1.csv"), index=False)
s26.head(500)[["cve_id"]].to_csv(os.path.join(SUB, "top500_v1.csv"), index=False)
t5 = s26.head(500)
v0 = set(pd.read_csv(os.path.join(SUB, "top500_v0.csv")).cve_id) if os.path.exists(os.path.join(SUB, "top500_v0.csv")) else set()
print(f"\nfinal: LightGBM [{best_gbm}]{' + exploit/EPSS rank ensemble' if use_ens else ''}")
print(f"top500_v1: {t5.any_flag.sum()} with an exploit flag, {t5.txt_wordpress.sum()} WordPress, "
      f"{(t5.assigner == 'secalert_us@oracle.com').sum()} Oracle, overlap with v0 = {len(v0 & set(t5.cve_id))}")
print(t5.snapshot_month.value_counts().sort_index().to_string())
