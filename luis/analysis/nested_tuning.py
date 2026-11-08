"""Nested hyperparameter tuning: does a grid search help, when it only ever looks at training data?

Tuning on the A/B test itself would inflate the result (winner's curse). So for each split the search runs INSIDE the
training year: fit on months 01-06, validate on months 07-11 (average precision, steadier than hits with ~20 positives).
The chosen settings are then refit on the whole training year and evaluated ONCE on the test year, against the default
settings used so far (v4f = XGBoost big + LR, rank average, Granite demotion filters, cap 3 per assigner-month).

Tuned inside training: XGBoost (random search over a grid), LightGBM (same), the XGBoost-vs-LR ensemble weight, and the
cap value. The LR's C is already tuned inside training (common.m_lr).

Outputs: reports/nested_tuning_choices.csv, reports/nested_tuning_results.csv, reports/nested_tuning_bootstrap.csv,
         reports/nested_tuning_log.txt (through Tee)
Usage:  python -E -P luis/analysis/nested_tuning.py
"""
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from common import (CHEN, CHEN_LITE, FULL, LGB, REP, SAFE, XGB_BIG, capped, density_k, hits, jan_sep, load, lr_pipe,  # noqa: E402
                    m_lr, pooled_eval, rank01)

T0 = time.time()
RNG = np.random.default_rng(11)
N_CONFIGS = 30


def log(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


XGB_GRID = dict(max_depth=[3, 4, 6, 8], learning_rate=[0.01, 0.03, 0.05], n_estimators=[300, 800, 1500],
                min_child_weight=[3, 10, 30], subsample=[0.6, 0.8], colsample_bytree=[0.3, 0.5, 0.7],
                reg_lambda=[1, 10, 30], scale_pos_weight=[5, 25, 100])
LGB_GRID = dict(num_leaves=[7, 15, 31, 63], learning_rate=[0.01, 0.03, 0.05], n_estimators=[300, 800, 1500],
                min_child_samples=[10, 30, 100], subsample=[0.6, 0.8], colsample_bytree=[0.3, 0.5, 0.7],
                reg_lambda=[1, 5, 30], scale_pos_weight=[5, 25, 100])


def sample_configs(grid, n):
    keys = list(grid)
    allc = list(itertools.product(*grid.values()))
    pick = RNG.choice(len(allc), size=n, replace=False)
    return [dict(zip(keys, allc[i])) for i in pick]


def fit_xgb(tr, F, p, seed):
    return xgb.XGBClassifier(tree_method="hist", device="cuda", eval_metric="aucpr", random_state=seed, **p).fit(tr[F], tr.y)


def fit_lgb(tr, F, p, seed):
    base = {k: v for k, v in LGB.items() if k not in p}
    return lgb.LGBMClassifier(random_state=seed, **base, **p).fit(tr[F], tr.y)


def inner(tr):
    return tr[tr.snapshot_month.str[-2:] <= "06"].reset_index(drop=True), tr[tr.snapshot_month.str[-2:] >= "07"].reset_index(drop=True)


ex = pd.read_parquet(os.path.join(CHEN, "processed", "llm_extract.parquet"))[["cve_id", "impact", "typical_attacker", "attacker_appeal", "product_type"]]


def demote_mask(df):
    d = df[["cve_id"]].merge(ex, on="cve_id", how="left")
    return ((d.impact == "denial_of_service") | (d.typical_attacker == "unlikely_target") | (d.attacker_appeal == "low")
            | (d.product_type == "wordpress_plugin_or_theme")).fillna(False).to_numpy()


def finish(df, xgb_pct, lr_pct, w, cap):
    s = w * xgb_pct + (1 - w) * lr_pct - 50 * demote_mask(df)
    return capped(df, s, cap)


tr24, tr25 = load(2024, "2024-12-31"), load(2025, "2025-12-31")
te25, te24 = jan_sep(load(2025, "2025-11-04"), 2025), jan_sep(load(2024, "2024-11-04"), 2024)
SPLITS = {"A": (tr24, te25), "B": (tr25, te24)}
choices, sc = [], {}
for name, (tr, te) in SPLITS.items():
    a, b = inner(tr)
    log(f"{name}: inner fit {len(a):,} rows / {int(a.y.sum())} pos, inner validation {len(b):,} rows / {int(b.y.sum())} pos")
    # XGBoost search (1 seed per config inside)
    best_x, best_ap = None, -1
    for p in sample_configs(XGB_GRID, N_CONFIGS):
        ap = average_precision_score(b.y, fit_xgb(a, FULL, p, 0).predict_proba(b[FULL])[:, 1])
        if ap > best_ap:
            best_x, best_ap = p, ap
    default_ap_x = average_precision_score(b.y, fit_xgb(a, FULL, dict(XGB_BIG, scale_pos_weight=25), 0).predict_proba(b[FULL])[:, 1])
    log(f"{name}: XGBoost search done (best inner AP {best_ap:.4f} vs default {default_ap_x:.4f})")
    # LightGBM search
    best_l, best_apl = None, -1
    for p in sample_configs(LGB_GRID, N_CONFIGS):
        ap = average_precision_score(b.y, fit_lgb(a, SAFE, p, 0).predict_proba(b[SAFE])[:, 1])
        if ap > best_apl:
            best_l, best_apl = p, ap
    default_ap_l = average_precision_score(b.y, fit_lgb(a, SAFE, {}, 0).predict_proba(b[SAFE])[:, 1])
    log(f"{name}: LightGBM search done (best inner AP {best_apl:.4f} vs default {default_ap_l:.4f})")
    # ensemble weight and cap, chosen on the inner split with the tuned XGBoost + LR
    xi = rank01(fit_xgb(a, FULL, best_x, 0).predict_proba(b[FULL])[:, 1])
    # fixed C here: the inner fit set (months 01-06) has no months 07-11 left for m_lr's own C search
    li = rank01(lr_pipe(1e-3).fit(a[SAFE + CHEN_LITE], a.y).predict_proba(b[SAFE + CHEN_LITE])[:, 1])
    kb = density_k(len(b))
    grid_wc = [(w, c) for w in (0.3, 0.4, 0.5, 0.6, 0.7) for c in (None, 2, 3, 5, 8)]
    score_wc = {(w, c): (average_precision_score(b.y, w * xi + (1 - w) * li), hits(b.y, finish(b, xi, li, w, c), kb)) for w, c in grid_wc}
    w_best = max((0.3, 0.4, 0.5, 0.6, 0.7), key=lambda w: score_wc[(w, None)][0])
    c_best = max((None, 2, 3, 5, 8), key=lambda c: (score_wc[(w_best, c)][1], c == 3))   # ties -> keep 3
    choices.append({"split": name, "xgb": best_x, "xgb inner AP": round(best_ap, 4), "xgb default inner AP": round(default_ap_x, 4),
                    "lgbm": best_l, "lgbm inner AP": round(best_apl, 4), "lgbm default inner AP": round(default_ap_l, 4),
                    "weight on XGBoost": w_best, "cap": c_best})
    # refit on the whole training year, score the test year once
    xd = rank01(np.mean([fit_xgb(tr, FULL, dict(XGB_BIG, scale_pos_weight=25), s).predict_proba(te[FULL])[:, 1] for s in range(5)], axis=0))
    xt = rank01(np.mean([fit_xgb(tr, FULL, best_x, s).predict_proba(te[FULL])[:, 1] for s in range(5)], axis=0))
    lt = rank01(np.mean([fit_lgb(tr, SAFE, best_l, s).predict_proba(te[SAFE])[:, 1] for s in range(5)], axis=0))
    ld = rank01(np.mean([fit_lgb(tr, SAFE, {}, s).predict_proba(te[SAFE])[:, 1] for s in range(5)], axis=0))
    lr_ = rank01(m_lr(tr, te, SAFE + CHEN_LITE))
    sc[("Exploit code, then EPSS", name)] = (te.any_flag * 10 + te.epss).to_numpy()
    sc[("v4f default (XGBoost default + LR, w 0.5, cap 3)", name)] = finish(te, xd, lr_, 0.5, 3)
    sc[("v4f tuned XGBoost only", name)] = finish(te, xt, lr_, 0.5, 3)
    sc[("v4f tuned XGBoost + tuned weight + tuned cap", name)] = finish(te, xt, lr_, w_best, c_best)
    sc[("LightGBM default + LR (filters, cap 3)", name)] = finish(te, ld, lr_, 0.5, 3)
    sc[("LightGBM tuned + LR (filters, cap 3)", name)] = finish(te, lt, lr_, 0.5, 3)
    log(f"{name}: test scored (weight {w_best}, cap {c_best})")

ch = pd.DataFrame(choices)
ch.to_csv(os.path.join(REP, "nested_tuning_choices.csv"), index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 200)
print("\nCHOSEN INSIDE TRAINING (compare A vs B: stable choices are a good sign)")
print(ch.to_string(index=False))

MODELS = list(dict.fromkeys(m for m, _ in sc))
rows = []
for m in MODELS:
    r = {"model": m}
    for name, (tr, te) in SPLITS.items():
        r[f"{name}@{density_k(len(te))}"] = hits(te.y_nov4, sc[(m, name)], density_k(len(te)))
    r["A+B"] = r["A@243"] + r["B@198"]
    rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(os.path.join(REP, "nested_tuning_results.csv"), index=False)
print("\nTEST (evaluated once)")
print(res.to_string(index=False))
both = pd.concat([te25, te24], ignore_index=True)
boot, _ = pooled_eval(both, {m: np.r_[sc[(m, "A")], sc[(m, "B")]] for m in MODELS}, "Exploit code, then EPSS",
                      "v4f default (XGBoost default + LR, w 0.5, cap 3)", caps=(None,))
boot = boot.drop(columns="cap")
boot.to_csv(os.path.join(REP, "nested_tuning_bootstrap.csv"), index=False)
print(f"\npooled top {density_k(len(both))} of {len(both):,} ('vs reference' = vs v4f default)")
print(boot.to_string(index=False))
log("done")
