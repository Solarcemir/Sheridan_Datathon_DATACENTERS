"""Static prior: learn what KEV-bound CVEs look like from MORE years, without exploit signals.

Problem: the A/B models train on one replica year (40-55 positives), and 40 of 75 test positives are invisible to every
model (no exploit code, ordinary EPSS). Exploit signals only exist for 2024-25, but everything else does for older years.
Idea: build cohorts for 2022, 2023, 2024 (and 2025 for the final model) with the same rules as the replicas, keep only
features that don't need signals (text, TF-IDF KEV model, assigner/product history, CVSS, siblings by count, Chen-lite),
and train a "static prior" with labels = KEV added after the snapshot month AND before Jan 1 of the test year, so the
test stays strictly forward (also for B: 2024 is predicted from 2022-23 only). KEV history features use KEV before Jan 1
of each cohort's own year; cohorts before 2022 are skipped because the KEV catalog started in Nov 2021.
Then: static prior alone, combined with v4's components (XGBoost big + LR), and how many no-signal positives each catches.

Outputs: derived/static/static_cohort_{year}.parquet (~1 MB each, committed), reports/static_prior_results.csv,
         reports/static_prior_bootstrap.csv, reports/static_prior_invisible.csv, reports/static_prior_log.txt (through Tee)

Usage:  python -E -P luis/analysis/static_prior.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

import build_features as bf  # noqa: E402
from common import (CHEN_LITE, DER, FULL, LGB, REP, SAFE, add_chen_lite, capped, density_k, fd, hits, jan_sep, load,  # noqa: E402
                    lr_pipe, m_lr, m_xgb, pooled_eval, rank01)

T0 = time.time()
CACHE = os.path.join(DER, "static")
os.makedirs(CACHE, exist_ok=True)


def log(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


G = fd.set_index("feature").group.to_dict()
STATIC = ([f for f in fd.feature if f.startswith("txt_") and f != "txt_in_wild"] + ["desc_len", "text_kev_lr", "text_kev_sim",
          "n_same_day", "n_adjacent_ids", "age_days"] + [f for f in fd.feature if G.get(f) == "history"]
          + [f for f in fd.feature if G.get(f) == "cvss"] + CHEN_LITE)


def static_cohort(Y):
    path = os.path.join(CACHE, f"static_cohort_{Y}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    v = bf.v
    pub = v[v.published.dt.year == Y].copy()
    end = pub.published + pd.offsets.MonthEnd(0)
    added = pub.cve_id.map(bf.kev_date)
    gap = pub[added <= end].assign(kev_by=end[added <= end])[["cve_id", "assigner", "published", "kev_by"]]
    c = pub[~(added <= end)].copy()
    c["snapshot_month"] = c.published.dt.strftime("%Y-%m")
    c["snapshot_end"] = c.published + pd.offsets.MonthEnd(0)
    c["epss"] = np.nan
    for f in bf.FLAGS:
        c[f] = 0
    c["kev_added"] = c.cve_id.map(bf.kev_date)
    c = bf.add_basic(c)
    c = bf.add_siblings(c, gap)
    c = bf.add_history(c, Y)
    c = bf.add_text_model(c, Y)
    c = add_chen_lite(c.reset_index(drop=True), Y)
    out = c[["cve_id", "snapshot_month", "snapshot_end", "published", "assigner", "kev_added"] + STATIC].copy()
    out.to_parquet(path, index=False)
    log(f"static cohort {Y}: {len(out):,} CVEs, ever-KEV after the snapshot month: {int(out.kev_added.notna().sum())}")
    return out


def labeled(years, T):
    """Stack cohorts; y = KEV added after the snapshot month and before Jan 1 of the test year T."""
    df = pd.concat([static_cohort(Y).assign(cohort=Y) for Y in years], ignore_index=True)
    df["y"] = ((df.kev_added > df.snapshot_end) & (df.kev_added < pd.Timestamp(f"{T}-01-01"))).astype(int)
    return df


def static_lr(tr, te):
    """C chosen on the last training cohort (fit on the earlier ones), then refit on all."""
    last = tr.cohort.max()
    a, b = tr[tr.cohort < last], tr[tr.cohort == last]
    grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    ap = [average_precision_score(b.y, lr_pipe(C).fit(a[STATIC], a.y).predict_proba(b[STATIC])[:, 1]) for C in grid]
    C = grid[int(np.argmax(ap))]
    return lr_pipe(C).fit(tr[STATIC], tr.y).predict_proba(te[STATIC])[:, 1], C


def static_lgbm(tr, te, seeds=10):
    return np.mean([lgb.LGBMClassifier(random_state=s, **LGB).fit(tr[STATIC], tr.y).predict_proba(te[STATIC])[:, 1]
                    for s in range(seeds)], axis=0)


# ----------------------------------------------------------------------------- data
for Y in (2022, 2023, 2024):
    static_cohort(Y)
st = {"A": labeled([2022, 2023, 2024], 2025), "B": labeled([2022, 2023], 2024)}
log("static training sets: " + ", ".join(f"{k}: {len(d):,} CVEs, {int(d.y.sum())} positives "
                                         f"({d.groupby('cohort').y.sum().to_dict()})" for k, d in st.items()))
tr24, tr25 = load(2024, "2024-12-31"), load(2025, "2025-12-31")
te25, te24 = jan_sep(load(2025, "2025-11-04"), 2025), jan_sep(load(2024, "2024-11-04"), 2024)
SPLITS = {"A": (tr24, te25), "B": (tr25, te24)}

sc = {}
for name, (tr, te) in SPLITS.items():
    R = {}
    sc[("Exploit code, then EPSS", name)] = (te.any_flag * 10 + te.epss).to_numpy()
    R["xgb"] = rank01(m_xgb(tr, te, FULL))
    R["lr"] = rank01(m_lr(tr, te, SAFE + CHEN_LITE))
    p, C = static_lr(st[name], te)
    R["slr"] = rank01(p)
    R["slgb"] = rank01(static_lgbm(st[name], te))
    log(f"{name}: models done (static LR C={C})")
    for k, parts in {
        "Static prior LR (no signals, 2022+ cohorts)": ["slr"],
        "Static prior LightGBM": ["slgb"],
        "v4: XGBoost + LR  [reference]": ["xgb", "lr"],
        "v4 + static LR": ["xgb", "lr", "slr"],
        "v4 + static LightGBM": ["xgb", "lr", "slgb"],
        "v4 + both static": ["xgb", "lr", "slr", "slgb"],
        "XGBoost + static LR": ["xgb", "slr"],
        "LR + static LR": ["lr", "slr"],
    }.items():
        sc[(k, name)] = np.sum([R[x] for x in parts], axis=0)
    sc[("Exploit code + static LR (no ML on signals)", name)] = rank01(sc[("Exploit code, then EPSS", name)]) + R["slr"]

MODELS = list(dict.fromkeys(m for m, _ in sc))
rows = []
for m in MODELS:
    for cap in (None, 3):
        r = {"model": m, "cap": cap or "none"}
        for name, (tr, te) in SPLITS.items():
            r[f"{name}@{density_k(len(te))}"] = hits(te.y_nov4, capped(te, sc[(m, name)], cap), density_k(len(te)))
        r["A+B @density"] = r["A@243"] + r["B@198"]
        rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(os.path.join(REP, "static_prior_results.csv"), index=False)
pd.set_option("display.width", 250)
print(res.to_string(index=False))

both = pd.concat([te25, te24], ignore_index=True)
boot, pooled = pooled_eval(both, {m: np.r_[sc[(m, "A")], sc[(m, "B")]] for m in MODELS},
                           "Exploit code, then EPSS", "v4: XGBoost + LR  [reference]")
boot.to_csv(os.path.join(REP, "static_prior_bootstrap.csv"), index=False)
K = density_k(len(both))
print(f"\npooled: top {K} of {len(both):,}, {both.y_nov4.sum()} positives ('vs reference' = vs v4 with cap 3)")
print(boot.to_string(index=False))

# ----------------------------------------------------------------------------- the invisible positives
yv, inv = both.y_nov4.to_numpy() == 1, both.any_flag.to_numpy() == 0
inv_rows = []
for m in MODELS:
    top = np.argsort(-pooled[(m, None)], kind="stable")[:K]
    found = np.zeros(len(both), bool)
    found[top] = True
    inv_rows.append({"model": m, "positives found": int((found & yv).sum()), "of which had NO exploit code": int((found & yv & inv).sum()),
                     "no-code positives in test": int((yv & inv).sum())})
inv_df = pd.DataFrame(inv_rows)
inv_df.to_csv(os.path.join(REP, "static_prior_invisible.csv"), index=False)
print("\nPositives without public exploit code at the snapshot, found in each model's top", K)
print(inv_df.to_string(index=False))
t_v4 = set(np.argsort(-pooled[("v4: XGBoost + LR  [reference]", 3)], kind="stable")[:K])
t_st = set(np.argsort(-pooled[("Static prior LR (no signals, 2022+ cohorts)", None)], kind="stable")[:K])
pos = set(np.where(yv)[0])
print(f"\nv4 (cap 3) and static LR top {K}: {len(t_v4 & t_st)} CVEs in common; positives: v4 {len(t_v4 & pos)}, static {len(t_st & pos)}, "
      f"both {len(t_v4 & t_st & pos)}, only static {len((t_st - t_v4) & pos)}")
log("done")
