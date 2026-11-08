"""Forward-only (blocks 1-4, each trained on earlier blocks, purged) versions of the proxies for Luis's voters:
lr_luis_only (LR on Luis's features, cv.fit_lr) and xgb_fixed_luis (XGBoost, Luis's XGB_BIG, 5 seeds rank-averaged).
  python scripts/fwd_extra.py   -> results/cv_fwd_extra.parquet
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import cv, xgb_tune

def lr_job(X, y, g, tr, te):
    sc, lr, C = cv.fit_lr(X[tr], y[tr], g[tr]); return "lr_luis_only", te, lr.decision_function(sc.transform(X[te]))

def xgb_job(X, y, tr, te, seed):
    xgb_tune.THREADS = 4
    return f"xgb{seed}", te, pd.Series(xgb_tune.fit_predict(X, y, tr, te, xgb_tune.XGB_LUIS, seed)).rank(pct=True).to_numpy()

h = cv.load(); f = pd.read_parquet(cv.PROC / "cv_folds.parquet").set_index("cve_id")
h["fold"], h["group"] = h.cve_id.map(f.fold), h.cve_id.map(f.group)
X = cv.luis_feats(h).to_numpy(np.float32); y = h.label.to_numpy(); g = h.group.fillna(h.cve_id).to_numpy()
jobs = []
for k in range(1, 5):
    te = (h.fold == k).to_numpy(); purge = h.group.isin(set(h.group[te].dropna())).to_numpy()
    tr = (h.fold < k).to_numpy() & ~purge; tei = np.flatnonzero(te)
    jobs.append(delayed(lr_job)(X, y, g, tr, tei))
    jobs += [delayed(xgb_job)(X, y, tr, te, s) for s in range(5)]
res = Parallel(n_jobs=len(jobs))(jobs)
out = h[["cve_id", "fold"]].copy(); out["lr_luis_only"] = np.nan; xs = {}
for name, te, s in res:
    te = np.flatnonzero(te) if te.dtype == bool else te
    if name == "lr_luis_only": out.loc[te, name] = s
    else: xs.setdefault(tuple(te[:3]), []).append((te, s))
out["xgb_fixed_luis"] = np.nan
for parts in xs.values(): out.loc[parts[0][0], "xgb_fixed_luis"] = np.mean([s for _, s in parts], axis=0)
out.to_parquet(cv.RES / "cv_fwd_extra.parquet", index=False); print("wrote results/cv_fwd_extra.parquet", out.notna().sum().to_dict())
