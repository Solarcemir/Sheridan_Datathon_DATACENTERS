"""Nested tuning of XGBoost in the purged 5-fold CV (can more compute beat Luis's fixed settings?).

Outer: the 5 purged time-block folds of cv.py, used ONLY to measure. Inner: inside each outer fold's
training rows, 3 contiguous publish-date blocks (group-purged); a random search over N_CONFIGS settings
(Luis's XGB_BIG is config 0) is ranked by pooled inner average precision (Dec-31 label). The winner is
refit on the whole outer training set (5 seeds, rank-averaged) and scores the outer test block.
Also scores Luis's fixed settings on the same folds for an apples-to-apples comparison.

Feature sets: luis = Luis's safe features (as lgbm_luis_only); all = our best LR features + Luis's.

  python scripts/xgb_tune.py            # -> results/cv_xgb.parquet, results/xgb_tune_trials.csv
"""

import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from cv import K_FOLDS, PROC, RES, features, llm_feats, load, luis_feats, meta_features, team_feats  # noqa: E402

N_CONFIGS = int(os.environ.get("N_CONFIGS", 60))
ROUND = int(os.environ.get("ROUND", 1))          # 2 = extended space around the round-1 inner winners
DEVICE = os.environ.get("DEVICE", "cpu")         # "cuda": fits on the (single) visible GPU
SETS = os.environ.get("SETS", "luis,all").split(",")
THREADS, JOBS = (4, int(os.environ.get("JOBS", 12))) if DEVICE == "cuda" else (1, 88)
SUFFIX = "" if ROUND == 1 else f"{ROUND}"
SEEDS = 5
XGB_LUIS = dict(n_estimators=1500, max_depth=6, learning_rate=0.01, min_child_weight=10, subsample=0.7,
                colsample_bytree=0.5, reg_lambda=10, gamma=1.0, scale_pos_weight=25)


def configs(n: int) -> list[dict]:
    rng = np.random.default_rng(0)
    lu = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    out = [XGB_LUIS]
    for _ in range(n - 1):
        lr = lu(0.005, 0.1)
        out.append(dict(
            learning_rate=lr,
            n_estimators=int(np.clip(round(lu(15, 60) / lr / 10) * 10, 100, 3000)),  # ~ trees x lr in [15, 60]
            max_depth=int(rng.integers(2, 9)),
            min_child_weight=lu(1, 100),
            subsample=float(rng.uniform(0.5, 1.0)),
            colsample_bytree=float(rng.uniform(0.2, 1.0)),
            reg_lambda=lu(0.1, 100),
            reg_alpha=float(rng.choice([0, lu(0.01, 10)])),
            gamma=float(rng.choice([0, lu(0.01, 5)])),
            scale_pos_weight=lu(1, 300),
        ))
    return out


def configs2(n: int) -> list[dict]:
    """Round 2. Round-1 inner winners sat at the edge of the space (depth 8, 3,000 trees, lr 0.005,
    scale_pos_weight ~1), chosen on inner AP only (outer results not seen). Extend in that direction.
    Config 0 = Luis, 1 = round-1 winner."""
    rng = np.random.default_rng(1)
    lu = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    r1 = configs(3)[2]
    out = [XGB_LUIS, r1]
    for _ in range(n - 2):
        lr = lu(0.001, 0.01)
        out.append(dict(
            learning_rate=lr,
            n_estimators=int(np.clip(round(lu(10, 40) / lr / 10) * 10, 1000, 12000)),
            max_depth=int(rng.integers(6, 15)),
            min_child_weight=lu(0.2, 10),
            subsample=float(rng.uniform(0.4, 1.0)),
            colsample_bytree=float(rng.uniform(0.1, 0.6)),
            reg_lambda=lu(0.1, 100),
            reg_alpha=float(rng.choice([0, lu(0.01, 10)])),
            gamma=float(rng.choice([0, lu(0.01, 5)])),
            scale_pos_weight=lu(0.5, 5),
        ))
    return out


def fit_predict(X, y, tr, te, params, seed=0):
    import xgboost as xgb
    m = xgb.XGBClassifier(tree_method="hist", device=DEVICE, n_jobs=THREADS, random_state=seed, eval_metric="aucpr", **params)
    m.fit(X[tr], y[tr])
    return m.predict_proba(X[te])[:, 1]


def inner_splits(h: pd.DataFrame, train: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """3 contiguous publish-date blocks of the outer training rows, purged by KEV group."""
    idx = np.flatnonzero(train)
    order = idx[np.argsort(h.published.to_numpy()[idx], kind="stable")]
    blocks = np.array_split(order, 3)
    out = []
    for b in blocks:
        te = np.zeros(len(h), bool); te[b] = True
        purge = h.group.isin(set(h.group[te].dropna())).to_numpy()
        out.append((train & ~te & ~purge, te))
    return out


def inner_job(fs, f, ci, ii, X, y, tr, te, params):
    t = time.time()
    return fs, f, ci, ii, np.flatnonzero(te), fit_predict(X, y, tr, te, params), time.time() - t


def outer_job(fs, f, tag, X, y, tr, te, params, seed):
    return fs, f, tag, seed, np.flatnonzero(te), fit_predict(X, y, tr, te, params, seed)


def main() -> None:
    t0 = time.time()
    h = load()
    folds = pd.read_parquet(PROC / "cv_folds.parquet").set_index("cve_id")
    h["fold"], h["group"] = h.cve_id.map(folds.fold), h.cve_id.map(folds.group)
    LU = luis_feats(h)
    sets = {"luis": LU.to_numpy(np.float32),
            "all": pd.concat([features(h), meta_features(h), team_feats(h), llm_feats(h), LU], axis=1).to_numpy(np.float32)}
    sets = {k: v for k, v in sets.items() if k in SETS}
    y = h.label.to_numpy()
    cfgs = configs(N_CONFIGS) if ROUND == 1 else configs2(N_CONFIGS)
    outer = []
    for f in range(K_FOLDS):
        te = (h.fold == f).to_numpy()
        purge = h.group.isin(set(h.group[te].dropna())).to_numpy()
        outer.append(((~te) & ~purge, te))
    print(f"{len(h):,} rows; features " + " ".join(f"{k}={v.shape[1]}" for k, v in sets.items()) + "; "
          f"round {ROUND} on {DEVICE}: {N_CONFIGS} configs x 3 inner x {K_FOLDS} outer x {len(sets)} sets", flush=True)

    # 1) inner search
    jobs = []
    for fs, X in sets.items():
        for f, (tr, _) in enumerate(outer):
            for ii, (itr, ite) in enumerate(inner_splits(h, tr)):
                for ci, p in enumerate(cfgs):
                    jobs.append(delayed(inner_job)(fs, f, ci, ii, X, y, itr, ite, p))
    res = Parallel(n_jobs=JOBS, verbose=5)(jobs)
    pooled: dict = {}
    secs = []
    for fs, f, ci, ii, rows, s, dt in res:
        pooled.setdefault((fs, f, ci), []).append((rows, s))
        secs.append(dt)
    trials = []
    for (fs, f, ci), parts in pooled.items():
        rows = np.concatenate([r for r, _ in parts])
        # rank within each inner block so blocks are comparable when pooled
        s = np.concatenate([pd.Series(v).rank(pct=True).to_numpy() for _, v in parts])
        trials.append({"set": fs, "fold": f, "config": ci, "inner_ap": average_precision_score(y[rows], s), **cfgs[ci]})
    trials = pd.DataFrame(trials)
    trials.to_csv(RES / f"xgb_tune{SUFFIX}_trials_{'_'.join(sets)}.csv", index=False)
    print(f"inner search done in {time.time() - t0:.0f}s (median fit {np.median(secs):.0f}s)", flush=True)
    best = trials.loc[trials.groupby(["set", "fold"]).inner_ap.idxmax()]
    print(best[["set", "fold", "config", "inner_ap", "max_depth", "learning_rate", "n_estimators",
                "min_child_weight", "scale_pos_weight", "colsample_bytree"]].to_string(index=False), flush=True)
    luis_rank = trials[trials.config.isin([0, 1] if ROUND > 1 else [0])].set_index(["set", "fold", "config"]).inner_ap
    print("reference configs (0 = Luis, 1 = round-1 winner) inner AP:", luis_rank.round(4).to_dict(), flush=True)

    # 2) outer refits: tuned winner and Luis's fixed config, 5 seeds each
    jobs = []
    for fs, X in sets.items():
        for f, (tr, te) in enumerate(outer):
            bc = int(best[(best.set == fs) & (best.fold == f)].config.iloc[0])
            for tag, p in (("tuned", cfgs[bc]), ("fixed", XGB_LUIS)):
                for sd in range(SEEDS):
                    jobs.append(delayed(outer_job)(fs, f, tag, X, y, tr, te, p, sd))
    res = Parallel(n_jobs=JOBS, verbose=5)(jobs)
    out = h[["cve_id"]].copy()
    acc: dict = {}
    for fs, f, tag, sd, rows, s in res:
        acc.setdefault((fs, tag, f), []).append((rows, pd.Series(s).rank(pct=True).to_numpy()))
    for (fs, tag, f), parts in acc.items():
        col = f"xgb{SUFFIX}_{tag}_{fs}"
        if col not in out:
            out[col] = np.nan
        out.loc[parts[0][0], col] = np.mean([v for _, v in parts], axis=0)
    name = f"cv_xgb{SUFFIX}_{'_'.join(sets)}.parquet" if SUFFIX else "cv_xgb.parquet"
    out.to_parquet(RES / name, index=False)
    print(f"wrote results/{name} in {time.time() - t0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
