"""One unified logistic regression to replace avg(Chen LR, Luis LR) as the linear voter (luis/context_Luis.md 14.16).

Columns: Chen's 100 (base 15 + meta 27 + team 3 + Granite 30B 55) plus the Luis-only columns of Luis's LR that have
no counterpart in Chen's (lr_columns_luis_chen.csv), with the same log1p rule as cv.luis_feats:
  epss (raw), n_flags, age_days, desc_len, 11 regex text flags, assigner_kev_3y, assigner_volume_3y, txt_vuldb_disclosed.
Left out on purpose: cv4_e_poc (CVSS 4 exploit maturity, excluded as after-the-fact since the start) and
days_to_cutoff (in training it mostly encodes the publish year: 2024 rows ~400-700 days to the Dec-31-2025 label
cutoff, 2026 rows 40-300 days to Nov 4; extrapolation, and the time-left adjustment was neutral).

Three versions, fixed before running (all reported):
  v1  curated union, L2, C grid extended down to 1e-5 (Chen's C sat at the grid's lower edge, 1e-4)
  v2  same columns, group-balanced L2: after standardizing, each block (base / meta / team / Granite / Luis-only)
      is scaled by 1/sqrt(block size), so a 55-column block doesn't get 55x the say of a 1-column signal
  v3  same columns, elastic net (saga, l1_ratio 0.5), same C grid
C picked by inner grouped 3-fold CV on the training rows only (all positives + 15k negatives, AP), as in cv.fit_lr.

  python scripts/lr_unified.py cv            # OOF columns lr_unified_v1..v3 -> results/cv_oof_lr.parquet, cv_fwd_lr.parquet
  python scripts/lr_unified.py final v1      # train on all 2024-01..2025-09, score scoreboard
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import cv  # noqa: E402
from stack import features, meta_features  # noqa: E402

ROOT = cv.ROOT
C_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
LUIS_ONLY = ["epss", "n_flags", "age_days", "desc_len", "txt_remote", "txt_sqli", "txt_xss", "txt_traversal",
             "txt_deser", "txt_cmdi", "txt_memory", "txt_privesc", "txt_ssrf", "txt_upload", "txt_kernel",
             "assigner_kev_3y", "assigner_volume_3y"]
VULDB = (r"exploit has been (?:disclosed|made available) to the public|"
         r"has been disclosed to the public and may be used")   # Luis's regex (luis/analysis/common.py)


def blocks(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    lo = cv.luis_feats(df, LUIS_ONLY)
    lo["luis_txt_vuldb_disclosed"] = df.description.fillna("").str.contains(VULDB, case=False, regex=True).astype(int).to_numpy()
    return {"base": features(df), "meta": meta_features(df), "team": cv.team_feats(df), "granite": cv.llm_feats(df),
            "luis_only": lo}


def matrix(B: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X = pd.concat(B.values(), axis=1)
    gid = np.concatenate([[i] * b.shape[1] for i, b in enumerate(B.values())])
    return np.nan_to_num(X.to_numpy(float)), gid, list(X.columns)


class Model:
    """StandardScaler (+ optional block weights) + logistic regression."""

    def __init__(self, version: str, gid: np.ndarray, C: float):
        self.v, self.C = version, C
        sizes = np.bincount(gid)
        self.w = 1 / np.sqrt(sizes[gid]) if version == "v2" else np.ones(len(gid))

    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        kw = dict(penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=3000) if self.v == "v3" else dict(max_iter=3000)
        self.lr = LogisticRegression(C=self.C, class_weight="balanced", **kw).fit(self.sc.transform(X) * self.w, y)
        return self

    def score(self, X):
        return self.lr.decision_function(self.sc.transform(X) * self.w)


def fit(version, X, y, groups, gid):
    rng = np.random.default_rng(0)
    neg = np.flatnonzero(y == 0)
    sub = np.sort(np.concatenate([np.flatnonzero(y == 1), rng.choice(neg, min(cv.INNER_NEG, len(neg)), replace=False)]))
    Xs, ys, gs = X[sub], y[sub], groups[sub]
    best, best_ap = None, -1
    for C in C_GRID:
        oof = np.zeros(len(ys))
        for a, b in GroupKFold(n_splits=3, shuffle=True, random_state=0).split(Xs, ys, gs):
            oof[b] = Model(version, gid, C).fit(Xs[a], ys[a]).score(Xs[b])
        ap = average_precision_score(ys, oof)
        if ap > best_ap:
            best, best_ap = C, ap
    return Model(version, gid, best).fit(X, y), best


def job(version, mode, f, X, y, groups, gid, train, test):
    m, C = fit(version, X[train], y[train], groups[train], gid)
    return version, mode, f, np.flatnonzero(test), m.score(X[test]), C


def run_cv() -> None:
    h = cv.load()
    folds = pd.read_parquet(cv.PROC / "cv_folds.parquet").set_index("cve_id")
    h["fold"], h["group"] = h.cve_id.map(folds.fold), h.cve_id.map(folds.group)
    X, gid, cols = matrix(blocks(h))
    print(f"{X.shape[1]} columns: " + ", ".join(f"{k} {n}" for k, n in zip(["base", "meta", "team", "granite", "luis_only"],
                                                                            np.bincount(gid))), flush=True)
    y, groups = h.label.to_numpy(), h.group.fillna(h.cve_id).to_numpy()
    jobs = []
    for v in ("v1", "v2", "v3"):
        for f in range(cv.K_FOLDS):
            test = (h.fold == f).to_numpy()
            purge = h.group.isin(set(h.group[test].dropna())).to_numpy()
            jobs.append(delayed(job)(v, "oof", f, X, y, groups, gid, (~test) & ~purge, test))
            if f > 0:
                jobs.append(delayed(job)(v, "fwd", f, X, y, groups, gid, (h.fold < f).to_numpy() & ~purge, test))
    res = Parallel(n_jobs=len(jobs))(jobs)
    for mode, fname in (("oof", "cv_oof_lr.parquet"), ("fwd", "cv_fwd_lr.parquet")):
        out = pd.read_parquet(cv.RES / fname)
        assert (out.cve_id.to_numpy() == h.cve_id.to_numpy()).all()
        for v in ("v1", "v2", "v3"):
            out[f"lr_unified_{v}"] = np.nan
        for v, md, f, rows, s, C in res:
            if md == mode:
                out.loc[rows, f"lr_unified_{v}"] = s
                print(f"{mode} {v} fold {f}: C={C}", flush=True)
        out.to_parquet(cv.RES / fname, index=False)


def run_final(version: str) -> None:
    tr = cv.load()
    g = pd.read_csv(cv.PROC / "kev_groups.csv").set_index("cve_id").group
    groups = tr.cve_id.map(g).fillna(tr.cve_id).to_numpy()
    sb = pd.read_parquet(cv.PROC / "scoreboard_2026_view.parquet").reset_index(drop=True)
    Btr, Bsb = blocks(tr), blocks(sb)
    X, gid, cols = matrix(Btr)
    Xs, _, cols_sb = matrix(Bsb)
    assert cols == cols_sb
    m, C = fit(version, X, tr.label.to_numpy(), groups, gid)
    print(f"{version}: {len(cols)} columns, C={C}, trained on {len(tr):,} rows ({int(tr.label.sum())} positives)")
    coef = pd.DataFrame({"feature": cols, "block": np.array(["base", "meta", "team", "granite", "luis_only"])[gid],
                         "coef_per_sd": m.lr.coef_[0] * m.w})
    coef.reindex(coef.coef_per_sd.abs().sort_values(ascending=False).index).to_csv(
        cv.RES / f"lr_unified_{version}_coefficients.csv", index=False)
    out = pd.DataFrame({"cve_id": sb.cve_id.astype(str), "lr_score": m.score(Xs).astype(float)})
    assert len(out) == 72670 and out.cve_id.is_unique and out.lr_score.notna().all()
    out.to_parquet(cv.RES / f"lr_unified_{version}_scoreboard.parquet", index=False)
    print(f"wrote results/lr_unified_{version}_scoreboard.parquet and _coefficients.csv; nonzero coefs {(m.lr.coef_ != 0).sum()}")


if __name__ == "__main__":
    {"cv": lambda: run_cv(), "final": lambda: run_final(sys.argv[2])}[sys.argv[1]]()
