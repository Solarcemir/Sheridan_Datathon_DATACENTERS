"""5-fold time-blocked, group-purged cross-validation over 2024-01..2025-09 (all 119 positives).

Folds: 5 consecutive publish-date blocks with equal CVE counts. When scoring fold f, training
drops fold f AND every CVE whose hand-made KEV group (processed/kev_groups.csv) has a member in
fold f (purging), so exploit-chain siblings never straddle train/test.

  python scripts/cv.py folds          # write processed/cv_folds.parquet + per-fold id lists
  python scripts/cv.py lr             # linear models, out-of-fold scores -> results/cv_oof_lr.parquet
  python scripts/cv.py report [files] # hits@517 / @1000 / AP with 95% bootstrap CIs
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from stack import FLAGS, features, meta_features

ROOT = Path(__file__).resolve().parent.parent
PROC, RES = ROOT / "processed", ROOT / "results"
K_FOLDS = 5
SCOREBOARD_ROWS = 72_670
import os as _os

def _both(proc, name: str) -> pd.DataFrame:
    """Historical + scoreboard versions of a per-CVE table (CVE ids don't overlap)."""
    parts = [proc / f"{name}_historical.parquet", proc / f"{name}_scoreboard.parquet"]
    return pd.concat([pd.read_parquet(x) for x in parts if x.exists()], ignore_index=True).drop_duplicates("cve_id")

FORWARD = _os.environ.get("CV_FORWARD") == "1"
OUT_FILE = "cv_fwd_lr.parquet" if FORWARD else "cv_oof_lr.parquet"
C_GRID = [0.0001, 0.0003, 0.001, 0.003, 0.01]
INNER_NEG = 15_000   # inner CV uses all positives + this many sampled negatives (speed)


def load() -> pd.DataFrame:
    h = pd.read_parquet(PROC / "historical_2024_2025.parquet")
    return h[h.snapshot_month.between("2024-01", "2025-09")].reset_index(drop=True)


def make_folds() -> None:
    h = load().sort_values(["published", "cve_id"]).reset_index(drop=True)
    h["fold"] = (np.arange(len(h)) * K_FOLDS // len(h)).astype(int)
    g = pd.read_csv(PROC / "kev_groups.csv").set_index("cve_id").group
    h["group"] = h.cve_id.map(g)
    h[["cve_id", "fold", "group"]].to_parquet(PROC / "cv_folds.parquet", index=False)
    for f in range(K_FOLDS):
        test = h[h.fold == f]
        purge_groups = set(test.group.dropna())
        train = h[(h.fold != f) & ~h.group.isin(purge_groups)]
        purged = h[(h.fold != f) & h.group.isin(purge_groups)]
        (PROC / f"cv_fold{f}_train.txt").write_text("\n".join(train.cve_id))
        (PROC / f"cv_fold{f}_test.txt").write_text("\n".join(test.cve_id))
        print(f"fold {f}: {test.published.min():%Y-%m-%d}..{test.published.max():%Y-%m-%d} "
              f"test {len(test):,} ({int(test.label.sum())} pos) | train {len(train):,} "
              f"({int(train.label.sum())} pos) | purged {len(purged)} ({sorted(purge_groups)})")


def granite_zero(h: pd.DataFrame) -> np.ndarray:
    s = pd.concat([pd.read_parquet(RES / "all2024_zero.parquet"), pd.read_parquet(RES / "holdout_zero.parquet")])
    return h.cve_id.map(s.drop_duplicates("cve_id").set_index("cve_id").score).clip(lower=-30).to_numpy()


def fit_lr(X, y, groups):
    """Pick C by inner grouped 3-fold CV (average precision) on the training data, refit on all of it."""
    rng = np.random.default_rng(0)
    neg = np.flatnonzero(y == 0)
    sub = np.sort(np.concatenate([np.flatnonzero(y == 1), rng.choice(neg, min(INNER_NEG, len(neg)), replace=False)]))
    Xs, ys, gs = X[sub], y[sub], groups[sub]
    best, best_ap = None, -1
    for C in C_GRID:
        oof = np.zeros(len(ys))
        for a, b in GroupKFold(n_splits=3, shuffle=True, random_state=0).split(Xs, ys, gs):
            sc = StandardScaler().fit(Xs[a])
            oof[b] = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(
                sc.transform(Xs[a]), ys[a]).decision_function(sc.transform(Xs[b]))
        ap = average_precision_score(ys, oof)
        if ap > best_ap:
            best, best_ap = C, ap
    sc = StandardScaler().fit(X)
    return sc, LogisticRegression(C=best, class_weight="balanced", max_iter=2000).fit(sc.transform(X), y), best


def emb_features(h: pd.DataFrame) -> pd.DataFrame:
    f = _both(PROC, "emb_feat").set_index("cve_id").reindex(h.cve_id)
    return pd.DataFrame({c: f[c].to_numpy() for c in ["kev_sim_max", "kev_sim_top5", "typ_sim_top5", "kev_contrast"]},
                        index=h.index)


def team_feats(h: pd.DataFrame) -> pd.DataFrame:
    f = _both(PROC, "team_feat").set_index("cve_id").reindex(h.cve_id)
    return pd.DataFrame({"text_kev_lr_logit": np.log(f.text_kev_lr / (1 - f.text_kev_lr)).to_numpy(),
                         "text_kev_sim": f.text_kev_sim.to_numpy(),
                         "log1p_adjacent_ids": np.log1p(f.n_adjacent_ids).to_numpy()}, index=h.index)


LGB_PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, scale_pos_weight=25,
                  verbose=-1, n_jobs=2, deterministic=True, force_row_wise=True)


def fit_lgbm_score(X, y, train, test):
    """Luis's LightGBM settings (deterministic), mean over 10 seeds (rank-averaged)."""
    import lightgbm as lgb
    preds = [lgb.LGBMClassifier(random_state=sd, **LGB_PARAMS).fit(X[train], y[train]).predict_proba(X[test])[:, 1]
             for sd in range(10)]
    return np.mean([pd.Series(p).rank(pct=True).to_numpy() for p in preds], axis=0)


def age_feats(h: pd.DataFrame) -> pd.DataFrame:
    """Days the CVE had been public at the end of its snapshot month; and the same for CVEs with no
    exploit code yet (no code after a long exposure is evidence; after 2 days it isn't)."""
    end = pd.PeriodIndex(h.snapshot_month, freq="M").to_timestamp(how="end").normalize()
    age = (end - h.published).dt.days.clip(lower=0).to_numpy() / 30.0
    no_code = (h[FLAGS].max(axis=1) == 0).to_numpy()
    return pd.DataFrame({"age_months": age, "age_months_no_code": age * no_code}, index=h.index)


def prior_feats(h: pd.DataFrame) -> pd.DataFrame:
    f = _both(PROC, "prior_sig").set_index("cve_id").reindex(h.cve_id).fillna(0)
    out = {}
    for pre in ("asg_", "ven_"):
        if f"{pre}n" not in f:
            continue
        out[f"{pre}log_n"] = np.log1p(f[f"{pre}n"])
        out[f"{pre}epss_p90"] = np.log(f[f"{pre}epss_p90"] + 1e-4)
        out[f"{pre}epss_max"] = f[f"{pre}epss_max"]
        out[f"{pre}log_epss_gt_0_1"] = np.log1p(f[f"{pre}n_epss_gt_0_1"])
        out[f"{pre}share_nuclei"] = f[f"{pre}share_nuclei"]
        out[f"{pre}share_metasploit"] = f[f"{pre}share_metasploit"]
        out[f"{pre}log_any_code"] = np.log1p(f[f"{pre}n_any_code"])
    return pd.DataFrame({k: v.to_numpy() for k, v in out.items()}, index=h.index)


def llm_feats(h: pd.DataFrame) -> pd.DataFrame:
    f = _both(PROC, "llm_feat").set_index("cve_id").reindex(h.cve_id)
    return pd.DataFrame(f.fillna(0).to_numpy(), index=h.index, columns=f.columns)


LLM_COMPACT = ["llm_attacker_appeal", "llm_install_base", "llm_product_kev", "llm_vendor_kev",
               "llm_product_ransomware_kev", "llm_typical_attacker=unlikely_target",
               "llm_typical_attacker=ransomware_or_extortion_groups", "llm_impact=denial_of_service",
               "llm_impact=remote_code_execution", "llm_impact=file_write_or_upload",
               "llm_product_type=enterprise_server_application", "llm_product_type=remote_access_or_it_management",
               "llm_product_type=email_or_collaboration", "llm_product_type=industrial_or_ot"]


def fresh_feats(h: pd.DataFrame) -> pd.DataFrame:
    f = _both(PROC, "fresh").set_index("cve_id").reindex(h.cve_id).fillna(0)
    return pd.DataFrame({f"log1p_{c}": np.log1p(f[c]).to_numpy() for c in f.columns}, index=h.index)


LUIS_DIR = next((p for p in (ROOT.parent / "luis" / "derived" / "for_chen",
                 ROOT / "team" / "luis" / "luis" / "derived" / "for_chen") if p.exists()),
                ROOT.parent / "luis" / "derived" / "for_chen")
LUIS_DROP = {"cve_id", "snapshot_month", "snapshot_end", "published", "assigner", "cwe_ids",
             "luis_y_dec31_of_year", "y_nov4", "txt_in_wild", "assigner_kev_ytd", "sib_kev_known"}
LUIS_SIB = ["n_same_day", "sib_flags", "sib_max_epss_pct", "n_adjacent_ids"]


def luis_feats(h: pd.DataFrame, cols=None) -> pd.DataFrame:
    """Luis's safe features from his shared Parquet (txt_in_wild and the hidden-KEV columns dropped)."""
    f = pd.concat([pd.read_parquet(LUIS_DIR / "luis_features_hist_2024_2025.parquet"),
                   pd.read_parquet(LUIS_DIR / "luis_features_scoreboard_2026.parquet")]).drop_duplicates("cve_id")
    f = f.set_index("cve_id").reindex(h.cve_id)
    cols = cols or [c for c in f.columns if c not in LUIS_DROP]
    X = f[cols].astype(float)
    for c in cols:  # counts on a log scale
        if X[c].max() > 10:
            X[c] = np.log1p(X[c].clip(lower=0))
    return pd.DataFrame(X.fillna(0).to_numpy(), index=h.index, columns=[f"luis_{c}" for c in cols])


def emb_pca(h: pd.DataFrame, n: int = 64) -> pd.DataFrame:
    """Description embedding compressed to n dims. PCA is unsupervised and fit on 2021-23 CVEs only
    (never on labels or on the evaluated rows), so it is fold-safe."""
    from sklearn.decomposition import PCA
    ids = pd.read_parquet(PROC / "emb_historical_ids.parquet").cve_id
    E = np.load(PROC / "emb_historical.npy").astype(np.float32)
    row = pd.Series(np.arange(len(ids)), index=ids)
    pca = PCA(n_components=n, random_state=0).fit(np.load(PROC / "emb_extra.npy").astype(np.float32))
    Z = pca.transform(E[row[h.cve_id].to_numpy()])
    return pd.DataFrame(Z, index=h.index, columns=[f"pc{i}" for i in range(n)])





def _one(name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, test: np.ndarray, train: np.ndarray):
    if name.startswith("lgbm"):
        return name, test, fit_lgbm_score(X, y, train, test), None
    sc, lr, C = fit_lr(X[train], y[train], groups[train])
    return name, test, lr.decision_function(sc.transform(X[test])), C


def run_lr() -> None:
    import os
    from joblib import Parallel, delayed
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    h = load()
    folds = pd.read_parquet(PROC / "cv_folds.parquet").set_index("cve_id")
    h["fold"], h["group"] = h.cve_id.map(folds.fold), h.cve_id.map(folds.group)
    F, M, EF, EP = features(h), meta_features(h), emb_features(h), emb_pca(h)
    L = llm_feats(h) if (PROC / "llm_feat_historical.parquet").exists() else None
    T = team_feats(h)
    A = age_feats(h)
    P = prior_feats(h)
    FR = fresh_feats(h)
    LU = luis_feats(h)
    LS = luis_feats(h, LUIS_SIB)
    h["granite_zero"] = granite_zero(h)
    G = h[["granite_zero"]]
    variants = {
        "lr_features": [F],
        "lr_features_meta": [F, M],
        "lr_features_granite": [F, G],
        "lr_features_meta_granite": [F, M, G],
        "lr_features_meta_embknn": [F, M, EF],
        "lr_features_meta_embknn_granite": [F, M, EF, G],
        "lr_features_meta_embknn_pca64": [F, M, EF, EP],
        "lr_features_meta_embknn_pca64_granite": [F, M, EF, EP, G],
    }
    variants.update({
        "lr_features_team": [F, T],
        "lr_features_meta_embknn_pca64_team": [F, M, EF, EP, T],
        "lgbm_features_meta_embknn_team": [F, M, EF, T],
        "lr_full_age": [F, M, EF, EP, T, A],
        "lr_full_prior": [F, M, EF, EP, T, P],
        "lr_full_age_prior": [F, M, EF, EP, T, A, P],
    })
    if L is not None:
        variants.update({
            "lr_features_meta_embknn_pca64_team_llmx": [F, M, EF, EP, T, L],
            "lr_full_llmx_prior": [F, M, EF, EP, T, L, P],
            "lr_full_llmx_nopca": [F, M, EF, T, L],
            "lr_full_llmx_noemb": [F, M, T, L],
            "lr_compact": [F, M, T, L[LLM_COMPACT]],
            "lr_best_fresh": [F, M, T, L, FR],
            "lr_best_luis_sib": [F, M, T, L, LS],
            "lr_best_luis_all": [F, M, T, L, LU],
            "lr_luis_only": [LU],
            "lgbm_luis_only": [LU],
            "lgbm_best": [F, M, T, L],
            "lr_no_llm_noemb": [F, M, T],
            "lr_features_llmx": [F, L],
            "lgbm_features_meta_embknn_team_llmx": [F, M, EF, T, L],
            "lr_features_meta_llmx": [F, M, L],
            "lr_features_meta_embknn_llmx": [F, M, EF, L],
            "lr_features_meta_embknn_pca64_llmx": [F, M, EF, EP, L],
        })
    only = os.environ.get("CV_ONLY")  # comma list: run just these variants, merge into existing file
    if only:
        keep = set(only.split(","))
        missing = keep - set(variants)
        assert not missing, f"unknown variants: {missing}"
        variants = {k: v for k, v in variants.items() if k in keep}
    y = h.label.to_numpy()
    groups = h.group.fillna(h.cve_id).to_numpy()
    jobs = []
    for name, parts in variants.items():
        X = pd.concat(parts, axis=1).to_numpy()
        for f in range(K_FOLDS):
            test = (h.fold == f).to_numpy()
            purge = h.group.isin(set(h.group[test].dropna())).to_numpy()
            if FORWARD:  # only EARLIER blocks train; block 0 has no past and is skipped
                if f == 0:
                    continue
                train = (h.fold < f).to_numpy() & ~purge
            else:
                train = (~test) & ~purge
            jobs.append(delayed(_one)(name, X, y, groups, test, train))
    res = Parallel(n_jobs=min(len(jobs), 40), verbose=0)(jobs)
    prev = pd.read_parquet(RES / OUT_FILE) if only and (RES / OUT_FILE).exists() else None
    out = h[["cve_id", "fold", "label"]].copy()
    if prev is not None:
        out = prev.copy()
    out["baseline_code_epss"] = h[FLAGS].max(axis=1) * 10 + h.epss.fillna(0)
    out["granite_zero"] = h.granite_zero
    out["emb_kev_contrast"] = EF.kev_contrast
    for name in variants:
        out[name] = np.nan
    for name, test, score, C in res:
        out.loc[test, name] = score
        print(f"{name} fold {int(h.fold[test].iloc[0])}: C={C}", flush=True)
    out.to_parquet(RES / OUT_FILE, index=False)


def boot_ci(score: np.ndarray, y: np.ndarray, k: int, base: np.ndarray, reps=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    hits, diff, aps = [], [], []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        s, yy, b = score[i], y[i], base[i]
        top = np.argpartition(-s, k)[:k]
        topb = np.argpartition(-b, k)[:k]
        hits.append(yy[top].sum()); diff.append(yy[top].sum() - yy[topb].sum())
        aps.append(average_precision_score(yy, s))
    q = lambda a: np.percentile(a, [2.5, 97.5]).round(3 if a is aps else 0)
    return q(hits), q(diff), q(aps)


def report(files: list[str]) -> None:
    nov4 = "--nov4" in files
    cap = int(files[files.index("--cap") + 1]) if "--cap" in files else 0
    files = [f for i, f in enumerate(files) if f not in ("--nov4", "--cap") and not (i and files[i - 1] == "--cap")]
    df = pd.read_parquet(RES / "cv_oof_lr.parquet")
    for f in files:
        extra = pd.read_parquet(f)
        df = df.merge(extra, on="cve_id", how="left", suffixes=("", "_dup"))
    meta = load()[["cve_id", "published", "kev_date_added", "assigner", "snapshot_month"]]
    df = df.merge(meta, on="cve_id", how="left")
    if nov4:  # judges' window: KEV by Nov 4 of the publish year
        cut = pd.to_datetime(df.published.dt.year.astype(str) + "-11-04")
        df["label"] = (df.kev_date_added.notna() & (df.kev_date_added <= cut)).astype(int)
    y = df.label.to_numpy()
    k = round(500 * len(df) / SCOREBOARD_ROWS)
    base = df.baseline_code_epss.to_numpy()
    cols = [c for c in df.columns if c not in ("cve_id", "fold", "label", "published", "kev_date_added",
                                               "assigner", "snapshot_month") and not c.endswith("_dup")]
    if cap:  # keep at most `cap` per assigner per snapshot month: push the rest below everything
        for c in cols:
            r = df[c].rank(ascending=False, method="first")
            n = df.assign(_r=r).sort_values("_r").groupby(["assigner", "snapshot_month"]).cumcount()
            df[c] = np.where(n.reindex(df.index) < cap, df[c], df[c] - 1e6)
    print(f"labels: {'KEV by Nov 4 of publish year' if nov4 else 'KEV by 2025-12-31'}; cap: {cap or 'none'}")
    print(f"cross-validated 2024-01..2025-09: {len(df):,} CVEs, {y.sum()} positives; "
          f"scaled K = {k} (scoreboard top 500)\n")
    rows = []
    for c in cols:
        s = df[c].to_numpy()
        if np.isnan(s).any():
            print(f"(skipping {c}: {np.isnan(s).sum()} missing scores)"); continue
        order = np.argsort(-s, kind="stable")
        h_k, h_1k = int(y[order[:k]].sum()), int(y[order[:1000]].sum())
        ci_h, ci_d, ci_ap = boot_ci(s, y, k, base)
        per_fold = [int(df.label[(df.fold == f)].to_numpy()[np.argsort(-s[(df.fold == f).to_numpy()])][:k // K_FOLDS].sum())
                    for f in range(K_FOLDS)]
        rows.append({"model": c, f"hits@{k}": h_k, "95% CI": f"{ci_h[0]:.0f}-{ci_h[1]:.0f}",
                     "vs baseline (95% CI)": f"{ci_d[0]:+.0f} to {ci_d[1]:+.0f}",
                     "hits@1000": h_1k, "AP": round(average_precision_score(y, s), 4),
                     "AP 95% CI": f"{ci_ap[0]:.3f}-{ci_ap[1]:.3f}",
                     f"per-fold hits@{k // K_FOLDS}": per_fold})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"folds": make_folds, "lr": run_lr}.get(cmd, lambda: report(sys.argv[2:]))()
