"""Shared pieces for Luis's experiment scripts: data frames for the A/B replicas and 2026, Chen-style metadata
features, models, the per-assigner-month cap, and the density / pooled-bootstrap evaluation.

Conventions (see context_Luis.md section 4):
  A = train 2024 replica -> test 2025 Jan-Sep (48 positives by Nov 4)
  B = train 2025 replica -> test 2024 Jan-Sep (27 positives by Nov 4)
  hits at scoreboard density: K = round(500 * test rows / 72,670)  (A 243, B 198, pooled 441)
"""
import os

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
REPO = os.path.abspath(os.path.join(ROOT, ".."))
DATA = os.path.join(REPO, "IBM-20261107T174321Z-1-001", "IBM", "data", "data")
DER, REP, SUB = os.path.join(ROOT, "derived"), os.path.join(ROOT, "reports"), os.path.join(ROOT, "submissions")
CHEN = os.path.join(REPO, "chen")

_v = _kev = None


def vuln():
    global _v
    if _v is None:
        v = pd.read_csv(f"{DATA}/vulnerabilities.csv", dtype=str,
                        usecols=["cve_id", "published", "assigner", "description", "cvss4_vector", "cwe_ids"]).set_index("cve_id")
        v["published"] = pd.to_datetime(v.published).dt.normalize()
        v["assigner"] = v.assigner.fillna("unknown")
        v["description"] = v.description.fillna("")
        _v = v
    return _v


def kev():
    global _kev
    if _kev is None:
        k = pd.read_csv(f"{DATA}/known_exploited.csv", dtype=str)
        k["date_added"] = pd.to_datetime(k.date_added)
        _kev = k
    return _kev


fd = pd.read_csv(os.path.join(DER, "feature_dictionary.csv"))
SAFE = [f for f in fd[fd.risk == "safe"].feature if f != "txt_in_wild"] + ["days_to_cutoff"]
WP_CNAS = {"security@wordfence.com", "audit@patchstack.com", "contact@wpscan.com"}
CHEN_LITE = ["id_year_lag", "old_id", "wp_cna", "txt_eol", "txt_vuldb_disclosed", "txt_auth_bypass", "assigner_ransom_kev",
             "is_patch_tuesday", "twins", "cwe_unknown", "cv4_urgent", "cv4_auto", "cv4_e_poc"]
HEAT = ["heat_n_prev", "heat_flag_rate_prev", "heat_epss_mean_prev"]
FULL = SAFE + CHEN_LITE + HEAT


def add_chen_lite(df, year):
    v, k = vuln(), kev()
    start = pd.Timestamp(f"{year}-01-01")
    x = v.loc[df.cve_id]
    pub = pd.to_datetime(x.published.to_numpy())
    df["id_year_lag"] = pub.year - df.cve_id.str.split("-").str[1].astype(int).to_numpy()
    df["old_id"] = (df.id_year_lag >= 1).astype(int)
    df["wp_cna"] = df.assigner.isin(WP_CNAS).astype(int)
    d = x.description.str
    df["txt_eol"] = d.contains(r"end[- ]of[- ]life|\bEOL\b|no longer (?:supported|maintained)|end of (?:support|service)|"
                               r"unsupported (?:product|version|device)", case=False, regex=True).to_numpy().astype(int)
    df["txt_vuldb_disclosed"] = d.contains(r"exploit has been (?:disclosed|made available) to the public|"
                                           r"has been disclosed to the public and may be used", case=False, regex=True).to_numpy().astype(int)
    df["txt_auth_bypass"] = d.contains(r"authentication bypass|bypass(?:es|ing)? (?:the )?authentication|auth bypass",
                                       case=False, regex=True).to_numpy().astype(int)
    rk = k[(k.date_added < start) & (k.known_ransomware_use == "Known")].cve_id
    df["assigner_ransom_kev"] = df.assigner.map(rk.map(v.assigner).value_counts()).fillna(0).to_numpy()
    df["is_patch_tuesday"] = ((pub.weekday == 1) & (pub.day >= 8) & (pub.day <= 14)).astype(int)
    key = pd.DataFrame({"a": df.assigner.to_numpy(), "p": pub, "d": x.description.to_numpy()})
    df["twins"] = key.groupby(["a", "p", "d"]).d.transform("size").to_numpy() - 1
    df["cwe_unknown"] = x.cwe_ids.isna().to_numpy().astype(int)
    vec = x.cvss4_vector.fillna("")
    df["cv4_urgent"] = vec.str.contains(r"/U:(?:Red|Amber)").to_numpy().astype(int)
    df["cv4_auto"] = vec.str.contains(r"/AU:Y").to_numpy().astype(int)
    df["cv4_e_poc"] = vec.str.contains(r"/E:P").to_numpy().astype(int)
    return df


def add_heat(df):
    g = df.groupby(["assigner", "snapshot_month"]).agg(n=("cve_id", "size"), f=("any_flag", "sum"), e=("epss_pct", "sum")).reset_index()
    g = g.sort_values(["assigner", "snapshot_month"])
    for c in ("n", "f", "e"):
        g[c + "_prev"] = g.groupby("assigner")[c].cumsum() - g[c]
    m = df[["assigner", "snapshot_month"]].merge(g, on=["assigner", "snapshot_month"], how="left")
    df["heat_n_prev"] = m.n_prev.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        df["heat_flag_rate_prev"] = np.where(m.n_prev > 0, m.f_prev / m.n_prev, np.nan)
        df["heat_epss_mean_prev"] = np.where(m.n_prev > 0, m.e_prev / m.n_prev, np.nan)
    return df


def load(year, cutoff):
    """Luis's leak-free features for a replica year (Jan-Nov) or 2026, plus Chen-lite and heat features."""
    df = pd.read_csv(os.path.join(DER, f"features_{year}.csv"), parse_dates=["snapshot_end", "published"])
    df["days_to_cutoff"] = (pd.Timestamp(cutoff) - df.snapshot_end).dt.days
    return add_heat(add_chen_lite(df, year))


def jan_sep(df, year):
    return df[df.snapshot_month <= f"{year}-09"].reset_index(drop=True)


# ----------------------------------------------------------------------------- models
LGB = dict(objective="binary", n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, scale_pos_weight=25, verbose=-1, n_jobs=8,
           deterministic=True, force_row_wise=True)
XGB_BIG = dict(n_estimators=1500, max_depth=6, learning_rate=0.01, min_child_weight=10, subsample=0.7, colsample_bytree=0.5,
               reg_lambda=10, gamma=1.0)


def m_lgbm(tr, te, F, seeds=10):
    return np.mean([lgb.LGBMClassifier(random_state=s, **LGB).fit(tr[F], tr.y).predict_proba(te[F])[:, 1] for s in range(seeds)], axis=0)


def m_xgb(tr, te, F, seeds=5):
    import xgboost as xgb
    out = []
    for s in range(seeds):
        m = xgb.XGBClassifier(tree_method="hist", device="cuda", scale_pos_weight=25, eval_metric="aucpr", random_state=s, **XGB_BIG)
        m.fit(tr[F], tr.y)
        out.append(m.predict_proba(te[F])[:, 1])
    return np.mean(out, axis=0)


def lr_pipe(C):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def m_lr(tr, te, F, return_C=False):
    """C picked inside the training data only: fit months 1-6, score months 7-11 by average precision."""
    a, b = tr[tr.snapshot_month.str[-2:] <= "06"], tr[tr.snapshot_month.str[-2:] >= "07"]
    grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    ap = [average_precision_score(b.y, lr_pipe(C).fit(a[F], a.y).predict_proba(b[F])[:, 1]) for C in grid]
    C = grid[int(np.argmax(ap))]
    p = lr_pipe(C).fit(tr[F], tr.y).predict_proba(te[F])[:, 1]
    return (p, C) if return_C else p


def rank01(x):
    return pd.Series(np.asarray(x, dtype=float)).rank(pct=True).to_numpy()


def capped(df, s, cap):
    if not cap:
        return np.asarray(s, dtype=float)
    s = pd.Series(np.asarray(s, dtype=float), index=df.index)
    n = df.assign(_s=s).sort_values("_s", ascending=False, kind="stable").groupby(["assigner", "snapshot_month"]).cumcount()
    s = s.copy()
    s[n[n >= cap].index] -= 100
    return s.to_numpy()


def hits(y, s, k):
    return int(np.asarray(y)[np.argsort(-np.asarray(s, dtype=float), kind="stable")[:k]].sum())


def density_k(n):
    return round(500 * n / 72670)


def pooled_eval(both, scores, base_key, ref_key, caps=(None, 3), reps=1000, seed=0):
    """scores: {name: raw score over `both` rows}. Ranks within snapshot month, applies caps, paired bootstrap."""
    rng = np.random.default_rng(seed)
    K, yv = density_k(len(both)), both.y_nov4.to_numpy()
    pooled = {}
    for m, raw in scores.items():
        pct = both.assign(_s=np.asarray(raw, dtype=float)).groupby("snapshot_month")._s.rank(pct=True).to_numpy()
        for cap in caps:
            pooled[(m, cap)] = capped(both, pct, cap)
    idxs = [rng.integers(0, len(both), len(both)) for _ in range(reps)]

    def tk(s, i):
        ss = s[i]
        return yv[i][np.argpartition(-ss, K)[:K]].sum()

    rb = np.array([tk(pooled[(base_key, None)], i) for i in idxs])
    rr = np.array([tk(pooled[(ref_key, caps[-1])], i) for i in idxs])
    rows = []
    for (m, cap), s in pooled.items():
        hs = np.array([tk(s, i) for i in idxs])
        rows.append({"model": m, "cap": cap or "none", f"hits@{K}": hits(yv, s, K),
                     "95% CI": f"{np.percentile(hs, 2.5):.0f}-{np.percentile(hs, 97.5):.0f}",
                     "vs exploit+EPSS": f"{np.percentile(hs - rb, 2.5):+.0f} to {np.percentile(hs - rb, 97.5):+.0f}",
                     "vs reference": f"{np.percentile(hs - rr, 2.5):+.0f} to {np.percentile(hs - rr, 97.5):+.0f}",
                     "AP": round(average_precision_score(yv, s), 4)})
    return pd.DataFrame(rows), pooled
