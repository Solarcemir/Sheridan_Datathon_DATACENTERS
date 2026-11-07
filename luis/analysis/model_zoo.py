"""Model zoo: heavier and different models on Luis's A/B replicas, to find what adds signal.

Setup (same as chen_insights.py): A = train 2024 -> test 2025 Jan-Sep (48 pos), B = train 2025 -> test 2024 Jan-Sep
(27 pos), labels y_nov4, hits at scoreboard density (A top 243, B top 198), pooled view (top 441 of 64,120) with a
1,000-rep paired bootstrap vs exploit+EPSS AND vs the current best (LightGBM + LR(+Chen) rank average).
Features: Luis's safe features (minus txt_in_wild) + Chen's safe and careful features + a new "assigner heat" group
(Chen's to-do #5): for each CVE, what happened to the SAME assigner's CVEs in EARLIER snapshot months of the same
year (count, share with exploit code, mean EPSS percentile), each measured at its own snapshot.

Models: LightGBM (reference), XGBoost small and big (GPU), CatBoost with assigner and main CWE as categoricals
(ordered target statistics), LightGBM LambdaRank with one query per snapshot month, the tuned LR (reference),
and a big linear model (TF-IDF of the description + all tabular features, C tuned inside training).
Then fixed rank-average ensembles, prediction correlations, and top-K overlap.

Outputs: reports/model_zoo_results.csv, reports/model_zoo_bootstrap.csv, reports/model_zoo_correlation.csv,
         reports/model_zoo_catboost_importance.csv, reports/model_zoo_log.txt (if run through Tee)

Usage:  python -E -P luis/analysis/model_zoo.py
        python -E -P luis/analysis/model_zoo.py --score-2026   # XGBoost big + LR(+Chen), cap 3 -> submissions/top500_v4.csv
"""
import os
import sys
import time

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402
from scipy.sparse import csr_matrix, hstack  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

T0 = time.time()
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA = os.path.join(ROOT, "..", "IBM-20261107T174321Z-1-001", "IBM", "data", "data")
DER, REP = os.path.join(ROOT, "derived"), os.path.join(ROOT, "reports")
RNG = np.random.default_rng(0)


def log(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- data + features
v = pd.read_csv(f"{DATA}/vulnerabilities.csv", dtype=str,
                usecols=["cve_id", "published", "assigner", "description", "cvss4_vector", "cwe_ids"]).set_index("cve_id")
v["published"] = pd.to_datetime(v.published).dt.normalize()
v["assigner"] = v.assigner.fillna("unknown")
v["description"] = v.description.fillna("")
kev = pd.read_csv(f"{DATA}/known_exploited.csv", dtype=str)
kev["date_added"] = pd.to_datetime(kev.date_added)
fd = pd.read_csv(os.path.join(DER, "feature_dictionary.csv"))
SAFE = [f for f in fd[fd.risk == "safe"].feature if f != "txt_in_wild"] + ["days_to_cutoff"]
WP_CNAS = {"security@wordfence.com", "audit@patchstack.com", "contact@wpscan.com"}
CHEN = ["id_year_lag", "old_id", "wp_cna", "txt_eol", "txt_vuldb_disclosed", "txt_auth_bypass", "assigner_ransom_kev",
        "is_patch_tuesday", "twins", "cwe_unknown", "cv4_urgent", "cv4_auto", "cv4_e_poc"]
HEAT = ["heat_n_prev", "heat_flag_rate_prev", "heat_epss_mean_prev"]


def add_chen(df, year):
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
    rk = kev[(kev.date_added < start) & (kev.known_ransomware_use == "Known")].cve_id
    df["assigner_ransom_kev"] = df.assigner.map(rk.map(v.assigner).value_counts()).fillna(0).to_numpy()
    df["is_patch_tuesday"] = ((pub.weekday == 1) & (pub.day >= 8) & (pub.day <= 14)).astype(int)
    key = pd.DataFrame({"a": df.assigner.to_numpy(), "p": pub, "d": x.description.to_numpy()})
    df["twins"] = key.groupby(["a", "p", "d"]).d.transform("size").to_numpy() - 1
    df["cwe_unknown"] = x.cwe_ids.isna().to_numpy().astype(int)       # NVD placeholders show up as blank
    vec = x.cvss4_vector.fillna("")
    df["cv4_urgent"] = vec.str.contains(r"/U:(?:Red|Amber)").to_numpy().astype(int)
    df["cv4_auto"] = vec.str.contains(r"/AU:Y").to_numpy().astype(int)
    df["cv4_e_poc"] = vec.str.contains(r"/E:P").to_numpy().astype(int)
    df["cwe_main"] = x.cwe_ids.fillna("none").str.split(";").str[0].to_numpy()
    df["description"] = x.description.to_numpy()
    return df


def add_heat(df):
    """Same-assigner CVEs from EARLIER snapshot months of the same frame, each with its own snapshot signals."""
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
    df = pd.read_csv(os.path.join(DER, f"features_{year}.csv"), parse_dates=["snapshot_end", "published"])
    df["days_to_cutoff"] = (pd.Timestamp(cutoff) - df.snapshot_end).dt.days
    return add_heat(add_chen(df, year))


def jan_sep(df, year):
    return df[df.snapshot_month <= f"{year}-09"].reset_index(drop=True)


tr24, tr25 = load(2024, "2024-12-31"), load(2025, "2025-12-31")
te25, te24 = jan_sep(load(2025, "2025-11-04"), 2025), jan_sep(load(2024, "2024-11-04"), 2024)
SPLITS = {"A": (tr24, te25), "B": (tr25, te24)}
FULL = SAFE + CHEN + HEAT
log("data ready: " + str({k: (len(t), int(t.y.sum()), len(s), int(s.y_nov4.sum())) for k, (t, s) in SPLITS.items()}))

# ----------------------------------------------------------------------------- models
LGB = dict(objective="binary", n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, scale_pos_weight=25, verbose=-1, n_jobs=8,
           deterministic=True, force_row_wise=True)


def m_lgbm(tr, te, F, seeds=10):
    return np.mean([lgb.LGBMClassifier(random_state=s, **LGB).fit(tr[F], tr.y).predict_proba(te[F])[:, 1] for s in range(seeds)], axis=0)


def m_xgb(tr, te, F, big=False, seeds=5):
    p = dict(n_estimators=1500, max_depth=6, learning_rate=0.01, min_child_weight=10, subsample=0.7, colsample_bytree=0.5,
             reg_lambda=10, gamma=1.0) if big else \
        dict(n_estimators=400, max_depth=4, learning_rate=0.03, min_child_weight=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=5)
    out = []
    for s in range(seeds):
        m = xgb.XGBClassifier(tree_method="hist", device="cuda", scale_pos_weight=25, eval_metric="aucpr", random_state=s, **p)
        m.fit(tr[F], tr.y)
        out.append(m.predict_proba(te[F])[:, 1])
    return np.mean(out, axis=0)


CAT_COLS = ["assigner", "cwe_main"]
CAT_IMP = {}


def m_cat(tr, te, F, split, seeds=3):
    cols = F + CAT_COLS
    out = []
    for s in range(seeds):
        kw = dict(iterations=1000, depth=6, learning_rate=0.03, l2_leaf_reg=5, auto_class_weights="Balanced", random_seed=s,
                  verbose=0, cat_features=CAT_COLS, allow_writing_files=False)   # no catboost_info/ folder
        try:
            m = CatBoostClassifier(task_type="GPU", devices="0", **kw).fit(tr[cols], tr.y)
        except Exception as e:  # noqa: BLE001
            log(f"  catboost GPU failed ({str(e)[:60]}), using CPU")
            m = CatBoostClassifier(thread_count=8, **kw).fit(tr[cols], tr.y)
        out.append(m.predict_proba(te[cols])[:, 1])
        if s == 0:
            CAT_IMP[split] = pd.Series(m.get_feature_importance(), index=cols)
    return np.mean(out, axis=0)


def m_rank(tr, te, F, seeds=5):
    t = tr.sort_values("snapshot_month", kind="stable")
    groups = t.groupby("snapshot_month", sort=True).size().to_numpy()
    out = []
    for s in range(seeds):
        m = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, lambdarank_truncation_level=300,
                           random_state=s, verbose=-1, n_jobs=8, deterministic=True, force_row_wise=True)
        m.fit(t[F], t.y, group=groups)
        out.append(m.predict(te[F]))
    return np.mean(out, axis=0)


def lr_pipe(C):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def inner_split(tr):
    return tr[tr.snapshot_month.str[-2:] <= "06"], tr[tr.snapshot_month.str[-2:] >= "07"]


def m_lr(tr, te, F):
    a, b = inner_split(tr)
    grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    ap = [average_precision_score(b.y, lr_pipe(C).fit(a[F], a.y).predict_proba(b[F])[:, 1]) for C in grid]
    return lr_pipe(grid[int(np.argmax(ap))]).fit(tr[F], tr.y).predict_proba(te[F])[:, 1]


def m_biglinear(tr, te, F):
    """TF-IDF of the description (fit on the training CVEs) + all tabular features in one logistic model."""
    def mats(fit_df, apply_dfs):
        tf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100000, sublinear_tf=True, stop_words="english")
        tf.fit(fit_df.description)
        imp, sc = SimpleImputer(strategy="median").fit(fit_df[F]), None
        sc = StandardScaler().fit(imp.transform(fit_df[F]))
        return [hstack([tf.transform(d.description), csr_matrix(sc.transform(imp.transform(d[F])))]).tocsr() for d in apply_dfs]
    a, b = inner_split(tr)
    Xa, Xb = mats(a, [a, b])
    grid = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    ap = [average_precision_score(b.y, LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(Xa, a.y).predict_proba(Xb)[:, 1])
          for C in grid]
    C = grid[int(np.argmax(ap))]
    Xtr, Xte = mats(tr, [tr, te])
    return LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(Xtr, tr.y).predict_proba(Xte)[:, 1]


def rank01(x):
    return pd.Series(np.asarray(x, dtype=float)).rank(pct=True).to_numpy()


def capped(df, s, cap):
    if not cap:
        return np.asarray(s, dtype=float)
    s = pd.Series(np.asarray(s, dtype=float), index=df.index)
    n = df.assign(_s=s).sort_values("_s", ascending=False, kind="stable").groupby(["assigner", "snapshot_month"]).cumcount()
    s = s.copy(); s[n[n >= cap].index] -= 100
    return s.to_numpy()


def hits(y, s, k):
    return int(np.asarray(y)[np.argsort(-np.asarray(s), kind="stable")[:k]].sum())


if "--score-2026" in sys.argv:
    # Best pooled option in the zoo: XGBoost big (all features) + LR (Luis safe + Chen), rank average, cap 3.
    SUB = os.path.join(ROOT, "submissions")
    train = pd.concat([tr24, tr25], ignore_index=True)
    s26 = load(2026, "2026-11-04")
    s26["xgb_pct"] = rank01(m_xgb(train, s26, FULL, big=True))
    s26["lr_pct"] = rank01(m_lr(train, s26, SAFE + CHEN))
    s26["score"] = capped(s26, s26.xgb_pct + s26.lr_pct, 3)
    s26 = s26.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    s26["rank"] = np.arange(1, len(s26) + 1)
    s26[["rank", "cve_id", "score", "xgb_pct", "lr_pct", "snapshot_month", "assigner", "epss", "any_flag", "wp_cna",
         "cwe_unknown"]].to_csv(os.path.join(DER, "scored_2026_v4.csv"), index=False)
    s26.head(500)[["cve_id"]].to_csv(os.path.join(SUB, "top500_v4.csv"), index=False)
    t = s26.head(500)
    v3 = set(pd.read_csv(os.path.join(SUB, "top500_v3.csv")).cve_id)
    log(f"top500_v4: overlap with v3 = {len(v3 & set(t.cve_id))}, exploit flag = {int(t.any_flag.sum())}, "
        f"Oracle = {int((t.assigner == 'secalert_us@oracle.com').sum())}, WordPress CNAs = {int(t.wp_cna.sum())}, "
        f"blank CWE = {int(t.cwe_unknown.sum())}, assigners = {t.assigner.nunique()}")
    print(t.snapshot_month.value_counts().sort_index().to_dict())
    raise SystemExit(0)


sc = {}
for name, (tr, te) in SPLITS.items():
    sc[("Exploit code, then EPSS", name)] = (te.any_flag * 10 + te.epss).to_numpy()
    sc[("LightGBM: Luis safe", name)] = m_lgbm(tr, te, SAFE)
    sc[("LightGBM: all features", name)] = m_lgbm(tr, te, FULL)
    log(f"{name}: LightGBM done")
    sc[("LR: Luis safe + Chen", name)] = m_lr(tr, te, SAFE + CHEN)
    sc[("LR: all features (+heat)", name)] = m_lr(tr, te, FULL)
    log(f"{name}: LR done")
    sc[("XGBoost small (GPU)", name)] = m_xgb(tr, te, FULL)
    sc[("XGBoost big (GPU)", name)] = m_xgb(tr, te, FULL, big=True)
    log(f"{name}: XGBoost done")
    sc[("CatBoost (assigner, CWE categorical)", name)] = m_cat(tr, te, FULL, name)
    log(f"{name}: CatBoost done")
    sc[("LightGBM LambdaRank (query = month)", name)] = m_rank(tr, te, FULL)
    log(f"{name}: LambdaRank done")
    sc[("Big linear: TF-IDF + tabular", name)] = m_biglinear(tr, te, FULL)
    log(f"{name}: big linear done")
    R = {m: rank01(sc[(m, name)]) for m in [k for k, s in sc if s == name]}
    ens = {
        "ENS: LightGBM + LR(+Chen)  [current best]": ["LightGBM: Luis safe", "LR: Luis safe + Chen"],
        "ENS: LightGBM(all) + LR(all, +heat)": ["LightGBM: all features", "LR: all features (+heat)"],
        "ENS: XGBoost small + LR(+Chen)": ["XGBoost small (GPU)", "LR: Luis safe + Chen"],
        "ENS: XGBoost big + LR(+Chen)": ["XGBoost big (GPU)", "LR: Luis safe + Chen"],
        "ENS: CatBoost + LR(+Chen)": ["CatBoost (assigner, CWE categorical)", "LR: Luis safe + Chen"],
        "ENS: LambdaRank + LR(+Chen)": ["LightGBM LambdaRank (query = month)", "LR: Luis safe + Chen"],
        "ENS: LightGBM + LR(+Chen) + CatBoost": ["LightGBM: Luis safe", "LR: Luis safe + Chen", "CatBoost (assigner, CWE categorical)"],
        "ENS: LightGBM + LR(+Chen) + big linear": ["LightGBM: Luis safe", "LR: Luis safe + Chen", "Big linear: TF-IDF + tabular"],
        "ENS: everything (7 models)": ["LightGBM: Luis safe", "LR: Luis safe + Chen", "XGBoost small (GPU)", "XGBoost big (GPU)",
                                       "CatBoost (assigner, CWE categorical)", "LightGBM LambdaRank (query = month)",
                                       "Big linear: TF-IDF + tabular"],
    }
    for e, parts in ens.items():
        sc[(e, name)] = np.sum([R[p] for p in parts], axis=0)

MODELS = list(dict.fromkeys(m for m, _ in sc))
rows = []
for m in MODELS:
    for cap in (None, 3):
        r = {"model": m, "cap": cap or "none"}
        for name, (tr, te) in SPLITS.items():
            k = round(500 * len(te) / 72670)
            r[f"{name}@{k}"] = hits(te.y_nov4, capped(te, sc[(m, name)], cap), k)
        r["A+B @density"] = r["A@243"] + r["B@198"]
        rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(os.path.join(REP, "model_zoo_results.csv"), index=False)
pd.set_option("display.width", 250)
print(res.to_string(index=False))

# ----------------------------------------------------------------------------- pooled + paired bootstrap
both = pd.concat([te25.assign(split="A"), te24.assign(split="B")], ignore_index=True)
K, yv = round(500 * len(both) / 72670), both.y_nov4.to_numpy()
pooled = {}
for m in MODELS:
    raw = np.r_[sc[(m, "A")], sc[(m, "B")]]
    pct = both.assign(_s=raw).groupby("snapshot_month")._s.rank(pct=True).to_numpy()
    for cap in (None, 3):
        pooled[(m, cap)] = capped(both, pct, cap)
reps = [RNG.integers(0, len(both), len(both)) for _ in range(1000)]


def tk(s, idx):
    ss = s[idx]
    return yv[idx][np.argpartition(-ss, K)[:K]].sum()


ref_base = np.array([tk(pooled[("Exploit code, then EPSS", None)], i) for i in reps])
ref_best = np.array([tk(pooled[("ENS: LightGBM + LR(+Chen)  [current best]", None)], i) for i in reps])
boot = []
for (m, cap), s in pooled.items():
    hs = np.array([tk(s, i) for i in reps])
    boot.append({"model": m, "cap": cap or "none", f"hits@{K}": hits(yv, s, K),
                 "95% CI": f"{np.percentile(hs, 2.5):.0f}-{np.percentile(hs, 97.5):.0f}",
                 "vs exploit+EPSS": f"{np.percentile(hs - ref_base, 2.5):+.0f} to {np.percentile(hs - ref_base, 97.5):+.0f}",
                 "vs current best": f"{np.percentile(hs - ref_best, 2.5):+.0f} to {np.percentile(hs - ref_best, 97.5):+.0f}",
                 "AP": round(average_precision_score(yv, s), 4)})
boot = pd.DataFrame(boot)
boot.to_csv(os.path.join(REP, "model_zoo_bootstrap.csv"), index=False)
print(f"\npooled: top {K} of {len(both):,}, {yv.sum()} positives")
print(boot.to_string(index=False))

# ----------------------------------------------------------------------------- diversity
single = [m for m in MODELS if not m.startswith("ENS") and not m.startswith("Exploit")]
P = pd.DataFrame({m: pooled[(m, None)] for m in single})
corr = P.corr(method="spearman").round(2)
corr.to_csv(os.path.join(REP, "model_zoo_correlation.csv"))
tops = {m: set(np.argsort(-P[m].to_numpy(), kind="stable")[:K]) for m in single}
ov = pd.DataFrame({a: {b: len(tops[a] & tops[b]) for b in single} for a in single})
print("\nSpearman correlation of pooled scores:")
print(corr.to_string())
print(f"\ntop-{K} overlap (CVEs in common):")
print(ov.to_string())
hit_sets = {m: {i for i in tops[m] if yv[i]} for m in single}
union = set().union(*hit_sets.values())
print(f"\nhits found by at least one single model: {len(union)} of {yv.sum()} | per model: "
      + ", ".join(f"{m.split(':')[0].split(' (')[0]}={len(h)}" for m, h in hit_sets.items()))
imp = pd.concat(CAT_IMP, axis=1).mean(axis=1).sort_values(ascending=False)
imp.to_csv(os.path.join(REP, "model_zoo_catboost_importance.csv"), header=["importance"])
print("\nCatBoost top features:", ", ".join(f"{k} {v:.1f}" for k, v in imp.head(12).items()))
log("done")
