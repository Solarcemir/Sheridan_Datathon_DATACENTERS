"""Test Chen's GPU-box insights inside Luis's setup (A/B replicas, scoreboard-density K, paired bootstrap).

New features from chen/context (all computed from data known at the snapshot; history uses KEV before Jan 1 of Y):
  safe    id_year_lag, old_id (ID reserved in an earlier year), wp_cna (Wordfence/Patchstack/WPScan),
          txt_eol (end-of-life text), txt_vuldb_disclosed ("exploit has been disclosed to the public"),
          txt_auth_bypass, assigner_ransom_kev (past KEV with known ransomware use, by assigner),
          is_patch_tuesday (2nd Tuesday of the month), twins (identical description, same assigner and day)
  careful cwe_unknown (NVD-CWE-noinfo/Other), cv4_urgent (CVSS 4 U:Red/Amber), cv4_auto (AU:Y), cv4_e_poc (E:P)
Models: Luis's LightGBM (deterministic, 10 seeds), strongly regularized logistic regression with C picked
inside the training year only (months 1-6 -> 7-11, average precision), LR with 64 PCA dims of the Granite
embedding, and the LR + LightGBM rank average. txt_in_wild is dropped everywhere (agreed with Chen).
Evaluation: A = train 2024 -> test 2025 Jan-Sep (48 pos), B = train 2025 -> test 2024 Jan-Sep (27 pos), labels
y_nov4, hits at scoreboard density (K = 500 x rows / 72,670) with and without a cap per assigner-month,
plus a pooled view (both test years, top 441 of 64,120) with 1,000-rep paired bootstrap vs exploit-then-EPSS.

Outputs: reports/chen_insights_results.csv, reports/chen_insights_bootstrap.csv,
         reports/chen_features_lift.csv, reports/chen_missed_positives.csv
With --score-2026: trains the winning ensemble (LightGBM on Luis's safe features + LR on safe + Chen features,
rank average, cap 3 per assigner-month) on all 2024+2025 months and writes derived/scored_2026_v3.csv and
submissions/top500_v3.csv.

Usage:  python -E -P luis/analysis/chen_insights.py [--score-2026]
"""
import os
import sys

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA = os.path.join(ROOT, "..", "IBM-20261107T174321Z-1-001", "IBM", "data", "data")
DER, REP = os.path.join(ROOT, "derived"), os.path.join(ROOT, "reports")
RNG = np.random.default_rng(0)

# ----------------------------------------------------------------------------- data
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
CHEN_SAFE = ["id_year_lag", "old_id", "wp_cna", "txt_eol", "txt_vuldb_disclosed", "txt_auth_bypass",
             "assigner_ransom_kev", "is_patch_tuesday", "twins"]
CHEN_CAREFUL = ["cwe_unknown", "cv4_urgent", "cv4_auto", "cv4_e_poc"]


def add_chen(df, year):
    start = pd.Timestamp(f"{year}-01-01")
    x = v.loc[df.cve_id]
    pub = x.published.to_numpy()
    df["id_year_lag"] = pd.to_datetime(pub).year - df.cve_id.str.split("-").str[1].astype(int).to_numpy()
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
    ransom = rk.map(v.assigner).value_counts()
    df["assigner_ransom_kev"] = df.assigner.map(ransom).fillna(0).to_numpy()
    p = pd.to_datetime(pub)
    df["is_patch_tuesday"] = ((p.weekday == 1) & (p.day >= 8) & (p.day <= 14)).astype(int)
    key = pd.DataFrame({"a": df.assigner.to_numpy(), "p": pub, "d": x.description.to_numpy()})
    df["twins"] = key.groupby(["a", "p", "d"]).d.transform("size").to_numpy() - 1
    # In this dataset NIST's NVD-CWE-noinfo / NVD-CWE-Other placeholders appear as a blank cwe_ids (3,556 rows in 2024-01..2025-09)
    df["cwe_unknown"] = x.cwe_ids.isna().to_numpy().astype(int)
    vec = x.cvss4_vector.fillna("")
    df["cv4_urgent"] = vec.str.contains(r"/U:(?:Red|Amber)").to_numpy().astype(int)
    df["cv4_auto"] = vec.str.contains(r"/AU:Y").to_numpy().astype(int)
    df["cv4_e_poc"] = vec.str.contains(r"/E:P").to_numpy().astype(int)
    df["cv4_e_attacked"] = vec.str.contains(r"/E:A").to_numpy().astype(int)   # reported only, never used (leak-prone)
    return df


def load(year, cutoff):
    df = pd.read_csv(os.path.join(DER, f"features_{year}.csv"), parse_dates=["snapshot_end", "published"])
    df["days_to_cutoff"] = (pd.Timestamp(cutoff) - df.snapshot_end).dt.days
    return add_chen(df, year)


tr24, tr25 = load(2024, "2024-12-31"), load(2025, "2025-12-31")
te25 = load(2025, "2025-11-04"); te25 = te25[te25.snapshot_month <= "2025-09"].reset_index(drop=True)
te24 = load(2024, "2024-11-04"); te24 = te24[te24.snapshot_month <= "2024-09"].reset_index(drop=True)
SPLITS = {"A": (tr24, te25), "B": (tr25, te24)}
print({k: (len(t), int(t.y.sum()), len(s), int(s.y_nov4.sum())) for k, (t, s) in SPLITS.items()})

# Granite embedding PCA-64, fit on non-KEV CVEs published before 2024 (unsupervised, fold-safe)
E = np.load(os.path.join(DER, "granite", "emb_granite30m.npy")).astype(np.float32)
eids = pd.read_csv(os.path.join(DER, "granite", "emb_ids.csv")).cve_id
pos_of = pd.Series(np.arange(len(eids)), index=eids.values)
cohort_ids, kev_ids = set(tr24.cve_id) | set(tr25.cve_id), set(kev.cve_id)
pub_of = v.published
pool = [c for c in eids if c not in cohort_ids and c not in kev_ids and pub_of.get(c, pd.Timestamp("2099-01-01")) < pd.Timestamp("2024-01-01")]
pca = PCA(64, random_state=0).fit(E[pos_of.loc[pool].to_numpy()])
PCA_COLS = [f"emb_pc{i}" for i in range(64)]
for df in (tr24, tr25, te24, te25):
    Z = pca.transform(E[pos_of.loc[df.cve_id].to_numpy()])
    df[PCA_COLS] = Z
print(f"PCA-64 fit on {len(pool):,} pre-2024 non-KEV descriptions; explained variance {pca.explained_variance_ratio_.sum():.2f}")

# ----------------------------------------------------------------------------- models
LGB = dict(objective="binary", n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, scale_pos_weight=25, verbose=-1, n_jobs=8,
           deterministic=True, force_row_wise=True)


def gbm(tr, te, F, seeds=10):
    return np.mean([lgb.LGBMClassifier(random_state=s, **LGB).fit(tr[F], tr.y).predict_proba(te[F])[:, 1] for s in range(seeds)], axis=0)


def lr_pipe(C):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def lr(tr, te, F):
    """Pick C inside the training year only: fit months 1-6, score months 7-11 by average precision."""
    a, b = tr[tr.snapshot_month.str[-2:] <= "06"], tr[tr.snapshot_month.str[-2:] >= "07"]
    grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    ap = [average_precision_score(b.y, lr_pipe(C).fit(a[F], a.y).predict_proba(b[F])[:, 1]) if b.y.sum() else 0 for C in grid]
    C = grid[int(np.argmax(ap))]
    return lr_pipe(C).fit(tr[F], tr.y).predict_proba(te[F])[:, 1], C


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


S_CHEN = SAFE + CHEN_SAFE
S_CHEN_C = SAFE + CHEN_SAFE + CHEN_CAREFUL

if "--leak-test" in sys.argv:
    # Clean vs leak-only vs clean+leak, same splits, density K, pooled paired bootstrap, and top-K overlap.
    LEAK = ["LEAK_nvd_analyzed", "LEAK_n_references", "LEAK_n_affected_cpes"]
    for yr, frames in ((2024, (tr24, te24)), (2025, (tr25, te25))):
        lk = pd.read_csv(os.path.join(DER, f"leak_only_{yr}.csv")).set_index("cve_id")
        for df in frames:
            df[LEAK] = lk.loc[df.cve_id, LEAK].to_numpy()
    sc = {}
    for name, (tr, te) in SPLITS.items():
        sc[("Exploit code, then EPSS", name)] = (te.any_flag * 10 + te.epss).to_numpy()
        clean_g, clean_l = gbm(tr, te, SAFE), lr(tr, te, S_CHEN_C)[0]
        sc[("CLEAN: LightGBM + LR(+Chen) rank avg", name)] = rank01(clean_g) + rank01(clean_l)
        sc[("LEAK ONLY: LightGBM on 3 leak columns", name)] = gbm(tr, te, LEAK)
        sc[("LEAK ONLY: LR on 3 leak columns", name)] = lr(tr, te, LEAK)[0]
        sc[("LEAK ONLY: sort by n_references", name)] = te.LEAK_n_references.to_numpy() + te.LEAK_nvd_analyzed.to_numpy() * 1e-3
        sc[("LEAK + exploit signals: LightGBM", name)] = gbm(tr, te, LEAK + ["epss_pct", "epss_missing", "exploitdb", "metasploit",
                                                                              "nuclei", "poc_github", "n_flags", "any_flag"])
        sc[("CLEAN + LEAK: LightGBM + LR rank avg", name)] = rank01(gbm(tr, te, SAFE + LEAK)) + rank01(lr(tr, te, S_CHEN_C + LEAK)[0])
    models = list(dict.fromkeys(m for m, _ in sc))
    both = pd.concat([te25.assign(split="A"), te24.assign(split="B")], ignore_index=True)
    K, yv = round(500 * len(both) / 72670), both.y_nov4.to_numpy()
    pooled = {m: both.assign(_s=np.r_[sc[(m, "A")], sc[(m, "B")]]).groupby("snapshot_month")._s.rank(pct=True).to_numpy() for m in models}
    reps = [RNG.integers(0, len(both), len(both)) for _ in range(1000)]

    def tk(s, idx):
        ss = s[idx]
        return yv[idx][np.argpartition(-ss, K)[:K]].sum()

    rows = []
    for m in models:
        r = {"model": m}
        for name, (tr, te) in SPLITS.items():
            k = round(500 * len(te) / 72670)
            r[f"{name}@{k}"] = hits(te.y_nov4, sc[(m, name)], k)
        r["A+B @density"] = r["A@243"] + r["B@198"]
        s = pooled[m]
        hs = np.array([tk(s, i) for i in reps])
        d = np.array([tk(s, i) - tk(pooled["Exploit code, then EPSS"], i) for i in reps])
        r[f"pooled hits@{K}"] = hits(yv, s, K)
        r["pooled 95% CI"] = f"{np.percentile(hs, 2.5):.0f}-{np.percentile(hs, 97.5):.0f}"
        r["vs exploit+EPSS"] = f"{np.percentile(d, 2.5):+.0f} to {np.percentile(d, 97.5):+.0f}"
        r["AP"] = round(average_precision_score(yv, s), 4)
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(REP, "leak_vs_clean.csv"), index=False)
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))
    top = {m: set(np.argsort(-pooled[m], kind="stable")[:K]) for m in models}
    c, lo = top["CLEAN: LightGBM + LR(+Chen) rank avg"], top["LEAK ONLY: LightGBM on 3 leak columns"]
    hit = lambda S: {i for i in S if yv[i] == 1}  # noqa: E731
    print(f"\ntop {K} overlap clean vs leak-only: {len(c & lo)} CVEs in common | hits: clean {len(hit(c))}, leak-only {len(hit(lo))}, "
          f"both {len(hit(c) & hit(lo))}, only clean {len(hit(c) - hit(lo))}, only leak {len(hit(lo) - hit(c))}")
    an = both.LEAK_nvd_analyzed.to_numpy() == 1
    print(f"test rows NVD-analyzed: {an.mean():.1%} of all, {an[yv == 1].mean():.1%} of positives | "
          f"KEV rate analyzed {yv[an].mean():.3%} vs not {yv[~an].mean():.3%}")
    sys.exit(0)

if "--score-2026" in sys.argv:
    SUB = os.path.join(ROOT, "submissions")
    train = pd.concat([tr24, tr25], ignore_index=True)
    s26 = load(2026, "2026-11-04")
    g = gbm(train, s26, SAFE)
    l, C = lr(train, s26, S_CHEN_C)
    s26["gbm_pct"], s26["lr_pct"] = rank01(g), rank01(l)
    s26["score"] = capped(s26, s26.gbm_pct + s26.lr_pct, 3)
    s26 = s26.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    s26["rank"] = np.arange(1, len(s26) + 1)
    s26[["rank", "cve_id", "score", "gbm_pct", "lr_pct", "snapshot_month", "assigner", "epss", "any_flag", "wp_cna",
         "cwe_unknown", "txt_auth_bypass", "txt_eol"]].to_csv(os.path.join(DER, "scored_2026_v3.csv"), index=False)
    s26.head(500)[["cve_id"]].to_csv(os.path.join(SUB, "top500_v3.csv"), index=False)
    t = s26.head(500)
    v2 = set(pd.read_csv(os.path.join(SUB, "top500_v2.csv")).cve_id)
    print(f"top500_v3 (LR C={C}): overlap with v2 = {len(v2 & set(t.cve_id))}, exploit flag = {int(t.any_flag.sum())}, "
          f"Oracle = {int((t.assigner == 'secalert_us@oracle.com').sum())}, WordPress CNAs = {int(t.wp_cna.sum())}, "
          f"blank CWE = {int(t.cwe_unknown.sum())}, assigners = {t.assigner.nunique()}")
    print(t.snapshot_month.value_counts().sort_index().to_dict())
    print(t.assigner.value_counts().head(8).to_dict())
    sys.exit(0)

scores = {}           # (model, split) -> score on the test set
chosen_C = {}
for name, (tr, te) in SPLITS.items():
    scores[("Exploit code, then EPSS", name)] = (te.any_flag * 10 + te.epss).to_numpy()
    scores[("LightGBM: Luis safe", name)] = gbm(tr, te, SAFE)
    scores[("LightGBM: + Chen safe", name)] = gbm(tr, te, S_CHEN)
    scores[("LightGBM: + Chen safe + careful", name)] = gbm(tr, te, S_CHEN_C)
    for label, F in [("LR: Luis safe", SAFE), ("LR: + Chen safe + careful", S_CHEN_C), ("LR: + Chen + PCA-64", S_CHEN_C + PCA_COLS)]:
        scores[(label, name)], chosen_C[(label, name)] = lr(tr, te, F)
    scores[("Rank avg: LightGBM(+Chen) + LR(+Chen+PCA)", name)] = rank01(scores[("LightGBM: + Chen safe + careful", name)]) + \
        rank01(scores[("LR: + Chen + PCA-64", name)])
    scores[("Rank avg: LightGBM(Luis) + LR(+Chen+PCA)", name)] = rank01(scores[("LightGBM: Luis safe", name)]) + \
        rank01(scores[("LR: + Chen + PCA-64", name)])
    scores[("Rank avg: LightGBM(Luis) + LR(+Chen)", name)] = rank01(scores[("LightGBM: Luis safe", name)]) + \
        rank01(scores[("LR: + Chen safe + careful", name)])
    scores[("Rank avg of 3: LightGBM(Luis) + LR(+Chen) + LR(+Chen+PCA)", name)] = rank01(scores[("LightGBM: Luis safe", name)]) + \
        rank01(scores[("LR: + Chen safe + careful", name)]) + rank01(scores[("LR: + Chen + PCA-64", name)])
print("LR C chosen inside training years:", {f"{k[0]} [{k[1]}]": c for k, c in chosen_C.items()})

MODELS = list(dict.fromkeys(m for m, _ in scores))
rows = []
for m in MODELS:
    for cap in (None, 3, 5):
        r = {"model": m, "cap": cap or "none"}
        for name, (tr, te) in SPLITS.items():
            k = round(500 * len(te) / 72670)
            s = capped(te, scores[(m, name)], cap)
            r[f"{name}@{k}"] = hits(te.y_nov4, s, k)
            r[f"{name}@500"] = hits(te.y_nov4, s, 500)
        r["A+B @density"] = r["A@243"] + r["B@198"]
        r["A+B @500"] = r["A@500"] + r["B@500"]
        rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(os.path.join(REP, "chen_insights_results.csv"), index=False)
pd.set_option("display.width", 250)
print(res.to_string(index=False))

# ----------------------------------------------------------------------------- pooled view + paired bootstrap
both = pd.concat([te25.assign(split="A"), te24.assign(split="B")], ignore_index=True)
K = round(500 * len(both) / 72670)
y = both.y_nov4.to_numpy()
pooled = {}
for m in MODELS:
    raw = np.r_[scores[(m, "A")], scores[(m, "B")]]
    pct = both.assign(_s=raw).groupby("snapshot_month")._s.rank(pct=True).to_numpy()   # comparable across the two models
    for cap in (None, 3):
        pooled[(m, cap)] = capped(both, pct, cap)
base = pooled[("Exploit code, then EPSS", None)]


def topk_hits(y, s, idx, k):
    ss = s[idx]
    top = np.argpartition(-ss, k)[:k]
    return y[idx][top].sum()


boot = []
reps = [RNG.integers(0, len(both), len(both)) for _ in range(1000)]
for (m, cap), s in pooled.items():
    h = hits(y, s, K)
    diffs = np.array([topk_hits(y, s, i, K) - topk_hits(y, base, i, K) for i in reps])
    hs = np.array([topk_hits(y, s, i, K) for i in reps])
    boot.append({"model": m, "cap": cap or "none", f"hits@{K}": h, "95% CI": f"{np.percentile(hs, 2.5):.0f}-{np.percentile(hs, 97.5):.0f}",
                 "vs exploit+EPSS (95% CI)": f"{np.percentile(diffs, 2.5):+.0f} to {np.percentile(diffs, 97.5):+.0f}",
                 "AP": round(average_precision_score(y, s), 4)})
boot = pd.DataFrame(boot)
boot.to_csv(os.path.join(REP, "chen_insights_bootstrap.csv"), index=False)
print(f"\npooled: top {K} of {len(both):,}, {y.sum()} positives")
print(boot.to_string(index=False))

# ----------------------------------------------------------------------------- lift of Chen's features on our replicas
allrep = pd.concat([te25, te24], ignore_index=True)
base_rate = allrep.y_nov4.mean()
lift = []
for f in CHEN_SAFE + CHEN_CAREFUL + ["cv4_e_attacked"]:
    on = allrep[f] > 0
    lift.append({"feature": f, "rows": int(on.sum()), "kev": int(allrep[on].y_nov4.sum()),
                 "kev_rate": round(allrep[on].y_nov4.mean(), 5) if on.any() else 0,
                 "lift_vs_base": round(allrep[on].y_nov4.mean() / base_rate, 2) if on.any() else 0})
lift = pd.DataFrame(lift)
lift.to_csv(os.path.join(REP, "chen_features_lift.csv"), index=False)
print(f"\nChen's features on our test replicas (base rate {base_rate:.4%}):")
print(lift.to_string(index=False))

# ----------------------------------------------------------------------------- missed positives (Chen's analysis, our data)
best = boot.sort_values(f"hits@{K}", ascending=False).iloc[0]
s = pooled[(best.model, None if best.cap == "none" else best.cap)]
rank = pd.Series(-s).rank(method="first").to_numpy()
p = both.assign(rank=rank)[both.y_nov4 == 1].copy()
p["found"] = p["rank"] <= 2 * K
summary = p.groupby("found").agg(n=("cve_id", "size"), exploit_code=("any_flag", "mean"), median_epss_pct=("epss_pct", "median"),
                                 median_cvss=("cvss3", "median"), ui_required=("cv_user_interaction", lambda x: (x == 0).mean()))
summary.to_csv(os.path.join(REP, "chen_missed_positives.csv"))
print(f"\npositives found in the top {2 * K} by [{best.model}, cap {best.cap}] vs missed:")
print(summary.round(3).to_string())
missed = p[~p.found].merge(v[["assigner"]].reset_index().rename(columns={"assigner": "a2"}), on="cve_id", how="left")
print("missed examples:", ", ".join(missed.sort_values("rank").cve_id.head(15)))
