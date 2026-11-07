"""Batch-aware ranking: if the model ranks one CVE high, should its siblings move up too?

Siblings = same assigner, same snapshot month, same ID year, ID numbers within K, published within 3 days.
Only information known at the snapshot is used: the siblings' own stage-1 scores (never their KEV outcome).

  final = s1_pct + lambda * max(0, sib_max_pct - s1_pct)   if sib_max_pct >= gate, else s1_pct
          (lambda = 1 means "inherit your strongest sibling's rank"; lambda = 0 is the plain model)

Stage 1 is the v1 LightGBM (safe features + TF-IDF text), scored out-of-sample:
  A: trained on 2024, scores 2025 (the realistic test)   B: trained on 2025, scores 2024 (robustness only)

Outputs: reports/sibling_boost_results.csv, reports/sibling_retro_check.csv,
         derived/backtest_rankings_A.csv, derived/backtest_rankings_B.csv  (every candidate with its rank and label)

Usage:  python -E -P analysis/sibling_boost.py
"""
import os

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DER, REP = os.path.join(ROOT, "derived"), os.path.join(ROOT, "reports")
fd = pd.read_csv(os.path.join(DER, "feature_dictionary.csv"))
FEATS = [f for f in fd[fd.risk == "safe"].feature] + ["days_to_cutoff"]
LGB = dict(objective="binary", n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, scale_pos_weight=25, verbose=-1, n_jobs=4)


def load(year, cutoff):
    df = pd.read_csv(os.path.join(DER, f"features_{year}.csv"), parse_dates=["snapshot_end", "published"])
    df["days_to_cutoff"] = (pd.Timestamp(cutoff) - df.snapshot_end).dt.days
    return df


def stage1(train, test):
    ms = [lgb.LGBMClassifier(random_state=s, **LGB).fit(train[FEATS], train.y) for s in range(5)]
    return np.mean([m.predict_proba(test[FEATS])[:, 1] for m in ms], axis=0)


def jan_sep(df, year):
    return df[df.snapshot_month <= f"{year}-09"].reset_index(drop=True)


SPLITS = {}
for name, tr_y, te_y in [("A", 2024, 2025), ("B", 2025, 2024)]:
    tr = load(tr_y, f"{tr_y}-12-31")
    te = jan_sep(load(te_y, f"{te_y}-11-04"), te_y)
    te["s1"] = stage1(tr, te)
    te["s1_pct"] = te.s1.rank(pct=True)
    SPLITS[name] = te
    print(f"{name}: test {te_y} Jan-Sep, {len(te):,} rows, {te.y_nov4.sum()} positives by Nov 4")


def sibling_max(df, K, window=25):
    """Max stage-1 percentile among siblings (excluding self) and the number of siblings."""
    parts = df.cve_id.str.split("-", expand=True)
    s = df.assign(id_year=parts[1], id_num=pd.to_numeric(parts[2])) \
          .sort_values(["assigner", "snapshot_month", "id_year", "id_num"])
    best = pd.Series(-1.0, index=s.index)
    count = pd.Series(0, index=s.index)
    for off in [o for k in range(1, window + 1) for o in (k, -k)]:
        o = s.shift(off)
        ok = ((o.assigner == s.assigner) & (o.snapshot_month == s.snapshot_month) & (o.id_year == s.id_year)
              & ((o.id_num - s.id_num).abs() <= K) & ((o.published - s.published).abs() <= pd.Timedelta(days=3)))
        best = best.where(~ok | (best >= o.s1_pct), o.s1_pct)
        count += ok.astype(int)
    return best.reindex(df.index).to_numpy(), count.reindex(df.index).to_numpy()


def hits(df, score, k=500):
    return int(df.y_nov4.to_numpy()[np.argsort(-np.asarray(score), kind="stable")[:k]].sum())


rows = []
for K in (3, 10, 50):
    for name, te in SPLITS.items():
        te[f"sib_{K}"], te[f"nsib_{K}"] = sibling_max(te, K)
    for gate in (0.95, 0.99):
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            r = {"K (ID distance)": K, "gate": gate, "lambda": lam}
            for name, te in SPLITS.items():
                sib = te[f"sib_{K}"].to_numpy()
                boost = np.where(sib >= gate, lam * np.maximum(0, sib - te.s1_pct.to_numpy()), 0)
                final = te.s1_pct.to_numpy() + boost
                base_top = set(np.argsort(-te.s1_pct.to_numpy(), kind="stable")[:500])
                new_top = set(np.argsort(-final, kind="stable")[:500])
                added = list(new_top - base_top)
                r[f"hits {name}"] = hits(te, final)
                r[f"added {name}"] = len(added)
                r[f"added that were KEV {name}"] = int(te.y_nov4.to_numpy()[added].sum()) if added else 0
            r["hits A+B"] = r["hits A"] + r["hits B"]
            rows.append(r)
res = pd.DataFrame(rows).drop_duplicates(subset=["K (ID distance)", "lambda", "hits A", "hits B", "added A", "added B"])
res.to_csv(os.path.join(REP, "sibling_boost_results.csv"), index=False)
pd.set_option("display.width", 220)
print(res.to_string(index=False))

# ----------------------------------------------------------------------------- retro check (ChatGPT's suggestion)
# Are siblings of a model top-500 CVE more likely to reach KEV than other CVEs?
chk = []
for name, te in SPLITS.items():
    top_mask = te.s1_pct.rank(ascending=False, method="first") <= 500
    thresh = te.s1_pct[top_mask].min()
    for K in (3, 10, 50):
        has_top_sib = (te[f"sib_{K}"] >= thresh) & ~top_mask
        chk.append({"split": name, "K": K, "outside top500 with a top500 sibling": int(has_top_sib.sum()),
                    "of which KEV": int(te[has_top_sib].y_nov4.sum()),
                    "KEV rate (sibling of top500)": round(te[has_top_sib].y_nov4.mean(), 4) if has_top_sib.any() else 0,
                    "KEV rate (other non-top500)": round(te[~top_mask & ~has_top_sib].y_nov4.mean(), 5),
                    "KEV rate inside top500": round(te[top_mask].y_nov4.mean(), 4)})
chk = pd.DataFrame(chk)
chk.to_csv(os.path.join(REP, "sibling_retro_check.csv"), index=False)
print(chk.to_string(index=False))

# ----------------------------------------------------------------------------- Oracle check
for name, te in SPLITS.items():
    t = te.loc[np.argsort(-te.s1_pct.to_numpy(), kind="stable")[:500]]
    o = t.assigner == "secalert_us@oracle.com"
    print(f"{name}: Oracle CVEs in model top 500 = {o.sum()}, of which KEV = {t[o].y_nov4.sum()} | "
          f"Oracle in whole test set = {(te.assigner == 'secalert_us@oracle.com').sum()}, KEV = {te[te.assigner == 'secalert_us@oracle.com'].y_nov4.sum()}")

# ----------------------------------------------------------------------------- full backtest rankings, row by row
best = res.sort_values(["hits A", "hits A+B", "lambda"], ascending=[False, False, True]).iloc[0]
K, gate, lam = int(best["K (ID distance)"]), best.gate, best["lambda"]
for name, te in SPLITS.items():
    sib = te[f"sib_{K}"].to_numpy()
    te["sibling_score"] = np.where(sib >= 0, sib, np.nan)
    te["final_score"] = te.s1_pct + np.where(sib >= gate, lam * np.maximum(0, sib - te.s1_pct), 0)
    te["model_rank"] = te.s1_pct.rank(ascending=False, method="first").astype(int)
    te["final_rank"] = te.final_score.rank(ascending=False, method="first").astype(int)
    te["in_top500_model"] = (te.model_rank <= 500).astype(int)
    te["in_top500_final"] = (te.final_rank <= 500).astype(int)
    out = te.rename(columns={"s1": "model_score", "y_nov4": "y_kev_by_nov4"})
    out.sort_values("final_rank")[["final_rank", "model_rank", "cve_id", "snapshot_month", "assigner", "model_score",
                                    "sibling_score", "final_score", "in_top500_model", "in_top500_final", "y_kev_by_nov4"]] \
        .to_csv(os.path.join(DER, f"backtest_rankings_{name}.csv"), index=False)
print(f"saved backtest rankings with the A-best setting: K={K}, gate={gate}, lambda={lam}")
