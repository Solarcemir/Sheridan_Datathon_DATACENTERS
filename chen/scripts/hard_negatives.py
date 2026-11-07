"""Out-of-fold hard negatives for LoRA training.

Refit the stacking LR (features + Granite zero-shot) with 5-fold CV on 2024, grouped so a
hand-made KEV group (processed/kev_groups_2024.csv) never spans folds. Each 2024 CVE gets the
score from the fold that did not train on it. Hard negatives = the highest-scored non-KEV CVEs.

Writes processed/oof_2024.parquet (cve_id, oof_score, oof_rank) and prints a balance check
of positives vs hard vs random negatives.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from stack import FLAGS, WP, features

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "processed"


def main() -> None:
    hist = pd.read_parquet(PROC / "historical_2024_2025.parquet")
    tr = hist[hist.snapshot_month.between("2024-01", "2024-12")].reset_index(drop=True)
    z = pd.read_parquet(ROOT / "results" / "all2024_zero.parquet").set_index("cve_id").score
    tr["granite_zero"] = tr.cve_id.map(z).clip(lower=-30)
    X = pd.concat([features(tr), tr[["granite_zero"]]], axis=1).to_numpy()
    y = tr.label.to_numpy()

    g = pd.read_csv(PROC / "kev_groups_2024.csv").set_index("cve_id").group
    groups = tr.cve_id.map(g).fillna(tr.cve_id).to_numpy()

    oof = np.zeros(len(tr))
    for k, (a, b) in enumerate(GroupKFold(n_splits=5, shuffle=True, random_state=0).split(X, y, groups)):
        sc = StandardScaler().fit(X[a])
        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(sc.transform(X[a]), y[a])
        oof[b] = lr.decision_function(sc.transform(X[b]))
    tr["oof_score"] = oof
    tr["oof_rank"] = tr.oof_score.rank(ascending=False, method="first").astype(int)
    tr[["cve_id", "oof_score", "oof_rank", "label"]].to_parquet(PROC / "oof_2024.parquet", index=False)

    K = 274
    print(f"out-of-fold LR on 2024: hits@{K} = {int(tr.nsmallest(K, 'oof_rank').label.sum())}, "
          f"hits@500 = {int(tr.nsmallest(500, 'oof_rank').label.sum())} (of {y.sum()})\n")

    neg = tr[tr.label == 0].sort_values("oof_rank")
    rng = np.random.default_rng(0)
    sets = {
        "positives": tr[tr.label == 1],
        "hard top 670": neg.head(670),
        "hard 671-2000": neg.iloc[670:2000],
        "random 1340": neg.iloc[rng.choice(len(neg), 1340, replace=False)],
        "all negatives": neg,
    }
    cna = lambda d: d.assigner.isin(WP)
    rows = {}
    for n, d in sets.items():
        rows[n] = {
            "n": len(d),
            "any code %": round(100 * d[FLAGS].max(axis=1).mean(), 1),
            "EPSS pct median": round(100 * d.epss.rank(pct=True).median(), 0) if n == "all negatives" else None,
            "CVSS median": d.cvss3_score.median(),
            "network %": round(100 * (d.attack_vector == "NETWORK").mean(), 0),
            "no priv %": round(100 * (d.privileges_required == "NONE").mean(), 0),
            "WordPress CNA %": round(100 * cna(d).mean(), 0),
            "granite zero median": round(d.granite_zero.median(), 1),
        }
    t = pd.DataFrame(rows).T.drop(columns="EPSS pct median")
    print(t.to_string())

    print("\ntop hard negatives (assigner | description):")
    for r in neg.head(12).itertuples():
        print(f"  {r.oof_rank:5d} {r.assigner[:26]:26} {str(r.description)[:105]}")
    print("\nassigners among hard top 670:")
    print(neg.head(670).assigner.value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
