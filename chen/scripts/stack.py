"""Stacking baseline: logistic regression on publish-time features + Granite scores.

Train on all 2024 CVEs, test on the 2025 holdout (2025-01..09). Granite scores used for
training are out-of-sample: zero-shot never sees labels; many-shot 2024 scores are
leave-one-group-out (llm_score_logo.py). 2025 many-shot uses all-2024 examples.

  python scripts/stack.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FLAGS = ["exploitdb", "metasploit", "nuclei", "poc_github"]
WP = {"security@wordfence.com", "audit@patchstack.com", "contact@wpscan.com"}
SCOREBOARD_ROWS = 72_670

SCORES = {  # name -> (2024 train file, 2025 test file)
    "granite_zero": ("all2024_zero.parquet", "holdout_zero.parquet"),
    "granite_many": ("all2024_logo.parquet", "holdout_many.parquet"),
}


def features(df: pd.DataFrame) -> pd.DataFrame:
    epss_pct = df.groupby("snapshot_month")["epss"].rank(pct=True)
    cvss = df.cvss3_score.fillna(df.cvss4_score)
    return pd.DataFrame({
        **{f: df[f].astype(float) for f in FLAGS},
        "any_code": df[FLAGS].max(axis=1).astype(float),
        "epss_pct": epss_pct.fillna(0.5),
        "epss_missing": epss_pct.isna().astype(float),
        "cvss": cvss.fillna(cvss.median()),
        "cvss_missing": cvss.isna().astype(float),
        "av_network": (df.attack_vector == "NETWORK").astype(float),
        "ac_low": (df.attack_complexity == "LOW").astype(float),
        "pr_none": (df.privileges_required == "NONE").astype(float),
        "ui_none": (df.user_interaction == "NONE").astype(float),
        "scope_changed": (df.scope == "CHANGED").astype(float),
        "wordpress_cna": df.assigner.isin(WP).astype(float),
    }, index=df.index)


def meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fair publish-time metadata (scripts/meta_features.py), log-scaled where counts drift."""
    m = pd.read_parquet(ROOT / "processed" / "meta_historical.parquet").set_index("cve_id").reindex(df.cve_id)
    return pd.DataFrame({
        "log_assigner_kev_rate": np.log(m.assigner_kev_rate.to_numpy()),
        "log_assigner_kev": np.log1p(m.assigner_kev.to_numpy()),
        "log_days_since_assigner_kev": np.log1p(m.assigner_days_since_kev.to_numpy()),
        "log_cwe_kev_rate": np.log(m.cwe_kev_rate.to_numpy()),
        "log_batch_size": np.log(m.batch_size.to_numpy()),
        "out_of_band": m.out_of_band.astype(float).to_numpy(),
        "log1p_twins": np.log1p(m.twins.to_numpy()),
        "sibling_has_code": m.sibling_has_code.astype(float).to_numpy(),
        "old_id": (m.id_year_lag >= 1).astype(float).to_numpy(),
        "patch_tuesday": m.patch_tuesday.astype(float).to_numpy(),
        "eol_text": m.eol_text.astype(float).to_numpy(),
        "unauth_text": m.unauth_text.astype(float).to_numpy(),
        "cvss4_urgent": m.cvss4_urgency.isin(["Red", "Amber"]).astype(float).to_numpy(),
        "cvss4_automatable": m.cvss4_automatable.astype(float).to_numpy(),
        # second batch
        "log1p_vendor_name_kev": np.log1p(m.vendor_name_kev.to_numpy()),
        "log1p_product_name_kev": np.log1p(m.product_name_kev.to_numpy()),
        "log1p_named_ransomware_kev": np.log1p(m.named_ransomware_kev.to_numpy()),
        "log1p_assigner_ransomware_kev": np.log1p(m.assigner_ransomware_kev.to_numpy()),
        "sibling_max_epss_pct": m.sibling_max_epss_pct.to_numpy(),
        "rce_text": m.rce_text.astype(float).to_numpy(),
        "auth_bypass_text": m.auth_bypass_text.astype(float).to_numpy(),
        "preauth_text": m.preauth_text.astype(float).to_numpy(),
        "log1p_n_versions": np.log1p(m.n_versions.to_numpy()),
        "cvss4_at_none": m.cvss4_at_none.astype(float).to_numpy(),
        "cvss4_subsequent_high": m.cvss4_subsequent_high.astype(float).to_numpy(),
        "cwe_unknown": m.cwe_unknown.astype(float).to_numpy(),
        "cwe_broad": m.cwe_broad.astype(float).to_numpy(),
    }, index=df.index)


C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def pick_C(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, dict]:
    """Regularisation chosen by 5-fold grouped CV on the training year only (average precision)."""
    scores = {}
    for C in C_GRID:
        oof = np.zeros(len(y))
        for a, b in GroupKFold(n_splits=5, shuffle=True, random_state=0).split(X, y, groups):
            sc = StandardScaler().fit(X[a])
            m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(sc.transform(X[a]), y[a])
            oof[b] = m.decision_function(sc.transform(X[b]))
        scores[C] = average_precision_score(y, oof)
    return max(scores, key=scores.get), scores


def hits(score: np.ndarray, y: np.ndarray, ks) -> dict:
    order = np.argsort(-score, kind="stable")
    return {f"hits@{k}": int(y[order[:k]].sum()) for k in ks}


def main() -> None:
    hist = pd.read_parquet(ROOT / "processed" / "historical_2024_2025.parquet")
    tr = hist[hist.snapshot_month.between("2024-01", "2024-12")].copy()
    te = hist[hist.snapshot_month.between("2025-01", "2025-09")].copy()

    available = []
    for name, (f_tr, f_te) in SCORES.items():
        if (RES / f_tr).exists() and (RES / f_te).exists():
            for df, f in ((tr, f_tr), (te, f_te)):
                s = pd.read_parquet(RES / f).set_index("cve_id").score
                df[name] = df.cve_id.map(s).clip(lower=-30).to_numpy()
            available.append(name)
        else:
            print(f"(skipping {name}: missing {f_tr} or {f_te})")

    Xtr, Xte = features(tr), features(te)
    ytr, yte = tr.label.to_numpy(), te.label.to_numpy()
    ks = [round(500 * len(te) / SCOREBOARD_ROWS), 500, 1000]
    print(f"train 2024: {len(tr):,} CVEs, {ytr.sum()} positives | test 2025-01..09: "
          f"{len(te):,} CVEs, {yte.sum()} positives\n")

    rows = {}
    te_code = te[FLAGS].max(axis=1).to_numpy()
    rows["exploit code, then EPSS"] = hits(te_code * 10 + te.epss.fillna(0).to_numpy(), yte, ks)
    for name in available:
        rows[f"{name} alone"] = hits(te[name].to_numpy(), yte, ks)

    Mtr, Mte = meta_features(tr), meta_features(te)
    kg = pd.read_csv(ROOT / "processed" / "kev_groups_2024.csv").set_index("cve_id").group
    groups = tr.cve_id.map(kg).fillna(tr.cve_id).to_numpy()
    g = available
    variants = {  # label -> (use meta, extra Granite score columns)
        "features only": (False, []),
        "features + Granite": (False, g),
        "features + meta": (True, []),
        "features + meta + Granite": (True, g),
    }
    coefs = {}
    for label, (use_meta, extra) in variants.items():
        A_tr = pd.concat([Xtr] + ([Mtr] if use_meta else []) + [tr[extra]], axis=1)
        A_te = pd.concat([Xte] + ([Mte] if use_meta else []) + [te[extra]], axis=1)
        sc = StandardScaler().fit(A_tr)
        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
        lr.fit(sc.transform(A_tr), ytr)
        rows[f"LR C=1: {label}"] = hits(lr.decision_function(sc.transform(A_te)), yte, ks)
        C, cv = pick_C(A_tr.to_numpy(), ytr, groups)
        lr = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(sc.transform(A_tr), ytr)
        rows[f"LR C={C} (CV on 2024): {label}"] = hits(lr.decision_function(sc.transform(A_te)), yte, ks)
        print(f"{label}: CV average precision by C " + ", ".join(f"{c}:{v:.4f}" for c, v in cv.items()), flush=True)
        coefs[label] = pd.Series(lr.coef_[0], index=A_tr.columns)
        # gradient boosting: shallow, heavily regularised (67 positives)
        gb = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                            min_samples_leaf=200, l2_regularization=1.0,
                                            class_weight="balanced", random_state=0)
        gb.fit(A_tr, ytr)
        rows[f"GBM: {label}"] = hits(gb.predict_proba(A_te)[:, 1], yte, ks)

    print(pd.DataFrame(rows).T.to_string())
    print("\nstandardised coefficients (log-odds per SD):")
    print(pd.DataFrame(coefs).round(2).to_string())


if __name__ == "__main__":
    main()
