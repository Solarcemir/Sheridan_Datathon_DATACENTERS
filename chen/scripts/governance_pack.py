"""Files for registering the final LR in IBM watsonx.governance (factsheet + drift evaluation).

Writes governance/:
  feature_schema.csv       name, group, type, description, why it's known at the snapshot
  training_sample.csv      10,000 training rows (all positives + random negatives): the 100 LR features,
                           label (KEV by Nov 4 of publish year), out-of-fold LR score
  scoreboard_sample.csv    10,000 random 2026 scoreboard rows: the same features + final LR score
                           (the "payload" for a drift evaluation)
  drift_table.csv          our own drift check per feature: mean/std in training vs 2026, standardized
                           difference, population stability index (PSI)

Only derived numeric features and scores (no IBM descriptions or raw data files).

  python scripts/governance_pack.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import cv
import final

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "governance"

GROUPS = [("base", "stack.features: exploit flags, EPSS percentile, CVSS parts, WordPress CNA",
           "signals at the end of the snapshot month; CVSS from the CNA at publish"),
          ("meta", "meta_features.py: assigner / CWE KEV history, release batch, text flags, CVSS 4 extras",
           "publish-day facts; KEV history up to the end of the previous year"),
          ("tfidf", "team_features.py: TF-IDF 'reads like past KEV' model, similarity, adjacent IDs",
           "KEV entries before Jan 1 of the publish year"),
          ("granite", "llm_features.py: Granite 4.2 30B world-knowledge answers + vendor/product KEV history",
           "Granite reads description, assigner and CWE only; KEV history up to the end of the previous year")]


def psi(a: np.ndarray, b: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near-)constant or binary feature: compare category shares
        va = np.unique(np.r_[a, b])
        pa = np.array([(a == v).mean() for v in va]) + 1e-4
        pb = np.array([(b == v).mean() for v in va]) + 1e-4
    else:
        ia = np.clip(np.searchsorted(edges, a, side="right") - 1, 0, len(edges) - 2)
        ib = np.clip(np.searchsorted(edges, b, side="right") - 1, 0, len(edges) - 2)
        pa = np.bincount(ia, minlength=len(edges) - 1) / len(a) + 1e-4
        pb = np.bincount(ib, minlength=len(edges) - 1) / len(b) + 1e-4
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    tr = cv.load()
    tr["nov4"] = (tr.kev_date_added.notna()
                  & (tr.kev_date_added <= pd.to_datetime(tr.published.dt.year.astype(str) + "-11-04"))).astype(int)
    sb = final.scoreboard()
    Xtr, Xsb = final.blocks(tr)["lr"], final.blocks(sb)["lr"].fillna(0)

    # feature schema: which group each column came from
    from stack import features, meta_features
    sizes = [features(tr.head(5)).shape[1], meta_features(tr.head(5)).shape[1],
             cv.team_feats(tr.head(5)).shape[1], cv.llm_feats(tr.head(5)).shape[1]]
    grp = sum(([g] * n for g, n in zip(GROUPS, sizes)), [])
    assert len(grp) == Xtr.shape[1]
    binary = [set(np.unique(Xtr[c])) <= {0.0, 1.0} for c in Xtr.columns]
    pd.DataFrame({"feature": Xtr.columns, "group": [g[0] for g in grp],
                  "type": ["binary" if b else "numeric" for b in binary],
                  "source": [g[1] for g in grp], "known_at_snapshot_because": [g[2] for g in grp]}
                 ).to_csv(OUT / "feature_schema.csv", index=False)

    # training sample with out-of-fold LR scores (honest scores for training rows)
    oof = pd.read_parquet(ROOT / "results" / "cv_oof_lr.parquet").set_index("cve_id").lr_full_llmx_noemb
    pos = np.flatnonzero(tr.label.to_numpy() == 1)
    neg = rng.choice(np.flatnonzero(tr.label.to_numpy() == 0), 10_000 - len(pos), replace=False)
    i = np.sort(np.r_[pos, neg])
    ts = pd.concat([tr.iloc[i][["cve_id", "snapshot_month"]].reset_index(drop=True),
                    Xtr.iloc[i].reset_index(drop=True)], axis=1)
    ts["label_kev_by_nov4"] = tr.nov4.to_numpy()[i]
    ts["label_kev_by_2025_12_31"] = tr.label.to_numpy()[i]
    ts["lr_score_out_of_fold"] = tr.cve_id.iloc[i].map(oof).to_numpy()
    ts.to_csv(OUT / "training_sample.csv", index=False, float_format="%.6g")

    fs = pd.read_parquet(ROOT / "results" / "final_scores.parquet").set_index("cve_id")
    j = np.sort(rng.choice(len(sb), 10_000, replace=False))
    ss = pd.concat([sb.iloc[j][["cve_id", "snapshot_month"]].reset_index(drop=True),
                    Xsb.iloc[j].reset_index(drop=True)], axis=1)
    ss["lr_score_final"] = sb.cve_id.iloc[j].map(fs.lr).to_numpy()
    ss.to_csv(OUT / "scoreboard_sample.csv", index=False, float_format="%.6g")

    # our own drift table (all rows, not just samples)
    rows = []
    for c in Xtr.columns:
        a, b = Xtr[c].to_numpy(float), Xsb[c].to_numpy(float)
        sd = np.sqrt((a.var() + b.var()) / 2) or 1.0
        rows.append({"feature": c, "train_mean": a.mean(), "scoreboard_mean": b.mean(),
                     "train_std": a.std(), "scoreboard_std": b.std(),
                     "std_mean_diff": (b.mean() - a.mean()) / sd, "psi": psi(a, b)})
    d = pd.DataFrame(rows).sort_values("psi", ascending=False)
    d.to_csv(OUT / "drift_table.csv", index=False, float_format="%.4g")
    print(f"wrote {OUT}: schema {len(Xtr.columns)} features; training sample {len(ts):,} "
          f"({len(pos)} positives); scoreboard sample {len(ss):,}")
    print(f"PSI > 0.25 (major shift): {int((d.psi > 0.25).sum())} features; 0.1-0.25: {int(d.psi.between(0.1, 0.25).sum())}")
    print(d.head(12)[["feature", "train_mean", "scoreboard_mean", "std_mean_diff", "psi"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
