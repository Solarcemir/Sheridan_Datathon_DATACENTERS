"""Train and export the servable artifacts for the repo scanner: LR A (56 features, the same model
luis/analysis/final_build.py uses as half of its linear voter) and the text_kev_lr/text_kev_sim TF-IDF
model, both refit so they can score ONE live CVE at a time instead of a fixed replica cohort.

Why re-export rather than reuse derived/scored_2026_final.csv: that file only has scores for the 72,670
scoreboard CVEs. A repo scan needs to score arbitrary CVEs (whatever a user's dependencies turn up), so we
need the fitted model object itself, not its past predictions.

days_to_cutoff: at training time this was "days of runway left before the judge's Nov-4 deadline" -- a live
tool has no such deadline, so common_live.DAYS_TO_CUTOFF_DEFAULT (180 days) is used for every live row. That
constant is baked into the live feature builder, not into this export, so it can be tuned without retraining.

Outputs (git-ignored, same as luis/derived/*.csv -- see .gitignore):
  model/lr_a.joblib          sklearn Pipeline (imputer, scaler, logistic regression), 56 features
  model/tfidf_kev.joblib     {vec, lr, pos_matrix, pos_ids} for text_kev_lr / text_kev_sim on a live description
  model/feature_order.json   the 56 column names, in the order lr_a.joblib expects them
  model/meta.json            training summary (rows, positives, chosen C, build time) for the scanner's "about" panel

Usage: python -E -P luis/repo_scanner/train_export.py   (run once; re-run after `git pull`s that touch the IBM data)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from common import CHEN_LITE, DATA, SAFE, kev, load, lr_pipe, vuln  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
os.makedirs(MODEL_DIR, exist_ok=True)
T0 = time.time()
FEATURES = SAFE + CHEN_LITE
assert len(FEATURES) == 56, f"expected 56 features, got {len(FEATURES)}"


def log(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- LR A: fit on both replica years
train = pd.concat([load(2024, "2024-12-31"), load(2025, "2025-12-31")], ignore_index=True)
log(f"training rows {len(train):,}, positives {int(train.y.sum())}")

a, b = train[train.snapshot_month.str[-2:] <= "06"], train[train.snapshot_month.str[-2:] >= "07"]
from sklearn.metrics import average_precision_score  # noqa: E402
grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
ap = [average_precision_score(b.y, lr_pipe(c).fit(a[FEATURES], a.y).predict_proba(b[FEATURES])[:, 1]) for c in grid]
C = grid[int(np.argmax(ap))]
log(f"C chosen inside training (fit Jan-Jun, validate Jul-Nov): {C} (inner AP {max(ap):.4f})")
pipe = lr_pipe(C).fit(train[FEATURES], train.y)
joblib.dump(pipe, os.path.join(MODEL_DIR, "lr_a.joblib"))
log(f"wrote model/lr_a.joblib ({len(FEATURES)} features)")

# ----------------------------------------------------------------------------- text_kev_lr / text_kev_sim
# Same recipe as build_features.py's add_text_model, with "today" standing in for the training year's Jan 1:
# every CVE on KEV as of now is a positive, same negative sample size and TF-IDF settings.
v, k = vuln(), kev()
pos_ids = set(k.cve_id)
base = v[v.published >= "2015-01-01"]
pos = base[base.index.isin(pos_ids)]
neg = base[~base.index.isin(pos_ids)].sample(n=min(40000, len(base)), random_state=0)
corpus = pd.concat([pos.description, neg.description])
yv = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=150000, sublinear_tf=True, stop_words="english")
Xt = vec.fit_transform(corpus)
lr = LogisticRegression(C=2.0, class_weight="balanced", max_iter=3000).fit(Xt, yv)
pos_matrix = Xt[:len(pos)].tocsr()
joblib.dump({"vec": vec, "lr": lr, "pos_matrix": pos_matrix, "pos_ids": pos.index.tolist()},
            os.path.join(MODEL_DIR, "tfidf_kev.joblib"))
log(f"wrote model/tfidf_kev.joblib ({len(pos):,} KEV positives, {len(neg):,} negatives, {Xt.shape[1]:,} terms)")

json.dump(FEATURES, open(os.path.join(MODEL_DIR, "feature_order.json"), "w"))
json.dump({
    "built": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    "training_rows": int(len(train)), "training_positives": int(train.y.sum()),
    "lr_a_C": C, "kev_positives_for_text_model": int(len(pos)), "features": len(FEATURES),
    "data_snapshot": {"vulnerabilities_rows": int(len(v)), "kev_rows": int(len(k))},
}, open(os.path.join(MODEL_DIR, "meta.json"), "w"), indent=2)
log("done")
