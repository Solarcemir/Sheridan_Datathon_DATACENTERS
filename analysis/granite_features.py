"""IBM Granite Embedding features from the vulnerability description.

1. Embeds descriptions with ibm-granite/granite-embedding-30m-english (384 dims, normalized):
   every CVE in derived/features_{2024,2025,2026}.csv, every KEV CVE published since 2015,
   and a 60,000-CVE sample of non-KEV CVEs (2015-2025) used as negatives.
   Cached in derived/granite/ (re-runs skip this step).
2. For each year Y, using only KEV entries added before Jan 1 of Y:
   gr_kev_lr        logistic regression on the embedding: "does this read like past KEV entries?"
   gr_kev_sim       max cosine similarity to any past KEV description
   gr_kev_knn5      mean of the 5 highest similarities
   gr_cluster_rate  smoothed KEV rate of the semantic cluster (k-means, 60 clusters) the CVE falls in
   gr_hybrid_lr     logistic regression on TF-IDF words + Granite embedding (64 PCA dims) together
3. reports/granite_clusters_2026.csv: what each cluster is about, how often it reached KEV,
   and how many 2026 scoreboard CVEs fall in it.

Usage:  python -E -P analysis/granite_features.py   (needs sentence-transformers; uses the GPU if present)
"""
import os
import sys
import time

os.environ.setdefault("USE_TF", "0")          # TensorFlow is installed on this machine; keep transformers on PyTorch
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import csr_matrix, hstack  # noqa: E402
from sklearn.cluster import MiniBatchKMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

T0 = time.time()
MODEL = "ibm-granite/granite-embedding-30m-english"
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "IBM-20261107T174321Z-1-001", "IBM", "data", "data")
DER, OUT, REP = os.path.join(ROOT, "derived"), os.path.join(ROOT, "derived", "granite"), os.path.join(ROOT, "reports")
os.makedirs(OUT, exist_ok=True)
EMB, IDS = os.path.join(OUT, "emb_granite30m.npy"), os.path.join(OUT, "emb_ids.csv")


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


v = pd.read_csv(f"{DATA}/vulnerabilities.csv", dtype=str, usecols=["cve_id", "published", "description"])
v["published"] = pd.to_datetime(v.published)
v["description"] = v.description.fillna("")
v = v.set_index("cve_id")
kev = pd.read_csv(f"{DATA}/known_exploited.csv", dtype=str)
kev["date_added"] = pd.to_datetime(kev.date_added)
kev_ids = set(kev.cve_id)

cohorts = {y: pd.read_csv(os.path.join(DER, f"features_{y}.csv"), usecols=["cve_id"]).cve_id for y in (2024, 2025, 2026)}
recent = v[v.published >= "2015-01-01"]
pos_all = recent[recent.index.isin(kev_ids)]
neg_all = recent[(recent.published < "2026-01-01") & ~recent.index.isin(kev_ids)].sample(n=60000, random_state=0)
want = pd.Index(sorted(set().union(*[set(c) for c in cohorts.values()], pos_all.index, neg_all.index)))

# ----------------------------------------------------------------------------- 1. embed (cached)
if os.path.exists(EMB) and os.path.exists(IDS) and len(pd.read_csv(IDS)) == len(want):
    E = np.load(EMB).astype(np.float32)
    ids = pd.read_csv(IDS).cve_id
    log(f"loaded cached embeddings {E.shape}")
else:
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"embedding {len(want):,} descriptions with {MODEL} on {dev}")
    model = SentenceTransformer(MODEL, device=dev)
    texts = v.loc[want, "description"].tolist()
    order = np.argsort([len(t) for t in texts])               # length-sorted batches run faster
    emb = model.encode([texts[i] for i in order], batch_size=256, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False)
    E = np.empty_like(emb)
    E[order] = emb
    np.save(EMB, E.astype(np.float16))
    pd.DataFrame({"cve_id": want}).to_csv(IDS, index=False)
    ids = pd.Series(want)
    log(f"saved embeddings {E.shape}")
pos_of = pd.Series(np.arange(len(ids)), index=ids.values)


def rows(cves):
    return E[pos_of.loc[list(cves)].to_numpy()]


# ----------------------------------------------------------------------------- 2. per-year features
FEATS = {
    "gr_kev_lr": "Granite embedding + logistic regression trained on past KEV vs non-KEV descriptions",
    "gr_kev_sim": "Max cosine similarity (Granite) to any past KEV description",
    "gr_kev_knn5": "Mean of the 5 highest Granite similarities to past KEV descriptions",
    "gr_cluster_rate": "Smoothed KEV rate of the Granite semantic cluster (60 k-means clusters, fit on past data)",
    "gr_hybrid_lr": "Logistic regression on TF-IDF words + Granite embedding (64 PCA dims), trained on past KEV",
}
pd.DataFrame([(k, "granite", "safe", d) for k, d in FEATS.items()],
             columns=["feature", "group", "risk", "description"]).to_csv(os.path.join(OUT, "feature_dictionary_granite.csv"), index=False)

for year, cohort in cohorts.items():
    start = pd.Timestamp(f"{year}-01-01")
    p_ids = kev[(kev.date_added < start) & kev.cve_id.isin(pos_all.index)].cve_id
    n_ids = neg_all[neg_all.published < start].index
    P, N, X = rows(p_ids), rows(n_ids), rows(cohort)
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
    lr.fit(np.vstack([P, N]), np.r_[np.ones(len(P)), np.zeros(len(N))])
    out = pd.DataFrame({"cve_id": cohort.values, "gr_kev_lr": lr.predict_proba(X)[:, 1]})
    sim_max, sim5 = np.zeros(len(X)), np.zeros(len(X))
    for i in range(0, len(X), 8192):
        S = X[i:i + 8192] @ P.T
        top = np.partition(S, -5, axis=1)[:, -5:]
        sim_max[i:i + 8192], sim5[i:i + 8192] = top.max(axis=1), top.mean(axis=1)
    out["gr_kev_sim"], out["gr_kev_knn5"] = sim_max, sim5
    km = MiniBatchKMeans(n_clusters=60, random_state=0, batch_size=4096, n_init=3).fit(np.vstack([P, N]))
    cp, cn = np.bincount(km.predict(P), minlength=60), np.bincount(km.predict(N), minlength=60)
    prior = len(P) / (len(P) + len(N))
    rate = (cp + prior * 50) / (cp + cn + 50)
    lab = km.predict(X)
    out["gr_cluster_rate"] = rate[lab]
    # Hybrid text model: TF-IDF words (exact product names) + Granite semantics (64 PCA dims, small weight)
    corpus = list(p_ids) + list(n_ids)
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=150000, sublinear_tf=True, stop_words="english")
    Tt = tf.fit_transform(v.loc[corpus, "description"])
    pca = PCA(64, random_state=0).fit(np.vstack([P, N]))
    sd = pca.transform(np.vstack([P, N])).std(axis=0)
    Zt, Zc = pca.transform(np.vstack([P, N])) / sd * 0.1, pca.transform(X) / sd * 0.1
    hy = LogisticRegression(C=2.0, class_weight="balanced", max_iter=4000)
    hy.fit(hstack([Tt, csr_matrix(Zt)]).tocsr(), np.r_[np.ones(len(P)), np.zeros(len(N))])
    out["gr_hybrid_lr"] = hy.predict_proba(hstack([tf.transform(v.loc[cohort, "description"]), csr_matrix(Zc)]).tocsr())[:, 1]
    out.to_csv(os.path.join(OUT, f"granite_features_{year}.csv"), index=False)
    log(f"granite_features_{year}.csv: {len(out):,} rows, {len(P):,} KEV refs, {len(N):,} negatives")

    if year == 2026:     # cluster report for the story: which technologies get exploited
        train_ids = list(p_ids) + list(n_ids)
        train_lab = np.r_[km.predict(P), km.predict(N)]
        desc = v.loc[train_ids, "description"].tolist()
        tf = TfidfVectorizer(stop_words="english", min_df=5, max_df=0.3, ngram_range=(1, 2), max_features=40000)
        T = tf.fit_transform(desc)
        vocab = np.array(tf.get_feature_names_out())
        kp = kev.set_index("cve_id")
        p_arr, p_lab = np.asarray(p_ids), km.predict(P)
        rep = []
        for k in range(60):
            m = train_lab == k
            terms = vocab[np.asarray(T[m].mean(axis=0)).ravel().argsort()[::-1][:8]]
            prods = kp.loc[p_arr[p_lab == k]]
            top_prod = ", ".join((prods.vendor + " " + prods["product"]).value_counts().head(4).index)
            rep.append({"cluster": k, "kev_refs": int(cp[k]), "sampled_non_kev": int(cn[k]),
                        "kev_rate_smoothed": round(rate[k], 4), "scoreboard_2026_cves": int((lab == k).sum()),
                        "top_terms": ", ".join(terms), "top_kev_products": top_prod})
        pd.DataFrame(rep).sort_values("kev_rate_smoothed", ascending=False) \
            .to_csv(os.path.join(REP, "granite_clusters_2026.csv"), index=False)
        log("reports/granite_clusters_2026.csv written")
log("done")
