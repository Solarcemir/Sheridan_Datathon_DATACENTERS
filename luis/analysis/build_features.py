"""Build leak-aware feature tables for two historical replicas of the benchmark and for 2026.

Replica Y (Y = 2024, 2025) copies how scoreboard_2026.csv was built:
  * cohort for month m  = CVEs published in m that were NOT on KEV at the end of m
  * signals             = EPSS + exploit flags from exploit_signals_Y.csv at the end of m
  * label y             = added to KEV after the end of m and on or before Dec 31 of Y
  y_nov4                = same, but on or before Nov 4 of Y (the cutoff day the judges use)

Every "history" feature for year Y only uses KEV entries added before Jan 1 of Y,
so the 2024 replica can train a model that is tested on 2025, and 2024+2025 can train the 2026 model.

Outputs:
  derived/features_2024.csv, derived/features_2025.csv, derived/features_2026.csv   (no leak columns)
  derived/leak_only_{year}.csv    (cve_id + LEAK_ columns, only for the "how much does leakage inflate" demo)
  derived/feature_dictionary.csv  (every feature with its group and leakage risk)

Usage:  python analysis/build_features.py [DATA_DIR]
"""
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

T0 = time.time()
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "..", "IBM-20261107T174321Z-1-001", "IBM", "data", "data")
OUT = os.path.join(ROOT, "derived")
os.makedirs(OUT, exist_ok=True)
FLAGS = ["exploitdb", "metasploit", "nuclei", "poc_github"]


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- load
v = pd.read_csv(f"{DATA}/vulnerabilities.csv", dtype=str, usecols=[
    "cve_id", "published", "nvd_status", "assigner", "description", "cvss3_score", "attack_vector",
    "attack_complexity", "privileges_required", "user_interaction", "scope", "confidentiality_impact",
    "integrity_impact", "availability_impact", "cvss4_score", "cwe_ids", "n_references", "n_affected_cpes"])
v["published"] = pd.to_datetime(v.published).dt.normalize()
v["assigner"] = v.assigner.fillna("unknown")
v["description"] = v.description.fillna("")
kev = pd.read_csv(f"{DATA}/known_exploited.csv", dtype=str)
kev["date_added"] = pd.to_datetime(kev.date_added)
kev_date = kev.set_index("cve_id").date_added
v_idx = v.set_index("cve_id")
log(f"loaded vulnerabilities={len(v):,} kev={len(kev):,}")


# ----------------------------------------------------------------------------- cohorts
def replica_cohort(year, months, cutoff, signals_file):
    sig = pd.read_csv(f"{DATA}/{signals_file}", dtype={"month": str, "cve_id": str})
    sig = sig[sig.month.isin(months)]
    parts, gap = [], []
    for m in months:
        end = pd.Timestamp(m) + pd.offsets.MonthEnd(0)
        pub = v[v.published.dt.strftime("%Y-%m") == m]
        added = pub.cve_id.map(kev_date)
        gap.append(pub[added <= end].assign(kev_by=end))          # already on KEV by month-end
        c = pub[~(added <= end)].merge(sig[sig.month == m][["cve_id", "epss"] + FLAGS], on="cve_id", how="left")
        c["snapshot_month"], c["snapshot_end"] = m, end
        a = c.cve_id.map(kev_date)
        c["y"] = ((a > end) & (a <= pd.Timestamp(cutoff))).astype(int)
        c["y_nov4"] = ((a > end) & (a <= pd.Timestamp(f"{year}-11-04"))).astype(int)   # judges' cutoff day
        parts.append(c)
    return pd.concat(parts, ignore_index=True), pd.concat(gap, ignore_index=True)[["cve_id", "assigner", "published", "kev_by"]]


def cohort_2026():
    sb = pd.read_csv(f"{DATA}/scoreboard_2026.csv", dtype=str)
    sb["epss"] = pd.to_numeric(sb.epss)
    for f in FLAGS:
        sb[f] = pd.to_numeric(sb[f])
    sb["snapshot_end"] = pd.to_datetime(sb.snapshot_month) + pd.offsets.MonthEnd(0)
    c = sb.drop(columns=["published"]).merge(v, on="cve_id", how="left")
    # Gap: published Jan-Sep 2026 but missing from the scoreboard = already on KEV at their snapshot.
    pub = v[(v.published >= "2026-01-01") & (v.published < "2026-10-01")]
    g = pub[~pub.cve_id.isin(sb.cve_id)].copy()
    snap = g.published + pd.offsets.MonthEnd(0)
    g["kev_by"] = np.where(g.published.dt.month == 6, pd.Timestamp("2026-07-31"), snap)
    g["kev_by"] = pd.to_datetime(g.kev_by)
    return c, g[["cve_id", "assigner", "published", "kev_by"]]


# ----------------------------------------------------------------------------- feature blocks
TEXT_FLAGS = {
    "txt_rce": r"remote code execution|\bRCE\b|execute arbitrary (?:code|commands)|arbitrary code execution",
    "txt_unauth": r"unauthenticated|without authentication|authentication bypass|bypass authentication",
    "txt_in_wild": r"in the wild|actively exploited|may have been exploited|limited, targeted attack",
    "txt_remote": r"remote attacker|remote, unauthenticated|over the network|via the network",
    "txt_wordpress": r"wordpress",
    "txt_plugin": r"\bplugin\b",
    "txt_sqli": r"sql injection",
    "txt_xss": r"cross[- ]site scripting|\bXSS\b",
    "txt_traversal": r"path traversal|directory traversal",
    "txt_deser": r"deserializ",
    "txt_cmdi": r"command injection|os command",
    "txt_memory": r"use[- ]after[- ]free|out[- ]of[- ]bounds|buffer overflow|heap overflow|type confusion|memory corruption",
    "txt_privesc": r"privilege escalation|elevation of privilege|escalate (?:their )?privileges",
    "txt_edge": r"\b(?:vpn|firewall|gateway|router|appliance|load balancer)\b",
    "txt_ssrf": r"server[- ]side request forgery|\bSSRF\b",
    "txt_upload": r"file upload|upload arbitrary",
    "txt_dos": r"denial of service",
    "txt_kernel": r"\bkernel\b",
}
ORD = {
    "attack_vector": {"NETWORK": 3, "ADJACENT_NETWORK": 2, "LOCAL": 1, "PHYSICAL": 0},
    "attack_complexity": {"LOW": 1, "HIGH": 0},
    "privileges_required": {"NONE": 2, "LOW": 1, "HIGH": 0},
    "user_interaction": {"NONE": 1, "REQUIRED": 0},
    "scope": {"CHANGED": 1, "UNCHANGED": 0},
    "confidentiality_impact": {"HIGH": 2, "LOW": 1, "NONE": 0},
    "integrity_impact": {"HIGH": 2, "LOW": 1, "NONE": 0},
    "availability_impact": {"HIGH": 2, "LOW": 1, "NONE": 0},
}
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\.]*")


def ngram_lookup(texts, names, max_n=4):
    """For each text, return (max KEV count, number of distinct names) among names found as word n-grams."""
    best, n_hit = np.zeros(len(texts)), np.zeros(len(texts))
    for i, t in enumerate(texts):
        toks = [w.rstrip(".") for w in TOKEN.findall(t)]
        seen = set()
        for n in range(1, max_n + 1):
            for j in range(len(toks) - n + 1):
                k = " ".join(toks[j:j + n])
                if k in names:
                    seen.add(k)
        if seen:
            best[i] = max(names[k] for k in seen)
            n_hit[i] = len(seen)
    return best, n_hit


def add_basic(c):
    c["epss"] = pd.to_numeric(c.epss)
    c["epss_missing"] = c.epss.isna().astype(int)
    c["epss"] = c.epss.fillna(0.0)
    c[FLAGS] = c[FLAGS].fillna(0).astype(int)
    c["n_flags"] = c[FLAGS].sum(axis=1)
    c["any_flag"] = (c.n_flags > 0).astype(int)
    c["epss_pct"] = c.groupby("snapshot_month").epss.rank(pct=True)   # EPSS model changed in Mar 2025
    c["age_days"] = (c.snapshot_end - c.published).dt.days
    c["desc_len"] = c.description.str.len()
    for k, p in TEXT_FLAGS.items():
        c[k] = c.description.str.contains(p, case=False, regex=True).astype(int)
    for col, mp in ORD.items():                                         # CAREFUL group
        c["cv_" + col] = c[col].map(mp)
    c["cvss3"] = pd.to_numeric(c.cvss3_score)
    c["has_cvss4"] = c.cvss4_score.notna().astype(int)
    c["LEAK_nvd_analyzed"] = c.nvd_status.isin(["Analyzed", "Modified"]).astype(int)
    c["LEAK_n_references"] = pd.to_numeric(c.n_references).fillna(0)
    c["LEAK_n_affected_cpes"] = pd.to_numeric(c.n_affected_cpes).fillna(0)
    return c


def add_siblings(c, gap):
    c["pub_day"] = c.published
    parts = c.cve_id.str.split("-", expand=True)
    c["id_year"], c["id_num"] = parts[1], pd.to_numeric(parts[2])
    key = ["assigner", "pub_day"]
    g = c.groupby(key)
    c["n_same_day"] = g.cve_id.transform("size")
    c["sib_flags"] = g.n_flags.transform("sum") - c.n_flags
    c["_r"] = g.epss_pct.rank(method="first", ascending=False)
    top = c[c._r == 1].set_index(key).epss_pct
    sec = c[c._r == 2].set_index(key).epss_pct
    mi = pd.MultiIndex.from_frame(c[key])
    c["sib_max_epss_pct"] = np.where(c._r == 1, sec.reindex(mi).to_numpy(), top.reindex(mi).to_numpy())
    s = c.sort_values(key + ["id_year", "id_num"])
    adj = pd.Series(0, index=s.index)
    for k in (1, 2, 3):
        for sh in (k, -k):
            o = s.shift(sh)
            adj += ((o.assigner == s.assigner) & (o.pub_day == s.pub_day) & (o.id_year == s.id_year)
                    & ((o.id_num - s.id_num).abs() <= 3)).astype(int)
    c["n_adjacent_ids"] = adj.reindex(c.index)
    # "Already on KEV" CVEs from the same year, only once they were on KEV by this CVE's snapshot.
    m = c[["cve_id", "assigner", "published", "snapshot_end"]].merge(gap, on="assigner", suffixes=("", "_g"))
    m = m[m.kev_by <= m.snapshot_end]
    c["assigner_kev_ytd"] = c.cve_id.map(m.groupby("cve_id").size()).fillna(0)
    near = m[(m.published - m.published_g).abs() <= pd.Timedelta(days=3)]
    c["sib_kev_known"] = c.cve_id.map(near.groupby("cve_id").size()).fillna(0)
    return c.drop(columns=["_r"])


def add_history(c, year):
    start = pd.Timestamp(f"{year}-01-01")
    kev_before = kev[kev.date_added < start]
    kev_ids = set(kev_before.cve_id)
    # assigner: KEV counts and smoothed rate over the 3 years before Y
    win = v[(v.published >= start - pd.DateOffset(years=3)) & (v.published < start)]
    g = win.assign(k=win.cve_id.isin(kev_ids)).groupby("assigner").k.agg(["sum", "size"])
    prior = g["sum"].sum() / g["size"].sum()
    rate = (g["sum"] + prior * 300) / (g["size"] + 300)
    c["assigner_kev_3y"] = c.assigner.map(g["sum"]).fillna(0)
    c["assigner_rate_3y"] = c.assigner.map(rate).fillna(prior)
    c["assigner_volume_3y"] = np.log1p(c.assigner.map(g["size"]).fillna(0))
    kev_ass = kev_before.cve_id.map(v_idx.assigner)
    c["assigner_kev_all"] = c.assigner.map(kev_ass.value_counts()).fillna(0)
    prev = kev_before[kev_before.date_added >= start - pd.DateOffset(years=1)]
    c["assigner_kev_prev_year"] = c.assigner.map(prev.cve_id.map(v_idx.assigner).value_counts()).fillna(0)
    # CWE KEV rate (CAREFUL: NIST can add CWEs during analysis)
    ex = win[["cve_id", "cwe_ids"]].assign(cwe=win.cwe_ids.fillna("").str.split(";")).explode("cwe")
    ex["k"] = ex.cve_id.isin(kev_ids)
    cg = ex.groupby("cwe").k.agg(["sum", "size"])
    cprior = cg["sum"].sum() / cg["size"].sum()
    crate = ((cg["sum"] + cprior * 200) / (cg["size"] + 200)).to_dict()
    c["cwe_kev_rate"] = [max((crate.get(x, cprior) for x in str(s).split(";")), default=cprior) if isinstance(s, str) else np.nan
                         for s in c.cwe_ids]
    # KEV product / vendor names that appear in the description (case-sensitive word n-grams)
    prod = kev_before["product"].dropna().value_counts()
    prod = {k: int(n) for k, n in prod.items() if n >= 3 and len(k) >= 3 and "Multiple" not in k}
    vend = kev_before["vendor"].dropna().value_counts()
    vend = {k: int(n) for k, n in vend.items() if n >= 3 and len(k) >= 3}
    c["kev_product_hist"], c["kev_product_hits"] = ngram_lookup(c.description.tolist(), prod)
    c["kev_vendor_hist"], _ = ngram_lookup(c.description.tolist(), vend, max_n=2)
    return c


def add_text_model(c, year):
    """TF-IDF + logistic regression trained on KEV descriptions known before Jan 1 of Y,
    plus the max cosine similarity to any of those KEV descriptions (baseline for Granite embeddings)."""
    start = pd.Timestamp(f"{year}-01-01")
    pos_ids = set(kev[kev.date_added < start].cve_id)
    base = v[(v.published >= "2015-01-01") & (v.published < start)]
    pos = base[base.cve_id.isin(pos_ids)]
    neg = base[~base.cve_id.isin(pos_ids)].sample(n=min(40000, len(base)), random_state=0)
    corpus = pd.concat([pos.description, neg.description])
    yv = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=150000, sublinear_tf=True, stop_words="english")
    Xt = vec.fit_transform(corpus)
    lr = LogisticRegression(C=2.0, class_weight="balanced", max_iter=3000)
    lr.fit(Xt, yv)
    Xc = vec.transform(c.description)
    c["text_kev_lr"] = lr.predict_proba(Xc)[:, 1]
    P = Xt[: len(pos)].T.tocsc()
    sim = np.zeros(Xc.shape[0])
    for i in range(0, Xc.shape[0], 4000):
        sim[i:i + 4000] = (Xc[i:i + 4000] @ P).max(axis=1).toarray().ravel()
    c["text_kev_sim"] = sim
    log(f"  text model {year}: {len(pos):,} KEV positives, {len(neg):,} negatives")
    return c


KEEP_ID = ["cve_id", "snapshot_month", "snapshot_end", "published", "assigner", "cwe_ids"]
FEATURES = {
    # name: (group, risk, description)
    "epss": ("signals", "safe", "EPSS at the end of snapshot_month (0 if blank)"),
    "epss_pct": ("signals", "safe", "EPSS percentile within the snapshot month (comparable across the 2025 model change)"),
    "epss_missing": ("signals", "safe", "1 if there was no EPSS score that month"),
    **{f: ("signals", "safe", f"1 if {f} code existed at the snapshot") for f in FLAGS},
    "n_flags": ("signals", "safe", "Number of exploit flags"),
    "any_flag": ("signals", "safe", "1 if any exploit flag"),
    "age_days": ("time", "safe", "Days between publication and the end of snapshot_month"),
    "desc_len": ("text", "safe", "Description length in characters"),
    **{k: ("text", "safe", f"Description matches /{p}/") for k, p in TEXT_FLAGS.items()},
    "text_kev_lr": ("text", "safe", "TF-IDF logistic model: how much the description reads like past KEV entries"),
    "text_kev_sim": ("text", "safe", "Max cosine similarity to any past KEV description"),
    "n_same_day": ("siblings", "safe", "CVEs in the cohort from the same assigner on the same day"),
    "sib_flags": ("siblings", "safe", "Exploit flags among same-day siblings"),
    "sib_max_epss_pct": ("siblings", "safe", "Highest EPSS percentile among same-day siblings"),
    "n_adjacent_ids": ("siblings", "safe", "Same-day siblings within 3 ID numbers"),
    "assigner_kev_ytd": ("kev_now", "careful", "Same-year CVEs by this assigner already on KEV by the snapshot (2026: inferred from CVEs missing from the scoreboard)"),
    "sib_kev_known": ("kev_now", "careful", "Same-assigner CVEs published within 3 days that were already on KEV by the snapshot"),
    "assigner_kev_3y": ("history", "safe", "KEV entries among this assigner's CVEs from the previous 3 years"),
    "assigner_rate_3y": ("history", "safe", "Smoothed KEV rate of this assigner over the previous 3 years"),
    "assigner_volume_3y": ("history", "safe", "log(1 + CVEs published by this assigner in the previous 3 years)"),
    "assigner_kev_all": ("history", "safe", "All KEV entries before Jan 1 of the year whose CVE has this assigner"),
    "assigner_kev_prev_year": ("history", "safe", "KEV entries added in the previous 12 months for this assigner"),
    "kev_product_hist": ("history", "safe", "Max past KEV count of a KEV product name found in the description"),
    "kev_product_hits": ("history", "safe", "Distinct past KEV product names found in the description"),
    "kev_vendor_hist": ("history", "safe", "Max past KEV count of a KEV vendor name found in the description"),
    **{"cv_" + k: ("cvss", "careful", f"CVSS 3 {k} as an ordinal") for k in ORD},
    "cvss3": ("cvss", "careful", "CVSS 3 base score"),
    "has_cvss4": ("cvss", "careful", "1 if a CVSS 4 score exists"),
    "cwe_kev_rate": ("cvss", "careful", "Highest smoothed KEV rate among the CVE's CWEs (previous 3 years)"),
    "LEAK_nvd_analyzed": ("leak", "leak", "nvd_status is Analyzed or Modified. Leaks the answer"),
    "LEAK_n_references": ("leak", "leak", "n_references. Leaks the answer"),
    "LEAK_n_affected_cpes": ("leak", "leak", "n_affected_cpes. Leaks the answer"),
}


def build(c, gap, year):
    c = add_basic(c)
    c = add_siblings(c, gap)
    c = add_history(c, year)
    c = add_text_model(c, year)
    labels = [x for x in ("y", "y_nov4") if x in c.columns]
    leak = [f for f in FEATURES if FEATURES[f][1] == "leak"]
    clean = [f for f in FEATURES if f not in leak]
    out = c[KEEP_ID + labels + clean]                       # leak-free by construction
    out.to_csv(os.path.join(OUT, f"features_{year}.csv"), index=False)
    c[["cve_id"] + leak].to_csv(os.path.join(OUT, f"leak_only_{year}.csv"), index=False)   # demo only
    pos = f", positives y={int(out.y.sum())}, y_nov4={int(out.y_nov4.sum())}" if "y" in out else ""
    log(f"features_{year}.csv rows={len(out):,} gap(already on KEV)={len(gap)}{pos}")


pd.DataFrame([(k, *v_) for k, v_ in FEATURES.items()], columns=["feature", "group", "risk", "description"]) \
    .to_csv(os.path.join(OUT, "feature_dictionary.csv"), index=False)

c24, g24 = replica_cohort(2024, [f"2024-{i:02d}" for i in range(1, 12)], "2024-12-31", "exploit_signals_2024.csv")
build(c24, g24, 2024)
c25, g25 = replica_cohort(2025, [f"2025-{i:02d}" for i in range(1, 12)], "2025-12-31", "exploit_signals_2025.csv")
build(c25, g25, 2025)
c26, g26 = cohort_2026()
build(c26, g26, 2026)
log("done")
