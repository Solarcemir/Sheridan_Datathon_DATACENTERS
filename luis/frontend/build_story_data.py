"""Build the data file for the Breach Story frontend: luis/frontend/story_data.json.

Reads only derived team files plus aggregates of the local IBM data (counts, shares, word lifts). No raw IBM rows
are written except the 500 CVE ids of our own top 500 with their explanations (already public in the team hub).

Inputs:  luis/derived/scored_2026_final.csv, luis/reports/top500_final_explained.csv, luis/reports/lr_columns_luis_chen.csv,
         luis/reports/univariate_lift.csv, luis/reports/granite_clusters_2026.csv, chen/processed/llm_extract.parquet,
         IBM data at the repo root (scoreboard_2026.csv, vulnerabilities.csv, known_exploited.csv)
Usage:   python -E -P luis/frontend/build_story_data.py
"""
import glob
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LUIS = os.path.dirname(HERE)
REPO = os.path.dirname(LUIS)
DATA = glob.glob(os.path.join(REPO, "IBM-*", "IBM", "data", "data"))[0]
OUT = os.path.join(HERE, "story_data.json")


def pct(x, d=1):
    return round(100 * float(x), d)


s = pd.read_csv(os.path.join(LUIS, "derived", "scored_2026_final.csv"))
top = s.head(500).copy()
ex = pd.read_parquet(os.path.join(REPO, "chen", "processed", "llm_extract.parquet")).drop_duplicates("cve_id").set_index("cve_id")
sb = pd.read_csv(os.path.join(DATA, "scoreboard_2026.csv"))
FLAGS = ["exploitdb", "metasploit", "nuclei", "poc_github"]
N = len(sb)

# ----------------------------------------------------------------------------- before the model: the raw signal
anyf = sb[FLAGS].max(axis=1) > 0
pre = {
    "n_scoreboard": N,
    "share_any_flag": pct(anyf.mean()),
    "share_no_flag_low_epss": pct(((~anyf) & (sb.epss.fillna(0) < 0.01)).mean()),
    "share_epss_ge_0_1": pct((sb.epss.fillna(0) >= 0.1).mean()),
    "median_epss": round(float(sb.epss.median()), 4),
    "flags": {f: int(sb[f].sum()) for f in FLAGS},
    "months": sb.snapshot_month.value_counts().sort_index().to_dict(),
    "granite_demoted_share": pct(s.granite_demoted.mean()),
}

# where the signal lives: univariate AUC per feature (training replicas), leak-free features only
u = pd.read_csv(os.path.join(LUIS, "reports", "univariate_lift.csv"))
u = u[u.risk == "safe"].sort_values("univariate_auc", ascending=False)
pre["univariate"] = [{"feature": r.feature, "group": r.group, "auc": round(float(r.univariate_auc), 3),
                      "lift_top5": round(float(r.lift_top5pct), 1)} for r in u.head(14).itertuples()]
pre["group_best_auc"] = (u.groupby("group").univariate_auc.max().sort_values(ascending=False).round(3).to_dict())

# Granite Embedding clusters: technologies that past KEV entries concentrate in
c = pd.read_csv(os.path.join(LUIS, "reports", "granite_clusters_2026.csv")).sort_values("kev_rate_smoothed", ascending=False)
pre["clusters"] = [{"id": int(r.cluster), "kev_rate": pct(r.kev_rate_smoothed), "kev_refs": int(r.kev_refs),
                    "cves_2026": int(r.scoreboard_2026_cves),
                    "terms": [t.strip() for t in str(r.top_terms).split(",")][:5],
                    "products": [t.strip() for t in str(r.top_kev_products).split(",")][:3]} for r in c.head(10).itertuples()]

# KEV catalog: who gets exploited
k = pd.read_csv(os.path.join(DATA, "known_exploited.csv"))
k["year"] = pd.to_datetime(k.date_added).dt.year
pre["kev_total"] = len(k)
pre["kev_vendors"] = k.vendor.value_counts().head(10).to_dict()
pre["kev_ransomware_share"] = pct((k.known_ransomware_use == "Known").mean())
pre["kev_by_year"] = {int(y): int(n) for y, n in k.year.value_counts().sort_index().items()}

# ----------------------------------------------------------------------------- after the model: our top 500
def share(col, ids):
    v = ex.reindex(ids)[col].fillna("unknown")
    return v.value_counts(normalize=True)


post = {}
for col in ("product_type", "impact", "typical_attacker"):
    a, b = share(col, top.cve_id), share(col, s.cve_id)
    keys = [x for x in a.index if x != "unknown"][:9]
    post[col] = [{"key": x, "top": pct(a.get(x, 0)), "all": pct(b.get(x, 0)),
                  "lift": round(float(a.get(x, 0) / b.get(x, 1e-9)), 1)} for x in keys]
vend = ex.reindex(top.cve_id).vendor.fillna("unknown").str.strip()
vend = vend[~vend.str.lower().isin(["unknown", "", "n/a", "none"])]
post["vendors"] = vend.value_counts().head(12).to_dict()
post["months"] = {"top": top.snapshot_month.value_counts().sort_index().to_dict(), "all": pre["months"]}
post["exploit_code"] = {"top": int(top.any_flag.sum()), "all_share": pre["share_any_flag"]}
bins = [-1, 0.001, 0.01, 0.1, 2]
lab = ["< 0.001", "0.001-0.01", "0.01-0.1", ">= 0.1"]
post["epss_buckets"] = {"labels": lab,
                        "top": pd.cut(top.epss.fillna(0), bins, labels=lab).value_counts().reindex(lab).astype(int).tolist(),
                        "all": [pct(x) for x in pd.cut(s.epss.fillna(0), bins, labels=lab).value_counts(normalize=True).reindex(lab)]}

# voters: who put each CVE into the top 500 (each voter's own top 500 over the full scoreboard)
for v in ("tree", "linear", "llm"):
    s[f"in_{v}"] = s[v].rank(ascending=False, method="first") <= 500
top = s.head(500).copy()
combo = Counter(tuple(int(x) for x in r) for r in top[["in_tree", "in_linear", "in_llm"]].itertuples(index=False))
post["voters"] = {"+".join(n for n, f in zip(("tree", "linear", "llm"), key) if f) or "none": int(v) for key, v in combo.items()}
post["voter_corr"] = s[["tree", "linear", "llm"]].corr(method="spearman").round(2).to_dict()

# post-processing effects: what the Granite filters and the cap changed
raw = s[["tree", "linear", "llm"]].mean(axis=1)
nofilt = set(s.assign(r=raw).nlargest(500, "r").cve_id)
filt = s.assign(r=raw - 50 * s.granite_demoted).nlargest(500, "r")
post["filters_pushed_out"] = int(s[s.cve_id.isin(nofilt)].granite_demoted.sum())
precap = set(filt.cve_id)
post["cap_pushed_out"] = len(precap - set(top.cve_id))
over = filt.groupby(["assigner", "snapshot_month"]).size().sort_values(ascending=False)
post["cap_biggest_batches"] = [{"assigner": a, "month": m, "would_have": int(n)} for (a, m), n in over.head(5).items() if n > 3]

# words that our top 500 over-represents (description bigrams vs the whole scoreboard)
v = pd.read_csv(os.path.join(DATA, "vulnerabilities.csv"), usecols=["cve_id", "description"]).set_index("cve_id")
STOP = set("a an the of in to and or for by via on with is be can that this from as at an allows allow could may which "
           "it its are was were has have been vulnerability issue versions version prior before through when affected".split())


def bigrams(t):
    w = [x for x in re.findall(r"[a-z][a-z0-9\-]+", str(t).lower())]
    return {f"{a} {b}" for a, b in zip(w, w[1:]) if a not in STOP and b not in STOP}


all_c, top_c = Counter(), Counter()
tops = set(top.cve_id)
for cid, d in v.reindex(s.cve_id).description.items():
    g = bigrams(d)
    all_c.update(g)
    if cid in tops:
        top_c.update(g)
rows = [(b, n, n / 500 / (all_c[b] / N)) for b, n in top_c.items() if n >= 12 and all_c[b] >= 20]
post["bigrams"] = [{"term": b, "top": n, "all": all_c[b], "lift": round(l, 1)} for b, n, l in sorted(rows, key=lambda r: -r[2])[:14]]
post["nearest_kev_ge_085"] = None  # filled below

# ----------------------------------------------------------------------------- feature weights (both LRs)
w = pd.read_csv(os.path.join(LUIS, "reports", "lr_columns_luis_chen.csv"))
weights = [{"lr": r.lr, "f": r.feature, "g": r.group, "c": float(r.coef_per_sd),
            "d": "" if r.lr == "Chen" else str(r.definition), "eq": "" if str(r.closest_in_other_lr) == "none" else str(r.closest_in_other_lr)}
           for r in w.itertuples()]

# ----------------------------------------------------------------------------- top 500 table
t = pd.read_csv(os.path.join(LUIS, "reports", "top500_final_explained.csv"))
t = t.merge(s[["cve_id", "tree", "linear", "llm", "in_tree", "in_linear", "in_llm"]], on="cve_id", how="left")
t = t.join(ex[["impact", "typical_attacker", "attacker_appeal", "vendor"]], on="cve_id")
post["nearest_kev_ge_085"] = int((t.kev_sim_max >= 0.85).sum())
rows = []
for r in t.itertuples():
    rows.append({"r": int(r.rank), "id": r.cve_id, "m": r.snapshot_month, "p": "" if str(r.product) in ("unknown", "nan") else str(r.product),
                 "cat": "" if pd.isna(r.category) else str(r.category), "imp": "" if pd.isna(r.impact) else str(r.impact).replace("_", " "),
                 "att": "" if pd.isna(r.typical_attacker) else str(r.typical_attacker).replace("_", " "),
                 "e": round(float(r.epss), 4) if pd.notna(r.epss) else None,
                 "fl": "".join(str(int(getattr(r, f))) for f in FLAGS),
                 "tx": round(float(r.text_kev_lr), 3), "nk": "" if pd.isna(r.nearest_kev) else r.nearest_kev,
                 "ks": round(float(r.kev_sim_max), 2) if pd.notna(r.kev_sim_max) else None,
                 "v": [round(float(r.tree), 3), round(float(r.linear), 3), round(float(r.llm), 3)],
                 "why": r.reason})

out = {"built": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "pre": pre, "post": post, "weights": weights, "top500": rows}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(f"wrote {OUT}: {os.path.getsize(OUT) / 1024:.0f} KB")
print(json.dumps({k: v for k, v in pre.items() if k not in ("univariate", "clusters")}, indent=0, default=str)[:1500])
print(json.dumps({k: v for k, v in post.items() if k not in ("months",)}, indent=0, default=str)[:3500])
