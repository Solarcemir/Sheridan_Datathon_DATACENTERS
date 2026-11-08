"""One command to build the team's final 2026 ranking: luis/submissions/top500_final.csv.

Voters (equal-weight rank average), chosen from the A/B results and the statistical review (context_Luis.md §14.8-14.13):
  tree    XGBoost big on Luis's features, 40 seeds averaged (8 GPU runs x 5 seeds) for a stable top-500 boundary
  linear  Luis's LR (safe + Chen-lite features). If Chen pushed his best LR's 2026 scores, the linear voter is the average
          of both LRs (Chen's uses CVSS + Granite 30B features: a hedge against the 2026 shift where CVSS-first beats EPSS-first)
  LLM     Chen's Granite 4.2 8B LoRA, if he pushed its 2026 scores (the only addition that survives Holm correction)
Then the 4 Granite-30B demotion filters (validated on training years; §14.10) and a cap of 3 per assigner-month.

Optional inputs from Chen (columns exactly as listed; higher score = more likely to reach KEV):
  chen/submissions/lora8b_scoreboard_2026.parquet   cve_id, lora_score
  chen/submissions/lr_best_scoreboard_2026.parquet  cve_id, lr_score
  chen/submissions/lr_unified_scoreboard_2026.parquet  cve_id, lr_score   (if present: the ONLY linear voter; §14.16)
Outputs: derived/xgb_bag_2026.parquet (cache), derived/scored_2026_final.csv, submissions/top500_final.csv,
         reports/final_build_info.txt, reports/top500_final_explained.csv, site/top500.json (hub table)

Usage:  python -E -P luis/analysis/final_build.py
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from common import CHEN, DER, FULL, REP, SUB, XGB_BIG, capped, load, rank01  # noqa: E402

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- tree voter: bagged XGBoost (cached)
s26 = load(2026, "2026-11-04")
cache = os.path.join(DER, "xgb_bag_2026.parquet")
if os.path.exists(cache):
    bag = pd.read_parquet(cache)
else:
    train = pd.concat([load(2024, "2024-12-31"), load(2025, "2025-12-31")], ignore_index=True)
    preds = []
    for r in range(8):
        for s in range(5):
            m = xgb.XGBClassifier(tree_method="hist", device="cuda", scale_pos_weight=25, eval_metric="aucpr",
                                  random_state=1000 * r + s, **XGB_BIG).fit(train[FULL], train.y)
            preds.append(m.predict_proba(s26[FULL])[:, 1])
        log(f"XGBoost run {r + 1}/8 done")
    bag = pd.DataFrame({"cve_id": s26.cve_id, "xgb_bag": np.mean(preds, axis=0)})
    bag.to_parquet(cache, index=False)
s26 = s26.merge(bag, on="cve_id", how="left")
s26["tree"] = rank01(s26.xgb_bag)

# ----------------------------------------------------------------------------- linear voter
v4 = pd.read_csv(os.path.join(DER, "scored_2026_v4.csv"), usecols=["cve_id", "lr_pct"])   # Luis's LR, same recipe as v4
s26 = s26.merge(v4, on="cve_id", how="left")
chen_lr_path = os.path.join(CHEN, "submissions", "lr_best_scoreboard_2026.parquet")
unified_path = os.path.join(CHEN, "submissions", "lr_unified_scoreboard_2026.parquet")   # context_Luis.md §14.16
lora_path = os.path.join(CHEN, "submissions", "lora8b_scoreboard_2026.parquet")
used = ["tree: XGBoost big, 40 seeds averaged"]
if os.path.exists(unified_path):   # Chen pushes it only if it passed the decision rule in §14.16
    u = pd.read_parquet(unified_path)[["cve_id", "lr_score"]].rename(columns={"lr_score": "lr_unified"})
    s26 = s26.merge(u, on="cve_id", how="left")
    s26["linear"] = rank01(s26.lr_unified.fillna(s26.lr_unified.min()))
    used.append(f"linear: one unified LR with Luis's and Chen's features ({u.lr_unified.notna().sum():,} scores)")
elif os.path.exists(chen_lr_path):
    c = pd.read_parquet(chen_lr_path)[["cve_id", "lr_score"]]
    s26 = s26.merge(c, on="cve_id", how="left")
    s26["linear"] = (rank01(s26.lr_pct) + rank01(s26.lr_score.fillna(s26.lr_score.min()))) / 2
    used.append(f"linear: average of Luis's LR and Chen's best LR ({c.lr_score.notna().sum():,} scores)")
else:
    s26["linear"] = rank01(s26.lr_pct)
    used.append("linear: Luis's LR (Chen's LR file not found)")
voters = ["tree", "linear"]
if os.path.exists(lora_path):
    lo = pd.read_parquet(lora_path)[["cve_id", "lora_score"]]
    s26 = s26.merge(lo, on="cve_id", how="left")
    s26["llm"] = rank01(s26.lora_score.fillna(s26.lora_score.min()))
    voters.append("llm")
    used.append(f"LLM: Chen's Granite 4.2 8B LoRA ({lo.lora_score.notna().sum():,} scores)")
else:
    used.append("LLM: Chen's LoRA file not found, so not included yet")

# ----------------------------------------------------------------------------- Granite filters + cap
ex = pd.read_parquet(os.path.join(CHEN, "processed", "llm_extract.parquet"))[["cve_id", "impact", "typical_attacker", "attacker_appeal", "product_type"]]
s26 = s26.merge(ex, on="cve_id", how="left")
s26["granite_demoted"] = ((s26.impact == "denial_of_service") | (s26.typical_attacker == "unlikely_target") | (s26.attacker_appeal == "low")
                          | (s26.product_type == "wordpress_plugin_or_theme")).fillna(False)
raw = s26[voters].mean(axis=1).to_numpy() - 50 * s26.granite_demoted.to_numpy()
s26["score_final"] = capped(s26, raw, 3)
s26 = s26.sort_values("score_final", ascending=False, kind="stable").reset_index(drop=True)
s26["rank_final"] = np.arange(1, len(s26) + 1)
s26[["rank_final", "cve_id", "score_final"] + voters + ["granite_demoted", "snapshot_month", "assigner", "epss", "any_flag"]] \
    .to_csv(os.path.join(DER, "scored_2026_final.csv"), index=False)
top = s26.head(500)
top[["cve_id"]].to_csv(os.path.join(SUB, "top500_final.csv"), index=False)

v4f = set(pd.read_csv(os.path.join(SUB, "top500_v4f.csv")).cve_id)
mix = top.snapshot_month.value_counts().sort_index().to_dict()
info = [f"top500_final built {time.strftime('%Y-%m-%d %H:%M')}", "voters (equal-weight rank average): " + ", ".join(voters)] + \
       [f"  - {u}" for u in used] + \
       ["then: 4 Granite-30B demotion filters (DoS only, unlikely target, low appeal, WordPress), cap 3 per assigner-month",
        f"rows {len(top)} | unique {top.cve_id.nunique()} | with exploit code {int(top.any_flag.sum())} | "
        f"Oracle {int((top.assigner == 'secalert_us@oracle.com').sum())} | Granite-demoted inside top 500 {int(top.granite_demoted.sum())} | "
        f"assigners {top.assigner.nunique()} | max per assigner-month {int(top.groupby(['assigner', 'snapshot_month']).size().max())}",
        f"month mix {mix}", f"overlap with top500_v4f.csv: {len(v4f & set(top.cve_id))}"]
open(os.path.join(REP, "final_build_info.txt"), "w", encoding="utf-8").write("\n".join(info) + "\n")
print("\n".join(info))

# ----------------------------------------------------------------------------- explanations + hub data
py = [sys.executable, "-E", "-P"]
subprocess.run(py + [os.path.join(HERE, "team_merge.py"), "--score-2026", "--ranking", "derived/scored_2026_final.csv", "rank_final", "final"],
               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
label = "final · " + " + ".join({"tree": "XGBoost", "linear": "LR", "llm": "Granite LoRA"}[v] for v in voters) + " · Granite filters · cap 3"
note = ("The 500 CVEs from the 2026 scoreboard the team would patch first. Ranking: equal-weight rank average of "
        + ", ".join(used) + "; CVEs that IBM Granite 4.2 30B reads as denial-of-service only, unlikely targets, low attacker appeal or "
        "WordPress plugins pushed down; at most 3 per vendor per month. Product, type and reason come from Granite 30B's reading of "
        "each description plus the most similar past KEV entry (Granite Embedding r2). Expected on 2026: about 12 hits "
        "(80% interval 7-22) vs 8 for the best README baseline.")
subprocess.run(py + [os.path.join(HERE, "export_top500_json.py"), label, "reports/top500_final_explained.csv", note], check=True)
log("done")
