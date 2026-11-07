"""Combine Luis's and Chen's approaches on Luis's A/B replicas (the forward test), then score 2026.

Chen's inputs (chen/processed/*.parquet, chen/results/*.parquet):
  meta_{historical,scoreboard}     31 metadata features (assigner KEV rate, release size, twins, text flags, CVSS 4 parts...)
  llm_feat_{historical,scoreboard} 55 features from the Granite 4.2 30B world-knowledge extraction (2024-01..2025-09 + 2026)
  emb_feat_{historical,scoreboard} Granite Embedding r2 nearest-KEV similarity (+ the nearest KEV CVE, for explanations)
  llm_extract                      raw Granite 30B answers (vendor, product, type, exposure, attacker...), for explanations
  cv_oof_lora / cv_oof_avg         his out-of-fold LoRA 8B scores and his best 3-model average (5-fold purged CV)
Models on A/B: Luis's XGBoost big and LightGBM, Chen-style LR (his base + meta + Luis TF-IDF + Granite 30B features, about
his 100-feature "best LR"), the same LR plus Luis's safe features, XGBoost with Chen's features, and rank averages.
LoRA is added from Chen's out-of-fold scores (trained in his 5-fold CV, so a different regime; flagged in the tables).

Outputs: reports/team_merge_results.csv, reports/team_merge_bootstrap.csv, reports/team_merge_log.txt (through Tee)
With --score-2026: reports/top500_v4_explained.csv (v4 ranking explained with Granite 30B fields + nearest KEV). If Chen has
pushed chen/submissions/lora8b_scoreboard_2026.parquet (cve_id, lora_score), it builds v5 = v4 components + LoRA, cap 3:
derived/scored_2026_v5.csv, submissions/top500_v5.csv, reports/top500_v5_explained.csv

Usage:  python -E -P luis/analysis/team_merge.py [--score-2026]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -P drops the script dir from sys.path
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import (CHEN, DER, FULL, REP, SAFE, SUB, CHEN_LITE, capped, density_k, hits, jan_sep, load, m_lgbm,  # noqa: E402
                    m_lr, m_xgb, pooled_eval, rank01)

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


def cp(name):
    return pd.read_parquet(os.path.join(CHEN, name))


META = {k: cp(f"processed/meta_{k}.parquet") for k in ("historical", "scoreboard")}
LLMF = {k: cp(f"processed/llm_feat_{k}.parquet") for k in ("historical", "scoreboard")}
EMB = {k: cp(f"processed/emb_feat_{k}.parquet") for k in ("historical", "scoreboard")}
META_COLS = [c for c in META["historical"].columns if c != "cve_id" and pd.api.types.is_numeric_dtype(META["historical"][c])]
LLM_COLS = [c for c in LLMF["historical"].columns if c != "cve_id"]
EMB_COLS = ["kev_sim_max", "kev_sim_top5", "typ_sim_top5", "kev_contrast"]
CM = ["cm_" + c for c in META_COLS]


def add_chen_full(df, kind):
    m = META[kind][["cve_id"] + META_COLS].rename(columns={c: "cm_" + c for c in META_COLS})
    df = df.merge(m, on="cve_id", how="left").merge(LLMF[kind], on="cve_id", how="left") \
           .merge(EMB[kind][["cve_id", "nearest_kev"] + EMB_COLS], on="cve_id", how="left")
    # Chen stores some flags as bool; rows missing from his files turn them into object. Make everything numeric.
    num = CM + LLM_COLS + EMB_COLS
    df[num] = df[num].apply(lambda s: pd.to_numeric(s.map(lambda x: float(x) if isinstance(x, (bool, np.bool_)) else x), errors="coerce"))
    # Chen's 15 base features, rebuilt from Luis's columns
    df["b_av_net"] = (df.cv_attack_vector == 3).astype(int)
    df["b_ac_low"] = (df.cv_attack_complexity == 1).astype(int)
    df["b_pr_none"] = (df.cv_privileges_required == 2).astype(int)
    df["b_ui_none"] = (df.cv_user_interaction == 1).astype(int)
    df["b_scope"] = (df.cv_scope == 1).astype(int)
    df["b_cvss"] = df.cvss3
    df["b_cvss_missing"] = df.cvss3.isna().astype(int)
    p = df.text_kev_lr.clip(1e-6, 1 - 1e-6)
    df["text_kev_logit"] = np.log(p / (1 - p))           # Chen feeds the TF-IDF score to the LR as a logit
    return df


BASE = ["exploitdb", "metasploit", "nuclei", "poc_github", "any_flag", "epss_pct", "epss_missing", "b_cvss", "b_cvss_missing",
        "b_av_net", "b_ac_low", "b_pr_none", "b_ui_none", "b_scope", "wp_cna"]
TEAM = ["text_kev_logit", "text_kev_sim", "n_adjacent_ids"]
LR_CHEN = BASE + CM + TEAM + LLM_COLS                                   # ~ Chen's best LR (100 features)
LR_CHEN_LUIS = LR_CHEN + [f for f in SAFE if f not in LR_CHEN]          # ~ his "LR best + all of Luis's safe features"
XGB_CHEN = FULL + CM + LLM_COLS

tr24 = add_chen_full(load(2024, "2024-12-31"), "historical")
tr25 = add_chen_full(load(2025, "2025-12-31"), "historical")
te25 = add_chen_full(jan_sep(load(2025, "2025-11-04"), 2025), "historical")
te24 = add_chen_full(jan_sep(load(2024, "2024-11-04"), 2024), "historical")
SPLITS = {"A": (tr24, te25), "B": (tr25, te24)}
cov = {k: round(float(t[LLM_COLS[0]].notna().mean()), 3) for k, t in (("train24", tr24), ("train25", tr25), ("test25", te25), ("test24", te24))}
log(f"Chen features merged: {len(CM)} meta, {len(LLM_COLS)} Granite-30B, {len(EMB_COLS)} embedding; Granite coverage {cov}")

LORA_2026 = os.path.join(CHEN, "submissions", "lora8b_scoreboard_2026.parquet")   # expected from Chen: cve_id, lora_score

if "--score-2026" in sys.argv:
    # A/B result (team_merge_bootstrap.csv): without LoRA, v4 (XGBoost big + LR(Luis + Chen-lite), cap 3) stays best;
    # adding Chen's LoRA 8B adds +2..+4 hits to every ensemble. So: v4's 2026 components (+ LoRA if Chen's file exists).
    s26 = add_chen_full(load(2026, "2026-11-04"), "scoreboard")
    v4 = pd.read_csv(os.path.join(DER, "scored_2026_v4.csv"), usecols=["cve_id", "xgb_pct", "lr_pct"])
    s26 = s26.merge(v4, on="cve_id", how="left")
    parts = ["xgb_pct", "lr_pct"]
    if os.path.exists(LORA_2026):
        lo = pd.read_parquet(LORA_2026)[["cve_id", "lora_score"]]
        s26 = s26.merge(lo, on="cve_id", how="left")
        s26["lora_pct"] = rank01(s26.lora_score.fillna(s26.lora_score.min()))
        parts.append("lora_pct")
        VER = "v5"
    else:
        VER = "v4"
    log(f"2026 ranking {VER}: rank average of {parts}, cap 3" + ("" if VER == "v5" else f" (no Chen LoRA file at {LORA_2026})"))
    s26["score"] = capped(s26, s26[parts].sum(axis=1), 3)
    s26 = s26.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    s26["rank"] = np.arange(1, len(s26) + 1)
    C = "from v4"
    if VER == "v5":
        s26[["rank", "cve_id", "score"] + parts + ["snapshot_month", "assigner", "epss", "any_flag", "wp_cna"]] \
            .to_csv(os.path.join(DER, "scored_2026_v5.csv"), index=False)
        s26.head(500)[["cve_id"]].to_csv(os.path.join(SUB, "top500_v5.csv"), index=False)

    # Explanations from Granite 30B's answers + nearest KEV entry (Granite Embedding r2) + the signals at the snapshot
    ex = cp("processed/llm_extract.parquet").set_index("cve_id")
    t = s26.head(500).copy().join(ex, on="cve_id", rsuffix="_g")
    h = lambda s: "" if pd.isna(s) else str(s).replace("_", " ")  # noqa: E731

    def product(r):
        v, p = h(r.vendor), h(r["product"])
        return p if (not v or v.lower() in p.lower()) else f"{v} {p}"

    def reason(r):
        parts = []
        code = [n for f, n in (("exploitdb", "Exploit-DB"), ("metasploit", "Metasploit"), ("nuclei", "Nuclei"), ("poc_github", "GitHub PoC")) if r[f]]
        if code:
            parts.append("public exploit code (" + ", ".join(code) + ")")
        if r.epss_pct >= 0.95:
            parts.append(f"EPSS in the top {max(1, round((1 - r.epss_pct) * 100))}% of its month")
        if h(r.impact) and r.impact != "other":
            parts.append(h(r.impact) + (" with no login needed" if r.precondition == "none" else ""))
        if r.attacker_appeal == "high":
            parts.append(f"high attacker appeal ({h(r.typical_attacker)})")
        if r.end_of_life == "yes":
            parts.append("end-of-life product")
        if pd.notna(r.kev_sim_max) and r.kev_sim_max >= 0.85 and pd.notna(r.nearest_kev):
            parts.append(f"resembles KEV entry {r.nearest_kev} (similarity {r.kev_sim_max:.2f})")
        if pd.notna(r.cm_assigner_kev) and r.cm_assigner_kev >= 3:
            parts.append(f"{int(r.cm_assigner_kev)} past KEV entries from this assigner")
        s = "; ".join(parts) or "ranked high by the combined model"
        return s[0].upper() + s[1:] + "."

    t["product_name"] = t.apply(product, axis=1)
    t["category"] = t.product_type.map(h)
    t["granite"] = t.apply(lambda r: " · ".join(x for x in (h(r.product_type), f"internet: {h(r.internet_exposed)}" if h(r.internet_exposed) else "",
                                                           f"appeal: {h(r.attacker_appeal)}" if h(r.attacker_appeal) else "",
                                                           f"attacker: {h(r.typical_attacker)}" if h(r.typical_attacker) else "") if x), axis=1)
    t["reason"] = t.apply(reason, axis=1)
    out = t.rename(columns={"product_name": "product_label"})
    out = out.assign(product=out.product_label)[["rank", "cve_id", "snapshot_month", "assigner", "product", "category", "granite",
                                                  "epss", "exploitdb", "metasploit", "nuclei", "poc_github", "text_kev_lr",
                                                  "nearest_kev", "kev_sim_max", "reason"]]
    out.to_csv(os.path.join(REP, f"top500_{VER}_explained.csv"), index=False)
    v4 = set(pd.read_csv(os.path.join(SUB, "top500_v4.csv")).cve_id)
    log(f"top500_{VER}: overlap with top500_v4.csv = {len(v4 & set(t.cve_id))}, exploit flag = {int(t.any_flag.sum())}, "
        f"Oracle = {int((t.assigner == 'secalert_us@oracle.com').sum())}, WordPress CNAs = {int(t.wp_cna.sum())}, "
        f"assigners = {t.assigner.nunique()}, Granite fields missing = {int(t.product_type.isna().sum())}")
    print(t.snapshot_month.value_counts().sort_index().to_dict())
    print(out.head(8)[["rank", "cve_id", "product", "category", "reason"]].to_string(index=False))
    raise SystemExit(0)

# ----------------------------------------------------------------------------- A/B
lora = cp("results/cv_oof_lora.parquet")[["cve_id", "lora_control"]]
chen_best = cp("results/cv_oof_avg.parquet")[["cve_id", "avg_lrteam_lora_lgbm"]]
sc = {}
for name, (tr, te) in SPLITS.items():
    te = te.merge(lora, on="cve_id", how="left").merge(chen_best, on="cve_id", how="left")
    SPLITS[name] = (tr, te)
    R = {}
    sc[("Exploit code, then EPSS", name)] = (te.any_flag * 10 + te.epss).to_numpy()
    R["xgb"] = rank01(m_xgb(tr, te, FULL))
    R["lgbm"] = rank01(m_lgbm(tr, te, SAFE))
    R["lr_lite"] = rank01(m_lr(tr, te, SAFE + CHEN_LITE))
    log(f"{name}: Luis models done")
    R["lr_chen"] = rank01(m_lr(tr, te, LR_CHEN))
    R["lr_chen_luis"] = rank01(m_lr(tr, te, LR_CHEN_LUIS))
    R["xgb_chen"] = rank01(m_xgb(tr, te, XGB_CHEN))
    log(f"{name}: Chen-feature models done")
    R["lora"] = rank01(te.lora_control.fillna(te.lora_control.min()))
    for k, parts in {
        "XGBoost big (Luis)": ["xgb"],
        "LR, Chen-style (base + meta + TF-IDF + Granite 30B)": ["lr_chen"],
        "LR, Chen-style + Luis safe features": ["lr_chen_luis"],
        "XGBoost big with Chen features": ["xgb_chen"],
        "LoRA 8B alone (Chen OOF)": ["lora"],
        "v4: XGBoost + LR(Luis + Chen-lite)  [reference]": ["xgb", "lr_lite"],
        "XGBoost + LR(Chen-style)": ["xgb", "lr_chen"],
        "XGBoost + LR(Chen-style + Luis)": ["xgb", "lr_chen_luis"],
        "LightGBM + LR(Chen-style)": ["lgbm", "lr_chen"],
        "XGBoost + LightGBM + LR(Chen-style)": ["xgb", "lgbm", "lr_chen"],
        "XGBoost(Chen feats) + LR(Chen-style)": ["xgb_chen", "lr_chen"],
        "+LoRA: v4 + LoRA": ["xgb", "lr_lite", "lora"],
        "+LoRA: XGBoost + LR(Chen-style) + LoRA": ["xgb", "lr_chen", "lora"],
        "+LoRA: XGBoost + LR(Chen-style + Luis) + LoRA": ["xgb", "lr_chen_luis", "lora"],
        "+LoRA: LightGBM + LR(Chen-style) + LoRA  (Chen's recipe)": ["lgbm", "lr_chen", "lora"],
        "+LoRA: LR(Chen-style) + LoRA": ["lr_chen", "lora"],
    }.items():
        sc[(k, name)] = np.sum([R[p] for p in parts], axis=0)
    sc[("Chen's own OOF: LR + LoRA + LightGBM(Luis), 5-fold", name)] = te.avg_lrteam_lora_lgbm.fillna(0).to_numpy()

MODELS = list(dict.fromkeys(m for m, _ in sc))
rows = []
for m in MODELS:
    for cap in (None, 3):
        r = {"model": m, "cap": cap or "none"}
        for name, (tr, te) in SPLITS.items():
            k = density_k(len(te))
            r[f"{name}@{k}"] = hits(te.y_nov4, capped(te, sc[(m, name)], cap), k)
        r["A+B @density"] = r["A@243"] + r["B@198"]
        rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(os.path.join(REP, "team_merge_results.csv"), index=False)
pd.set_option("display.width", 250)
print(res.to_string(index=False))

both = pd.concat([SPLITS["A"][1], SPLITS["B"][1]], ignore_index=True)
boot, _ = pooled_eval(both, {m: np.r_[sc[(m, "A")], sc[(m, "B")]] for m in MODELS},
                      "Exploit code, then EPSS", "v4: XGBoost + LR(Luis + Chen-lite)  [reference]")
boot.to_csv(os.path.join(REP, "team_merge_bootstrap.csv"), index=False)
print(f"\npooled: top {density_k(len(both))} of {len(both):,}, {both.y_nov4.sum()} positives  ('vs reference' = vs v4 with cap 3)")
print(boot.to_string(index=False))
log("done")
