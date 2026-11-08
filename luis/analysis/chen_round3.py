"""Chen's 3rd push on our forward A/B test: does anything beat the final recipe?

New from Chen (Nov 7, ~8 pm): out-of-fold scores for a Granite 4.2 30B LoRA (cv_lora30b_f*), an 8B LoRA with Granite's
30B answers written into the prompt (cv_llmx8b_f*), and the OOF of his final LR (cv_oof_lr.lr_full_llmx_noemb, the
model behind chen/submissions/lr_best_scoreboard_2026.parquet). His CV says neither LoRA variant beats the 8B; this
re-checks it on our split and checks the exact recipe final_build.py builds once his 2026 files are in.

All candidates: equal-weight rank average of the voters, Granite-30B demotion filters, cap 3 per assigner-month,
pooled 2024 + 2025 Jan-Sep test cohorts ranked within snapshot month, top 441 of 64,120, KEV by Nov 4.
Caveat (Chen's §8.5): his OOF scores come from a 5-fold CV over 2024-25, so on our A/B they are not strictly forward
(a 2024 test CVE's LoRA saw 2025 folds). Treat rows with Chen voters as estimates.

Outputs: reports/chen_round3.csv, reports/chen_round3_log.txt (through Tee), derived/xgb_bag_ab.parquet (cache)
Usage:  python -E -P luis/analysis/chen_round3.py
"""
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from common import CHEN, CHEN_LITE, DER, FULL, REP, SAFE, XGB_BIG, capped, density_k, hits, jan_sep, load, m_lgbm, m_lr, rank01  # noqa: E402

T0 = time.time()
RNG = np.random.default_rng(3)


def log(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


tr24, tr25 = load(2024, "2024-12-31"), load(2025, "2025-12-31")
te25, te24 = jan_sep(load(2025, "2025-11-04"), 2025), jan_sep(load(2024, "2024-11-04"), 2024)
both = pd.concat([te25, te24], ignore_index=True)

# ----------------------------------------------------------------------------- Luis's voters (forward: train one year, test the other)
cache = os.path.join(DER, "xgb_bag_ab.parquet")
if os.path.exists(cache):
    comp = pd.read_parquet(cache)
else:
    parts = []
    for name, tr, te in (("A", tr24, te25), ("B", tr25, te24)):
        ps = [xgb.XGBClassifier(tree_method="hist", device="cuda", scale_pos_weight=25, eval_metric="aucpr",
                                random_state=1000 * r + s, **XGB_BIG).fit(tr[FULL], tr.y).predict_proba(te[FULL])[:, 1]
              for r in range(4) for s in range(5)]
        parts.append(pd.DataFrame({"cve_id": te.cve_id, "xgb_bag": np.mean(ps, axis=0), "lr_luis": m_lr(tr, te, SAFE + CHEN_LITE)}))
        log(f"{name}: XGBoost 20 seeds + LR done")
    comp = pd.concat(parts, ignore_index=True)
    comp.to_parquet(cache, index=False)
if "lgbm" not in comp:   # LightGBM on Luis's safe features (10 deterministic seeds): the tree in Chen's own recipe
    comp["lgbm"] = np.r_[m_lgbm(tr24, te25, SAFE), m_lgbm(tr25, te24, SAFE)]
    comp.to_parquet(cache, index=False)
both = both.merge(comp, on="cve_id", how="left")

# ----------------------------------------------------------------------------- Chen's out-of-fold voters
def folds(pattern, col):
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(CHEN, "results", pattern)))])[["cve_id", "score"]] \
        .rename(columns={"score": col})


ex = pd.read_parquet(os.path.join(CHEN, "processed", "llm_extract.parquet"))[["cve_id", "impact", "typical_attacker", "attacker_appeal", "product_type"]]
both = (both.merge(ex, on="cve_id", how="left")
        .merge(pd.read_parquet(os.path.join(CHEN, "results", "cv_oof_lora.parquet"))[["cve_id", "lora_control"]].rename(columns={"lora_control": "lora8b"}), on="cve_id", how="left")
        .merge(folds("cv_lora30b_f*.parquet", "lora30b"), on="cve_id", how="left")
        .merge(folds("cv_llmx8b_f*.parquet", "llmx8b"), on="cve_id", how="left")
        .merge(pd.read_parquet(os.path.join(CHEN, "results", "cv_oof_lr.parquet"))[["cve_id", "lr_full_llmx_noemb"]].rename(columns={"lr_full_llmx_noemb": "lr_chen"}), on="cve_id", how="left")
        .merge(pd.read_parquet(os.path.join(CHEN, "results", "cv_oof_avg.parquet"))[["cve_id", "avg_lrteam_lora_lgbm"]], on="cve_id", how="left"))
miss = {c: int(both[c].isna().sum()) for c in ("lora8b", "lora30b", "llmx8b", "lr_chen", "avg_lrteam_lora_lgbm")}
log(f"test rows {len(both):,}, positives {int(both.y_nov4.sum())}; missing Chen scores {miss}")


nA = len(te25)
yv = both.y_nov4.to_numpy().astype(int)
K = density_k(len(both))
demote = ((both.impact == "denial_of_service") | (both.typical_attacker == "unlikely_target") | (both.attacker_appeal == "low")
          | (both.product_type == "wordpress_plugin_or_theme")).fillna(False).to_numpy()


# How each voter is put on a 0-1 scale before averaging. MAIN = stats_review.py's convention: Luis's voters ranked within
# each split (one model per test year, like one model for all of 2026), Chen's OOF ranked over the pooled rows (one CV).
# Two alternatives are reported as a sensitivity check: implementation details alone move hits by a few.
def norm(col, how):
    x = both[col].fillna(both[col].min()).to_numpy()
    if how == "month":
        return both.groupby("snapshot_month")[col].rank(pct=True).fillna(0).to_numpy()
    if how == "split" or col in ("xgb_bag", "lr_luis", "lgbm"):
        return np.r_[rank01(x[:nA]), rank01(x[nA:])]
    return rank01(x)


def voters(how):
    V = {k: norm(c, how) for k, c in (("tree", "xgb_bag"), ("lgbm", "lgbm"), ("lr_luis", "lr_luis"), ("lr_chen", "lr_chen"), ("lora8b", "lora8b"),
                                      ("lora30b", "lora30b"), ("llmx8b", "llmx8b"), ("chen3", "avg_lrteam_lora_lgbm"))}
    V["linear"] = (V["lr_luis"] + V["lr_chen"]) / 2
    V["llm_8b_30b"] = (V["lora8b"] + V["lora30b"]) / 2
    V["llm_all3"] = (V["lora8b"] + V["lora30b"] + V["llmx8b"]) / 3
    return V


def finish(raw, filt=True, cap=3):
    """Pooled view: re-rank within snapshot month (A and B come from different models), push filtered CVEs down, cap."""
    pct = both.assign(_s=np.asarray(raw, dtype=float)).groupby("snapshot_month")._s.rank(pct=True).to_numpy()
    return capped(both, pct - 50 * demote * filt, cap)


RECIPES = {   # name: (voters, Granite filters, cap)
    "Exploit code, then EPSS (no cap)": (None, False, None),
    "v4f = current final without Chen files (XGB + LR Luis)": (("tree", "lr_luis"), True, 3),
    "FINAL with Chen files (XGB + avg LR + LoRA 8B) [est.]": (("tree", "linear", "lora8b"), True, 3),
    "  same, no Granite filters [est.]": (("tree", "linear", "lora8b"), False, 3),
    "  same, LR Luis only as linear [est.]": (("tree", "lr_luis", "lora8b"), True, 3),
    "  same, LR Chen only as linear [est.]": (("tree", "lr_chen", "lora8b"), True, 3),
    "XGB + avg LR, no LLM [est.]": (("tree", "linear"), True, 3),
    "LLM voter = LoRA 30B [est.]": (("tree", "linear", "lora30b"), True, 3),
    "LLM voter = LoRA 8B + Granite answers (llmx) [est.]": (("tree", "linear", "llmx8b"), True, 3),
    "LLM voter = avg(8B, 30B) [est.]": (("tree", "linear", "llm_8b_30b"), True, 3),
    "LLM voter = avg(8B, 30B, llmx) [est.]": (("tree", "linear", "llm_all3"), True, 3),
    "4 voters: XGB + avg LR + LoRA 8B + LoRA 30B [est.]": (("tree", "linear", "lora8b", "lora30b"), True, 3),
    "Chen's recipe: LightGBM + LR Chen + LoRA 8B, no filters [est.]": (("lgbm", "lr_chen", "lora8b"), False, 3),
    "  Chen's recipe + Granite filters [est.]": (("lgbm", "lr_chen", "lora8b"), True, 3),
    "  LightGBM + avg LR + LoRA 8B (filters) [est.]": (("lgbm", "linear", "lora8b"), True, 3),
    "  LightGBM + LR Luis + LoRA 8B (filters) [est.]": (("lgbm", "lr_luis", "lora8b"), True, 3),
    # cv_oof_avg.avg_lrteam_lora_lgbm: an older saved average (earlier "lrteam" LR + LightGBM trained inside Chen's CV);
    # NOT his final recipe, which is the row above
    "Chen's saved OOF average avg_lrteam_lora_lgbm (older LR) [est.]": (("chen3",), False, 3),
}


def candidates(how):
    V = voters(how)
    out = {}
    for m, (vs, filt, cap) in RECIPES.items():
        out[m] = (both.any_flag * 10 + both.epss).to_numpy() if vs is None else finish(np.sum([V[v] for v in vs], axis=0), filt, cap)
    return out


V0 = voters("main")
for k in ("lora30b", "llmx8b", "lr_chen"):
    ref_k = "lr_luis" if k == "lr_chen" else "lora8b"
    print(f"   Spearman {k} vs {ref_k}: {pd.Series(V0[k]).corr(pd.Series(V0[ref_k]), method='spearman'):.2f}")
CAND = candidates("main")
SENS = {how: candidates(how) for how in ("split", "month")}
names = list(CAND)
base, ref = names[0], names[1]
reps = 2000
idxs = [RNG.integers(0, len(both), len(both)) for _ in range(reps)]


def tk(s, i):
    return yv[i][np.argpartition(-s[i], K)[:K]].sum()


B = {m: np.array([tk(CAND[m], i) for i in idxs]) for m in names}
TOP = {m: set(np.argsort(-CAND[m], kind="stable")[:K]) for m in names}
rows = []
for m in names:
    d_b, d_r = B[m] - B[base], B[m] - B[ref]
    rows.append({"model": m, f"hits@{K}": hits(yv, CAND[m], K), "95% CI": f"{np.percentile(B[m], 2.5):.0f}-{np.percentile(B[m], 97.5):.0f}",
                 "vs exploit+EPSS": f"{np.percentile(d_b, 2.5):+.0f} to {np.percentile(d_b, 97.5):+.0f}",
                 "vs v4f": f"{np.percentile(d_r, 2.5):+.0f} to {np.percentile(d_r, 97.5):+.0f}",
                 "P(better than v4f)": round(float((d_r > 0).mean()), 2),
                 "AP": round(average_precision_score(yv, CAND[m]), 4),
                 "top overlap with FINAL": len(TOP[m] & TOP[names[2]]),
                 "hits, all ranked per split": hits(yv, SENS["split"][m], K), "hits, all ranked within month": hits(yv, SENS["month"][m], K)})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(REP, "chen_round3.csv"), index=False)
pd.set_option("display.width", 300)
print(f"\npooled top {K} of {len(both):,} ({int(yv.sum())} positives, KEV by Nov 4); filters + cap 3 unless stated; [est.] = uses Chen OOF")
print("main normalization = stats_review.py's; last two columns = same recipes with other voter normalizations (sensitivity)")
print(res.to_string(index=False))
log("done")
