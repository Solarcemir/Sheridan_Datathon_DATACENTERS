"""Statistical review: which of our differences are real, and what to expect on 2026.

After 50+ variants evaluated on the same 75 positives, two risks: the winner's curse (the best-looking variant is
inflated by luck) and 2-3 hit gaps that are noise. This script:
  1. Run-to-run variability: XGBoost (GPU, not bit-reproducible) re-run 8 times x 5 seeds; spread of v4 / v4f hits,
     and whether averaging all 40 seeds ("bagged") helps.
  2. Paired tests on the same positives vs exploit-then-EPSS and vs v4f: exact McNemar on discordant positives and a
     2,000-rep paired bootstrap, Holm-corrected across the comparisons.
  3. Romano-Wolf style "not significantly worse than the best" set (single-step max-t, FWER 5%).
  4. Power: minimum detectable difference with 75 positives, and positives needed to detect +2 hits.
  5. 2026 projection with uncertainty: Poisson-Gamma posteriors, anchored on the README's 2026 baselines
     (CVSS 7, EPSS 6, exploit-then-EPSS 8) vs the same baselines on our pooled replicas.
Pooled view = 2024 + 2025 Jan-Sep test cohorts, ranks within snapshot month, top 441 of 64,120 (scoreboard rate).

Outputs: reports/stats_review_*.csv, reports/stats_review_log.txt (through Tee)
Usage:  python -E -P luis/analysis/stats_review.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest, norm  # noqa: E402
import xgboost as xgb  # noqa: E402
from common import (CHEN, CHEN_LITE, FULL, REP, SAFE, XGB_BIG, capped, density_k, hits, jan_sep, load, m_lgbm,  # noqa: E402
                    m_lr, rank01)

T0 = time.time()
RNG = np.random.default_rng(7)


def log(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- scores on A/B
tr24, tr25 = load(2024, "2024-12-31"), load(2025, "2025-12-31")
te25, te24 = jan_sep(load(2025, "2025-11-04"), 2025), jan_sep(load(2024, "2024-11-04"), 2024)
ex = pd.read_parquet(os.path.join(CHEN, "processed", "llm_extract.parquet"))[["cve_id", "impact", "typical_attacker", "attacker_appeal", "product_type"]]
lora = pd.read_parquet(os.path.join(CHEN, "results", "cv_oof_lora.parquet"))[["cve_id", "lora_control"]]
chen_best = pd.read_parquet(os.path.join(CHEN, "results", "cv_oof_avg.parquet"))[["cve_id", "avg_lrteam_lora_lgbm"]]
RUNS = 8
comp = {}
for name, tr, te in (("A", tr24, te25), ("B", tr25, te24)):
    c = {"lgbm": rank01(m_lgbm(tr, te, SAFE)), "lr": rank01(m_lr(tr, te, SAFE + CHEN_LITE))}
    xs = []
    for r in range(RUNS):   # different seed blocks per run
        ps = []
        for s in range(5):
            m = xgb.XGBClassifier(tree_method="hist", device="cuda", scale_pos_weight=25, eval_metric="aucpr",
                                  random_state=1000 * r + s, **XGB_BIG)
            m.fit(tr[FULL], tr.y)
            ps.append(m.predict_proba(te[FULL])[:, 1])
        xs.append(np.mean(ps, axis=0))
        c[f"xgb{r}"] = rank01(xs[-1])
    c["xgb_bag"] = rank01(np.mean(xs, axis=0))
    comp[name] = c
    log(f"{name}: components done ({RUNS} XGBoost runs x 5 seeds)")

both = pd.concat([te25, te24], ignore_index=True)
both = both.merge(ex, on="cve_id", how="left").merge(lora, on="cve_id", how="left").merge(chen_best, on="cve_id", how="left")
C = {k: np.r_[comp["A"][k], comp["B"][k]] for k in comp["A"]}
yv = both.y_nov4.to_numpy().astype(int)
K = density_k(len(both))
demote = ((both.impact == "denial_of_service") | (both.typical_attacker == "unlikely_target") | (both.attacker_appeal == "low")
          | (both.product_type == "wordpress_plugin_or_theme")).fillna(False).to_numpy()


def pooled(raw, cap=3, filt=False):
    """Rank within snapshot month (the two test years come from different models), push filtered CVEs down, cap."""
    pct = both.assign(_s=np.asarray(raw, dtype=float)).groupby("snapshot_month")._s.rank(pct=True).to_numpy()
    if filt:
        pct = pct - 50 * demote
    return capped(both, pct, cap)


# ----------------------------------------------------------------------------- 1. run-to-run variability
var = []
for r in range(RUNS):
    var.append({"run": r, "v4 (cap 3)": hits(yv, pooled(C[f"xgb{r}"] + C["lr"]), K),
                "v4f (filters + cap 3)": hits(yv, pooled(C[f"xgb{r}"] + C["lr"], filt=True), K)})
var = pd.DataFrame(var)
bag_v4, bag_v4f = hits(yv, pooled(C["xgb_bag"] + C["lr"]), K), hits(yv, pooled(C["xgb_bag"] + C["lr"], filt=True), K)
var.to_csv(os.path.join(REP, "stats_review_seed_variability.csv"), index=False)
print(f"\n1) RUN-TO-RUN VARIABILITY (pooled hits@{K}, {RUNS} XGBoost runs of 5 seeds each)")
print(var.to_string(index=False))
for c in ("v4 (cap 3)", "v4f (filters + cap 3)"):
    print(f"   {c}: mean {var[c].mean():.1f}, sd {var[c].std():.2f}, range {var[c].min()}-{var[c].max()}")
print(f"   bagged (all {RUNS * 5} seeds averaged): v4 {bag_v4}, v4f {bag_v4f}")

# ----------------------------------------------------------------------------- candidate rankings
L = rank01(both.lora_control.fillna(both.lora_control.min()))
CAND = {
    "Exploit code, then EPSS": pooled((both.any_flag * 10 + both.epss).to_numpy(), cap=None),
    "Exploit code, then EPSS + cap 3": pooled((both.any_flag * 10 + both.epss).to_numpy()),
    "LightGBM (cap 3)": pooled(C["lgbm"]),
    "XGBoost bagged (cap 3)": pooled(C["xgb_bag"]),
    "LR Luis + Chen-lite (cap 3)": pooled(C["lr"]),
    "v3: LightGBM + LR (cap 3)": pooled(C["lgbm"] + C["lr"]),
    "v4 bagged: XGBoost + LR (cap 3)": pooled(C["xgb_bag"] + C["lr"]),
    "v4f bagged: + Granite filters": pooled(C["xgb_bag"] + C["lr"], filt=True),
    "v4f + LightGBM (3 models, filters)": pooled(C["xgb_bag"] + C["lgbm"] + C["lr"], filt=True),
    "v4f + Chen LoRA [estimate]": pooled(C["xgb_bag"] + C["lr"] + L, filt=True),
    "Chen's own OOF 3-model avg (cap 3)": pooled(both.avg_lrteam_lora_lgbm.fillna(0).to_numpy()),
}
names = list(CAND)
TOP = {m: np.argsort(-s, kind="stable")[:K] for m, s in CAND.items()}
FOUND = {m: np.zeros(len(both), bool) for m in names}
for m in names:
    FOUND[m][TOP[m]] = True
H = {m: int(yv[TOP[m]].sum()) for m in names}

# paired bootstrap over test CVEs
reps = 2000
idxs = [RNG.integers(0, len(both), len(both)) for _ in range(reps)]


def tk(s, i):
    ss = s[i]
    return yv[i][np.argpartition(-ss, K)[:K]].sum()


B = {m: np.array([tk(CAND[m], i) for i in idxs]) for m in names}
log("bootstrap done")

# ----------------------------------------------------------------------------- 2. paired tests, Holm-corrected
def mcnemar(a, b):
    pa, pb = FOUND[a] & (yv == 1), FOUND[b] & (yv == 1)
    only_a, only_b = int((pa & ~pb).sum()), int((pb & ~pa).sum())
    n = only_a + only_b
    p = binomtest(only_a, n, 0.5).pvalue if n else 1.0
    return only_a, only_b, p


def holm(ps):
    order = np.argsort(ps)
    adj, m, run = np.empty(len(ps)), len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i]))
        adj[i] = run
    return adj


rows = []
for ref in ("Exploit code, then EPSS", "v4f bagged: + Granite filters"):
    block = []
    for m in names:
        if m == ref:
            continue
        oa, ob, p = mcnemar(m, ref)
        d = B[m] - B[ref]
        p_boot = 2 * min((d <= 0).mean(), (d >= 0).mean())
        block.append({"model": m, "vs": ref, "hits": H[m], "ref hits": H[ref], "positives only model": oa, "positives only ref": ob,
                      "McNemar p": p, "bootstrap diff 95% CI": f"{np.percentile(d, 2.5):+.0f} to {np.percentile(d, 97.5):+.0f}",
                      "bootstrap p": min(1.0, p_boot)})
    adj_m, adj_b = holm(np.array([r["McNemar p"] for r in block])), holm(np.array([r["bootstrap p"] for r in block]))
    for r, a, b in zip(block, adj_m, adj_b):
        r["McNemar p (Holm)"], r["bootstrap p (Holm)"] = round(a, 4), round(b, 4)
        r["McNemar p"], r["bootstrap p"] = round(r["McNemar p"], 4), round(r["bootstrap p"], 4)
    rows += block
tests = pd.DataFrame(rows)
tests.to_csv(os.path.join(REP, "stats_review_paired_tests.csv"), index=False)
pd.set_option("display.width", 250)
print("\n2) PAIRED TESTS ON THE SAME 75 POSITIVES (Holm-corrected within each block)")
print(tests.to_string(index=False))

# ----------------------------------------------------------------------------- 3. Romano-Wolf "not worse than best" set
best = max(names, key=lambda m: B[m].mean())
stats, centered = {}, []
for m in names:
    if m == best:
        continue
    d = B[best] - B[m]
    sd = d.std(ddof=1) or 1e-9
    stats[m] = d.mean() / sd
    centered.append((d - d.mean()) / sd)
tmax = np.max(np.vstack(centered), axis=0)
crit = np.percentile(tmax, 95)
mcs = pd.DataFrame([{"model": m, "mean boot hits": round(B[m].mean(), 1), "t vs best": round(stats.get(m, 0.0), 2),
                     "in set (not significantly worse than best)": (m == best) or stats[m] <= crit} for m in names]) \
    .sort_values("mean boot hits", ascending=False)
mcs.to_csv(os.path.join(REP, "stats_review_best_set.csv"), index=False)
print(f"\n3) ROMANO-WOLF SET vs the best ({best}); single-step max-t critical value {crit:.2f}")
print(mcs.to_string(index=False))

# ----------------------------------------------------------------------------- 4. power
d_base = B["v4f bagged: + Granite filters"] - B["Exploit code, then EPSS"]
d_v4 = B["v4f bagged: + Granite filters"] - B["v4 bagged: XGBoost + LR (cap 3)"]
d_lora = B["v4f + Chen LoRA [estimate]"] - B["v4f bagged: + Granite filters"]
z = norm.ppf(0.975) + norm.ppf(0.80)
print("\n4) POWER (80% power, two-sided 5%)")
for lab, d in (("v4f vs exploit+EPSS", d_base), ("v4f vs v4 (similar models)", d_v4), ("v4f+LoRA vs v4f", d_lora)):
    sd = d.std(ddof=1)
    mde = z * sd
    need = 75 * (z * sd / 2) ** 2 if sd > 0 else float("nan")
    print(f"   {lab:28s} sd of paired diff {sd:4.2f} -> minimum detectable difference {mde:4.1f} hits; "
          f"positives needed to detect +2: ~{need:,.0f}")

# ----------------------------------------------------------------------------- 5. 2026 projection (Poisson-Gamma)
anchors = {"CVSS 3 first": (7, hits(yv, pooled(both.cvss3.fillna(-1).to_numpy(), cap=None), K)),
           "EPSS first": (6, hits(yv, pooled(both.epss.to_numpy(), cap=None), K)),
           "Exploit code, then EPSS": (8, H["Exploit code, then EPSS"])}
print("\n5) 2026 PROJECTION: projected = README 2026 anchor x (model hits / anchor hits) on our replicas, Gamma(h + 0.5) posteriors")
proj_rows = []
for model in ("v4f bagged: + Granite filters", "v4f + Chen LoRA [estimate]", "Exploit code, then EPSS"):
    draws_all = []
    for a, (h26, hrep) in anchors.items():
        n = 20000
        lam_model = RNG.gamma(H[model] + 0.5, 1, n)
        lam_anchor_rep = RNG.gamma(hrep + 0.5, 1, n)
        lam_anchor_26 = RNG.gamma(h26 + 0.5, 1, n)
        dr = lam_anchor_26 * lam_model / lam_anchor_rep
        draws_all.append(dr)
        proj_rows.append({"model": model, "anchor": a, "anchor 2026 (README)": h26, "anchor on replicas": hrep,
                          "model on replicas": H[model], "median": round(np.median(dr), 1),
                          "80% interval": f"{np.percentile(dr, 10):.1f}-{np.percentile(dr, 90):.1f}"})
    pooled_draws = np.concatenate(draws_all)
    proj_rows.append({"model": model, "anchor": "all three anchors pooled", "median": round(np.median(pooled_draws), 1),
                      "80% interval": f"{np.percentile(pooled_draws, 10):.1f}-{np.percentile(pooled_draws, 90):.1f}"})
proj = pd.DataFrame(proj_rows)
proj.to_csv(os.path.join(REP, "stats_review_projection_2026.csv"), index=False)
print(proj.to_string(index=False))
log("done")
