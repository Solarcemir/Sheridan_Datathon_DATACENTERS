"""Post-processing on top of v4: cheap, interpretable ways to clean the top 500.

1. Hard "demotion" filters from Chen's Granite 4.2 30B extraction (impact = denial of service only, typical attacker =
   unlikely target, attacker appeal = low, product = WordPress plugin, install base = small/niche). To avoid choosing
   filters on the test data, each filter is first checked on the TRAINING year of each split (A: 2024, B: 2025):
   it is applied only if its KEV rate there is below 0.25x the base rate.
2. Caps per vendor: per assigner-month (current), per Granite-normalized vendor-month (handles MITRE / VulnCheck /
   GitHub, which assign CVEs for many vendors), per assigner-day, and both.
v4 scores on A/B are computed once and cached in derived/ab_scores_v4.parquet.

Outputs: reports/postprocess_filters_check.csv, reports/postprocess_results.csv, reports/postprocess_bootstrap.csv,
         reports/postprocess_log.txt (through Tee)

Usage:  python -E -P luis/analysis/postprocess.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import (CHEN, CHEN_LITE, DER, FULL, REP, SAFE, density_k, hits, jan_sep, load, m_lr, m_xgb,  # noqa: E402
                    pooled_eval, rank01)

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


CACHE = os.path.join(DER, "ab_scores_v4.parquet")
ex = pd.read_parquet(os.path.join(CHEN, "processed", "llm_extract.parquet"))
GCOLS = ["vendor", "product", "product_type", "impact", "attacker_appeal", "typical_attacker", "install_base"]

tr24, tr25 = load(2024, "2024-12-31"), load(2025, "2025-12-31")
te25, te24 = jan_sep(load(2025, "2025-11-04"), 2025), jan_sep(load(2024, "2024-11-04"), 2024)
if os.path.exists(CACHE):
    sc = pd.read_parquet(CACHE)
else:
    parts = []
    for name, tr, te in (("A", tr24, te25), ("B", tr25, te24)):
        parts.append(pd.DataFrame({"cve_id": te.cve_id, "split": name,
                                   "xgb_pct": rank01(m_xgb(tr, te, FULL)), "lr_pct": rank01(m_lr(tr, te, SAFE + CHEN_LITE))}))
        log(f"{name}: v4 components scored")
    sc = pd.concat(parts, ignore_index=True)
    sc.to_parquet(CACHE, index=False)
TE = {"A": te25.merge(sc[sc.split == "A"], on="cve_id"), "B": te24.merge(sc[sc.split == "B"], on="cve_id")}
TR = {"A": tr24, "B": tr25}
for d in list(TE.values()) + list(TR.values()):
    pass
TE = {k: d.merge(ex[["cve_id"] + GCOLS], on="cve_id", how="left") for k, d in TE.items()}
TR = {k: d.merge(ex[["cve_id"] + GCOLS], on="cve_id", how="left") for k, d in TR.items()}
for d in list(TE.values()) + list(TR.values()):
    d["g_vendor"] = d.vendor.fillna("").str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip().replace("", np.nan) \
        .fillna(d.assigner)
log("Granite fields merged: coverage " + str({k: round(float(d.product_type.notna().mean()), 3) for k, d in TE.items()}))

FILTERS = {
    "impact = denial of service": lambda d: d.impact == "denial_of_service",
    "typical attacker = unlikely target": lambda d: d.typical_attacker == "unlikely_target",
    "attacker appeal = low": lambda d: d.attacker_appeal == "low",
    "product = WordPress plugin/theme": lambda d: d.product_type == "wordpress_plugin_or_theme",
    "install base = small/niche": lambda d: d.install_base == "small_or_niche",
}

# ----------------------------------------------------------------------------- 1. check filters on the TRAINING year only
chk, ok = [], {}
for split, tr in TR.items():
    base = tr.y.mean()
    ok[split] = []
    for f, fn in FILTERS.items():
        m = fn(tr).fillna(False).to_numpy()
        rate = tr.y[m].mean() if m.any() else np.nan
        keep = bool(m.any() and rate < 0.25 * base)
        if keep:
            ok[split].append(f)
        chk.append({"split (training year)": f"{split} ({2024 if split == 'A' else 2025})", "filter": f, "CVEs": int(m.sum()),
                    "KEV": int(tr.y[m].sum()), "KEV rate / base": round(rate / base, 2) if m.any() else None, "use it": keep})
chk = pd.DataFrame(chk)
chk.to_csv(os.path.join(REP, "postprocess_filters_check.csv"), index=False)
pd.set_option("display.width", 250)
print(chk.to_string(index=False))


def demote(d, s, mask):
    s = np.asarray(s, dtype=float).copy()
    s[np.asarray(mask, dtype=bool)] -= 50
    return s


def cap_by(d, s, keys, cap):
    s = pd.Series(np.asarray(s, dtype=float), index=d.index)
    n = d.assign(_s=s).sort_values("_s", ascending=False, kind="stable").groupby(keys).cumcount()
    s = s.copy()
    s[n[n >= cap].index] -= 100
    return s.to_numpy()


def variants(split, d):
    base = (d.xgb_pct + d.lr_pct).to_numpy()
    d = d.assign(pub_day=d.published.dt.normalize())
    fmask = np.zeros(len(d), bool)
    for f in ok[split]:
        fmask |= FILTERS[f](d).fillna(False).to_numpy()
    out = {"Exploit code, then EPSS": (d.any_flag * 10 + d.epss).to_numpy(),
           "v4, no cap": base,
           "v4 + cap 3 per assigner-month  [reference]": cap_by(d, base, ["assigner", "snapshot_month"], 3),
           "v4 + cap 3 per Granite vendor-month": cap_by(d, base, ["g_vendor", "snapshot_month"], 3),
           "v4 + cap 5 per Granite vendor-month": cap_by(d, base, ["g_vendor", "snapshot_month"], 5),
           "v4 + cap 3 per assigner-day": cap_by(d, base, ["assigner", "pub_day"], 3),
           "v4 + both caps (assigner-month 3, vendor-month 3)": cap_by(d, cap_by(d, base, ["assigner", "snapshot_month"], 3),
                                                                     ["g_vendor", "snapshot_month"], 3),
           "v4 + Granite filters (train-checked)": demote(d, base, fmask),
           "v4 + Granite filters + cap 3 assigner-month": cap_by(d, demote(d, base, fmask), ["assigner", "snapshot_month"], 3)}
    for f in FILTERS:
        out[f"v4 + cap 3 + only filter: {f}"] = cap_by(d, demote(d, base, FILTERS[f](d).fillna(False).to_numpy()),
                                                       ["assigner", "snapshot_month"], 3)
    return out


V = {s: variants(s, d) for s, d in TE.items()}
rows = []
for m in V["A"]:
    r = {"variant": m}
    for s, d in TE.items():
        r[f"{s}@{density_k(len(d))}"] = hits(d.y_nov4, V[s][m], density_k(len(d)))
    r["A+B"] = r["A@243"] + r["B@198"]
    rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(os.path.join(REP, "postprocess_results.csv"), index=False)
print(f"\nfilters used (passed the training-year check): A {ok['A']} | B {ok['B']}")
print(res.to_string(index=False))

both = pd.concat([TE["A"], TE["B"]], ignore_index=True)
boot, _ = pooled_eval(both, {m: np.r_[V["A"][m], V["B"][m]] for m in V["A"]}, "Exploit code, then EPSS",
                      "v4 + cap 3 per assigner-month  [reference]", caps=(None,))
boot = boot.drop(columns="cap")
boot.to_csv(os.path.join(REP, "postprocess_bootstrap.csv"), index=False)
print(f"\npooled top {density_k(len(both))} of {len(both):,}, {both.y_nov4.sum()} positives ('vs reference' = vs v4 + cap 3 assigner-month)")
print(boot.to_string(index=False))
log("done")
