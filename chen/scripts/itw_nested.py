"""Is the "exploited in the wild at month-end" promotion rule p-hacking? Nested check: inside each fold's TRAINING rows only,
pick the binary point-in-time flag with the highest KEV rate (>= 3 positives), promote it to the top in the held-out fold
(ensemble = unified LR + XGBoost + LoRA 8B, Granite filters, cap 3; promoted CVEs bypass the cap). 5-fold and forward-only.
  python scripts/itw_nested.py   -> reports/itw_nested.txt
"""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lr_unified_compare import base_frame

R = Path(__file__).resolve().parent.parent
P, RES = R / "processed", R / "results"
a = pd.read_parquet(P / "asof_nvd_historical.parquet").merge(pd.read_parquet(P / "asof_cvelist_historical.parquet"), on="cve_id")
flags = [c for c in a.columns if c != "cve_id" and set(pd.unique(a[c].dropna())) <= {0, 1}]
folds = pd.read_parquet(P / "cv_folds.parquet")[["cve_id", "fold"]]
lab = pd.read_parquet(P / "historical_2024_2025.parquet", columns=["cve_id", "label"])
ren = lambda f, c, n: pd.read_parquet(f)[["cve_id", c]].rename(columns={c: n})
def lora(pat): return pd.concat([pd.read_parquet(f) for f in glob.glob(str(RES / pat))]).drop_duplicates("cve_id")[["cve_id", "score"]].rename(columns={"score": "lora"})
def frame(lin, tree, lo):
    return base_frame(lin.merge(tree, on="cve_id").merge(lo, on="cve_id").merge(lab, on="cve_id")).merge(a, on="cve_id").merge(folds, on="cve_id")
o5 = frame(ren(RES / "cv_oof_lr.parquet", "lr_unified_v1", "lin"), ren(RES / "cv_xgb.parquet", "xgb_fixed_luis", "tree"), lora("cv_control_f*.parquet"))
oF = frame(ren(RES / "cv_fwd_lr.parquet", "lr_unified_v1", "lin"), ren(RES / "cv_fwd_extra.parquet", "xgb_fixed_luis", "tree"), lora("cv_fwdlora_f*.parquet")).dropna(subset=["lin", "tree", "lora"])
def pick(train):
    best, br, base = None, 0, train.label.mean()
    for f in flags:
        on = train[f] == 1
        if train.label[on].sum() >= 3 and train.label[on].mean() > br:
            best, br = f, train.label[on].mean()
    return best, round(br / base, 1)
def topk(o, sc, keep, K):
    d = o.assign(_s=sc, _k=keep).sort_values("_s", ascending=False)
    return int(d[(d.groupby(["assigner", "snapshot_month"]).cumcount() < 3) | d._k].head(K).nov4.sum())
lines = []
for name, o, K, mode in (("5-fold purged CV", o5, 517, "oof"), ("forward-only blocks 1-4", oF, 414, "fwd")):
    r = lambda c: o[c].rank(pct=True)
    s = ((r("lin") + r("tree") + r("lora")) / 3).rank(pct=True) - 50 * o.filt
    promote, fixed, chosen = np.zeros(len(o), bool), (o.nv_desc_itw == 1).to_numpy(), []
    for f in sorted(o.fold.unique()):
        flag, lift = pick(o5[o5.fold < f] if mode == "fwd" else o5[o5.fold != f])
        chosen.append(f"block {int(f)}: {flag} ({lift}x)")
        if flag:
            promote |= ((o.fold == f) & (o[flag] == 1)).to_numpy()
    lines += [f"== {name} (K={K}, {int(o.nov4.sum())} Nov-4 positives)", "  flag picked from training rows only: " + "; ".join(chosen),
              f"  hits: current {topk(o, s, np.zeros(len(o), bool), K)} | + flag picked in training {topk(o, s + 100 * promote, promote, K)} | "
              f"+ fixed in-the-wild rule (chosen after seeing all data, optimistic) {topk(o, s + 100 * fixed, fixed, K)}", ""]
print("\n".join(lines)); (R / "reports" / "itw_nested.txt").write_text("\n".join(lines) + "\n")
