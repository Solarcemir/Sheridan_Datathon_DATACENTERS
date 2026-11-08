"""Unified LR vs avg(Chen LR, Luis LR) as the linear voter of the final ensemble (luis/context_Luis.md 14.16 rule).

Ensemble = rank average of tree (XGBoost, Luis's settings, his features) + linear + LoRA 8B, then the 4 Granite 30B
demotion filters, then cap 3 per assigner-month. Two checks:
  5-fold purged CV (K 517): tree = xgb_fixed_luis (cv_xgb), LoRA = cv_control OOF, linear OOF from cv_oof_lr.
  forward-only blocks 1-4 (K 414): every voter trained only on earlier blocks (cv_fwd_extra, cv_fwdlora, cv_fwd_lr).
Nov-4 labels; paired bootstrap (1,000) of the hit difference vs the current 2-LR ensemble.

  python scripts/lr_unified_compare.py   -> reports/lr_unified_compare.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap

ROOT = Path(__file__).resolve().parent.parent
RES, PROC = ROOT / "results", ROOT / "processed"
VERS = ["lr_unified_v1", "lr_unified_v2", "lr_unified_v3"]


def base_frame(scores: pd.DataFrame) -> pd.DataFrame:
    h = pd.read_parquet(PROC / "historical_2024_2025.parquet",
                        columns=["cve_id", "published", "snapshot_month", "assigner", "kev_date_added"])
    o = scores.merge(h, on="cve_id")
    o["nov4"] = (o.kev_date_added.notna()
                 & (o.kev_date_added <= pd.to_datetime(o.published.dt.year.astype(str) + "-11-04"))).astype(int)
    e = pd.read_parquet(PROC / "llm_extract.parquet").set_index("cve_id").reindex(o.cve_id)
    o["filt"] = ((e.impact == "denial_of_service") | (e.typical_attacker == "unlikely_target")
                 | (e.attacker_appeal == "low") | (e.product_type == "wordpress_plugin_or_theme")).fillna(False).to_numpy()
    return o.reset_index(drop=True)


def evaluate(o: pd.DataFrame, title: str) -> list[str]:
    r = lambda c: o[c].rank(pct=True)
    two = ((r("lr_full_llmx_noemb") + r("lr_luis_only")) / 2).rename("avg2")
    linear = {"avg(Chen LR, Luis LR) [current]": two, "Chen LR only": o.lr_full_llmx_noemb, "Luis LR only": o.lr_luis_only}
    linear.update({v.replace("lr_", ""): o[v] for v in VERS})
    n, K = len(o), round(500 * len(o) / 72670)
    y = o.nov4.to_numpy()
    rng = np.random.default_rng(0)
    B = [rng.integers(0, n, n) for _ in range(1000)]

    def topk(s, filters=True, cap=3):
        s = pd.Series(s).rank(pct=True).to_numpy() - (50 * o.filt.to_numpy() if filters else 0)
        d = o.assign(_s=s).sort_values("_s", ascending=False)
        d = d[d.groupby(["assigner", "snapshot_month"]).cumcount() < cap].head(K)
        m = np.zeros(n, bool); m[d.index] = True
        return m

    ens = {k: (r_lin.rank(pct=True) + r("tree") + r("lora")) / 3 for k, r_lin in ((k, pd.Series(v)) for k, v in linear.items())}
    ref = topk(ens["avg(Chen LR, Luis LR) [current]"])
    rows = []
    for k in linear:
        for kind, s in (("ensemble", ens[k]), ("LR alone", linear[k])):
            m = topk(s, filters=(kind == "ensemble"))
            d = [(y[i] * m[i]).sum() - (y[i] * ref[i]).sum() for i in B]
            rows.append({"linear voter": k, "as": kind, "hits Nov4": int(y[m].sum()),
                         "vs current ensemble 95% CI": f"{np.percentile(d, 2.5):+.0f}..{np.percentile(d, 97.5):+.0f}",
                         "AP Nov4 (raw score)": round(ap(y, pd.Series(s).to_numpy()), 4)})
    df = pd.DataFrame(rows).sort_values(["as", "linear voter"], ascending=[False, True])
    return [f"== {title}: {n:,} CVEs, {int(y.sum())} Nov-4 positives, K={K}; ensemble = tree + linear + LoRA 8B, "
            f"Granite filters, cap 3; 'LR alone' = cap 3, no filters", df.to_string(index=False), ""]


def main() -> None:
    oof = pd.read_parquet(RES / "cv_oof_lr.parquet")[["cve_id", "lr_full_llmx_noemb", "lr_luis_only"] + VERS]
    oof = oof.merge(pd.read_parquet(RES / "cv_xgb.parquet")[["cve_id", "xgb_fixed_luis"]].rename(
        columns={"xgb_fixed_luis": "tree"}), on="cve_id")
    lo = pd.concat([pd.read_parquet(f) for f in sorted(RES.glob("cv_control_f*.parquet"))]).drop_duplicates("cve_id")
    oof["lora"] = oof.cve_id.map(lo.set_index("cve_id").score)
    out = evaluate(base_frame(oof), "5-fold purged CV")

    fwd = pd.read_parquet(RES / "cv_fwd_lr.parquet")[["cve_id", "lr_full_llmx_noemb"] + VERS]
    ex = pd.read_parquet(RES / "cv_fwd_extra.parquet")[["cve_id", "lr_luis_only", "xgb_fixed_luis"]].rename(
        columns={"xgb_fixed_luis": "tree"})
    fl = pd.concat([pd.read_parquet(f) for f in sorted(RES.glob("cv_fwdlora_f*.parquet"))]).drop_duplicates("cve_id")
    fwd = fwd.merge(ex, on="cve_id")
    fwd["lora"] = fwd.cve_id.map(fl.set_index("cve_id").score)
    fwd = fwd.dropna(subset=["lr_full_llmx_noemb", "tree", "lora", "lr_luis_only"] + VERS)
    out += evaluate(base_frame(fwd), "forward-only blocks 1-4 (every voter trained on earlier blocks only)")
    text = "\n".join(out)
    print(text)
    (ROOT / "reports" / "lr_unified_compare.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
