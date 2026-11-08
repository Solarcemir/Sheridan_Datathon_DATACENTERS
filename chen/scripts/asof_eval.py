"""Do point-in-time public-history features help? (NVD change history + CVE List git history, as of snapshot month-end)

Pre-set rule: adopt only if the ensemble (tree + linear + LoRA 8B, Granite filters, cap 3) gains >= +2 Nov-4 hits on BOTH
the 5-fold purged CV (K 517) and the strictly forward blocks (K 414), with no drift red flag.
  A  unified LR + as-of features (trees unchanged)
  B  A, and XGBoost (Luis's settings, 5 seeds) also gets the as-of features

  python scripts/asof_eval.py   -> reports/asof_eval.txt, results/cv_asof.parquet
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

import cv  # noqa: E402
import lr_unified as U  # noqa: E402
import xgb_tune  # noqa: E402
from lr_unified_compare import base_frame  # noqa: E402

COUNTS = ["nv_days_to_anal", "nv_n_refs", "nv_refs_added", "nv_n_modified", "nv_n_cpe", "cl_n_refs"]


def asof(df: pd.DataFrame) -> pd.DataFrame:
    a = cv._both(cv.PROC, "asof_nvd").set_index("cve_id")
    b = cv._both(cv.PROC, "asof_cvelist").set_index("cve_id")
    x = a.join(b, how="outer").reindex(df.cve_id)
    x["nv_not_analyzed_days"] = x.nv_days_to_anal.isna().astype(int)
    for c in COUNTS:
        x[c] = np.log1p(x[c].clip(lower=0))
    return pd.DataFrame(x.fillna(0).to_numpy(float), index=df.index, columns=[f"asof_{c}" for c in x.columns])


def lr_job(X, y, groups, gid, tr, te, tag, f):
    m, C = U.fit("v1", X[tr], y[tr], groups[tr], gid)
    return tag, f, np.flatnonzero(te), m.score(X[te])


def xgb_job(X, y, tr, te, tag, f, seed):
    xgb_tune.THREADS = 2
    return tag, f, np.flatnonzero(te), pd.Series(xgb_tune.fit_predict(X, y, tr, te, xgb_tune.XGB_LUIS, seed)).rank(pct=True).to_numpy()


def main() -> None:
    h = cv.load()
    folds = pd.read_parquet(cv.PROC / "cv_folds.parquet").set_index("cve_id")
    h["fold"], h["group"] = h.cve_id.map(folds.fold), h.cve_id.map(folds.group)
    A = asof(h)
    Bk = U.blocks(h); Bk["asof"] = A
    X, gid, cols = U.matrix(Bk)
    XT = np.hstack([cv.luis_feats(h).to_numpy(np.float32), A.to_numpy(np.float32)])
    y, groups = h.label.to_numpy(), h.group.fillna(h.cve_id).to_numpy()
    lines = [f"as-of features: {A.shape[1]} ({', '.join(c.replace('asof_', '') for c in A.columns)})", ""]

    # univariate lifts + drift
    lab = base_frame(h[["cve_id"]].copy())
    base = lab.nov4.mean()
    sb = pd.read_parquet(cv.PROC / "scoreboard_2026_view.parquet", columns=["cve_id"])
    A26 = asof(sb)
    rows = []
    for c in A.columns:
        v = A[c].to_numpy(); on = v > (np.median(v) if v.max() > 1 else 0.5)
        yr = h.published.dt.year.to_numpy()
        rows.append({"feature": c.replace("asof_", ""), "share 2024": round(on[yr == 2024].mean(), 3), "share 2025": round(on[yr == 2025].mean(), 3),
                     "share 2026": round((A26[c].to_numpy() > (np.median(v) if v.max() > 1 else 0.5)).mean(), 3),
                     "KEV-by-Nov4 lift": round(lab.nov4[on].mean() / base, 1) if on.any() else np.nan, "positives with it": int(lab.nov4[on].sum())})
    lines += ["Univariate (2024-25 practice rows; 'on' = above median for counts, = 1 for flags) and drift:", pd.DataFrame(rows).to_string(index=False), ""]

    jobs = []
    for f in range(cv.K_FOLDS):
        te = (h.fold == f).to_numpy(); purge = h.group.isin(set(h.group[te].dropna())).to_numpy()
        for mode, tr in (("oof", (~te) & ~purge), ("fwd", (h.fold < f).to_numpy() & ~purge)):
            if mode == "fwd" and f == 0:
                continue
            jobs.append(delayed(lr_job)(X, y, groups, gid, tr, te, f"lr_{mode}", f))
            jobs += [delayed(xgb_job)(XT, y, tr, te, f"xgb_{mode}", f, s) for s in range(5)]
    res = Parallel(n_jobs=min(len(jobs), 80))(jobs)
    out = h[["cve_id"]].copy()
    acc = {}
    for tag, f, rows_, s in res:
        acc.setdefault((tag, f), []).append((rows_, s))
    for (tag, f), parts in acc.items():
        col = f"asof_{tag}"
        if col not in out:
            out[col] = np.nan
        out.loc[parts[0][0], col] = np.mean([p[1] for p in parts], axis=0) if tag.startswith("xgb") else parts[0][1]
    out.to_parquet(cv.RES / "cv_asof.parquet", index=False)

    # ensembles
    def ens(o, linear, tree):
        r = lambda c: o[c].rank(pct=True)
        return (r(linear) + r(tree) + r("lora")) / 3

    def topk(o, s, K):
        s = pd.Series(s).rank(pct=True).to_numpy() - 50 * o.filt.to_numpy()
        d = o.assign(_s=s).sort_values("_s", ascending=False)
        d = d[d.groupby(["assigner", "snapshot_month"]).cumcount() < 3].head(K)
        m = np.zeros(len(o), bool); m[d.index] = True
        return m

    lo = pd.concat([pd.read_parquet(f) for f in sorted(cv.RES.glob("cv_control_f*.parquet"))]).drop_duplicates("cve_id").set_index("cve_id").score
    fl = pd.concat([pd.read_parquet(f) for f in sorted(cv.RES.glob("cv_fwdlora_f*.parquet"))]).drop_duplicates("cve_id").set_index("cve_id").score
    oof = pd.read_parquet(cv.RES / "cv_oof_lr.parquet")[["cve_id", "lr_unified_v1"]].merge(
        pd.read_parquet(cv.RES / "cv_xgb.parquet")[["cve_id", "xgb_fixed_luis"]], on="cve_id").merge(out, on="cve_id")
    oof["lora"] = oof.cve_id.map(lo)
    fwd = pd.read_parquet(cv.RES / "cv_fwd_lr.parquet")[["cve_id", "lr_unified_v1"]].merge(
        pd.read_parquet(cv.RES / "cv_fwd_extra.parquet")[["cve_id", "xgb_fixed_luis"]], on="cve_id").merge(out, on="cve_id")
    fwd["lora"] = fwd.cve_id.map(fl)
    fwd = fwd.dropna(subset=["lr_unified_v1", "xgb_fixed_luis", "lora", "asof_lr_fwd", "asof_xgb_fwd"])
    rng = np.random.default_rng(0)
    for name, o, sfx in (("5-fold purged CV", base_frame(oof), "oof"), ("forward-only blocks 1-4", base_frame(fwd), "fwd")):
        n, K = len(o), round(500 * len(o) / 72670); yv = o.nov4.to_numpy(); Bs = [rng.integers(0, n, n) for _ in range(1000)]
        V = {"current (unified LR + XGBoost + LoRA)": ens(o, "lr_unified_v1", "xgb_fixed_luis"),
             "A: LR + as-of": ens(o, f"asof_lr_{sfx}", "xgb_fixed_luis"),
             "B: LR + as-of, XGBoost + as-of": ens(o, f"asof_lr_{sfx}", f"asof_xgb_{sfx}")}
        ref = topk(o, V["current (unified LR + XGBoost + LoRA)"], K)
        tab = []
        for k, s in V.items():
            m = topk(o, s, K); d = [(yv[i] * m[i]).sum() - (yv[i] * ref[i]).sum() for i in Bs]
            tab.append({"ensemble": k, "hits Nov4": int(yv[m].sum()), "vs current 95% CI": f"{np.percentile(d, 2.5):+.0f}..{np.percentile(d, 97.5):+.0f}"})
        for k, c in (("LR alone, current", "lr_unified_v1"), ("LR alone + as-of", f"asof_lr_{sfx}"), ("XGBoost alone, current", "xgb_fixed_luis"), ("XGBoost alone + as-of", f"asof_xgb_{sfx}")):
            m = topk(o.assign(filt=False), o[c], K); tab.append({"ensemble": k, "hits Nov4": int(yv[m].sum()), "vs current 95% CI": ""})
        lines += [f"== {name}: {n:,} CVEs, {int(yv.sum())} Nov-4 positives, K={K}", pd.DataFrame(tab).to_string(index=False), ""]
    text = "\n".join(lines)
    print(text)
    (cv.ROOT / "reports" / "asof_eval.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
