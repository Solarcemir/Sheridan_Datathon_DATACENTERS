"""The final Top-500 model applied to every CVE in the IBM snapshot (lookup table for Luis's repo scanner).

Every CVE that is not on the 2026 scoreboard is scored the way the final model scores a 2026 bug: as of the end of
its publish month (EPSS / exploit flags only where the IBM signal files have that month, else unknown), with KEV
history up to Dec 31 2025 (the same history the 2026 model sees). CVEs already on KEV by the end of their publish
month are skipped (the scanner shows those as "already exploited"). Scoreboard CVEs keep their real final scores.

Voters, exactly as in the final (luis/analysis/final_build.py with the unified LR):
  tree    XGBoost (Luis's XGB_BIG settings, 40 seeds, 59 features) trained on the 2024 + 2025 replicas
  linear  unified LR v1 (scripts/lr_unified.py), refit on 2024-01..2025-09
  llm     Granite 4.2 8B LoRA, 4 seeds (results/allcves_lora_s*.parquet, scored with scripts/llm_score.py)
Each voter is turned into a percentile against its own 2026-scoreboard scores, the 3 are averaged, and the
4 Granite 30B demotion rules subtract 50 (as in the final). `top500` = the CVE would rank inside our 2026 Top 500
(ignoring the per-vendor cap).

  python scripts/all_cves.py view | luis | tree | lr | combine
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC, RES = ROOT / "processed", ROOT / "results"
DATA = ROOT / "IBM" / "data"
REPO = ROOT / "team" / "luis"
LUIS_AN = REPO / "luis" / "analysis"
FLAGS = ["exploitdb", "metasploit", "nuclei", "poc_github"]
CUT = pd.Timestamp("2025-12-31")
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


# ----------------------------------------------------------------------------- 1. the view
def view():
    v = pd.read_csv(DATA / "vulnerabilities.csv", dtype={"cvss3_version": str})
    v["published"] = pd.to_datetime(v.published)
    sb = pd.read_parquet(PROC / "scoreboard_2026_view.parquet")
    kev = pd.read_csv(DATA / "known_exploited.csv", usecols=["cve_id", "date_added"])
    kd = kev.set_index("cve_id").date_added.pipe(pd.to_datetime)
    a = v[~v.cve_id.isin(sb.cve_id)].copy()
    a["snapshot_month"] = a.published.dt.strftime("%Y-%m")
    sig = pd.concat([pd.read_csv(DATA / f"exploit_signals_{y}.csv") for y in (2024, 2025)]).rename(columns={"month": "snapshot_month"})
    a = a.merge(sig, on=["cve_id", "snapshot_month"], how="left")
    a[FLAGS] = a[FLAGS].fillna(0).astype("int8")
    end = pd.PeriodIndex(a.snapshot_month, freq="M").to_timestamp(how="end").normalize()
    on_kev = (a.in_kev.fillna(0) == 1) | (a.cve_id.map(kd) <= end)
    a = a[~on_kev & a.description.notna()].drop(columns=["in_kev"])
    cols = list(sb.columns)
    a = a[cols].sort_values("cve_id").reset_index(drop=True)
    a.to_parquet(PROC / "allcves_view.parquet", index=False)
    log(f"allcves_view: {len(a):,} CVEs (skipped {int(on_kev.sum()):,} already on KEV at their month-end); "
        f"with EPSS {a.epss.notna().mean():.1%}; years {a.published.dt.year.min()}-{a.published.dt.year.max()}")


# ----------------------------------------------------------------------------- 2. Luis's 59 features
def luis():
    sys.argv = [sys.argv[0], str(DATA)]
    sys.path.insert(0, str(LUIS_AN))
    import build_features as bf   # noqa: E402  (loads the IBM data on import; read-only use of Luis's functions)
    import common as cm           # noqa: E402
    a = pd.read_parquet(PROC / "allcves_view.parquet", columns=["cve_id", "snapshot_month", "epss"] + FLAGS)
    c = a.merge(bf.v, on="cve_id", how="left")
    c["snapshot_end"] = pd.PeriodIndex(c.snapshot_month, freq="M").to_timestamp(how="end").normalize()
    # "already on KEV" siblings: CVEs on KEV by their own month-end (what Luis calls the gap)
    gp = bf.v.assign(kev_by=bf.v.cve_id.map(bf.kev_date))
    gp = gp[gp.kev_by <= gp.published + pd.offsets.MonthEnd(0)][["cve_id", "assigner", "published", "kev_by"]]
    log(f"luis: {len(c):,} rows; building basic + siblings")
    c = bf.add_basic(c)
    c = bf.add_siblings(c, gp)
    log("luis: history (KEV before 2026-01-01)")
    c = bf.add_history(c, 2026)
    tf = pd.read_parquet(PROC / "team_feat_allcves.parquet").set_index("cve_id")   # same TF-IDF recipe, year 2026
    c["text_kev_lr"], c["text_kev_sim"] = c.cve_id.map(tf.text_kev_lr).to_numpy(), c.cve_id.map(tf.text_kev_sim).to_numpy()
    keep = bf.KEEP_ID + [f for f in bf.FEATURES if bf.FEATURES[f][1] != "leak"]
    out = c[keep].copy()
    out["days_to_cutoff"] = 180   # same constant as the scanner: no judging deadline for an old CVE
    out = cm.add_heat(cm.add_chen_lite(out, 2026))
    out.to_parquet(PROC / "luis_feat_allcves.parquet", index=False)
    log(f"luis_feat_allcves: {len(out):,} rows, {len(cm.FULL)} model features present: {all(f in out for f in cm.FULL)}")


# ----------------------------------------------------------------------------- 3. tree voter
def _xgb_job(seed, Xtr, ytr, Xs):
    import xgboost as xgb
    sys.path.insert(0, str(LUIS_AN))
    from common import XGB_BIG
    m = xgb.XGBClassifier(tree_method="hist", n_jobs=8, scale_pos_weight=25, eval_metric="aucpr", random_state=seed, **XGB_BIG)
    m.fit(Xtr, ytr)
    return [m.predict_proba(X)[:, 1] for X in Xs]


def tree():
    from joblib import Parallel, delayed
    sys.path.insert(0, str(LUIS_AN))
    import common as cm
    train = pd.concat([cm.load(2024, "2024-12-31"), cm.load(2025, "2025-12-31")], ignore_index=True)
    s26 = cm.load(2026, "2026-11-04")
    al = pd.read_parquet(PROC / "luis_feat_allcves.parquet")
    log(f"tree: train {len(train):,} ({int(train.y.sum())} pos), scoreboard {len(s26):,}, allcves {len(al):,}")
    seeds = [1000 * r + s for r in range(8) for s in range(5)]   # final_build.py's 40 seeds
    res = Parallel(n_jobs=8)(delayed(_xgb_job)(sd, train[cm.FULL], train.y, [s26[cm.FULL], al[cm.FULL]]) for sd in seeds)
    sb = np.mean([r[0] for r in res], axis=0)
    ac = np.mean([r[1] for r in res], axis=0)
    ref = pd.read_parquet(REPO / "luis" / "derived" / "xgb_bag_2026.parquet").set_index("cve_id").xgb_bag
    rho = pd.Series(sb).rank().corr(pd.Series(s26.cve_id.map(ref).to_numpy()).rank())
    log(f"tree: rank corr of our 40-seed CPU bag with the final's GPU bag on the scoreboard: {rho:.4f}")
    pd.DataFrame({"cve_id": s26.cve_id, "tree_raw": sb}).to_parquet(RES / "allcves_tree_scoreboard.parquet", index=False)
    pd.DataFrame({"cve_id": al.cve_id, "tree_raw": ac}).to_parquet(RES / "allcves_tree.parquet", index=False)


# ----------------------------------------------------------------------------- 4. linear voter
def lr():
    sys.path.insert(0, str(Path(__file__).parent))
    import cv
    import lr_unified as lu
    tr = cv.load()
    g = pd.read_csv(cv.PROC / "kev_groups.csv").set_index("cve_id").group
    groups = tr.cve_id.map(g).fillna(tr.cve_id).to_numpy()
    X, gid, cols = lu.matrix(lu.blocks(tr))
    m, C = lu.fit("v1", X, tr.label.to_numpy(), groups, gid)
    log(f"lr: unified v1 refit, C={C}")
    out = {}
    for name, f in (("scoreboard", "scoreboard_2026_view.parquet"), ("allcves", "allcves_view.parquet")):
        df = pd.read_parquet(PROC / f).reset_index(drop=True)
        Xs, _, c2 = lu.matrix(lu.blocks(df))
        assert c2 == cols
        out[name] = pd.DataFrame({"cve_id": df.cve_id, "lr_raw": m.score(Xs)})
    pushed = pd.read_parquet(REPO / "chen" / "submissions" / "lr_unified_scoreboard_2026.parquet").set_index("cve_id").lr_score
    d = np.abs(out["scoreboard"].lr_raw.to_numpy() - out["scoreboard"].cve_id.map(pushed).to_numpy()).max()
    log(f"lr: max |difference| vs the pushed final scoreboard scores: {d:.2e}")
    out["scoreboard"].to_parquet(RES / "allcves_lr_scoreboard.parquet", index=False)
    out["allcves"].to_parquet(RES / "allcves_lr.parquet", index=False)


# ----------------------------------------------------------------------------- 5. combine
def pct_vs(ref, x):
    """Percentile of x within the reference (2026 scoreboard) distribution, 0..1."""
    r = np.sort(np.asarray(ref, float))
    return np.searchsorted(r, np.asarray(x, float), side="right") / len(r)


def combine():
    final = pd.read_csv(REPO / "luis" / "derived" / "scored_2026_final.csv", usecols=["cve_id", "tree", "llm", "granite_demoted"])
    s = pd.read_parquet(PROC / "scoreboard_2026_view.parquet", columns=["cve_id"]).merge(final, on="cve_id")
    u = pd.read_parquet(REPO / "chen" / "submissions" / "lr_unified_scoreboard_2026.parquet")
    s = s.merge(u, on="cve_id")
    s["linear"] = s.lr_score.rank(pct=True)
    s["raw"] = s[["tree", "linear", "llm"]].mean(axis=1) - 50 * s.granite_demoted
    thr = s.raw.nlargest(500).min()

    a = pd.read_parquet(PROC / "allcves_view.parquet", columns=["cve_id", "published"])
    tr_sb, tr = pd.read_parquet(RES / "allcves_tree_scoreboard.parquet"), pd.read_parquet(RES / "allcves_tree.parquet")
    lr_sb, lrr = pd.read_parquet(RES / "allcves_lr_scoreboard.parquet"), pd.read_parquet(RES / "allcves_lr.parquet")
    lo_sb = pd.read_parquet(REPO / "chen" / "submissions" / "lora8b_scoreboard_2026.parquet")   # 4-seed mean pct rank
    ref_seeds = [RES / "final_lora8b_scoreboard.parquet"] + [RES / f"final_lora8b_s{k}_scoreboard.parquet" for k in (1, 2, 3)]
    lo = []
    for k, rf in enumerate(ref_seeds):
        parts = sorted((RES / "allcves_lora_parts").glob(f"s{k}_*of*.parquet"))
        if not parts:
            continue
        x = pd.concat([pd.read_parquet(p) for p in parts]).drop_duplicates("cve_id").set_index("cve_id").score
        log(f"LoRA seed {k}: {len(x):,} of {len(a):,} scored")
        r = pd.read_parquet(rf).score
        lo.append(pct_vs(r, a.cve_id.map(x)))
    a["tree"] = pct_vs(tr_sb.tree_raw, a.cve_id.map(tr.set_index("cve_id").tree_raw))
    a["linear"] = pct_vs(lr_sb.lr_raw, a.cve_id.map(lrr.set_index("cve_id").lr_raw))
    a["llm"] = pct_vs(lo_sb.lora_score, np.mean(lo, axis=0)) if lo else np.nan
    ex = pd.concat([pd.read_parquet(PROC / "llm_extract.parquet")] + [pd.read_parquet(f) for f in sorted(RES.glob("xall_*of3.parquet"))]).drop_duplicates("cve_id").set_index("cve_id")
    e = ex.reindex(a.cve_id)
    a["granite_demoted"] = ((e.impact == "denial_of_service") | (e.typical_attacker == "unlikely_target") | (e.attacker_appeal == "low")
                            | (e.product_type == "wordpress_plugin_or_theme")).fillna(False).to_numpy()
    a["raw"] = a[["tree", "linear", "llm"]].mean(axis=1) - 50 * a.granite_demoted
    a["source"] = "rescored as a 2026 bug"
    s["source"] = "2026 scoreboard (final model)"
    cols = ["cve_id", "tree", "linear", "llm", "granite_demoted", "raw", "source"]
    t = pd.concat([s[cols], a[cols]], ignore_index=True)
    sb_raw = np.sort(s.raw.to_numpy())
    t["score"] = (100 * np.searchsorted(sb_raw, t.raw.to_numpy(), side="right") / len(sb_raw)).round(2)   # percentile vs 2026
    t["top500"] = t.raw >= thr
    e2 = ex.reindex(t.cve_id)
    for k in ("vendor", "product", "product_type", "impact", "typical_attacker", "attacker_appeal"):
        t[k] = e2[k].to_numpy()
    for k in ("tree", "linear", "llm"):
        t[k] = t[k].round(4)
    out = REPO / "chen" / "submissions" / "final_model_all_cves.parquet"
    t.drop(columns=["raw"]).to_parquet(out, index=False)
    log(f"wrote {out.relative_to(REPO)}: {len(t):,} CVEs ({len(s):,} scoreboard + {len(a):,} rescored), "
        f"Top-500-level {int(t.top500.sum()):,} (scoreboard {int(t.top500[:len(s)].sum())}), "
        f"{out.stat().st_size / 1e6:.1f} MB; LoRA seeds used: {len(lo)}")


# ----------------------------------------------------------------------------- 6. browser files for the static scanner
CATS = {"product_type": None, "impact": None, "typical_attacker": None, "attacker_appeal": None}


def export():
    """chen/frontend/scan_data/{year}.json: {"<number>": [score10, bits, epss_milli, tree, linear, llm, pt, im, ta, ap]}
    score10 = 10 x percentile vs the 2026 scoreboard (-1: not scored, already on KEV at publish); bits: 1 Top-500 level,
    2 Granite-demoted, 4 on CISA KEV (IBM snapshot), 8/16/32/64 Exploit-DB/Metasploit/Nuclei/GitHub PoC at the snapshot,
    128 on the 2026 scoreboard. Only derived numbers and Granite's labels: no IBM text leaves this script."""
    import json
    t = pd.read_parquet(REPO / "chen" / "submissions" / "final_model_all_cves.parquet")
    views = pd.concat([pd.read_parquet(PROC / "scoreboard_2026_view.parquet", columns=["cve_id", "epss"] + FLAGS),
                       pd.read_parquet(PROC / "allcves_view.parquet", columns=["cve_id", "epss"] + FLAGS)]).drop_duplicates("cve_id").set_index("cve_id")
    kev = set(pd.read_csv(DATA / "known_exploited.csv", usecols=["cve_id"]).cve_id)
    t = t.set_index("cve_id")
    ids = sorted(set(t.index) | kev)
    cats = {k: sorted(t[k].dropna().unique().tolist()) for k in CATS}
    idx = {k: {v: i for i, v in enumerate(vals)} for k, vals in cats.items()}
    out_dir = REPO / "chen" / "frontend" / "scan_data"
    out_dir.mkdir(exist_ok=True)
    by_year: dict = {}
    vv = views.reindex(ids)
    tt = t.reindex(ids)
    for cid, r, v in zip(ids, tt.itertuples(), vv.itertuples()):
        y, n = cid.split("-")[1], cid.split("-")[2]
        scored = pd.notna(r.score)
        bits = (1 if scored and r.top500 else 0) | (2 if scored and r.granite_demoted else 0) | (4 if cid in kev else 0) \
            | sum(b for f, b in zip(FLAGS, (8, 16, 32, 64)) if pd.notna(getattr(v, f)) and getattr(v, f) == 1) \
            | (128 if scored and str(r.source).startswith("2026") else 0)
        rec = [int(round(r.score * 10)) if scored else -1, bits, int(round(v.epss * 1000)) if pd.notna(v.epss) else -1,
               *[int(round(100 * getattr(r, k))) if pd.notna(getattr(r, k)) else -1 for k in ("tree", "linear", "llm")],
               *[idx[k].get(getattr(r, k), -1) for k in CATS]]
        by_year.setdefault(y, {})[n] = rec
    for y, d in by_year.items():
        (out_dir / f"{y}.json").write_text(json.dumps(d, separators=(",", ":")))
    sb = t[t.source.str.startswith("2026")]
    meta = {"cats": cats, "years": sorted(by_year), "n": len(ids), "built": time.strftime("%Y-%m-%d %H:%M"),
            "top500_min_score": float(sb[sb.top500].score.min())}
    (out_dir / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    size = sum(f.stat().st_size for f in out_dir.glob("*.json")) / 1e6
    log(f"scan_data: {len(ids):,} CVEs in {len(by_year)} year files, {size:.1f} MB; Top-500 level from score {meta['top500_min_score']}")


if __name__ == "__main__":
    {"view": view, "luis": luis, "tree": tree, "lr": lr, "combine": combine, "export": export}[sys.argv[1]]()
