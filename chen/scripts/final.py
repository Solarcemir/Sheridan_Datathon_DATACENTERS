"""Final models: train on ALL of 2024-01..2025-09, score the 2026 scoreboard ONCE.

  python scripts/final.py train      # LR (best, 100 features) + LightGBM (Luis's features) -> results/final_scores.parquet
  python scripts/final.py rank [--lora results/final_lora.parquet] [--cap 3]
                                     # rank average + cap -> submissions/top500_final.csv + checks

Same recipe as the CV winners (context_Chen.md §8.2):
  LR       features + meta + TF-IDF + Granite extraction; C by inner grouped CV on the training rows
  LightGBM Luis's settings (deterministic), 10 seeds, on Luis's safe features
  LoRA     Granite LoRA scores on the scoreboard, produced separately (cv_lora-style, all training rows)
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "8")
import cv  # noqa: E402
from stack import features, meta_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROC, RES, SUB = ROOT / "processed", ROOT / "results", ROOT / "submissions"
WP = {"security@wordfence.com", "audit@patchstack.com", "contact@wpscan.com"}


def scoreboard() -> pd.DataFrame:
    sb = pd.read_parquet(PROC / "scoreboard_2026_view.parquet")
    sb["label"] = 0
    return sb.reset_index(drop=True)


def blocks(df: pd.DataFrame) -> dict:
    F, M, T, L = features(df), meta_features(df), cv.team_feats(df), cv.llm_feats(df)
    return {"lr": pd.concat([F, M, T, L], axis=1), "lgbm": cv.luis_feats(df)}


def train() -> None:
    tr = cv.load()
    g = pd.read_csv(PROC / "kev_groups.csv").set_index("cve_id").group
    groups = tr.cve_id.map(g).fillna(tr.cve_id).to_numpy()
    sb = scoreboard()
    Btr, Bsb = blocks(tr), blocks(sb)
    for k in Btr:
        assert list(Btr[k].columns) == list(Bsb[k].columns), k
        nan = Bsb[k].isna().sum().sum()
        print(f"{k}: {Btr[k].shape[1]} features; scoreboard NaNs {nan}", flush=True)
    y = tr.label.to_numpy()
    sc, lr, C = cv.fit_lr(Btr["lr"].to_numpy(), y, groups)
    s_lr = lr.decision_function(sc.transform(Bsb["lr"].fillna(0).to_numpy()))
    print(f"LR trained on {len(tr):,} rows ({y.sum()} positives), C={C}", flush=True)
    Xa = np.vstack([Btr["lgbm"].to_numpy(), Bsb["lgbm"].to_numpy()])
    tr_idx, te_idx = np.arange(len(tr)), np.arange(len(tr), len(Xa))
    s_gb = cv.fit_lgbm_score(Xa, np.r_[y, np.zeros(len(sb))].astype(int), tr_idx, te_idx)
    print("LightGBM (10 seeds) done", flush=True)
    out = sb[["cve_id"]].assign(lr=s_lr, lgbm=s_gb)
    out.to_parquet(RES / "final_scores.parquet", index=False)
    coef = pd.Series(lr.coef_[0], index=Btr["lr"].columns).sort_values()
    coef.to_csv(RES / "final_lr_coefficients.csv", header=["coef_per_sd"])
    print("top + / - LR weights:\n", pd.concat([coef.tail(8), coef.head(8)]).round(2).to_string())


def rank(lora: str | None, cap: int) -> None:
    sb = scoreboard()
    s = sb[["cve_id", "assigner", "snapshot_month", "epss"] + cv.FLAGS].merge(
        pd.read_parquet(RES / "final_scores.parquet"), on="cve_id")
    cols = ["lr", "lgbm"]
    if lora:
        s = s.merge(pd.read_parquet(lora)[["cve_id", "score"]].rename(columns={"score": "lora"}), on="cve_id", how="left")
        assert s.lora.notna().all(), "LoRA scores missing for some scoreboard CVEs"
        cols.append("lora")
    s["final"] = np.mean([s[c].rank(pct=True) for c in cols], axis=0)
    s = s.sort_values("final", ascending=False).reset_index(drop=True)
    if cap:
        s = s[s.groupby(["assigner", "snapshot_month"]).cumcount() < cap].reset_index(drop=True)
    top = s.head(500).copy()
    top["rank"] = np.arange(1, 501)
    SUB.mkdir(exist_ok=True)
    tag = "final" if lora else "provisional_lr_lgbm"
    top[["cve_id"]].to_csv(SUB / f"top500_{tag}.csv", index=False)
    top.to_parquet(RES / f"top500_{tag}_scored.parquet", index=False)

    # checks
    print(f"models averaged: {cols}; cap {cap or 'none'}; wrote submissions/top500_{tag}.csv")
    print("by snapshot month:", top.snapshot_month.value_counts().sort_index().to_dict())
    print("Oracle:", int(top.assigner.str.contains("oracle", case=False).sum()),
          "| WordPress CNAs:", int(top.assigner.isin(WP).sum()),
          "| distinct assigners:", top.assigner.nunique(),
          "| max per assigner-month:", int(top.groupby(["assigner", "snapshot_month"]).size().max()))
    print("with any exploit flag:", int(top[cv.FLAGS].max(axis=1).sum()), "| EPSS median:", round(top.epss.median(), 5))
    print("top assigners:", top.assigner.value_counts().head(10).to_dict())
    luis = ROOT.parent / "luis" / "submissions" / "top500_v4.csv"
    if not luis.exists():
        luis = ROOT / "team" / "luis" / "luis" / "submissions" / "top500_v4.csv"
    if luis.exists():
        print("overlap with luis/top500_v4:", len(set(top.cve_id) & set(pd.read_csv(luis).cve_id)))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "train":
        train()
    else:
        rank(a[a.index("--lora") + 1] if "--lora" in a else None, int(a[a.index("--cap") + 1]) if "--cap" in a else 3)
