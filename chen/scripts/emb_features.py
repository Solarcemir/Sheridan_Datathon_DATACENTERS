"""Nearest-exploited-neighbour features from Granite embeddings (granite-embedding-english-r2).

For each CVE, with the same fairness cutoff as meta_features.py (end of the previous year):
  kev_sim_max / kev_sim_top5   cosine similarity to the most similar KEV entries added by the cutoff
                               (any CVE year, so ~1,100 pre-2024 KEV entries are used too)
  typ_sim_top5                 similarity to the most similar "typical" CVEs published in the
                               3 years before the cutoff (10k random sample)
  kev_contrast                 kev_sim_top5 - typ_sim_top5
  nearest_kev                  ID of the most similar KEV entry (for explanations)

  CUDA_VISIBLE_DEVICES=2 python scripts/emb_features.py embed    # embed KEV + 2021-23 sample
  python scripts/emb_features.py features                       # -> processed/emb_feat_{historical,scoreboard}.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = next((p for p in (ROOT / "IBM" / "data", ROOT.parent / "IBM-20261107T174321Z-1-001" / "IBM" / "data" / "data") if p.exists()), ROOT / "IBM" / "data")
PROC = ROOT / "processed"
N_TYP = 10_000


def embed_cmd() -> None:
    import torch
    from embed import embed
    from transformers import AutoModel, AutoTokenizer
    MODEL = "ibm-granite/granite-embedding-english-r2"
    vul = pd.read_csv(DATA / "vulnerabilities.csv", usecols=["cve_id", "published", "description"])
    vul["published"] = pd.to_datetime(vul.published)
    kev_ids = set(pd.read_csv(DATA / "known_exploited.csv").cve_id)
    sample = vul[vul.published.between("2021-01-01", "2023-12-31")].sample(40_000, random_state=0)
    todo = pd.concat([vul[vul.cve_id.isin(kev_ids)], sample]).drop_duplicates("cve_id")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    e = embed([" ".join(str(d).split()) for d in todo.description], tok, model)
    np.save(PROC / "emb_extra.npy", e)
    todo[["cve_id", "published"]].to_parquet(PROC / "emb_extra_ids.parquet", index=False)
    print(f"embedded {len(todo):,} extra descriptions ({len(kev_ids & set(todo.cve_id))} KEV)")


def load_all_embeddings() -> tuple[pd.DataFrame, np.ndarray]:
    parts, mats = [], []
    for name in ("historical", "scoreboard", "extra"):
        ids = pd.read_parquet(PROC / f"emb_{name}_ids.parquet")[["cve_id"]]
        parts.append(ids); mats.append(np.load(PROC / f"emb_{name}.npy"))
    ids = pd.concat(parts, ignore_index=True)
    E = np.concatenate(mats).astype(np.float32)
    keep = ~ids.cve_id.duplicated().to_numpy()
    return ids[keep].reset_index(drop=True), E[keep]


def topk_mean(S: np.ndarray, k: int) -> np.ndarray:
    k = min(k, S.shape[1])
    return np.sort(np.partition(S, -k, axis=1)[:, -k:], axis=1).mean(1)


def features_cmd() -> None:
    ids, E = load_all_embeddings()
    row = pd.Series(np.arange(len(ids)), index=ids.cve_id)
    vul = pd.read_csv(DATA / "vulnerabilities.csv", usecols=["cve_id", "published"])
    vul["published"] = pd.to_datetime(vul.published)
    pub = vul.set_index("cve_id").published
    kev = pd.read_csv(DATA / "known_exploited.csv", usecols=["cve_id", "date_added"])
    kev["date_added"] = pd.to_datetime(kev.date_added)
    kev = kev[kev.cve_id.isin(row.index)]
    rng = np.random.default_rng(0)

    for name, file in (("historical", "historical_2024_2025.parquet"), ("scoreboard", "scoreboard_2026_view.parquet")):
        df = pd.read_parquet(PROC / file, columns=["cve_id", "published"])
        df["cutoff"] = pd.to_datetime((df.published.dt.year - 1).astype(str) + "-12-31")
        out = []
        for cut, g in df.groupby("cutoff"):
            kv = kev[kev.date_added <= cut]
            K = E[row[kv.cve_id].to_numpy()]
            pool = [c for c in ids.cve_id if c in pub.index and cut - pd.DateOffset(years=3) < pub[c] <= cut]
            pool = rng.choice(pool, min(N_TYP, len(pool)), replace=False)
            T = E[row[pool].to_numpy()]
            Q = E[row[g.cve_id].to_numpy()]
            kv_ids = kv.cve_id.to_numpy()
            for s in range(0, len(g), 8192):
                q = Q[s:s + 8192]
                gid = g.cve_id.to_numpy()[s:s + 8192]
                SK = q @ K.T
                SK[gid[:, None] == kv_ids[None, :]] = -1          # never match itself
                ST = q @ T.T
                ST[gid[:, None] == pool[None, :]] = -1
                out.append(pd.DataFrame({
                    "cve_id": gid,
                    "kev_sim_max": SK.max(1), "kev_sim_top5": topk_mean(SK, 5),
                    "typ_sim_top5": topk_mean(ST, 5),
                    "nearest_kev": kv_ids[SK.argmax(1)],
                }))
            print(f"{name} cutoff {cut:%Y-%m-%d}: {len(g):,} CVEs vs {len(kv):,} KEV and {len(pool):,} typical", flush=True)
        f = pd.concat(out, ignore_index=True)
        f["kev_contrast"] = f.kev_sim_top5 - f.typ_sim_top5
        f.to_parquet(PROC / f"emb_feat_{name}.parquet", index=False)


if __name__ == "__main__":
    {"embed": embed_cmd, "features": features_cmd}[sys.argv[1]]()
