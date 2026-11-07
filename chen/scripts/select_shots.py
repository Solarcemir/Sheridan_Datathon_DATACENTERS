"""Pick many-shot examples: all positives + semantic near-misses + representative negatives.

  near-miss       For each positive, the negative with the most similar description
                  (Granite embedding cosine), among negatives with the same attack vector
                  and CVSS 3 within +-CALIPER. Sampled without replacement.
  representative  k-medoid-style: KMeans (k = #positives) on the remaining negatives'
                  embeddings, then the real CVE closest to each centroid.

Also prints a balance table: for each feature, the positive-vs-negative gap (standardised
mean difference) in the whole pool and in the chosen examples. The sign should agree.

  python scripts/select_shots.py --months 2024-01:2024-06 --out processed/shots_2024h1.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import prompts

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "processed"
FLAGS = ["exploitdb", "metasploit", "nuclei", "poc_github"]
CALIPER = 1.5
WP_ASSIGNERS = {"security@wordfence.com", "audit@patchstack.com", "contact@wpscan.com"}


def load_pool(months: str) -> tuple[pd.DataFrame, np.ndarray]:
    hist = prompts.add_epss_percentile(pd.read_parquet(PROC / "historical_2024_2025.parquet"))
    ids = pd.read_parquet(PROC / "emb_historical_ids.parquet").cve_id
    emb = np.load(PROC / "emb_historical.npy").astype(np.float32)
    row = pd.Series(np.arange(len(ids)), index=ids)
    lo, hi = months.split(":")
    pool = hist[hist.snapshot_month.between(lo, hi)].reset_index(drop=True)
    return pool, emb[row[pool.cve_id].to_numpy()]


def near_misses(pool: pd.DataFrame, emb: np.ndarray) -> list[int]:
    """Returns one negative row index per positive, aligned with the positives' order."""
    pos = np.flatnonzero(pool.label.to_numpy() == 1)
    neg = np.flatnonzero(pool.label.to_numpy() == 0)
    sims = emb[pos] @ emb[neg].T                      # (n_pos, n_neg)
    cv = pool.cvss3_score.to_numpy()
    av = pool.attack_vector.fillna("?").to_numpy()
    taken, chosen = np.zeros(len(neg), bool), [None] * len(pos)
    # most constrained positives first: fewest eligible candidates
    elig = []
    for p in pos:
        ok = av[neg] == av[p]
        if not np.isnan(cv[p]):
            ok &= np.abs(cv[neg] - cv[p]) <= CALIPER
        elig.append(ok)
    for k in np.argsort([e.sum() for e in elig]):
        ok = elig[k] & ~taken
        if not ok.any():                              # caliper too strict: fall back to any
            ok = ~taken
        j = np.flatnonzero(ok)[np.argmax(sims[k, ok])]
        taken[j] = True
        chosen[k] = int(neg[j])
    return chosen


def representatives(pool: pd.DataFrame, emb: np.ndarray, exclude: list[int], k: int, seed: int) -> list[int]:
    neg = np.setdiff1d(np.flatnonzero(pool.label.to_numpy() == 0), exclude)
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(emb[neg])
    out = []
    for c in range(k):
        members = neg[km.labels_ == c]
        d = ((emb[members] - km.cluster_centers_[c]) ** 2).sum(1)
        out.append(int(members[np.argmin(d)]))
    return out


def features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "any exploit code": df[FLAGS].max(axis=1),
        "EPSS percentile": df.epss_pct,
        "CVSS 3 score": df.cvss3_score,
        "network attack": (df.attack_vector == "NETWORK").astype(float),
        "no privileges needed": (df.privileges_required == "NONE").astype(float),
        "no user interaction": (df.user_interaction == "NONE").astype(float),
        "WordPress-plugin CNA": df.assigner.isin(WP_ASSIGNERS).astype(float),
        "CVSS 4 present": df.cvss4_score.notna().astype(float),
    })


def smd(f: pd.DataFrame, y: np.ndarray) -> pd.Series:
    a, b = f[y == 1], f[y == 0]
    sd = np.sqrt((a.var() + b.var()) / 2).replace(0, np.nan)
    return (a.mean() - b.mean()) / sd


def balance(pool: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    fp, fs = features(pool), features(shots)
    yp, ys = pool.label.to_numpy(), shots.label.to_numpy()
    t = pd.DataFrame({
        "pool pos": fp[yp == 1].mean(), "pool neg": fp[yp == 0].mean(),
        "shot pos": fs[ys == 1].mean(),
        "near-miss": fs[(shots.group == "near-miss").to_numpy()].mean(),
        "representative": fs[(shots.group == "representative").to_numpy()].mean(),
        "SMD pool": smd(fp, yp), "SMD shots": smd(fs, ys),
    })
    t["same sign"] = np.sign(t["SMD pool"]) == np.sign(t["SMD shots"])
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", required=True, help="YYYY-MM:YYYY-MM, inclusive")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pool, emb = load_pool(args.months)
    pos = list(np.flatnonzero(pool.label.to_numpy() == 1))
    nm = near_misses(pool, emb)
    rep = representatives(pool, emb, nm, len(pos), args.seed)
    shots = pd.concat([
        pool.iloc[pos].assign(group="positive", pair=None),
        pool.iloc[nm].assign(group="near-miss", pair=pool.cve_id.to_numpy()[pos]),
        pool.iloc[rep].assign(group="representative", pair=None),
    ]).sample(frac=1, random_state=args.seed)
    shots.to_parquet(args.out, index=False)

    sim = (emb[pos] * emb[nm]).sum(1)
    print(f"pool {args.months}: {len(pool):,} CVEs, {len(pos)} positives -> {len(shots)} shots")
    print(f"near-miss cosine: median {np.median(sim):.3f}, min {sim.min():.3f}")
    same_cna = (pool.assigner.to_numpy()[pos] == pool.assigner.to_numpy()[nm]).mean()
    print(f"near-miss shares the positive's assigner: {100 * same_cna:.0f}%\n")
    pd.set_option("display.width", 200)
    print(balance(pool, shots).round(2).to_string())

    print("\nexample near-miss pairs:")
    for p, n in list(zip(pos, nm))[:4]:
        for tag, i in (("KEV ", p), ("not ", n)):
            print(f"  {tag} {pool.assigner[i][:28]:28} {str(pool.description[i])[:110]}")
        print()


if __name__ == "__main__":
    main()
