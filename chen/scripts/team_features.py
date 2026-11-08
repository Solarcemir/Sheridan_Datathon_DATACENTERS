"""Luis's strongest features, ported (team/luis/analysis/build_features.py):

  text_kev_lr     TF-IDF (1-2 grams, 150k, sublinear) + logistic regression (C=2, balanced): KEV
                  descriptions known before Jan 1 of the CVE's publish year (published >= 2015) vs
                  40k random non-KEV descriptions from the same span. Same year cutoff as our history
                  features, so 2026 rows use KEV <= 2025-12-31.
  text_kev_sim    max TF-IDF cosine to any of those KEV descriptions
  n_adjacent_ids  same assigner, same publish day, same ID year, ID number within 3

  python scripts/team_features.py   # -> processed/team_feat_{historical,scoreboard}.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
DATA = next((p for p in (ROOT / "IBM" / "data", ROOT.parent / "IBM-20261107T174321Z-1-001" / "IBM" / "data" / "data") if p.exists()), ROOT / "IBM" / "data")
PROC = ROOT / "processed"


def adjacent_ids(df: pd.DataFrame) -> pd.Series:
    parts = df.cve_id.str.split("-", expand=True)
    d = df.assign(id_year=parts[1], id_num=pd.to_numeric(parts[2]))
    s = d.sort_values(["assigner", "published", "id_year", "id_num"])
    adj = pd.Series(0, index=s.index)
    for k in (1, 2, 3):
        for sh in (k, -k):
            o = s.shift(sh)
            adj += ((o.assigner == s.assigner) & (o.published == s.published) & (o.id_year == s.id_year)
                    & ((o.id_num - s.id_num).abs() <= 3)).astype(int)
    return adj.reindex(df.index)


def main(targets=None, fixed_year=None) -> None:
    v = pd.read_csv(DATA / "vulnerabilities.csv", usecols=["cve_id", "published", "description"])
    v["published"] = pd.to_datetime(v.published)
    v["description"] = v.description.fillna("")
    kev = pd.read_csv(DATA / "known_exploited.csv", usecols=["cve_id", "date_added"])
    kev["date_added"] = pd.to_datetime(kev.date_added)

    models = {}
    for year in ((fixed_year,) if fixed_year else (2024, 2025, 2026)):
        start = pd.Timestamp(f"{year}-01-01")
        pos_ids = set(kev[kev.date_added < start].cve_id)
        base = v[(v.published >= "2015-01-01") & (v.published < start)]
        pos = base[base.cve_id.isin(pos_ids)]
        neg = base[~base.cve_id.isin(pos_ids)].sample(n=min(40000, len(base)), random_state=0)
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=150000, sublinear_tf=True, stop_words="english")
        Xt = vec.fit_transform(pd.concat([pos.description, neg.description]))
        lr = LogisticRegression(C=2.0, class_weight="balanced", max_iter=3000)
        lr.fit(Xt, np.r_[np.ones(len(pos)), np.zeros(len(neg))])
        models[year] = (vec, lr, Xt[: len(pos)].T.tocsc())
        print(f"text model {year}: {len(pos):,} KEV positives, {len(neg):,} negatives", flush=True)

    for name, file in (targets or (("historical", "historical_2024_2025.parquet"), ("scoreboard", "scoreboard_2026_view.parquet"))):
        df = pd.read_parquet(PROC / file, columns=["cve_id", "published", "assigner", "description"])
        df["description"] = df.description.fillna("")
        out = pd.DataFrame({"cve_id": df.cve_id.to_numpy(), "text_kev_lr": np.nan, "text_kev_sim": np.nan})
        for year, idx in (((fixed_year, df.index),) if fixed_year else df.groupby(df.published.dt.year).groups.items()):
            vec, lr, P = models[int(year)]
            Xc = vec.transform(df.loc[idx, "description"])
            pos_in = out.index[df.index.get_indexer(idx)]
            out.loc[pos_in, "text_kev_lr"] = lr.predict_proba(Xc)[:, 1]
            sim = np.zeros(Xc.shape[0])
            for i in range(0, Xc.shape[0], 4000):
                sim[i:i + 4000] = (Xc[i:i + 4000] @ P).max(axis=1).toarray().ravel()
            out.loc[pos_in, "text_kev_sim"] = sim
        out["n_adjacent_ids"] = adjacent_ids(df).to_numpy()
        out.to_parquet(PROC / f"team_feat_{name}.parquet", index=False)
        print(f"{name}: {len(out):,} rows; text_kev_lr median {out.text_kev_lr.median():.3f}", flush=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["allcves"]:
        main((("allcves", "allcves_view.parquet"),), 2026)
    else:
        main()
