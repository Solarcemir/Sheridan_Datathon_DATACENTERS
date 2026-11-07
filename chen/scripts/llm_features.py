"""Turn Granite's structured answers (processed/llm_extract.parquet) into linear-model features.

  - one-hot product_type / used_by / precondition / impact
  - ordinal internet_exposed, install_base, attacker_appeal; end_of_life flag
  - product-level KEV history: KEV entries (added by the end of the previous year, same fairness
    cutoff as meta_features.py) whose CISA vendor matches Granite's normalised vendor, and whose
    CISA product also matches Granite's product.

  python scripts/llm_features.py    # -> processed/llm_feat_{historical,scoreboard}.parquet
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from extract import FIELDS

ROOT = Path(__file__).resolve().parent.parent
DATA = next((p for p in (ROOT / "IBM" / "data", ROOT.parent / "IBM-20261107T174321Z-1-001" / "IBM" / "data" / "data") if p.exists()), ROOT / "IBM" / "data")
PROC = ROOT / "processed"
ORD = {
    "internet_exposed": {"typically_internet_facing": 2, "sometimes": 1, "rarely": 0},
    "install_base": {"very_large": 3, "large": 2, "moderate": 1, "small_or_niche": 0},
    "attacker_appeal": {"high": 2, "medium": 1, "low": 0},
}
ONEHOT = ["product_type", "used_by", "precondition", "impact", "fix_status", "typical_attacker",
          "exploitation_style"]


def norm(s) -> str:
    s = re.sub(r"\(.*?\)", " ", str(s).lower())
    s = re.sub(r"\b(inc|corp|corporation|ltd|llc|co|software|systems|technologies|networks|the)\b", " ", s)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def matches(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return a == b or (len(a) >= 4 and a in b) or (len(b) >= 4 and b in a)


def main() -> None:
    ex = pd.read_parquet(PROC / "llm_extract.parquet")
    ex["v"], ex["p"] = ex.vendor.map(norm), ex["product"].map(norm)
    kev = pd.read_csv(DATA / "known_exploited.csv", usecols=["cve_id", "vendor", "product", "date_added", "known_ransomware_use"])
    kev["date_added"] = pd.to_datetime(kev.date_added)
    kev["v"], kev["p"] = kev.vendor.map(norm), kev["product"].map(norm)
    kev["ransom"] = kev.known_ransomware_use.eq("Known")
    kev_v = {v: g for v, g in kev.groupby("v")}

    # cache per (vendor, product): the matching KEV rows
    cache: dict = {}

    def kev_rows(v: str, p: str):
        key = (v, p)
        if key not in cache:
            vs = [kv for kv in kev_v if matches(v, kv)]
            g = pd.concat([kev_v[kv] for kv in vs]) if vs else kev.iloc[:0]
            gp = g[[matches(p, kp) for kp in g.p]] if len(g) else g
            cache[key] = (g, gp)
        return cache[key]

    for name, file in (("historical", "historical_2024_2025.parquet"), ("scoreboard", "scoreboard_2026_view.parquet")):
        df = pd.read_parquet(PROC / file, columns=["cve_id", "published"]).merge(ex, on="cve_id", how="inner")
        cut = pd.to_datetime((df.published.dt.year - 1).astype(str) + "-12-31")
        vk, pk, pr = [], [], []
        for v, p, c in zip(df.v, df.p, cut):
            g, gp = kev_rows(v, p)
            vk.append(int((g.date_added <= c).sum()))
            pk.append(int((gp.date_added <= c).sum()))
            pr.append(int(((gp.date_added <= c) & gp.ransom).sum()))
        out = pd.DataFrame({"cve_id": df.cve_id.to_numpy()})
        out["llm_vendor_kev"] = np.log1p(vk)
        out["llm_product_kev"] = np.log1p(pk)
        out["llm_product_ransomware_kev"] = np.log1p(pr)
        for k, m in ORD.items():
            out[f"llm_{k}"] = df[k].map(m).fillna(np.mean(list(m.values()))).to_numpy()
        out["llm_end_of_life"] = (df.end_of_life == "yes").astype(float).to_numpy()
        for k in ONEHOT:
            for val in FIELDS[k]:
                out[f"llm_{k}={val}"] = (df[k] == val).astype(float).to_numpy()
        out.to_parquet(PROC / f"llm_feat_{name}.parquet", index=False)
        print(f"{name}: {len(out):,} rows, {out.shape[1] - 1} features; "
              f"product KEV history >0 for {100 * (np.array(pk) > 0).mean():.1f}%", flush=True)


if __name__ == "__main__":
    main()
