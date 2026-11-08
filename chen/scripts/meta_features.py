"""Publish-time metadata features, computed identically for 2024-25 training rows and 2026.

Fairness rules (README "Keep it fair"):
  - KEV history uses known_exploited.csv only up to the END OF THE PREVIOUS YEAR. 2026 CVEs can
    only see KEV <= 2025-12-31 (the file stops there), so 2024 rows see KEV <= 2023-12-31 and
    2025 rows KEV <= 2024-12-31. Same staleness everywhere.
  - Nothing NVD fills in later (status, products, references, CVSS source).
  - No "exploited in the wild" description text and no CVSS4 E (exploit maturity): descriptions
    and vectors are today's versions and may have been edited after the attacks.
  - Nothing that needs monthly exploit signals for other CVEs (there are none for 2026).

Writes processed/meta_historical.parquet and processed/meta_scoreboard.parquet (keyed by cve_id).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = next((p for p in (ROOT / "IBM" / "data", ROOT.parent / "IBM-20261107T174321Z-1-001" / "IBM" / "data" / "data") if p.exists()), ROOT / "IBM" / "data")
PROC = ROOT / "processed"
FLAGS = ["exploitdb", "metasploit", "nuclei", "poc_github"]
P0 = 0.0016            # overall KEV rate; prior for smoothing
M = 1 / P0             # prior strength: ~625 pseudo-CVEs


def smoothed(k, n):
    return (k + 1.0) / (n + M)


GENERIC_PRODUCTS = {"multiple products", "multiple devices", "multiple chipsets", "kernel",
                    "mobile devices", "multiple routers", "multiple firewalls", "server", "unknown"}


def name_index(kev: pd.DataFrame) -> dict:
    """KEV vendor and product names -> list of (date_added, cve_id, ransom). Product names are
    cleaned (parentheticals and 'and'-lists split) and generic ones dropped."""
    idx = {"vendor": {}, "product": {}}
    for r in kev.itertuples():
        if isinstance(r.vendor, str) and len(r.vendor) >= 3:
            idx["vendor"].setdefault(r.vendor.lower(), []).append((r.date_added, r.cve_id, r.ransom))
        if isinstance(r.product, str):
            base = re.sub(r"\(.*?\)", "", r.product).lower()
            for part in re.split(r",| and |/", base):
                part = part.strip()
                if len(part) >= 4 and part not in GENERIC_PRODUCTS:
                    idx["product"].setdefault(part, []).append((r.date_added, r.cve_id, r.ransom))
    return idx


def name_history(desc: pd.Series, cutoffs: pd.Series, idx: dict):
    """For each description: KEV entries (added by its cutoff) whose vendor / product name it mentions."""
    out = {}
    for kind in ("vendor", "product"):
        keys = sorted(idx[kind], key=len, reverse=True)
        rx = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keys) + r")\b")
        out[kind] = [set(rx.findall(t)) for t in desc]
    vk, pk, rk = [], [], []
    for vs, ps, c in zip(out["vendor"], out["product"], cutoffs):
        ev = {(cid, rn) for k in vs for (dt, cid, rn) in idx["vendor"][k] if dt <= c}
        ep = {(cid, rn) for k in ps for (dt, cid, rn) in idx["product"][k] if dt <= c}
        vk.append(len(ev)); pk.append(len(ep)); rk.append(sum(rn for _, rn in ev | ep))
    return vk, pk, rk


def sibling_max(df: pd.DataFrame, grp: list, col: str) -> pd.Series:
    """Max of col over same-group rows, excluding the row itself (0 if alone)."""
    g = df.groupby(grp)[col]
    top1 = g.transform("max")
    cnt_top = df[col].eq(top1).groupby([x for x in grp]).transform("sum")
    second = df[col].where(df[col] < top1).groupby([x for x in grp]).transform("max").fillna(0)
    alone = g.transform("size") == 1
    is_top = df[col].eq(top1)
    res = np.where(alone, 0.0, np.where(is_top & (cnt_top == 1), second, top1))
    return pd.Series(res, index=df.index)


def main(views=None, fixed_cutoff=None) -> None:
    vul = pd.read_csv(DATA / "vulnerabilities.csv",
                      usecols=["cve_id", "published", "assigner", "description", "cwe_ids"])
    vul["published"] = pd.to_datetime(vul.published)
    kev = pd.read_csv(DATA / "known_exploited.csv",
                      usecols=["cve_id", "date_added", "vendor", "product", "known_ransomware_use"])
    kev["date_added"] = pd.to_datetime(kev.date_added)
    kev = kev.merge(vul[["cve_id", "assigner", "cwe_ids"]], on="cve_id", how="left")
    kev["ransom"] = kev.known_ransomware_use.eq("Known")
    names = name_index(kev)
    wt = pd.read_csv(DATA / "weakness_types.csv", usecols=["cwe_id", "abstraction"])
    abstraction = dict(zip(wt.cwe_id, wt.abstraction))

    # same-day batch facts over every published CVE (publish-time information)
    day = vul.groupby(["assigner", "published"]).cve_id.transform("size")
    twins = vul.groupby(["assigner", "published", "description"]).cve_id.transform("size")
    vul["batch_size"], vul["twins"] = day, twins
    batch = vul.set_index("cve_id")[["batch_size", "twins"]]

    def history(cutoffs: list[pd.Timestamp]) -> dict:
        """Per cutoff: assigner and CWE KEV counts/rates, assigner recency."""
        out = {}
        for c in cutoffs:
            kv = kev[kev.date_added <= c]
            recent = vul[(vul.published <= c) & (vul.published > c - pd.DateOffset(years=3))]
            a_k = kv.groupby("assigner").size()
            a_n = recent.groupby("assigner").size()
            a_last = kv.groupby("assigner").date_added.max()
            a_ransom = kv[kv.ransom].groupby("assigner").size()
            cw_k = kv.cwe_ids.dropna().str.split(";").explode().value_counts()
            cw_n = recent.cwe_ids.dropna().str.split(";").explode().value_counts()
            out[c] = (a_k, a_n, a_last, cw_k, cw_n, a_ransom)
        return out

    def build(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["cutoff"] = pd.Timestamp(fixed_cutoff) if fixed_cutoff else pd.to_datetime((df.published.dt.year - 1).astype(str) + "-12-31")
        H = history(sorted(df.cutoff.unique()))
        a_k, a_n, a_rate, a_days, cwe_rate, a_ransom = [], [], [], [], [], []
        for r in df[["assigner", "cwe_ids", "cutoff"]].itertuples(index=False):
            ak, an, alast, cwk, cwn, arn = H[r.cutoff]
            a_ransom.append(int(arn.get(r.assigner, 0)))
            k, n = int(ak.get(r.assigner, 0)), int(an.get(r.assigner, 0))
            a_k.append(k); a_n.append(n); a_rate.append(smoothed(k, n))
            last = alast.get(r.assigner)
            a_days.append(min((r.cutoff - last).days, 2000) if last is not None and pd.notna(last) else 2000)
            cw = [c for c in str(r.cwe_ids).split(";")] if isinstance(r.cwe_ids, str) else []
            cwe_rate.append(max([smoothed(int(cwk.get(c, 0)), int(cwn.get(c, 0))) for c in cw], default=P0))
        d = df.description.fillna("").str.lower()
        b = batch.reindex(df.cve_id)
        code = df[FLAGS].max(axis=1)
        grp = [df.assigner, df.published]
        sib_code = (df.assign(c=code).groupby(grp).c.transform("sum") - code) > 0
        v4 = df.cvss4_vector.fillna("")
        vk, pk, rk = name_history(d, df.cutoff, names)
        epct = df.groupby("snapshot_month").epss.rank(pct=True).fillna(0)
        sib_epss = sibling_max(df.assign(_e=epct), grp, "_e")
        cw_abs = df.cwe_ids.fillna("").str.split(";").map(
            lambda cs: [abstraction.get(c) for c in cs if c.startswith("CWE-")])
        c4 = lambda key: v4.str.extract(r"/" + key + r":([A-Za-z]+)")[0]
        meta = pd.DataFrame({
            "cve_id": df.cve_id.to_numpy(),
            "assigner_kev": a_k, "assigner_cves_3y": a_n, "assigner_kev_rate": a_rate,
            "assigner_days_since_kev": a_days,
            "cwe_kev_rate": cwe_rate,
            "batch_size": b.batch_size.fillna(1).astype(int).to_numpy(),
            "out_of_band": (b.batch_size.fillna(1) == 1).to_numpy(),
            "twins": (b.twins.fillna(1).astype(int) - 1).to_numpy(),
            "sibling_has_code": sib_code.to_numpy(),
            "id_year_lag": (df.published.dt.year - df.cve_id.str[4:8].astype(int)).to_numpy(),
            "patch_tuesday": ((df.published.dt.weekday == 1) & df.published.dt.day.between(8, 14)).to_numpy(),
            "eol_text": d.str.contains(r"end[- ]of[- ]life|\beol\b|no longer supported|unsupported when assigned").to_numpy(),
            "unauth_text": d.str.contains("unauthenticated").to_numpy(),
            "cvss4_urgency": v4.str.extract(r"/U:([A-Za-z]+)")[0].fillna("").to_numpy(),
            "cvss4_automatable": (v4.str.extract(r"/AU:([A-Z])")[0] == "Y").to_numpy(),
            "kev_cutoff": df.cutoff.dt.strftime("%Y-%m-%d").to_numpy(),
            # --- second batch
            "vendor_name_kev": vk, "product_name_kev": pk, "named_ransomware_kev": rk,
            "assigner_ransomware_kev": a_ransom,
            "sibling_max_epss_pct": sib_epss.to_numpy(),
            "rce_text": d.str.contains(r"remote code execution|execute arbitrary (?:code|commands)|arbitrary code execution|command execution").to_numpy(),
            "auth_bypass_text": d.str.contains(r"authentication bypass|bypass(?:es)? (?:the )?authentication|bypass(?:es)? (?:the )?login").to_numpy(),
            "preauth_text": d.str.contains(r"pre-?auth|without authentication|unauthenticated|no authentication").to_numpy(),
            "zero_click_text": d.str.contains(r"zero[- ]click|without (?:any )?user interaction|no user interaction").to_numpy(),
            "n_versions": d.str.count(r"\b\d+\.\d+(?:\.\d+)*\b").to_numpy(),
            "cvss4_at_none": (c4("AT") == "N").to_numpy(),
            "cvss4_subsequent_high": ((c4("SC") == "H") | (c4("SI") == "H") | (c4("SA") == "H")).to_numpy(),
            "cvss4_value_concentrated": (c4("V") == "C").to_numpy(),
            "cwe_unknown": (df.cwe_ids.isna() | df.cwe_ids.fillna("").str.contains("NVD-CWE")).to_numpy(),
            "cwe_broad": cw_abs.map(lambda a: any(x in ("Class", "Pillar") for x in a)).to_numpy(),
        })
        return meta

    if views is None:
        views = (("historical", pd.read_parquet(PROC / "historical_2024_2025.parquet")),
                 ("scoreboard", pd.read_parquet(PROC / "scoreboard_2026_view.parquet")))
    for name, df in views:
        m = build(df)
        m.to_parquet(PROC / f"meta_{name}.parquet", index=False)
        print(f"{name}: {len(m):,} rows; KEV cutoffs {sorted(m.kev_cutoff.unique())}", flush=True)
        print(m.drop(columns=["cve_id", "kev_cutoff", "cvss4_urgency"]).astype(float).describe()
              .loc[["mean", "50%", "max"]].round(4).to_string(), "\n")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["allcves"]:   # every other CVE, scored like a 2026 bug: KEV history to Dec 31 2025
        main([("allcves", pd.read_parquet(PROC / "allcves_view.parquet"))], "2025-12-31")
    else:
        main()
