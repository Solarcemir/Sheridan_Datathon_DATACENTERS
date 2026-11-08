"""Point-in-time NVD features: each CVE's NVD record as it stood at the END of its snapshot month, rebuilt from the
public NVD CVE Change History API (external/nvd_history, scripts/fetch_nvd_history.py).

Leak guards:
  - only events dated <= the CVE's snapshot month-end (June uses the July snapshot, as on the scoreboard);
  - "CVE CISA KEV Update" events are dropped entirely (that is the label);
  - KEV-related reference/description content is not used (no cisa.gov/known-exploited links).

Features (per CVE, as of its snapshot end):
  nv_analyzed        NVD Initial Analysis done by then (0/1)
  nv_days_to_anal    days from receipt to Initial Analysis (NaN if not analyzed)
  nv_n_refs          references attached
  nv_refs_added      references added after the first record (people writing about it during the month)
  nv_ref_exploit     any reference NVD tagged "Exploit"
  nv_ref_news        reference to a security news site
  nv_ref_intel       reference to a threat-intel / exploitation write-up blog
  nv_ref_gov         reference to a government CERT advisory (not the KEV catalog)
  nv_n_modified      "CVE Modified" events (record edits by the CNA)
  nv_n_cpe           CPE configuration entries added by NVD
  nv_desc_itw        description at that time says exploited in the wild / actively exploited
  nv_desc_changed    description edited during the month
  nv_anal_rank       nv_analyzed relative to its month: share of that month's CVEs analyzed (drift guard)

  python scripts/asof_nvd.py  -> processed/asof_nvd_{historical,scoreboard}.parquet
"""

import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC, HIST = ROOT / "processed", ROOT / "external" / "nvd_history"
NEWS = re.compile(r"bleepingcomputer|securityweek|thehackernews|therecord\.media|darkreading|helpnetsecurity|theregister|"
                  r"securityaffairs|arstechnica|zdnet|techcrunch|cyberscoop|infosecurity-magazine|bankinfosecurity|wired\.com", re.I)
INTEL = re.compile(r"unit42|mandiant|rapid7\.com/blog|horizon3|watchtowr|attackerkb|greynoise|huntress|crowdstrike|sophos\.com/.*/blog|"
                   r"trendmicro|zerodayinitiative|volexity|sentinelone|securelist|welivesecurity|blog\.talosintelligence|"
                   r"cloud\.google\.com/blog/topics/threat|microsoft\.com/en-us/security/blog|censys|shadowserver|vulncheck\.com/blog", re.I)
GOV = re.compile(r"cert\.org|ncsc\.|cert\.europa|jvn\.jp|cert-in|bsi\.bund|cert\.ssi|cyber\.gov\.au|cisa\.gov/news-events/(ics-)?advisor", re.I)
ITW = re.compile(r"exploited in the wild|actively exploited|in-the-wild|has been exploited|may have been (actively )?exploited|"
                 r"limited,? targeted exploitation|exploitation (has been )?detected|aware of (reports|an exploit)", re.I)
URL = re.compile(r"https?://\S+")


def snap_end(m: pd.Series) -> pd.Series:
    return pd.PeriodIndex(m, freq="M").to_timestamp(how="end").to_series(index=m.index)


def main() -> None:
    h = pd.read_parquet(PROC / "historical_2024_2025.parquet", columns=["cve_id", "snapshot_month", "published"]).assign(which="historical")
    s = pd.read_parquet(PROC / "scoreboard_2026_view.parquet", columns=["cve_id", "snapshot_month", "published"]).assign(which="scoreboard")
    df = pd.concat([h, s], ignore_index=True)
    end = dict(zip(df.cve_id, snap_end(df.snapshot_month)))
    want = set(df.cve_id)
    st = defaultdict(lambda: {"recv": None, "anal": None, "refs": set(), "added": 0, "exploit": 0, "mod": 0, "cpe": 0, "desc": "", "desc_n": 0})
    n_ev = n_kev = n_late = 0
    for f in sorted(HIST.glob("*.jsonl.gz")):
        for line in gzip.open(f, "rt"):
            e = json.loads(line)
            c = e.get("cveId")
            if c not in want:
                continue
            n_ev += 1
            if e.get("eventName") == "CVE CISA KEV Update":
                n_kev += 1
                continue
            t = pd.Timestamp(e["created"])
            if t > end[c]:
                n_late += 1
                continue
            x, ev = st[c], e.get("eventName")
            if ev in ("New CVE Received", "CVE Received") and x["recv"] is None:
                x["recv"] = t
            if ev == "Initial Analysis" and x["anal"] is None:
                x["anal"] = t
            if ev == "CVE Modified":
                x["mod"] += 1
            for d in e.get("details", []):
                ty, act, nv = d.get("type"), d.get("action"), d.get("newValue") or ""
                if ty == "Reference":
                    u = URL.search(nv)
                    if u:
                        if act == "Removed":
                            x["refs"].discard(u.group(0))
                        else:
                            if x["recv"] is not None and ev not in ("New CVE Received", "CVE Received"):
                                x["added"] += 1
                            x["refs"].add(u.group(0))
                elif ty == "Reference Type" and re.search(r"\bExploit\b", nv):
                    x["exploit"] = 1
                elif ty == "CPE Configuration" and act == "Added":
                    x["cpe"] += 1
                elif ty == "Description" and act in ("Added", "Changed"):
                    x["desc"] = nv
                    x["desc_n"] += 1
    rows = []
    for c in df.cve_id:
        x = st.get(c)
        if x is None:
            rows.append({"cve_id": c}); continue
        refs = " ".join(x["refs"])
        rows.append({"cve_id": c, "nv_analyzed": int(x["anal"] is not None),
                     "nv_days_to_anal": (x["anal"] - x["recv"]).days if x["anal"] is not None and x["recv"] is not None else np.nan,
                     "nv_n_refs": len(x["refs"]), "nv_refs_added": x["added"], "nv_ref_exploit": x["exploit"],
                     "nv_ref_news": int(bool(NEWS.search(refs))), "nv_ref_intel": int(bool(INTEL.search(refs))), "nv_ref_gov": int(bool(GOV.search(refs))),
                     "nv_n_modified": x["mod"], "nv_n_cpe": x["cpe"], "nv_desc_itw": int(bool(ITW.search(x["desc"]))), "nv_desc_changed": int(x["desc_n"] > 1)})
    out = df[["cve_id", "snapshot_month", "which"]].merge(pd.DataFrame(rows), on="cve_id", how="left")
    out["nv_missing"] = out.nv_n_refs.isna().astype(int)
    out["nv_anal_rank"] = out.groupby("snapshot_month").nv_analyzed.transform(lambda v: v - v.mean())
    print(f"events for our CVEs {n_ev:,} | KEV events dropped {n_kev:,} | after snapshot end dropped {n_late:,} | CVEs with history {len(st):,}/{len(df):,}")
    for w in ("historical", "scoreboard"):
        o = out[out.which == w].drop(columns=["which", "snapshot_month"])
        o.to_parquet(PROC / f"asof_nvd_{w}.parquet", index=False)
        print(w, len(o), o.drop(columns="cve_id").mean(numeric_only=True).round(3).to_dict())


if __name__ == "__main__":
    main()
