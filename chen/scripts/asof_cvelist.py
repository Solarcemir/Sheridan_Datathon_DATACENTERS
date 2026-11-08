"""Point-in-time CVE List features: each CVE's official record (CVEProject/cvelistV5, public git history) as it stood
at the END of its snapshot month. Reads the file from the last commit before that moment, never today's version.

Features:
  cl_ssvc_present     CISA's ADP triage (Vulnrichment SSVC) exists by then (0/1)
  cl_ssvc_poc         SSVC Exploitation = poc
  cl_ssvc_active      SSVC Exploitation = active  (CISA saw reports of attacks; KEV not yet, by the cohort rule)
  cl_ssvc_auto        SSVC Automatable = yes
  cl_ssvc_total       SSVC Technical Impact = total
  cl_n_refs           references in the CNA record
  cl_ref_exploit      a CNA/ADP reference tagged exploit
  cl_missing          no record at that commit

  python scripts/asof_cvelist.py   -> processed/asof_cvelist_{historical,scoreboard}.parquet
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC, REPO = ROOT / "processed", ROOT / "external" / "cvelistV5"


def git(*a) -> str:
    return subprocess.run(["git", "-C", str(REPO), *a], check=True, capture_output=True, text=True).stdout.strip()


def path_of(cve: str) -> str:
    _, y, n = cve.split("-")
    return f"cves/{y}/{int(n) // 1000}xxx/{cve}.json"


def ssvc(rec: dict) -> dict:
    out = {}
    for adp in rec.get("containers", {}).get("adp", []) or []:
        if (adp.get("providerMetadata", {}) or {}).get("shortName") != "CISA-ADP":
            continue
        for m in adp.get("metrics", []) or []:
            o = m.get("other", {}) or {}
            if o.get("type") == "ssvc":
                for opt in (o.get("content", {}) or {}).get("options", []) or []:
                    out.update({k.lower(): str(v).lower() for k, v in opt.items()})
    return out


def features(rec: dict | None) -> dict:
    if rec is None:
        return {"cl_missing": 1}
    s = ssvc(rec)
    cna = rec.get("containers", {}).get("cna", {}) or {}
    refs = list(cna.get("references", []) or [])
    for adp in rec.get("containers", {}).get("adp", []) or []:
        refs += adp.get("references", []) or []
    tags = " ".join(" ".join(r.get("tags", []) or []) for r in refs).lower()
    return {"cl_missing": 0, "cl_ssvc_present": int(bool(s)), "cl_ssvc_poc": int(s.get("exploitation") == "poc"),
            "cl_ssvc_active": int(s.get("exploitation") == "active"), "cl_ssvc_auto": int(s.get("automatable") == "yes"),
            "cl_ssvc_total": int(s.get("technical impact") == "total"), "cl_n_refs": len(cna.get("references", []) or []),
            "cl_ref_exploit": int("exploit" in tags)}


def main() -> None:
    h = pd.read_parquet(PROC / "historical_2024_2025.parquet", columns=["cve_id", "snapshot_month"]).assign(which="historical")
    s = pd.read_parquet(PROC / "scoreboard_2026_view.parquet", columns=["cve_id", "snapshot_month"]).assign(which="scoreboard")
    df = pd.concat([h, s], ignore_index=True)
    branch = git("rev-parse", "--abbrev-ref", "HEAD") or "main"
    rows = []
    for m, g in df.groupby("snapshot_month"):
        end = pd.Period(m, freq="M").to_timestamp(how="end").strftime("%Y-%m-%dT23:59:59Z")
        commit = git("rev-list", "-1", f"--before={end}", branch)
        if not commit:
            rows += [{"cve_id": c, "cl_missing": 1} for c in g.cve_id]
            continue
        req = "".join(f"{commit}:{path_of(c)}\n" for c in g.cve_id)
        p = subprocess.run(["git", "-C", str(REPO), "cat-file", "--batch"], input=req.encode(), capture_output=True, check=True)
        buf, i = p.stdout, 0
        for c in g.cve_id:
            nl = buf.index(b"\n", i)
            head = buf[i:nl].split()
            if len(head) >= 2 and head[1] == b"missing" or head[-1] == b"missing":
                rows.append({"cve_id": c, **features(None)}); i = nl + 1; continue
            size = int(head[2]); body = buf[nl + 1: nl + 1 + size]; i = nl + 1 + size + 1
            try:
                rows.append({"cve_id": c, **features(json.loads(body))})
            except json.JSONDecodeError:
                rows.append({"cve_id": c, **features(None)})
        f = pd.DataFrame(rows[-len(g):])
        print(f"{m}: commit {commit[:8]} | missing {f.cl_missing.mean():.1%} | SSVC {f.get('cl_ssvc_present', pd.Series([0])).mean():.1%} | active {f.get('cl_ssvc_active', pd.Series([0])).sum()}", flush=True)
    out = df[["cve_id", "which"]].merge(pd.DataFrame(rows), on="cve_id", how="left")
    for w in ("historical", "scoreboard"):
        out[out.which == w].drop(columns="which").to_parquet(PROC / f"asof_cvelist_{w}.parquet", index=False)


if __name__ == "__main__":
    main()
