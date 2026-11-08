"""Orchestrates one repo scan end to end: lockfile -> matched CVEs -> our 56 live features -> LR A score ->
demotion stand-in -> ranked, explained list. This is the live-serving twin of luis/analysis/final_build.py;
see proposal_repo_scanner.md for why the MVP ships with one voter (LR A) instead of the full ensemble.

Usage (CLI, for testing without the server):
  python -E -P luis/repo_scanner/scan.py path/to/package-lock.json
  python -E -P luis/repo_scanner/scan.py path/to/requirements.txt
"""
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib  # noqa: E402
import pandas as pd  # noqa: E402

import common_live as cl  # noqa: E402
import live_sources as ls  # noqa: E402
import lockfile_parse as lf  # noqa: E402

MODEL_DIR = cl.MODEL_DIR
_lr_a = _tfidf = _feature_order = _hist = _bounds = None


def _ensure_models():
    global _lr_a, _tfidf, _feature_order, _hist, _bounds
    if _lr_a is None:
        for f in ("lr_a.joblib", "tfidf_kev.joblib", "feature_order.json", "feature_bounds.json"):
            if not os.path.exists(os.path.join(MODEL_DIR, f)):
                raise RuntimeError(f"model/{f} is missing. Run `python -E -P luis/repo_scanner/train_export.py` first.")
        _lr_a = joblib.load(os.path.join(MODEL_DIR, "lr_a.joblib"))
        _tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_kev.joblib"))
        _feature_order = json.load(open(os.path.join(MODEL_DIR, "feature_order.json")))
        _bounds = json.load(open(os.path.join(MODEL_DIR, "feature_bounds.json")))
    if _hist is None:
        _, local_kev = cl.local_data()
        live_kev_rows = ls.fetch_kev_live()
        merged_kev = cl.merge_live_kev(local_kev, live_kev_rows)
        _hist = cl.HistoryTables(merged_kev)
        _hist.kev_ids = set(merged_kev.cve_id)   # for the "already exploited" split below
        _hist.live_kev_added = len(live_kev_rows)
    return _lr_a, _tfidf, _feature_order, _hist


# ----------------------------------------------------------------------------- demotion stand-in (see §10 of the proposal)
# 2 of the 4 training-time Granite rules are regex-approximable from the description alone; the other two
# (typical_attacker=unlikely_target, attacker_appeal=low) need a live Granite 30B read, not in this build.
def demoted(desc, package_name):
    d = (desc or "").lower()
    dos_only = ("denial of service" in d) and not any(w in d for w in (
        "remote code execution", "sql injection", "cross-site scripting", "privilege escalation",
        "authentication bypass", "command injection", "path traversal", "arbitrary code"))
    wp_plugin = "wordpress" in d and "plugin" in d or re.search(r"wp[-_]?plugin", package_name or "", re.I)
    return bool(dos_only or wp_plugin), ("denial-of-service only" if dos_only else "WordPress plugin" if wp_plugin else "")


def build_upgrade_plan(rows, kev_rows):
    """Collapse CVE rows into one action per dependency: the version that clears every known CVE for it (the
    highest of the per-CVE fixed versions), how many are already on KEV, the model's top score among them, and
    any CVE with no published fix. Ordered: anything already exploited first, then by the model's top score."""
    plan = {}
    for row, on_kev in [(r, False) for r in rows] + [(r, True) for r in kev_rows]:
        for p in row["packages"]:
            e = plan.setdefault((p["package"], p["version"]), {"package": p["package"], "version": p["version"],
                                                                "cves": 0, "on_kev": 0, "top_probability": 0.0,
                                                                "top_cve": None, "fixes": [], "unfixed": [],
                                                                "source_lockfile": row.get("source_lockfile")})
            e["cves"] += 1
            e["on_kev"] += on_kev
            prob = 1.0 if on_kev else row.get("probability", 0)
            if prob >= e["top_probability"]:
                e["top_probability"], e["top_cve"] = prob, row["cve_id"]
            (e["fixes"] if p.get("fixed_in") else e["unfixed"]).append(p.get("fixed_in") or row["cve_id"])
    out = []
    for e in plan.values():
        e["upgrade_to"] = max(e.pop("fixes"), key=ls._vkey) if e["cves"] > len(e["unfixed"]) else None
        e["top_probability"] = round(e["top_probability"], 3)
        out.append(e)
    out.sort(key=lambda e: (-e["on_kev"], -e["top_probability"], -e["cves"]))
    return out


def reason_for(row, meta):
    parts = []
    code = [n for k, n in (("exploitdb", "Exploit-DB"), ("metasploit", "Metasploit"), ("nuclei", "Nuclei"),
                            ("poc_github", "GitHub PoC")) if row.get(k)]
    if code:
        parts.append("public exploit code (" + ", ".join(code) + ")")
    if row.get("epss", 0) >= 0.5:
        parts.append(f"high EPSS ({row['epss']:.2f})")
    elif row.get("epss", 0) >= 0.1:
        parts.append(f"elevated EPSS ({row['epss']:.2f})")
    for flag, label in (("txt_rce", "remote code execution"), ("txt_sqli", "SQL injection"), ("txt_xss", "cross-site scripting"),
                         ("txt_deser", "insecure deserialization"), ("txt_cmdi", "command injection"),
                         ("txt_privesc", "privilege escalation"), ("txt_traversal", "path traversal")):
        if row.get(flag):
            parts.append(label + (" with no authentication" if row.get("txt_unauth") else ""))
            break
    if row.get("assigner_kev_all", 0) >= 3:
        parts.append(f"{int(row['assigner_kev_all'])} past KEV entries from this assigner")
    if row.get("_nearest_kev") and row.get("_nearest_kev_sim", 0) >= 0.5:
        parts.append(f"resembles KEV entry {row['_nearest_kev']} (similarity {row['_nearest_kev_sim']:.2f})")
    s = "; ".join(parts) or "no strong individual signal; ranked by the model's combined read of the description and history"
    return s[0].upper() + s[1:] + "."


def run_scan(filename, text):
    t0 = time.time()
    lr_a, tfidf, feature_order, hist = _ensure_models()
    ecosystem, deps = lf.detect_and_parse(filename, text)
    osv = ls.osv_batch(deps, ecosystem)

    cve_to_pkgs = {}   # a CVE can affect more than one scanned dependency; keep every match
    for name, vulns in osv.items():
        for v in vulns:
            pkgs = cve_to_pkgs.setdefault(v["cve_id"], {})
            key = (name, deps[name])
            # OSV sometimes reports the same CVE under two advisory ids (e.g. a GHSA and a PYSEC entry that
            # both alias it) -- same package+version would otherwise show up twice in one row.
            if key not in pkgs or (not pkgs[key]["fixed_in"] and v.get("fixed_in")):
                pkgs[key] = {"package": name, "version": deps[name], "fixed_in": v.get("fixed_in")}
    cve_to_pkgs = {cve_id: list(pkgs.values()) for cve_id, pkgs in cve_to_pkgs.items()}
    cve_ids = list(cve_to_pkgs)

    def resolve_meta(cve_id):
        return cve_id, cl.local_cve_meta(cve_id) or ls.live_nvd_lookup(cve_id)

    already_kev, to_score = [], []
    epss = ls.epss_batch(cve_ids)
    # Local lookups dominate (fast, in-memory); the rare NVD fallback is the only slow case here, so a small
    # pool is enough to keep one slow miss from serializing behind every other CVE.
    with ThreadPoolExecutor(max_workers=8) as pool:
        metas = dict(pool.map(resolve_meta, cve_ids))
    rejected = []
    for cve_id in cve_ids:
        meta = metas.get(cve_id)
        if meta is None:
            continue   # unresolvable (very rare: a brand-new id NVD hasn't published metadata for yet)
        pkgs = cve_to_pkgs[cve_id]
        d = (meta.get("description") or "").lstrip()
        if d.startswith("Rejected reason") or d.startswith("** REJECT **"):
            # A rejected id isn't a vulnerability to patch; ranking it put noise at the top (seen on IBM/node-sdk-core).
            # Disputed CVEs are kept: still published, the vendor just disagrees.
            rejected.append({"cve_id": cve_id, "packages": pkgs, "description": d[:200]})
            continue
        if cve_id in hist.kev_ids:
            already_kev.append({"cve_id": cve_id, "packages": pkgs, "published": str(meta["published"])[:10],
                                 "description": meta["description"][:280]})
            continue
        to_score.append((cve_id, meta, pkgs))

    # exploit_flags makes several live HTTP calls per CVE (nuclei, trickest, exploitdb); run the whole
    # to-score batch concurrently so scan time is "slowest single check", not "sum of every check".
    with ThreadPoolExecutor(max_workers=32) as pool:
        flags_by_cve = dict(zip((c for c, _, _ in to_score), pool.map(ls.exploit_flags, (c for c, _, _ in to_score))))

    feats = [cl.build_feature_row(c, m, epss.get(c), flags_by_cve[c], hist.kev_ids, hist, tfidf) for c, m, _ in to_score]
    probs, clipped = [], []
    if feats:
        X = pd.DataFrame([{k: f[k] for k in feature_order} for f in feats])
        # Clip to the training range: an old CVE is otherwise far outside what the model was fit on (see train_export.py).
        lo = pd.Series({k: _bounds[k][0] for k in feature_order})
        hi = pd.Series({k: _bounds[k][1] for k in feature_order})
        out_of_range = (X < lo) | (X > hi)
        clipped = [list(X.columns[out_of_range.iloc[i]]) for i in range(len(X))]
        probs = lr_a.predict_proba(X.clip(lower=lo, upper=hi, axis=1))[:, 1]

    rows = []
    for (cve_id, meta, pkgs), feat, prob, clip_cols in zip(to_score, feats, probs, clipped):
        flags = flags_by_cve[cve_id]
        prob = float(prob)
        demote, demote_reason = demoted(meta["description"], pkgs[0]["package"])
        score = prob - (0.5 if demote else 0.0)
        rows.append({
            "cve_id": cve_id, "packages": pkgs, "score": score, "probability": round(prob, 4),
            "published": str(meta["published"])[:10], "epss": round(feat["epss"], 4),
            "flags": flags, "text_kev_lr": round(feat["text_kev_lr"], 3),
            "nearest_kev": feat.get("_nearest_kev"), "nearest_kev_sim": round(feat.get("_nearest_kev_sim") or 0, 3),
            "demoted": demote, "demoted_reason": demote_reason,
            "reason": reason_for({**feat, "_nearest_kev": feat.get("_nearest_kev"), "_nearest_kev_sim": feat.get("_nearest_kev_sim")}, meta),
            "description": meta["description"][:280],
            "clipped_features": clip_cols,
        })
    # Ties (common once many CVEs share an assigner and an age bucket) break on live exploitation signals.
    rows.sort(key=lambda r: (-r["score"], -sum(r["flags"].values()), -r["epss"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    ls.save_caches()
    return {
        "ecosystem": ecosystem, "elapsed_s": round(time.time() - t0, 1),
        "summary": {"dependencies": len(deps), "dependencies_with_cves": len(osv), "cves_found": len(cve_ids),
                    "already_in_kev": len(already_kev), "scored": len(rows), "demoted": sum(r["demoted"] for r in rows),
                    "rejected_ids_skipped": len(rejected),
                    "kev_feed": f"local snapshot + {hist.live_kev_added} added since (live CISA fetch)" if hist.live_kev_added
                                else "local snapshot only (live CISA fetch unavailable this run)"},
        "already_in_kev": already_kev,
        "rejected": rejected,
        "upgrade_plan": build_upgrade_plan(rows, already_kev),
        "rows": rows,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    path = sys.argv[1]
    result = run_scan(os.path.basename(path), open(path, encoding="utf-8").read())
    print(json.dumps(result, indent=2)[:4000])
    print(f"\n...{len(json.dumps(result))} bytes total. summary: {result['summary']}")
