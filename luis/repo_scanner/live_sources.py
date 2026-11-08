"""Live, public, free, no-API-key data sources. Every call has a short timeout and fails soft (returns an
empty/default result rather than raising), because a flaky third party should degrade one feature, never
break the scan. Each function caches by its key for the lifetime of the process (a demo runs many scans of
overlapping dependencies; no point re-fetching the same CVE twice in five minutes).
"""
import re
import time

import requests

TIMEOUT = 6
CVE_RE = re.compile(r"CVE-\d{4}-\d+")
_cache = {}


def _get(key, fn):
    if key not in _cache:
        _cache[key] = fn()
    return _cache[key]


# ----------------------------------------------------------------------------- OSV.dev: package@version -> CVEs
def osv_batch(deps, ecosystem):
    """deps: {name: version}. -> {name: [{cve_id, summary, published, severity_vector, fixed_in}]}"""
    if not deps:
        return {}
    items = list(deps.items())
    out = {}
    for i in range(0, len(items), 100):   # OSV's batch endpoint accepts up to 1000; 100 keeps one call snappy
        chunk = items[i:i + 100]
        queries = [{"package": {"name": name, "ecosystem": ecosystem}, "version": version} for name, version in chunk]
        try:
            r = requests.post("https://api.osv.dev/v1/querybatch", json={"queries": queries}, timeout=TIMEOUT)
            r.raise_for_status()
            results = r.json().get("results", [])
        except Exception as e:
            print(f"[live_sources] OSV batch query failed: {e}")
            results = [{} for _ in chunk]
        for (name, version), res in zip(chunk, results):
            vulns = []
            for v in res.get("vulns", []):
                vulns.append(_resolve_osv_vuln(v["id"]))
            out[name] = [v for v in vulns if v]
    return out


def _resolve_osv_vuln(osv_id):
    """The batch endpoint only returns ids; fetch each vuln once (cached) for its CVE alias, description,
    severity and fixed-version ranges."""
    def fetch():
        try:
            r = requests.get(f"https://api.osv.dev/v1/vulns/{osv_id}", timeout=TIMEOUT)
            r.raise_for_status()
            v = r.json()
        except Exception as e:
            print(f"[live_sources] OSV vuln fetch failed for {osv_id}: {e}")
            return None
        cve_id = v["id"] if CVE_RE.fullmatch(v["id"]) else next((a for a in v.get("aliases", []) if CVE_RE.fullmatch(a)), None)
        if not cve_id:
            return None
        fixed = None
        for aff in v.get("affected", []):
            for rng in aff.get("ranges", []):
                for ev in rng.get("events", []):
                    if "fixed" in ev:
                        fixed = ev["fixed"]
        severity_vector = next((s["score"] for s in v.get("severity", []) if "CVSS" in s.get("type", "")), None)
        return {"cve_id": cve_id, "osv_id": osv_id, "summary": v.get("summary") or v.get("details", "")[:300],
                "published": v.get("published"), "severity_vector": severity_vector, "fixed_in": fixed}
    return _get(("osv_vuln", osv_id), fetch)


# ----------------------------------------------------------------------------- EPSS
def epss_batch(cve_ids):
    """-> {cve_id: float in [0,1]}. Missing ids (e.g. an EPSS outage) are simply absent, not zero."""
    ids = sorted(set(cve_ids))
    out = {}
    for i in range(0, len(ids), 100):   # FIRST.org's API caps `cve=` lists; chunk to be safe
        chunk = ids[i:i + 100]
        try:
            r = requests.get("https://api.first.org/data/v1/epss", params={"cve": ",".join(chunk)}, timeout=TIMEOUT)
            r.raise_for_status()
            for row in r.json().get("data", []):
                out[row["cve"]] = float(row["epss"])
        except Exception as e:
            print(f"[live_sources] EPSS batch query failed: {e}")
    return out


# ----------------------------------------------------------------------------- CISA KEV (freshness on top of the local snapshot)
def fetch_kev_live():
    """-> list of {cve_id, vendor, product, date_added, known_ransomware_use}, or [] on any failure
    (the local known_exploited.csv snapshot is always the fallback -- see common_live.merge_live_kev)."""
    def fetch():
        try:
            r = requests.get("https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
                              timeout=TIMEOUT)
            r.raise_for_status()
            rows = r.json().get("vulnerabilities", [])
            return [{"cve_id": x["cveID"], "vendor": x.get("vendorProject"), "product": x.get("product"),
                      "date_added": x.get("dateAdded"), "known_ransomware_use": x.get("knownRansomwareCampaignUse")}
                     for x in rows]
        except Exception as e:
            print(f"[live_sources] CISA KEV live fetch failed (using the local snapshot only): {e}")
            return []
    return _get("kev_live", fetch)


# ----------------------------------------------------------------------------- NVD (fallback only)
def live_nvd_lookup(cve_id):
    """For a CVE too new to be in the local snapshot. -> the same shape as common_live.local_cve_meta(), or None.
    NVD's public rate limit (5 req/30s, no key) is fine for this: it's only called for cache misses, one at a time."""
    def fetch():
        try:
            r = requests.get("https://services.nvd.nist.gov/rest/json/cves/2.0", params={"cveId": cve_id}, timeout=TIMEOUT)
            r.raise_for_status()
            vulns = r.json().get("vulnerabilities", [])
            if not vulns:
                return None
            cve = vulns[0]["cve"]
            desc = next((d["value"] for d in cve.get("descriptions", []) if d["lang"] == "en"), "")
            cwe = ";".join(sorted({d["description"][0]["value"] for d in cve.get("weaknesses", [])
                                    if d.get("description")})) or None
            metrics = cve.get("metrics", {})
            cvss4 = next((m["cvssData"].get("vectorString") for m in metrics.get("cvssMetricV40", [])), None)
            return {"published": cve.get("published"), "assigner": cve.get("sourceIdentifier", "unknown"),
                    "description": desc, "cvss4_vector": cvss4, "cwe_ids": cwe}
        except Exception as e:
            print(f"[live_sources] NVD lookup failed for {cve_id}: {e}")
            return None
    return _get(("nvd", cve_id), fetch)


# ----------------------------------------------------------------------------- exploit-source flags (best-effort)
_exploitdb_csv = None


def _exploitdb_text():
    global _exploitdb_csv
    if _exploitdb_csv is None:
        try:
            r = requests.get("https://gitlab.com/exploit-database/exploitdb/-/raw/main/files_exploits.csv", timeout=10)
            r.raise_for_status()
            _exploitdb_csv = r.text
        except Exception as e:
            print(f"[live_sources] Exploit-DB mirror fetch failed: {e}")
            _exploitdb_csv = ""
    return _exploitdb_csv


def exploit_flags(cve_id):
    """-> {exploitdb, metasploit, nuclei, poc_github} booleans, each a best-effort live check. A failed check
    is False, never an error -- same convention as every other live_sources function."""
    def exploitdb():
        return cve_id in _exploitdb_text()

    def nuclei():
        # nuclei-templates has moved files over time; try the layouts that have been stable for years.
        year = cve_id.split("-")[1]
        for path in (f"http/cves/{year}/{cve_id}.yaml", f"cves/{year}/{cve_id}.yaml"):
            try:
                r = requests.head(f"https://raw.githubusercontent.com/projectdiscovery/nuclei-templates/main/{path}",
                                   timeout=4)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
        return False

    def poc_github():
        year = cve_id.split("-")[1]
        try:
            r = requests.head(f"https://raw.githubusercontent.com/trickest/cve/main/{year}/{cve_id}.md", timeout=4)
            return r.status_code == 200
        except Exception:
            return False

    def metasploit():
        # The module metadata file is tens of MB; fetching it per scan is too slow for a demo, and we have no
        # lighter official index. Left off in this build rather than faked -- see proposal_repo_scanner.md §10.
        return False

    return {
        "exploitdb": _get(("edb", cve_id), exploitdb),
        "metasploit": _get(("msf", cve_id), metasploit),
        "nuclei": _get(("nuclei", cve_id), nuclei),
        "poc_github": _get(("poc", cve_id), poc_github),
    }
