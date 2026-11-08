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
        pairs = [(name, version, v["id"]) for (name, version), res in zip(chunk, results) for v in res.get("vulns", [])]
        # The batch endpoint only returns ids; resolving each one is an HTTP call, so do them concurrently.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=16) as pool:
            resolved = list(pool.map(_resolve_osv_vuln, [p[2] for p in pairs]))
        for name, _version in chunk:
            out.setdefault(name, [])
        for (name, version, _), rec in zip(pairs, resolved):
            if not rec:
                continue
            fixes = rec["fixed_by_pkg"].get(name.lower()) or [f for fs in rec["fixed_by_pkg"].values() for f in fs]
            out[name].append({**{k: v for k, v in rec.items() if k != "fixed_by_pkg"},
                              "fixed_in": _lowest_fix_above(version, fixes)})
    return out


def _vkey(s):
    """Sortable version key: PEP 440 when it parses, else numeric dot-parts (good enough for npm semver)."""
    try:
        from packaging.version import Version
        return (0, Version(s))
    except Exception:
        nums = [int(x) for x in re.findall(r"\d+", s)[:4]]
        return (1, tuple(nums + [0] * (4 - len(nums))))


def _lowest_fix_above(installed, fixes):
    """Of every version an advisory says fixes it, the smallest one newer than what's installed -- the upgrade
    that actually applies to you. (An advisory often lists fixes for several release lines, e.g. 1.8.19, 1.11.11
    and 2.0.3 for Django; someone on 1.11.0 needs 1.11.11, not 1.8.19.)"""
    try:
        cur = _vkey(installed)
        above = [f for f in fixes if _vkey(f)[0] == cur[0] and _vkey(f) > cur]
        return min(above, key=_vkey) if above else None
    except Exception:
        return None


def _resolve_osv_vuln(osv_id):
    """Fetch one OSV record once (cached) for its CVE alias, description, severity and, per affected package,
    every version listed as fixing it."""
    def fetch():
        try:
            r = _session.get(f"https://api.osv.dev/v1/vulns/{osv_id}", timeout=TIMEOUT)
            r.raise_for_status()
            v = r.json()
        except Exception as e:
            print(f"[live_sources] OSV vuln fetch failed for {osv_id}: {e}")
            return None
        cve_id = v["id"] if CVE_RE.fullmatch(v["id"]) else next((a for a in v.get("aliases", []) if CVE_RE.fullmatch(a)), None)
        if not cve_id:
            return None
        fixed_by_pkg = {}
        for aff in v.get("affected", []):
            pkg = (aff.get("package", {}).get("name") or "").lower()
            for rng in aff.get("ranges", []):
                for ev in rng.get("events", []):
                    if "fixed" in ev:
                        fixed_by_pkg.setdefault(pkg, []).append(ev["fixed"])
        severity_vector = next((s["score"] for s in v.get("severity", []) if "CVSS" in s.get("type", "")), None)
        return {"cve_id": cve_id, "osv_id": osv_id, "summary": v.get("summary") or v.get("details", "")[:300],
                "published": v.get("published"), "severity_vector": severity_vector, "fixed_by_pkg": fixed_by_pkg}
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
# Exploit-DB and Nuclei both publish a single bulk index, so each is downloaded once, kept on disk for a day
# (cache/, git-ignored) and turned into an in-memory set: a lookup per CVE instead of an HTTP call per CVE.
# GitHub PoCs (via the Trickest/cve mirror) have no bulk index, so those stay one HEAD per CVE -- over a pooled
# session, and remembered on disk so a CVE is only ever checked once.
import json as _json  # noqa: E402
import os as _os  # noqa: E402
import threading as _threading  # noqa: E402

from requests.adapters import HTTPAdapter  # noqa: E402

CACHE_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "cache")
_os.makedirs(CACHE_DIR, exist_ok=True)
INDEX_MAX_AGE = 24 * 3600
_session = requests.Session()
_session.mount("https://", HTTPAdapter(pool_connections=8, pool_maxsize=48))
_index = {}
_index_lock = _threading.Lock()
_poc_path = _os.path.join(CACHE_DIR, "poc_github.json")
_poc_lock = _threading.Lock()
try:
    _poc = _json.load(open(_poc_path))
except Exception:
    _poc = {}


def _cached_download(name, url, timeout=30):
    path = _os.path.join(CACHE_DIR, name)
    fresh = _os.path.exists(path) and time.time() - _os.path.getmtime(path) < INDEX_MAX_AGE
    if not fresh:
        try:
            r = _session.get(url, timeout=timeout)
            r.raise_for_status()
            open(path, "wb").write(r.content)
        except Exception as e:
            print(f"[live_sources] {name} download failed ({e}); using the cached copy if there is one")
    return open(path, encoding="utf-8", errors="ignore").read() if _os.path.exists(path) else ""


def _cve_set(name):
    with _index_lock:
        if name not in _index:
            if name == "exploitdb":
                text = _cached_download("exploitdb.csv", "https://gitlab.com/exploit-database/exploitdb/-/raw/main/files_exploits.csv")
                _index[name] = set(CVE_RE.findall(text))
            elif name == "nuclei":
                text = _cached_download("nuclei_cves.json", "https://raw.githubusercontent.com/projectdiscovery/nuclei-templates/main/cves.json")
                _index[name] = set(CVE_RE.findall(text))
    return _index[name]


def warm_indexes():
    """Download/load both bulk indexes up front (server start), so the first scan doesn't pay for them."""
    return {k: len(_cve_set(k)) for k in ("exploitdb", "nuclei")}


def _poc_github(cve_id):
    if cve_id in _poc:
        return _poc[cve_id]
    year = cve_id.split("-")[1]
    try:
        r = _session.head(f"https://raw.githubusercontent.com/trickest/cve/main/{year}/{cve_id}.md", timeout=5)
        hit = r.status_code == 200
        ok = r.status_code in (200, 404)
    except Exception:
        hit, ok = False, False
    if ok:   # only remember definite answers; a timeout gets retried next scan
        with _poc_lock:
            _poc[cve_id] = hit
    return hit


def save_caches():
    with _poc_lock:
        try:
            _json.dump(_poc, open(_poc_path, "w"))
        except Exception as e:
            print(f"[live_sources] could not save PoC cache: {e}")


def exploit_flags(cve_id):
    """-> {exploitdb, metasploit, nuclei, poc_github} booleans. A failed check is False, never an error.
    Metasploit stays False in this build: its module index is tens of MB with no lighter official list."""
    return {
        "exploitdb": cve_id in _cve_set("exploitdb"),
        "metasploit": False,
        "nuclei": cve_id in _cve_set("nuclei"),
        "poc_github": _poc_github(cve_id),
    }
