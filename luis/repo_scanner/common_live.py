"""Live equivalents of luis/analysis/common.py + build_features.py's feature logic, for a CVE that is NOT one
of our training replica rows -- a real CVE found while scanning someone's dependencies.

Every regex, name list and formula below is copied verbatim from build_features.py / common.py so the live
feature row matches what the exported LR A model was trained on. The one deliberate difference: history
features (assigner/vendor/product KEV rates) are computed "as of today" from the full local NVD + KEV
snapshot, not "as of Jan 1 of a training year" -- correct for a live tool, which should use the freshest
history it has, not a frozen replica cutoff.

Local data used (never re-shipped, already excluded from git by the repo's .gitignore):
  IBM-*/IBM/data/data/vulnerabilities.csv   -- full NVD-style export, 1988-2026, ~401k rows
  IBM-*/IBM/data/data/known_exploited.csv   -- the CISA KEV catalog as of the datathon snapshot
"""
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LUIS = os.path.dirname(HERE)
REPO = os.path.dirname(LUIS)
DATA = glob.glob(os.path.join(REPO, "IBM-*", "IBM", "data", "data"))[0]
MODEL_DIR = os.path.join(HERE, "model")

FLAGS = ["exploitdb", "metasploit", "nuclei", "poc_github"]
WP_CNAS = {"security@wordfence.com", "audit@patchstack.com", "contact@wpscan.com"}
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\.]*")
TEXT_FLAGS = {
    "txt_rce": r"remote code execution|\bRCE\b|execute arbitrary (?:code|commands)|arbitrary code execution",
    "txt_unauth": r"unauthenticated|without authentication|authentication bypass|bypass authentication",
    "txt_remote": r"remote attacker|remote, unauthenticated|over the network|via the network",
    "txt_wordpress": r"wordpress",
    "txt_plugin": r"\bplugin\b",
    "txt_sqli": r"sql injection",
    "txt_xss": r"cross[- ]site scripting|\bXSS\b",
    "txt_traversal": r"path traversal|directory traversal",
    "txt_deser": r"deserializ",
    "txt_cmdi": r"command injection|os command",
    "txt_memory": r"use[- ]after[- ]free|out[- ]of[- ]bounds|buffer overflow|heap overflow|type confusion|memory corruption",
    "txt_privesc": r"privilege escalation|elevation of privilege|escalate (?:their )?privileges",
    "txt_edge": r"\b(?:vpn|firewall|gateway|router|appliance|load balancer)\b",
    "txt_ssrf": r"server[- ]side request forgery|\bSSRF\b",
    "txt_upload": r"file upload|upload arbitrary",
    "txt_dos": r"denial of service",
    "txt_kernel": r"\bkernel\b",
}
# SAFE (43) + CHEN_LITE (13) = 56, same order as context_Luis.md / common.py. days_to_cutoff has no live
# meaning (no fixed judge deadline), so it's fixed at DAYS_TO_CUTOFF_DEFAULT -- see train_export.py's docstring.
FEATURE_ORDER = [
    "epss", "epss_pct", "epss_missing", "exploitdb", "metasploit", "nuclei", "poc_github", "n_flags", "any_flag",
    "age_days", "desc_len", "txt_rce", "txt_unauth", "txt_remote", "txt_wordpress", "txt_plugin", "txt_sqli",
    "txt_xss", "txt_traversal", "txt_deser", "txt_cmdi", "txt_memory", "txt_privesc", "txt_edge", "txt_ssrf",
    "txt_upload", "txt_dos", "txt_kernel", "text_kev_lr", "text_kev_sim", "n_same_day", "sib_flags",
    "sib_max_epss_pct", "n_adjacent_ids", "assigner_kev_3y", "assigner_rate_3y", "assigner_volume_3y",
    "assigner_kev_all", "assigner_kev_prev_year", "kev_product_hist", "kev_product_hits", "kev_vendor_hist",
    "days_to_cutoff", "id_year_lag", "old_id", "wp_cna", "txt_eol", "txt_vuldb_disclosed", "txt_auth_bypass",
    "assigner_ransom_kev", "is_patch_tuesday", "twins", "cwe_unknown", "cv4_urgent", "cv4_auto", "cv4_e_poc",
]
DAYS_TO_CUTOFF_DEFAULT = 180   # see train_export.py: no fixed deadline live, so "plenty of runway ahead"

_v = _kev = None


def ngram_lookup(text, names, max_n=4):
    """Max KEV count + distinct-name count among `names` found as word n-grams in `text` (one string)."""
    toks = [w.rstrip(".") for w in TOKEN.findall(text)]
    seen = set()
    for n in range(1, max_n + 1):
        for j in range(len(toks) - n + 1):
            k = " ".join(toks[j:j + n])
            if k in names:
                seen.add(k)
    if not seen:
        return 0.0, 0
    return float(max(names[k] for k in seen)), len(seen)


def local_data():
    """Load (once) the full local NVD-style export and the local CISA KEV snapshot."""
    global _v, _kev
    if _v is None:
        v = pd.read_csv(f"{DATA}/vulnerabilities.csv", dtype=str, usecols=[
            "cve_id", "published", "assigner", "description", "cvss4_vector", "cwe_ids"]).set_index("cve_id")
        v["published"] = pd.to_datetime(v.published, errors="coerce").dt.normalize()
        v["assigner"] = v.assigner.fillna("unknown")
        v["description"] = v.description.fillna("")
        _v = v
    if _kev is None:
        k = pd.read_csv(f"{DATA}/known_exploited.csv", dtype=str)
        k["date_added"] = pd.to_datetime(k.date_added, errors="coerce")
        _kev = k
    return _v, _kev


def merge_live_kev(local_kev, live_rows):
    """Local KEV snapshot + any CVEs CISA has added since, from a live feed fetch (see live_sources.fetch_kev).
    live_rows: list of {cve_id, vendor, product, date_added (ISO str), known_ransomware_use} or None/[] on failure."""
    if not live_rows:
        return local_kev
    extra = pd.DataFrame(live_rows)
    extra["date_added"] = pd.to_datetime(extra.date_added, errors="coerce")
    extra = extra[~extra.cve_id.isin(set(local_kev.cve_id))]
    if extra.empty:
        return local_kev
    cols = [c for c in local_kev.columns if c in extra.columns]
    return pd.concat([local_kev, extra[cols]], ignore_index=True)


class HistoryTables:
    """Assigner/vendor/product KEV-history lookups, computed once "as of today" from the full local snapshot
    (optionally extended with live-fetched recent KEV rows). Mirrors build_features.py's add_history(), minus
    the "before Jan 1 of the training year" cutoff -- a live tool uses everything known up to now."""

    def __init__(self, kev_df):
        v, _ = local_data()
        self.v = v
        kev_ids = set(kev_df.cve_id)
        win = v  # live: the "window" is all history, not a 3-year slice before a fixed year
        g = win.assign(k=win.index.isin(kev_ids)).groupby("assigner").k.agg(["sum", "size"])
        prior = g["sum"].sum() / g["size"].sum()
        self.assigner_kev_3y = g["sum"]
        self.assigner_rate_3y = (g["sum"] + prior * 300) / (g["size"] + 300)
        self.assigner_rate_prior = prior
        self.assigner_volume_3y = np.log1p(g["size"])
        kev_ass = kev_df.cve_id.map(v.assigner)
        self.assigner_kev_all = kev_ass.value_counts()
        one_year_ago = pd.Timestamp.utcnow().tz_localize(None) - pd.DateOffset(years=1)
        prev = kev_df[kev_df.date_added >= one_year_ago]
        self.assigner_kev_prev_year = prev.cve_id.map(v.assigner).value_counts()
        rk = kev_df[kev_df.get("known_ransomware_use", pd.Series(dtype=str)).fillna("") == "Known"].cve_id
        self.assigner_ransom_kev = rk.map(v.assigner).value_counts()
        prod = kev_df["product"].dropna().value_counts() if "product" in kev_df else pd.Series(dtype=int)
        self.prod_names = {k: int(n) for k, n in prod.items() if n >= 3 and len(k) >= 3 and "Multiple" not in k}
        vend = kev_df["vendor"].dropna().value_counts() if "vendor" in kev_df else pd.Series(dtype=int)
        self.vend_names = {k: int(n) for k, n in vend.items() if n >= 3 and len(k) >= 3}

    def lookup(self, assigner):
        return {
            "assigner_kev_3y": float(self.assigner_kev_3y.get(assigner, 0)),
            "assigner_rate_3y": float(self.assigner_rate_3y.get(assigner, self.assigner_rate_prior)),
            "assigner_volume_3y": float(self.assigner_volume_3y.get(assigner, 0)),
            "assigner_kev_all": float(self.assigner_kev_all.get(assigner, 0)),
            "assigner_kev_prev_year": float(self.assigner_kev_prev_year.get(assigner, 0)),
            "assigner_ransom_kev": float(self.assigner_ransom_kev.get(assigner, 0)),
        }


def _s(x):
    """pandas leaves truly-missing text cells as float NaN, and `nan or ""` evaluates to `nan` (NaN is
    truthy), not ""  -- every text field read off a pandas row goes through this instead of `x or ""`."""
    return x if isinstance(x, str) else ""


def local_cve_meta(cve_id):
    """published/assigner/description/cvss4_vector/cwe_ids from the local snapshot, or None if not found."""
    v, _ = local_data()
    if cve_id not in v.index:
        return None
    row = v.loc[cve_id]
    if isinstance(row, pd.DataFrame):   # duplicate ids: take the first
        row = row.iloc[0]
    return {"published": row.published, "assigner": _s(row.assigner) or "unknown", "description": _s(row.description),
            "cvss4_vector": _s(row.cvss4_vector), "cwe_ids": _s(row.cwe_ids)}


def build_feature_row(cve_id, meta, epss_value, flags, kev_set, hist, tfidf):
    """Assemble the exact 56 SAFE+CHEN_LITE columns for one live CVE.
    meta: dict from local_cve_meta() or a live-NVD fallback, with published/assigner/description/cvss4_vector/cwe_ids.
    epss_value: float or None. flags: dict of the 4 FLAGS booleans.
    kev_set: set of cve_ids already on KEV (local+live merged) -- only used by the caller to flag "already exploited",
             not fed into the model (a KEV-flag column isn't one of our 56 features; see context_Luis.md's leak rules).
    hist: a HistoryTables instance. tfidf: dict with vec/lr/pos_matrix/pos_ids from train_export.py's export.
    """
    d = _s(meta.get("description"))
    now = pd.Timestamp.utcnow().tz_localize(None)
    # The local snapshot gives a Timestamp; the live NVD fallback gives an ISO string -- normalize both.
    published = pd.to_datetime(meta.get("published"), errors="coerce", utc=True)
    published = now if pd.isna(published) else published.tz_convert(None).normalize()
    assigner = _s(meta.get("assigner")) or "unknown"
    age_days = max(0, (now - published).days)
    epss = float(epss_value) if epss_value is not None else 0.0
    epss_missing = epss_value is None
    f = {k: int(bool(flags.get(k))) for k in FLAGS}
    n_flags = sum(f.values())

    row = {
        "epss": epss, "epss_pct": epss, "epss_missing": int(epss_missing),   # no cohort to rank within live; use raw EPSS
        "n_flags": n_flags, "any_flag": int(n_flags > 0), "age_days": age_days, "desc_len": len(d),
        "n_same_day": 0, "sib_flags": 0, "sib_max_epss_pct": 0.0, "n_adjacent_ids": 0,   # no cohort siblings for a one-off scan
        "days_to_cutoff": DAYS_TO_CUTOFF_DEFAULT,
        "id_year_lag": max(0, published.year - int(cve_id.split("-")[1])),
        "wp_cna": int(assigner in WP_CNAS), "is_patch_tuesday": int(published.weekday() == 1 and 8 <= published.day <= 14),
        "cwe_unknown": int(not _s(meta.get("cwe_ids")).strip()),
    }
    row.update(f)
    row["old_id"] = int(row["id_year_lag"] >= 1)
    for k, p in TEXT_FLAGS.items():
        row[k] = int(bool(re.search(p, d, re.IGNORECASE)))
    row["txt_eol"] = int(bool(re.search(
        r"end[- ]of[- ]life|\bEOL\b|no longer (?:supported|maintained)|end of (?:support|service)|"
        r"unsupported (?:product|version|device)", d, re.IGNORECASE)))
    row["txt_vuldb_disclosed"] = int(bool(re.search(
        r"exploit has been (?:disclosed|made available) to the public|has been disclosed to the public and may be used",
        d, re.IGNORECASE)))
    row["txt_auth_bypass"] = int(bool(re.search(
        r"authentication bypass|bypass(?:es|ing)? (?:the )?authentication|auth bypass", d, re.IGNORECASE)))
    vec = _s(meta.get("cvss4_vector"))
    row["cv4_urgent"] = int(bool(re.search(r"/U:(?:Red|Amber)", vec)))
    row["cv4_auto"] = int(bool(re.search(r"/AU:Y", vec)))
    row["cv4_e_poc"] = int(bool(re.search(r"/E:P", vec)))
    row["twins"] = 0   # same-assigner/day/description duplicates: not meaningful for a single scanned CVE

    row.update(hist.lookup(assigner))
    row["kev_product_hist"], row["kev_product_hits"] = ngram_lookup(d, hist.prod_names)
    row["kev_vendor_hist"], _ = ngram_lookup(d, hist.vend_names, max_n=2)

    Xc = tfidf["vec"].transform([d])
    row["text_kev_lr"] = float(tfidf["lr"].predict_proba(Xc)[:, 1][0])
    sims = (Xc @ tfidf["pos_matrix"].T).toarray().ravel()
    best = int(np.argmax(sims)) if len(sims) else -1
    row["text_kev_sim"] = float(sims[best]) if best >= 0 else 0.0
    row["_nearest_kev"] = tfidf["pos_ids"][best] if best >= 0 else None
    row["_nearest_kev_sim"] = row["text_kev_sim"]
    return row
