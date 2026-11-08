"""Per-CVE analyst note written by an LLM on IBM watsonx.ai.

Credentials come from Streamlit secrets or environment variables:
  WATSONX_APIKEY      IBM Cloud API key (required to generate new notes)
  WATSONX_PROJECT_ID  watsonx project id
  WATSONX_URL         region endpoint, default Toronto
  WATSONX_MODEL       chat model id available in that region

Notes are cached in data/explanations.json so the shared app shows them without a key.
They are display-only: nothing here feeds back into the ranking.
"""
import json
import os
import threading
import time
from pathlib import Path

import requests

CACHE = Path(__file__).parent / "data" / "explanations.json"
DEFAULTS = {
    "WATSONX_URL": "https://ca-tor.ml.cloud.ibm.com",
    "WATSONX_MODEL": "meta-llama/llama-3-3-70b-instruct",
    "WATSONX_PROJECT_ID": "95739332-9f41-48b9-95fc-0660de0ea43a",
}
_token = {"value": None, "expires": 0}
_lock = threading.Lock()


def setting(name: str) -> str | None:
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name) or DEFAULTS.get(name)


def has_credentials() -> bool:
    return bool(setting("WATSONX_APIKEY"))


def model_name() -> str:
    return setting("WATSONX_MODEL")


def load_cache() -> dict:
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def _save(cve: str, entry: dict):
    with _lock:
        c = load_cache()
        c[cve] = entry
        CACHE.write_text(json.dumps(c, indent=1, ensure_ascii=False))


def _iam_token() -> str:
    if _token["value"] and time.time() < _token["expires"] - 120:
        return _token["value"]
    r = requests.post("https://iam.cloud.ibm.com/identity/token", timeout=30,
                      data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                            "apikey": setting("WATSONX_APIKEY")})
    r.raise_for_status()
    j = r.json()
    _token.update(value=j["access_token"], expires=time.time() + j.get("expires_in", 3600))
    return _token["value"]


def _v(x):
    return "unknown" if x is None or (isinstance(x, float) and x != x) else str(x).replace("_", " ")


def _facts(r: dict) -> str:
    flags = [n for n, k in [("Exploit-DB", "exploitdb"), ("Metasploit", "metasploit"), ("Nuclei", "nuclei"),
                            ("GitHub PoC", "poc_github")] if r.get(k) == 1]
    status = (f"in our submitted top 500 at rank {int(r['submission_rank'])}" if r.get("in_submission")
              else f"NOT in our top 500 (model rank {int(r['rank_final'])}); on an unvalidated watchlist")
    return "\n".join([
        f"CVE: {r['cve_id']} (published {r['published']}; {status})",
        f"Vendor / product (per Granite): {_v(r.get('granite_vendor'))} / {_v(r.get('granite_product'))}",
        f"Advisory text: {_v(r.get('description'))}",
        f"CVSS 3 at publish: {_v(r.get('cvss3'))}; attack vector {_v(r.get('cv_attack_vector'))}; "
        f"privileges {_v(r.get('cv_privileges_required'))}; user interaction {_v(r.get('cv_user_interaction'))}; "
        f"CWE {_v(r.get('cwe_ids'))}",
        f"Public exploit code at snapshot: {', '.join(flags) if flags else 'none'}; "
        f"EPSS percentile within its month: {r['epss_pct'] * 100:.0f}",
        "Granite 30B profile: " + "; ".join(f"{a} = {_v(r.get(k))}" for a, k in [
            ("product type", "product_type"), ("internet exposure", "internet_exposed"),
            ("install base", "install_base"), ("used by", "used_by"), ("typical attacker", "typical_attacker"),
            ("attacker appeal", "attacker_appeal"), ("impact", "impact"), ("needs", "precondition"),
            ("end of life", "end_of_life"), ("fix status", "fix_status")]),
        f"Model ranks out of 72,670: trees #{int(r['tree_rank'])}, linear #{int(r['linear_rank'])}, "
        f"Granite 8B LoRA #{int(r['llm_rank'])}",
    ])


SYSTEM = ("You are a vulnerability analyst writing a short triage note for a patching team. Use ONLY the facts "
          "given. Do not invent products, versions, exploitation reports, or KEV status; if something is unknown, "
          "say so. Plain English, no hype.")
ASK = ("Write the note as exactly four markdown bullets, each one or two sentences:\n"
       "- **What it is:** the product and what an attacker can do.\n"
       "- **Why it could be exploited soon:** the strongest reasons from the facts.\n"
       "- **What argues against it:** the weakest points or missing evidence.\n"
       "- **Suggested action:** patch now, patch this cycle, or monitor, with one reason.")


def generate(record: dict) -> dict:
    body = {"model_id": model_name(), "project_id": setting("WATSONX_PROJECT_ID"),
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": _facts(record) + "\n\n" + ASK}],
            "max_tokens": 400, "temperature": 0}
    r = requests.post(setting("WATSONX_URL").rstrip("/") + "/ml/v1/text/chat?version=2024-10-08",
                      headers={"Authorization": f"Bearer {_iam_token()}", "Content-Type": "application/json"},
                      json=body, timeout=90)
    if r.status_code != 200:
        raise RuntimeError(f"watsonx {r.status_code}: {r.text[:300]}")
    text = r.json()["choices"][0]["message"]["content"].strip()
    entry = {"text": text, "model": model_name(), "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
    _save(record["cve_id"], entry)
    return entry


def explain_cve(record: dict) -> dict | None:
    return load_cache().get(record["cve_id"])
