"""Distil Granite's world knowledge into structured features (one JSON object per CVE).

Granite sees ONLY the description, assigner and CWE names: no EPSS, exploit flags, dates, CVE ID,
or KEV information, so it can add product / bug knowledge but cannot recycle signals the linear
model already has. Output is constrained by a JSON schema (vLLM structured outputs), so every
answer parses into fixed categories.

  CUDA_VISIBLE_DEVICES=1 python scripts/extract.py --shard 0 --nshards 3
  python scripts/extract.py --merge          # -> processed/llm_extract.parquet
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

import prompts

ROOT = Path(__file__).resolve().parent.parent
DATA = next((p for p in (ROOT / "IBM" / "data", ROOT.parent / "IBM-20261107T174321Z-1-001" / "IBM" / "data" / "data") if p.exists()), ROOT / "IBM" / "data")
PROC, RES = ROOT / "processed", ROOT / "results"

FIELDS = {
    "product_type": ["network_edge_device", "remote_access_or_it_management", "enterprise_server_application",
                     "web_application_or_cms", "wordpress_plugin_or_theme", "email_or_collaboration", "browser",
                     "mobile_os_or_app", "desktop_os", "desktop_application", "consumer_router_or_iot",
                     "industrial_or_ot", "virtualization_or_cloud_infra", "library_or_framework",
                     "developer_tool", "other"],
    "internet_exposed": ["typically_internet_facing", "sometimes", "rarely"],
    "install_base": ["very_large", "large", "moderate", "small_or_niche"],
    "used_by": ["enterprises_and_governments", "consumers", "both", "developers", "industrial"],
    "precondition": ["none", "needs_user_account", "needs_admin", "needs_local_access", "needs_user_action",
                     "needs_adjacent_network_or_physical", "needs_prior_compromise_or_unusual_config"],
    "impact": ["remote_code_execution", "authentication_bypass", "privilege_escalation",
               "sensitive_data_disclosure", "file_write_or_upload", "client_side_script_injection",
               "denial_of_service", "other"],
    "end_of_life": ["yes", "no", "unknown"],
    "fix_status": ["patch_available", "workaround_or_mitigation_only", "no_fix_end_of_life", "unclear"],
    "typical_attacker": ["ransomware_or_extortion_groups", "state_sponsored_espionage", "botnets",
                         "opportunistic_criminals", "unlikely_target"],
    "exploitation_style": ["mass_exploitable_at_scale", "targeted_attacks", "difficult_or_theoretical"],
    "attacker_appeal": ["high", "medium", "low"],
}

SCHEMA = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string", "maxLength": 40},
        "product": {"type": "string", "maxLength": 60},
        **{k: {"type": "string", "enum": v} for k, v in FIELDS.items()},
    },
    "required": ["vendor", "product", *FIELDS],
    "additionalProperties": False,
}

SYSTEM = (
    "You are a security analyst with broad knowledge of software products, how they are deployed, "
    "and which kinds of products attackers target. Classify the vulnerability below using the "
    "description and your general knowledge of the product. Do not try to recall this specific CVE.\n\n"
    "Fields:\n"
    "- vendor, product: the real vendor and product name (normalised, e.g. 'SSLVPN' on a SonicWall "
    "advisory is vendor SonicWall, product SonicOS).\n"
    "- product_type: what kind of product it is. Any WordPress plugin or theme is wordpress_plugin_or_theme.\n"
    "- internet_exposed: is this product usually reachable from the internet in real deployments?\n"
    "- install_base: how widely the product is deployed worldwide.\n"
    "- used_by: who mainly runs it.\n"
    "- precondition: the main thing an attacker needs before exploiting it.\n"
    "- impact: the main thing a successful attacker gets, as the description states it (cross-site "
    "scripting is client_side_script_injection, not code execution).\n"
    "- end_of_life: is the affected product or version end-of-life / unsupported?\n"
    "- fix_status: does the description indicate a patch, only a workaround, or no fix (end-of-life)?\n"
    "- typical_attacker: which kind of attacker typically goes after this product and bug type. Use "
    "state_sponsored_espionage only for products such actors are known to target (phones, browsers, "
    "messaging, mail servers, government networking gear); botnets for routers, cameras and other IoT; "
    "ransomware_or_extortion_groups for enterprise edge devices, file transfer, backup, remote management "
    "and widely used enterprise servers; unlikely_target when attackers rarely bother with this product.\n"
    "- exploitation_style: could this be exploited at scale across the internet, mainly in targeted "
    "attacks, or is it hard / theoretical?\n"
    "- attacker_appeal: how attractive this product and bug type typically are to ransomware groups, "
    "state actors or botnets. Most vulnerabilities are low: reserve high for products and bug types that "
    "attackers are well known to go after, and use medium sparingly.\n"
    "Answer with the JSON object only."
)


def user_text(r, cwe: dict) -> str:
    cw = "; ".join(f"{c} ({cwe.get(c, '?')})" for c in str(r.cwe_ids).split(";")) if isinstance(r.cwe_ids, str) else "unknown"
    desc = " ".join(str(r.description).split())[:1500]
    return f"Assigned by: {r.assigner}\nWeakness: {cw}\nDescription: {desc}"


def targets() -> pd.DataFrame:
    h = pd.read_parquet(PROC / "historical_2024_2025.parquet", columns=["cve_id", "assigner", "cwe_ids", "description", "snapshot_month"])
    s = pd.read_parquet(PROC / "scoreboard_2026_view.parquet", columns=["cve_id", "assigner", "cwe_ids", "description"])
    t = pd.concat([h[h.snapshot_month <= "2025-09"].drop(columns="snapshot_month"), s]).drop_duplicates("cve_id")
    return t.sort_values("cve_id").reset_index(drop=True)


def targets_all() -> pd.DataFrame:
    """Every CVE in the IBM snapshot that Granite hasn't read yet (for the repo-scanner lookup table)."""
    v = pd.read_csv(DATA / "vulnerabilities.csv", usecols=["cve_id", "assigner", "cwe_ids", "description"])
    done = set(pd.read_parquet(PROC / "llm_extract.parquet", columns=["cve_id"]).cve_id)
    return v[~v.cve_id.isin(done) & v.description.notna()].sort_values("cve_id").reset_index(drop=True)


def run(args) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    cwe = prompts.load_cwe_names(DATA)
    t = targets_all() if args.all else targets()
    if args.ids:
        t = t[t.cve_id.isin(set(Path(args.ids).read_text().split()))]
    t = t.iloc[args.shard::args.nshards]
    if args.limit:
        t = t.head(args.limit)
    tok = AutoTokenizer.from_pretrained(args.model)
    texts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text(r, cwe)}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
             for r in t.itertuples()]
    llm = LLM(model=args.model, max_model_len=4096, gpu_memory_utilization=args.gpu_util,
              enable_prefix_caching=True, seed=0)
    sp = SamplingParams(max_tokens=200, temperature=0.0, structured_outputs=StructuredOutputsParams(json=SCHEMA))
    t0 = time.time()
    outs = llm.generate(texts, sp)
    rows, bad = [], 0
    for r, o in zip(t.itertuples(), outs):
        try:
            d = json.loads(o.outputs[0].text)
        except json.JSONDecodeError:
            d, bad = {}, bad + 1
        rows.append({"cve_id": r.cve_id, **{k: d.get(k) for k in ["vendor", "product", *FIELDS]}})
    out = Path(args.out or RES / f"extract_{args.shard}of{args.nshards}.parquet")
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"{len(rows):,} rows in {time.time() - t0:.0f}s, {bad} unparsable -> {out}", flush=True)


def merge() -> None:
    parts = sorted(RES.glob("extract_*of*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts]).drop_duplicates("cve_id")
    df.to_parquet(PROC / "llm_extract.parquet", index=False)
    print(f"merged {len(parts)} shards: {len(df):,} rows")
    for k in FIELDS:
        print(f"\n{k}:", df[k].value_counts(dropna=False).head(16).to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibm-granite/granite-4.2-30b")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-util", type=float, default=0.92)
    ap.add_argument("--out")
    ap.add_argument("--ids", help="text file of cve_ids to extract (default: all targets)")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--all", action="store_true", help="every not-yet-read CVE in vulnerabilities.csv (scanner table)")
    a = ap.parse_args()
    merge() if a.merge else run(a)
