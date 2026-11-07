"""Turn rows of the publish-time views into text for Granite.

The CVE ID is left out on purpose so the model can't answer from memory of a
specific ID. EPSS is shown as a percentile within the CVE's snapshot month,
which also sidesteps the March 2025 EPSS model change.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DESC_CHARS = 600

INSTRUCTION = (
    "You are a vulnerability-intelligence analyst helping a security team decide what to "
    "patch first. For a newly published software vulnerability, predict whether attackers "
    "will be confirmed exploiting it in the wild: specifically, whether CISA will add it to "
    "the Known Exploited Vulnerabilities (KEV) catalog in the following months. Only about "
    "1 in 400 newly published vulnerabilities is ever added. The details shown are what was "
    "known at the end of the vulnerability's publication month. EPSS is the FIRST "
    "exploit-prediction score, shown as a percentile among vulnerabilities published that "
    "month (100 = highest). Answer with exactly one word: Yes or No."
)

QUESTION = "Will this vulnerability be added to the CISA KEV catalog? Answer Yes or No."


def load_cwe_names(data_dir: Path) -> dict[str, str]:
    w = pd.read_csv(data_dir / "weakness_types.csv", usecols=["cwe_id", "name"])
    return dict(zip(w.cwe_id, w.name))


def add_epss_percentile(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["epss_pct"] = df.groupby("snapshot_month")["epss"].rank(pct=True) * 100
    return df


def _yn(v) -> str:
    return "yes" if v == 1 else "no"


def format_cve(r, cwe_names: dict[str, str]) -> str:
    lines = [f"Published: {r.published:%Y-%m} | Assigned by: {r.assigner}"]

    if isinstance(r.cwe_ids, str):
        cwes = [f"{c} ({cwe_names[c]})" if c in cwe_names else c for c in r.cwe_ids.split(";")]
        lines.append("Weakness: " + "; ".join(cwes))
    else:
        lines.append("Weakness: unknown")

    if pd.notna(r.cvss3_score):
        lines.append(f"CVSS 3: {r.cvss3_score} {r.cvss3_severity} ({r.cvss3_vector})")
    if pd.notna(r.cvss4_score):
        # drop "not defined" (X) metrics; they are most of the string and carry nothing
        vec = "/".join(p for p in str(r.cvss4_vector).split("/") if not p.endswith(":X"))
        lines.append(f"CVSS 4: {r.cvss4_score} {r.cvss4_severity} ({vec})")
    if pd.isna(r.cvss3_score) and pd.isna(r.cvss4_score):
        lines.append("CVSS: not scored yet")

    epss = "unknown" if np.isnan(r.epss_pct) else f"{r.epss_pct:.0f}th percentile"
    lines.append(f"EPSS: {epss}")
    lines.append(
        f"Public exploit code: Exploit-DB {_yn(r.exploitdb)}, Metasploit {_yn(r.metasploit)}, "
        f"Nuclei template {_yn(r.nuclei)}, GitHub PoC {_yn(r.poc_github)}"
    )

    if hasattr(r, "assigner_kev"):  # metadata lines (attach_meta)
        lines += meta_lines(r)
    if hasattr(r, "attacker_appeal"):  # Granite 30B world-knowledge profile (attach_llmx)
        lines += llmx_lines(r)

    desc = " ".join(str(r.description).split())
    if len(desc) > DESC_CHARS:
        desc = desc[:DESC_CHARS].rsplit(" ", 1)[0] + " ..."
    lines.append(f"Description: {desc}")
    return "\n".join(lines)


def attach_meta(df: pd.DataFrame, which: str) -> pd.DataFrame:
    """Join publish-time metadata (scripts/meta_features.py). which: historical | scoreboard."""
    m = pd.read_parquet(Path(__file__).resolve().parent.parent / "processed" / f"meta_{which}.parquet")
    return df.merge(m, on="cve_id", how="left", validate="one_to_one")


def attach_llmx(df: pd.DataFrame) -> pd.DataFrame:
    """Join Granite 4.2 30B's structured answers (processed/llm_extract.parquet)."""
    m = pd.read_parquet(Path(__file__).resolve().parent.parent / "processed" / "llm_extract.parquet")
    m = m.rename(columns={"vendor": "llm_vendor", "product": "llm_product"})
    return df.merge(m, on="cve_id", how="left", validate="one_to_one")


def llmx_lines(r) -> list[str]:
    w = lambda x: "unknown" if not isinstance(x, str) else x.replace("_", " ")
    return [f"Analyst profile: {w(r.llm_vendor)} {w(r.llm_product)}; {w(r.product_type)}; used by {w(r.used_by)}; "
            f"internet exposure {w(r.internet_exposed)}; install base {w(r.install_base)}",
            f"Analyst view of the bug: precondition {w(r.precondition)}; impact {w(r.impact)}; end of life "
            f"{w(r.end_of_life)}; fix {w(r.fix_status)}; typical attacker {w(r.typical_attacker)}; "
            f"exploitation {w(r.exploitation_style)}; attacker appeal {w(r.attacker_appeal)}"]


def meta_lines(r) -> list[str]:
    cut = pd.Timestamp(r.kev_cutoff)
    out = [f"Assigner track record: {r.assigner_kev} of its CVEs were on the KEV list by {cut:%Y-%m-%d} "
           f"({r.assigner_cves_3y} CVEs from it in the 3 years before)",
           f"Weakness track record: about {1000 * r.cwe_kev_rate:.1f} in 1,000 CVEs with this weakness "
           f"end up on KEV"]
    out.append("Release: the only CVE this assigner published that day" if r.out_of_band
               else f"Release: one of {r.batch_size} CVEs this assigner published that day")
    if r.twins:
        out.append(f"Published together with {r.twins} near-identical CVEs")
    out.append(f"A CVE in the same release has public exploit code: {'yes' if r.sibling_has_code else 'no'}")
    if r.id_year_lag >= 1:
        out.append(f"CVE ID reserved {r.id_year_lag} year(s) before publication")
    if r.cvss4_urgency in ("Red", "Amber", "Green", "Clear"):
        out.append(f"Vendor-stated urgency (CVSS 4): {r.cvss4_urgency}")
    return out


def pick_shots(pool: pd.DataFrame, neg_per_pos: int, seed: int) -> pd.DataFrame:
    """All positives, plus negatives: half 'hard' (top by exploit code then EPSS), half random."""
    pos = pool[pool.label == 1]
    neg = pool[pool.label == 0]
    n_neg = neg_per_pos * len(pos)
    n_hard = n_neg // 2
    has_code = neg[["exploitdb", "metasploit", "nuclei", "poc_github"]].max(axis=1)
    hard = neg.assign(_c=has_code).sort_values(["_c", "epss"], ascending=False).head(n_hard)
    rand = neg.drop(hard.index).sample(n_neg - n_hard, random_state=seed)
    shots = pd.concat([pos, hard.drop(columns="_c"), rand])
    return shots.sample(frac=1, random_state=seed)


def many_shot_system(shots: pd.DataFrame, cwe_names: dict[str, str]) -> str:
    parts = [
        INSTRUCTION,
        "",
        f"Below are {len(shots)} past vulnerabilities from 2024 with their outcomes. "
        "They are not a random sample: exploited ones are over-represented so you can see "
        "what they look like.",
    ]
    for i, r in enumerate(shots.itertuples(), 1):
        parts += ["", f"### Example {i}", format_cve(r, cwe_names),
                  f"Added to KEV: {'Yes' if r.label == 1 else 'No'}"]
    return "\n".join(parts)
