"""Prepare grounded historical cases for the Blind Spot Investigator.

Save as: Amy/feature/prepare_investigations.py
Run from the repository root:
    python Amy/feature/prepare_investigations.py
Or specify the folder containing the original IBM data CSVs:
    python Amy/feature/prepare_investigations.py --data-dir "IBM/data"

Reads Chen-derived historical rankings from Amy/feature/output and original
IBM CSVs. Writes ONLY to Amy/feature/output. Does not run AI, retrain models,
or change anything in Chen's or Luis's folders.

Requires: pandas and numpy (usually installed with pandas).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FEATURE_DIR.parent.parent
OUTPUT_DIR = FEATURE_DIR / "output"
SIGNAL_FLAGS = ("exploitdb", "metasploit", "nuclei", "poc_github")
NVD_FIELDS = (
    "cve_id", "description", "assigner", "cvss3_score", "cvss3_severity",
    "cvss4_score", "cwe_ids", "attack_vector", "attack_complexity",
    "privileges_required", "user_interaction",
)


def find_data_dir(explicit: str | None) -> Path:
    """Find the existing IBM data; never copy it into the feature directory."""
    paths = ([Path(explicit).expanduser()] if explicit else []) + [
        PROJECT_ROOT / "IBM" / "data",
        PROJECT_ROOT / "IBM" / "data" / "data",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "data" / "data",
        PROJECT_ROOT / "chen" / "IBM" / "data",
        PROJECT_ROOT / "chen" / "IBM" / "data" / "data",
    ]
    needed = ("vulnerabilities.csv", "known_exploited.csv",
              "exploit_signals_2024.csv", "exploit_signals_2025.csv")
    for candidate in paths:
        if all((candidate / f).is_file() for f in needed):
            return candidate
    raise FileNotFoundError(
        "IBM data folder not found. It must contain vulnerabilities.csv, "
        "known_exploited.csv, exploit_signals_2024.csv and "
        "exploit_signals_2025.csv. Pass --data-dir with that folder's path."
    )


def require_columns(frame: pd.DataFrame, columns: set[str], filename: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{filename} is missing required columns: {missing}")


def optional_number(value, digits: int | None = None):
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return round(number, digits) if digits is not None else number


def optional_string(value, length: int | None = None):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:length] if length is not None else text


def validate_rankings(rankings: pd.DataFrame, misses: pd.DataFrame, meta: dict):
    voters = meta.get("models")
    if not isinstance(voters, list) or not voters:
        raise ValueError("summary.json does not list the historical model voters")
    needed = {"cve_id", "published", "snapshot_month", "kev_date_added",
              "kev_by_nov4", "rank", "ensemble_score", "selected_top_k", *voters}
    require_columns(rankings, needed, "historical_rankings.csv")
    require_columns(misses, needed, "blind_spots.csv")
    if rankings.cve_id.isna().any() or rankings.cve_id.duplicated().any():
        raise ValueError("historical_rankings.csv has duplicate or missing CVE IDs")
    k = int(meta["top_k"])
    rankings = rankings.copy()
    rankings["rank"] = pd.to_numeric(rankings["rank"], errors="raise").astype(int)
    rankings["kev_by_nov4"] = pd.to_numeric(rankings["kev_by_nov4"], errors="raise").astype(int)
    if rankings["rank"].duplicated().any() or rankings["rank"].min() != 1 or rankings["rank"].max() != len(rankings):
        raise ValueError("Historical ranks must be unique, consecutive and start at 1")
    actual_hits = int(((rankings.kev_by_nov4 == 1) & (rankings["rank"] <= k)).sum())
    actual_misses = rankings[(rankings.kev_by_nov4 == 1) & (rankings["rank"] > k)]
    if (len(rankings) != meta["historical_cves"] or
        int((rankings.kev_by_nov4 == 1).sum()) != meta["kev_positives_by_nov4"] or
        actual_hits != meta["caught_in_top_k"] or
        len(actual_misses) != meta["missed_outside_top_k"] or
        set(actual_misses.cve_id) != set(misses.cve_id)):
        raise ValueError("Rankings, misses and summary do not agree. Recheck build_blind_spots.py outputs.")
    for col in voters:
        if rankings[col].isna().any():
            raise ValueError(f"Model score column {col} contains missing values")
        # These are rankings across the entire historical evaluation population.
        rankings[f"_pct_{col}"] = rankings[col].rank(pct=True)
        rankings[f"_rank_{col}"] = rankings[col].rank(ascending=False, method="min").astype(int)
    return rankings, actual_misses, voters, k


def load_nvd_context(data_dir: Path, ids: set[str]) -> dict[str, dict]:
    """Only use current NIST records as RETROSPECTIVE context, never snapshot proof."""
    entries: dict[str, dict] = {}
    for chunk in pd.read_csv(
        data_dir / "vulnerabilities.csv",
        usecols=lambda col: col in NVD_FIELDS,
        dtype={"cve_id": "string"},
        chunksize=250_000,
    ):
        require_columns(chunk, set(NVD_FIELDS), "vulnerabilities.csv")
        subset = chunk[chunk.cve_id.isin(ids)]
        for rec in subset.to_dict("records"):
            cve_id = str(rec["cve_id"])
            if cve_id in entries:
                raise ValueError(f"Duplicate CVE in vulnerabilities.csv: {cve_id}")
            entries[cve_id] = {
                "description": optional_string(rec.get("description"), 1500),
                "assigner": optional_string(rec.get("assigner")),
                "cvss3_score": optional_number(rec.get("cvss3_score")),
                "cvss3_severity": optional_string(rec.get("cvss3_severity")),
                "cvss4_score": optional_number(rec.get("cvss4_score")),
                "cwe_ids": optional_string(rec.get("cwe_ids")),
                "attack_vector": optional_string(rec.get("attack_vector")),
                "attack_complexity": optional_string(rec.get("attack_complexity")),
                "privileges_required": optional_string(rec.get("privileges_required")),
                "user_interaction": optional_string(rec.get("user_interaction")),
            }
    if missing := (ids - entries.keys()):
        raise ValueError(f"Missing current NIST context for {len(missing)} CVEs, e.g. {sorted(missing)[:3]}")
    return entries


def load_kev_context(data_dir: Path, ids: set[str]) -> dict[str, dict]:
    wanted = {"cve_id", "vendor", "product", "name", "known_ransomware_use"}
    df = pd.read_csv(
        data_dir / "known_exploited.csv",
        usecols=lambda col: col in wanted,
        dtype={"cve_id": "string"},
    )
    require_columns(df, wanted, "known_exploited.csv")
    df = df[df.cve_id.isin(ids)]
    if df.cve_id.duplicated().any():
        raise ValueError("Duplicate CVE ID in known_exploited.csv")
    return {
        str(r["cve_id"]): {key: optional_string(r.get(key)) for key in wanted - {"cve_id"}}
        for r in df.to_dict("records")
    }


def load_snapshot_signals(data_dir: Path, by_id: dict[str, str]):
    """Join only signal data from each CVE's snapshot month.

    Also compute observed-month EPSS percentiles over all scored CVEs for that
    month, not merely over the 75 KEV-positive investigation cases.
    """
    selected_months = set(by_id.values())
    selected_ids = set(by_id)
    snapshots = {}
    monthly_epss_parts = defaultdict(list)
    selected_columns = {"month", "cve_id", "epss", "in_kev", *SIGNAL_FLAGS}
    for year in (2024, 2025):
        filepath = data_dir / f"exploit_signals_{year}.csv"
        for chunk in pd.read_csv(
            filepath,
            usecols=lambda col: col in selected_columns,
            dtype={"month": "string", "cve_id": "string"},
            chunksize=250_000,
        ):
            require_columns(chunk, selected_columns, filepath.name)
            chunk = chunk[chunk.month.isin(selected_months)]
            if chunk.empty:
                continue
            scored = pd.to_numeric(chunk["epss"], errors="coerce")
            chunk["_epss_num"] = scored.where(scored.between(0, 1))
            for month, group in chunk.groupby("month"):
                vals = group["_epss_num"].dropna().to_numpy(dtype=float)
                if len(vals):
                    monthly_epss_parts[str(month)].append(vals)
            for rec in chunk[chunk.cve_id.isin(selected_ids)].to_dict("records"):
                cve_id = str(rec["cve_id"])
                if str(rec["month"]) != by_id[cve_id]:
                    continue
                if cve_id in snapshots:
                    raise ValueError(f"Duplicate signal record for {cve_id} in {by_id[cve_id]}")
                flags = {
                    field: int(optional_number(rec.get(field)) == 1)
                    for field in SIGNAL_FLAGS
                }
                snapshots[cve_id] = {
                    "record_present": True,
                    "epss": optional_number(rec.get("_epss_num"), 8),
                    "code_flags": flags,
                    "any_recorded_code_flag": any(flags.values()),
                }

    month_dist = {
        m: np.sort(np.concatenate(chunks)) for m, chunks in monthly_epss_parts.items()
    }
    for cve_id, month in by_id.items():
        if cve_id not in snapshots:
            # Matches the benchmark's 'blank EPSS, zero flags' convention,
            # but means no snapshot row was observed, NOT proven no attack code.
            snapshots[cve_id] = {
                "record_present": False, "epss": None,
                "code_flags": {flag: 0 for flag in SIGNAL_FLAGS},
                "any_recorded_code_flag": False,
            }
        info = snapshots[cve_id]
        arr = month_dist.get(month)
        epss = info["epss"]
        info["epss_percentile_within_observed_month"] = (
            round(100.0 * np.searchsorted(arr, epss, side="right") / len(arr), 2)
            if epss is not None and arr is not None and len(arr) else None
        )
        info["snapshot_month"] = month
        info["interpretation_note"] = (
            "No monthly record was present; score missing and flags default to zero. "
            "This is not proof of no public exploit or no attack."
            if not info["record_present"] else
            "Recorded signals as of month end; a zero flag means no code recorded "
            "in this source, not that exploitation was impossible."
        )
    return snapshots


def median_or_none(values):
    valid = [float(x) for x in values if x is not None and np.isfinite(x)]
    return round(float(np.median(valid)), 2) if valid else None


def group_statistics(cases: list[dict]):
    total = len(cases)
    observed_rows = [c for c in cases if c["snapshot_signals"]["record_present"]]
    epss_cases = [c for c in cases if c["snapshot_signals"]["epss"] is not None]
    n_flagged = sum(c["snapshot_signals"]["any_recorded_code_flag"] for c in cases)
    return {
        "count": total,
        "monthly_signal_rows_present": len(observed_rows),
        "no_monthly_signal_row": total - len(observed_rows),
        "epss_available_count": len(epss_cases),
        "median_snapshot_epss": median_or_none(
            c["snapshot_signals"]["epss"] for c in epss_cases
        ),
        "median_snapshot_epss_percentile": median_or_none(
            c["snapshot_signals"]["epss_percentile_within_observed_month"] for c in cases
        ),
        "cases_with_any_recorded_public_code": n_flagged,
        "public_code_flag_share_percent": round(100 * n_flagged / total, 1) if total else None,
        "median_days_published_to_kev": median_or_none(
            c["confirmed_outcome"]["days_published_to_kev"] for c in cases
        ),
        "median_current_nvd_cvss3": median_or_none(
            c["current_nvd_context"]["cvss3_score"] for c in cases
        ),
    }


def make_cases(rankings, voters, k, snapshots, nvd, kev) -> list[dict]:
    rows = rankings[rankings.kev_by_nov4 == 1].sort_values("rank")
    cases = []
    for row in rows.to_dict("records"):
        cve_id = str(row["cve_id"])
        rank = int(row["rank"])
        published = pd.Timestamp(row["published"])
        added = pd.Timestamp(row["kev_date_added"])
        if pd.isna(published) or pd.isna(added):
            raise ValueError(f"Missing dates for positive CVE {cve_id}")
        snapshot_month = str(row["snapshot_month"])
        if published.strftime("%Y-%m") != snapshot_month:
            raise ValueError(f"Unexpected snapshot month for {cve_id}")
        snapshot_end = pd.Period(snapshot_month, freq="M").end_time.normalize()
        if added <= snapshot_end:
            raise ValueError(f"KEV added before or at snapshot cutoff for {cve_id}")
        voters_info = {
            name: {
                "raw_score": optional_number(row[name], 8),
                "historical_rank": int(row[f"_rank_{name}"]),
                "percentile_rank": optional_number(row[f"_pct_{name}"], 6),
            }
            for name in voters
        }
        status = "caught" if rank <= k else "missed"
        band = None if status == "caught" else (
            "near_miss_518_to_1000" if rank <= 1000 else
            "mid_miss_1001_to_5000" if rank <= 5000 else "deep_miss_over_5000"
        )
        cases.append({
            "cve_id": cve_id,
            "classification": status,
            "miss_band": band,
            "model_evaluation": {
                "historical_rank": rank,
                "top_k_cutoff": k,
                "places_outside_cutoff": max(0, rank - k),
                "ensemble_percentile_rank": optional_number(row["ensemble_score"], 8),
                "note": "Rank average is not a calibrated attack probability. "
                        "A miss means outside a fixed review budget, not a zero-risk prediction.",
                "voters": voters_info,
            },
            "snapshot_signals": snapshots[cve_id],
            "current_nvd_context": nvd[cve_id],
            "current_nvd_context_warning":
                "Current-version NIST data. Description, CVSS, CWE and attack fields "
                "may have been revised after the snapshot; do not claim they were known then.",
            "confirmed_outcome": {
                "published": published.strftime("%Y-%m-%d"),
                "snapshot_cutoff": snapshot_end.strftime("%Y-%m-%d"),
                "kev_added": added.strftime("%Y-%m-%d"),
                "days_published_to_kev": int((added - published).days),
                "days_after_snapshot_to_kev": int((added - snapshot_end).days),
                "kev_catalog_context": kev.get(cve_id),
                "note": "KEV outcome was learned after the snapshot. "
                        "KEV absence does not prove a vulnerability was not exploited.",
            },
        })
    return cases


def save_outputs(cases: list[dict], summary: dict) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    with (OUTPUT_DIR / "investigation_cases.json").open("w", encoding="utf-8") as fp:
        json.dump({"metadata": summary, "cases": cases}, fp, indent=2, ensure_ascii=False,
                  allow_nan=False)
    with (OUTPUT_DIR / "investigation_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False, allow_nan=False)

    flat_rows = []
    for c in cases:
        signals = c["snapshot_signals"]
        flat_rows.append({
            "cve_id": c["cve_id"],
            "classification": c["classification"],
            "miss_band": c["miss_band"],
            "historical_rank": c["model_evaluation"]["historical_rank"],
            "ensemble_percentile_rank": c["model_evaluation"]["ensemble_percentile_rank"],
            "published": c["confirmed_outcome"]["published"],
            "snapshot_month": signals["snapshot_month"],
            "kev_added": c["confirmed_outcome"]["kev_added"],
            "days_published_to_kev": c["confirmed_outcome"]["days_published_to_kev"],
            "signal_record_present": signals["record_present"],
            "epss_at_snapshot": signals["epss"],
            "epss_observed_month_percentile": signals["epss_percentile_within_observed_month"],
            **{f"snapshot_{flag}": signals["code_flags"][flag] for flag in SIGNAL_FLAGS},
            "any_recorded_public_code": signals["any_recorded_code_flag"],
            "current_nvd_cvss3_not_snapshot_verified": c["current_nvd_context"]["cvss3_score"],
            "current_nvd_description_not_snapshot_verified": c["current_nvd_context"]["description"],
        })
    pd.DataFrame(flat_rows).to_csv(OUTPUT_DIR / "investigation_cases.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="Location of the existing original IBM CSV files")
    args = parser.parse_args()

    rankings_path = OUTPUT_DIR / "historical_rankings.csv"
    misses_path = OUTPUT_DIR / "blind_spots.csv"
    summary_path = OUTPUT_DIR / "summary.json"
    for path in (rankings_path, misses_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing previous-stage output: {path}")

    old_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rankings = pd.read_csv(
        rankings_path,
        dtype={"cve_id": "string", "snapshot_month": "string"},
        low_memory=False,
    )
    misses = pd.read_csv(misses_path, dtype={"cve_id": "string"})
    rankings, _, voters, k = validate_rankings(rankings, misses, old_summary)
    positives = rankings[rankings.kev_by_nov4 == 1]
    by_id = dict(zip(positives.cve_id.astype(str), positives.snapshot_month.astype(str)))
    print(f"Verified {len(positives)} historical KEV positives: "
          f"{old_summary['caught_in_top_k']} caught, {old_summary['missed_outside_top_k']} missed.")

    data_dir = find_data_dir(args.data_dir)
    print(f"Reading IBM source data from: {data_dir}")
    nvd = load_nvd_context(data_dir, set(by_id))
    kev = load_kev_context(data_dir, set(by_id))
    snapshots = load_snapshot_signals(data_dir, by_id)
    cases = make_cases(rankings, voters, k, snapshots, nvd, kev)
    caught = [c for c in cases if c["classification"] == "caught"]
    missed = [c for c in cases if c["classification"] == "missed"]
    summary = {
        "evaluation": "Historical 5-fold out-of-fold evaluation (2024–2025), not 2026 outcomes",
        "historical_evaluation_population": int(len(rankings)),
        "top_k": k,
        "positive_count": len(cases),
        "caught_count": len(caught),
        "missed_count": len(missed),
        "performance": {
            "historical_recall_at_k": round(len(caught) / len(cases), 4),
        },
        "comparison": {
            "caught": group_statistics(caught),
            "missed": group_statistics(missed),
        },
        "missed_rank_bands": {
            "518_to_1000": sum(c["miss_band"] == "near_miss_518_to_1000" for c in missed),
            "1001_to_5000": sum(c["miss_band"] == "mid_miss_1001_to_5000" for c in missed),
            "over_5000": sum(c["miss_band"] == "deep_miss_over_5000" for c in missed),
        },
        "limitations": [
            "The 75 cases are confirmed KEV additions, not a representative sample of all CVEs.",
            "This is a descriptive caught-versus-missed comparison, not a causal feature-attribution analysis.",
            "EPSS and four exploit-code flags are joined at the CVE's snapshot month.",
            "NIST records are current-version data and may include edits made after the snapshot.",
            "A missing signal record is not evidence that exploit code or exploitation did not exist.",
            "EPSS percentiles use the observed, nonblank monthly signals and may differ from other percentile conventions.",
            "Historical model scores and percentile ranks are NOT probabilities of exploitation.",
            "A KEV addition after the snapshot is an outcome, not a feature available at prediction time.",
            "The 2026 benchmark ground truth is not known from these files.",
        ],
        "watsonx_usage": "Use as evidence for hypothesis generation, not proof of why a model missed a case. "
                         "Separate snapshot signals, retrospective NIST context and later outcomes.",
    }
    if (len(caught), len(missed), len(cases)) != (32, 43, 75):
        raise ValueError("Unexpected historical caught/missed counts. No files written.")
    save_outputs(cases, summary)
    print("\nComparison (confirmed historical KEV additions only):")
    print(json.dumps(summary["comparison"], indent=2))
    print("\nSaved to Amy/feature/output:")
    print("  investigation_cases.json   (all 75 case records for watsonx.ai)")
    print("  investigation_cases.csv    (flat comparison table)")
    print("  investigation_summary.json (aggregate descriptive findings)")
    print("\nChen's files have not been changed.")


if __name__ == "__main__":
    main()
