"""IBM Before the Breach: build time-safe historical KEV labels and ML tables.

Run after clean_ibm_data.py:
    python build_ibm_training_yearend.py --data-dir data

Optional, separate experiments (never overwrite the default outputs):
    python build_ibm_training_yearend.py --data-dir data --cutoff nov04
    python build_ibm_training_yearend.py --data-dir data --june-use-july
    python build_ibm_training_yearend.py --data-dir data --cutoff nov04 --june-use-july

The default uses publication-month snapshots and a Dec 31 KEV cutoff, as in
our initial backtest. Alternative Nov 4 labels and July snapshots for June
publications are optional *historical* sensitivity tests.

Features come only from the historical monthly signals. Present-day NIST
metadata is never used as a historical feature. KEV addition dates are used
only for eligibility and labels. 2026 scoreboard entries are never labeled.

Standard library only; scans the large historical signals one year at a time.
"""
from __future__ import annotations

import argparse
import calendar
import csv
from collections import Counter
from datetime import date
import json
from pathlib import Path
import re

FLAGS = ("exploitdb", "metasploit", "nuclei", "poc_github")
FEATURES = ("epss", *FLAGS)
META = ("cve_id", "published", "snapshot_month")
SCORE_COLS = (*META, *FEATURES)
LABEL_COLS = (*SCORE_COLS, "target")
CVE_PATTERN = re.compile(r"^CVE-\d{4}-\d{4,}$")
YEARS = (2024, 2025)


def month_end(month: str) -> date:
    try:
        y, m = map(int, month.split("-"))
        if len(month) != 7 or not 1 <= m <= 12:
            raise ValueError()
        return date(y, m, calendar.monthrange(y, m)[1])
    except (TypeError, ValueError):
        raise ValueError(f"Invalid YYYY-MM month: {month!r}") from None


def parse_date(value: str, filename: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError(f"{filename}: invalid date {value!r}") from None


def read_rows(path: Path, needed: tuple[str, ...]):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run clean_ibm_data.py first")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        missing = sorted(set(needed) - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path.name}: missing required columns {missing}")
        yield from reader


def check_id(cve: str, filename: str):
    if not CVE_PATTERN.fullmatch(cve or ""):
        raise ValueError(f"{filename}: malformed CVE ID {cve!r}")


def load_candidates(path: Path, exclude_feb: bool, june_use_july: bool):
    out = {}
    stats = Counter()
    for row in read_rows(path, ("cve_id", "published")):
        cve = row["cve_id"]
        check_id(cve, path.name)
        published = parse_date(row["published"], path.name)
        if published.year not in YEARS or published.month > 9:
            stats["outside_jan_sep_2024_2025"] += 1
            continue
        if exclude_feb and published.year == 2025 and published.month == 2:
            stats["excluded_feb_2025_sensitivity"] += 1
            continue
        if cve in out:
            raise ValueError(f"Duplicate eligible CVE publication record: {cve}")
        # June 2026 scoreboard candidates use July snapshots. Optionally mirror
        # that rule in the historical years to measure its effect on evaluation.
        snapshot_month = 7 if june_use_july and published.month == 6 else published.month
        snap = f"{published.year:04d}-{snapshot_month:02d}"
        if june_use_july and published.month == 6:
            stats[f"june_published_with_july_snapshot_{published.year}"] += 1
        out[cve] = {
            "published": published.isoformat(),
            "snapshot_month": snap,
            "snapshot_end": month_end(snap),
            "year": published.year,
        }
        stats[f"publication_candidates_{published.year}"] += 1
    return out, stats


def load_kev(path: Path):
    dates = {}
    for row in read_rows(path, ("cve_id", "date_added")):
        cve = row["cve_id"]
        check_id(cve, path.name)
        if cve in dates:
            raise ValueError(f"Duplicate KEV CVE ID: {cve}")
        d = parse_date(row["date_added"], path.name)
        if d > date(2025, 12, 31):
            raise ValueError(f"KEV data beyond training cutoff for {cve}: {d}")
        dates[cve] = d
    return dates


def extract_signal(row: dict, path: Path):
    epss = (row.get("epss") or "").strip()
    if epss:
        try:
            val = float(epss)
        except ValueError:
            raise ValueError(f"{path.name}: invalid EPSS {epss!r}") from None
        if not (0 <= val <= 1):
            raise ValueError(f"{path.name}: EPSS outside [0, 1]: {epss!r}")
    signal = {"epss": epss}
    for flag in (*FLAGS, "in_kev"):
        raw = (row.get(flag) or "").strip()
        if raw not in ("", "0", "1"):
            raise ValueError(f"{path.name}: invalid {flag}={raw!r}")
        signal[flag] = raw
    return signal


def match_historical_snapshot_signals(clean: Path, candidates: dict):
    matched = {}
    stats = Counter()
    for year in YEARS:
        path = clean / f"exploit_signals_{year}_clean.csv"
        print(f"  Scanning {path.name} ...", flush=True)
        for row in read_rows(path, ("month", "cve_id", "epss", "in_kev", *FLAGS)):
            stats[f"rows_scanned_{year}"] += 1
            cve = row["cve_id"]
            candidate = candidates.get(cve)
            if candidate is None or candidate["year"] != year:
                continue
            if row["month"] != candidate["snapshot_month"]:
                continue
            if cve in matched:
                raise ValueError(f"Duplicate historical snapshot for {cve}")
            matched[cve] = extract_signal(row, path)
            stats[f"matched_snapshot_{year}"] += 1
    return matched, stats


def cutoff_for_year(year: int, cutoff: str) -> date:
    return date(year, 11, 4) if cutoff == "nov04" else date(year, 12, 31)


def make_cohorts(candidates: dict, matched: dict, kev: dict, cutoff: str):
    cohorts = {2024: [], 2025: []}
    stats = {2024: Counter(), 2025: Counter()}
    for cve, candidate in candidates.items():
        year = candidate["year"]
        count = stats[year]
        snap_end = candidate["snapshot_end"]
        kev_date = kev.get(cve)
        signal = matched.get(cve)
        date_says_prior_kev = kev_date is not None and kev_date <= snap_end
        flag_says_prior_kev = signal is not None and signal["in_kev"] == "1"
        # Only compare two observed values. No signal row, or an empty
        # in_kev field, is unknown rather than evidence of disagreement.
        if signal is not None and signal["in_kev"] in ("0", "1"):
            if (signal["in_kev"] == "1") != date_says_prior_kev:
                count["snapshot_kev_flag_disagrees_with_kev_date"] += 1
        # Like the 2026 scoreboard, exclude cases already in KEV by snapshot.
        if date_says_prior_kev or flag_says_prior_kev:
            count["excluded_already_kev_at_snapshot"] += 1
            if kev_date is not None and kev_date < parse_date(candidate["published"], "published"):
                count["kev_date_precedes_publication"] += 1
            continue
        if signal is None:
            # IBM scoreboard convention for no signal record.
            features = {"epss": "", **{flag: "0" for flag in FLAGS}}
            count["no_publication_month_signal_record"] += 1
        else:
            features = {feature: signal[feature] for feature in FEATURES}
            count["with_publication_month_signal_record"] += 1
        label_cutoff = cutoff_for_year(year, cutoff)
        if snap_end >= label_cutoff:
            raise ValueError(
                f"Historical snapshot for {cve} ({snap_end}) must precede "
                f"the label cutoff ({label_cutoff})."
            )
        target = int(kev_date is not None and snap_end < kev_date <= label_cutoff)
        count["positive_kev_by_cutoff" if target else "negative_no_kev_by_cutoff"] += 1
        if kev_date is not None and kev_date > label_cutoff:
            count["known_kev_added_after_cutoff_counted_negative"] += 1
        if not features["epss"]:
            count["missing_epss"] += 1
        cohorts[year].append({
            "cve_id": cve, "published": candidate["published"],
            "snapshot_month": candidate["snapshot_month"],
            **features, "target": str(target),
        })
    for year in YEARS:
        cohorts[year].sort(key=lambda r: (r["snapshot_month"], r["published"], r["cve_id"]))
    return cohorts, stats


def write_rows(path: Path, rows: list[dict], columns: tuple[str, ...]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def summarize(rows: list[dict]):
    positives = sum(row["target"] == "1" for row in rows)
    months = Counter(row["snapshot_month"] for row in rows)
    positives_by_month = Counter(row["snapshot_month"] for row in rows if row["target"] == "1")
    return {
        "rows": len(rows),
        "positive": positives,
        "negative": len(rows) - positives,
        "positive_rate_pct": round(100 * positives / len(rows), 4) if rows else None,
        "counts_by_snapshot_month": dict(sorted(months.items())),
        "positives_by_snapshot_month": dict(sorted(positives_by_month.items())),
    }


def write_scoreboard(path: Path, dest: Path):
    count = 0
    seen = set()
    with dest.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=SCORE_COLS)
        writer.writeheader()
        for row in read_rows(path, SCORE_COLS):
            check_id(row["cve_id"], path.name)
            if row["cve_id"] in seen:
                raise ValueError(f"Repeated scoreboard CVE ID: {row['cve_id']}")
            seen.add(row["cve_id"])
            writer.writerow({c: row[c] for c in SCORE_COLS})
            count += 1
    if count != 72_670:
        raise ValueError(
            f"Expected exactly 72,670 scoreboard rows; found {count:,}. "
            "Do not make a submission from incomplete data."
        )
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Build IBM Jan-Sep historical CVE cohorts with time-safe KEV labels"
    )
    parser.add_argument("--data-dir", default="data", help="Folder with cleaned/ CSVs")
    parser.add_argument("--exclude-feb-2025", action="store_true",
                        help="Sensitivity test: exclude Feb 2025 due to documented signal gap (default: include)")
    parser.add_argument("--cutoff", choices=("yearend", "nov04"), default="yearend",
                        help="KEV label cutoff for each historical year (default: Dec 31)")
    parser.add_argument("--june-use-july", action="store_true",
                        help="Experimental: use July snapshots for historical June publications")
    args = parser.parse_args()
    cleaned = Path(args.data_dir) / "cleaned"
    output = Path(args.data_dir) / "processed"
    output.mkdir(parents=True, exist_ok=True)

    print("[1/5] Load publication dates: Jan-Sep 2024 and Jan-Sep 2025")
    candidates, pub_counts = load_candidates(
        cleaned / "vulnerabilities_publication_clean.csv",
        args.exclude_feb_2025,
        args.june_use_july,
    )
    print(f"  {len(candidates):,} eligible publication records before KEV exclusion")

    print("[2/5] Load KEV addition dates (labels and eligibility only)")
    kev = load_kev(cleaned / "known_exploited_clean.csv")
    print(f"  {len(kev):,} KEV catalog entries")

    print("[3/5] Retrieve historical snapshot features from monthly signals")
    matched, signal_stats = match_historical_snapshot_signals(cleaned, candidates)
    print(f"  {len(matched):,} matching historical snapshot signal rows")

    print("[4/5] Build historical labels and chronological backtest cohorts")
    cohorts, cohort_stats = make_cohorts(candidates, matched, kev, args.cutoff)
    train, validation = cohorts[2024], cohorts[2025]
    combined = train + validation

    # Default names stay EXACTLY as before for existing training scripts.
    # Optional experiments get their own suffix, avoiding accidental overwrite.
    suffix = "yearend" if args.cutoff == "yearend" else "nov04"
    if args.june_use_july:
        suffix += "_junejuly"
    if args.exclude_feb_2025:
        suffix += "_exclude_feb2025"

    write_rows(output / f"backtest_train_{suffix}_2024.csv", train, LABEL_COLS)
    write_rows(output / f"backtest_validation_{suffix}_2025.csv", validation, LABEL_COLS)
    write_rows(output / f"all_labeled_{suffix}.csv", combined, LABEL_COLS)

    print("[5/5] Copy the 2026 prediction inputs and write report")
    scoreboard_n = write_scoreboard(
        cleaned / "scoreboard_2026_clean.csv",
        output / f"scoreboard_features_2026_{suffix}.csv",
    )
    report = {
        "label_definition": (
            "1 iff historical KEV date_added is after snapshot_month's last day "
            f"and on/before {cutoff_for_year(2025, args.cutoff).strftime('%b %d')} "
            "of that historical year"
        ),
        "cutoff_mode": args.cutoff,
        "june_use_july": args.june_use_july,
        "output_suffix": suffix,
        "cohorts": {
            "2024": {
                "publication_months": "2024-01 to 2024-09",
                "kev_cutoff": cutoff_for_year(2024, args.cutoff).isoformat(),
                "backtest_role": "TRAIN", **summarize(train),
                "checks": dict(cohort_stats[2024]),
            },
            "2025": {
                "publication_months": "2025-01 to 2025-09",
                "kev_cutoff": cutoff_for_year(2025, args.cutoff).isoformat(),
                "backtest_role": "VALIDATION", **summarize(validation),
                "checks": dict(cohort_stats[2025]),
            },
        },
        "combined_after_model_selection": summarize(combined),
        "publication_filter": dict(pub_counts),
        "historical_signal_scan": dict(signal_stats),
        "feature_columns": list(FEATURES),
        "metadata_not_model_features": list(META),
        "target_column": "target",
        "scoreboard_candidates": scoreboard_n,
        "exclude_feb_2025": args.exclude_feb_2025,
        "notes": [
            "2024 labels through the selected 2024 cutoff can be learned before a 2025 backtest.",
            "2025 outcomes are never used to train a model evaluated on 2025.",
            "After model selection, optionally refit on the corresponding all_labeled CSV for 2026 predictions.",
            "Official 2026 scoring runs through Nov 4, 2026; the nov04 option is a closer historical analogue.",
            "Dec 31 remains the default for reproducibility; alternative label cutoffs are sensitivity tests.",
            "The june-use-july option also changes historical KEV eligibility at the July month-end snapshot.",
            "2025 has a known February signal-data gap; missing monthly rows do not prove no signals existed.",
            "EPSS model was revised in March 2025, so historical scores may shift in distribution.",
            "Missing signal records are represented with blank EPSS and four 0 exploit flags, as on the 2026 scoreboard.",
            "The absence of KEV addition is not proof of the absence of exploitation.",
            "in_kev is used for historical candidate exclusion only and never as a feature.",
            "Current NIST status, CVSS, CWE, products, references, and last_modified are excluded to limit future-data leakage.",
            "Model evaluation should compare hits@500 on the SAME 2025 candidate cohort and cutoff.",
            "2025 has already influenced teammate heuristic selection: do not treat that heuristic's 2025 score as untouched holdout evidence.",
            "Default validation expects 35,280 / 52 because snapshot in_kev excludes CVE-2025-4664.",
            "Alternate cutoff or snapshot experiments may change positive counts and cohort membership.",
        ],
    }
    report_path = output / f"training_report_{suffix}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for year, cohort in ((2024, train), (2025, validation)):
        s = summarize(cohort)
        print(f"  {year}: {s['rows']:,} candidates | {s['positive']:,} positives | {s['positive_rate_pct']}% positive")
        print(f"       Missing historical snapshot signal rows: {cohort_stats[year]['no_publication_month_signal_record']:,}")
    print(f"  2026 scoreboard candidates: {scoreboard_n:,}")
    if not args.exclude_feb_2025 and not args.june_use_july:
        if len(validation) != 35_280:
            print(f"  WARNING: Expected 35,280 validation CVEs; got {len(validation):,}.")
            print("           Check the historical eligibility and source data.")
        else:
            print("  Validation cohort size verified: 35,280 candidates.")
        expected_positive = 52 if args.cutoff == "yearend" else 48
        actual_positive = summarize(validation)["positive"]
        if actual_positive != expected_positive:
            print(f"  WARNING: Expected {expected_positive} positives; got {actual_positive}.")
    else:
        print("  NOTE: Alternate snapshot/month filtering is enabled; default cohort counts do not apply.")
    print(f"  Report: {report_path}")
    if summarize(train)["positive"] < 25:
        print("  CAUTION: fewer than 25 positives in 2024 training; LightGBM may overfit.")
    if summarize(validation)["positive"] < 10:
        print("  CAUTION: fewer than 10 positives in 2025 validation; metric has high variability.")
    print(f"DONE. Files written for variant: {suffix}")
    print("Existing cleaned files and previous 90-day outputs have not been modified.")


if __name__ == "__main__":
    main()
