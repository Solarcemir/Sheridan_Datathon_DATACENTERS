"""Clean the IBM Before the Breach Datathon CSVs, without creating training labels.

Usage:
    pip install pandas numpy
    python clean_ibm_data.py --data-dir data

Expected input files in data/: vulnerabilities.csv, known_exploited.csv,
exploit_signals_2024.csv, exploit_signals_2025.csv, scoreboard_2026.csv.
Optional input: weakness_types.csv.

Outputs are written to data/cleaned/. Inputs are never modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

FLAGS = ["exploitdb", "metasploit", "nuclei", "poc_github"]
SIGNAL_COLS = ["month", "cve_id", "epss", "in_kev", *FLAGS]
SCOREBOARD_COLS = ["cve_id", "published", "snapshot_month", "epss", *FLAGS]


def read_columns(path: Path, required: list[str], all_columns: bool = False) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Extract the original data.zip into your data/ folder.")
    headers = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(required) - set(headers))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    return pd.read_csv(
        path,
        usecols=None if all_columns else required,
        dtype={column: "string" for column in ("cve_id", "month", "snapshot_month") if column in headers},
        low_memory=False,
    )


def normalize_ids(frame: pd.DataFrame, column: str = "cve_id") -> None:
    frame[column] = frame[column].astype("string").str.strip().str.upper()
    invalid = frame[column].isna() | ~frame[column].str.fullmatch(r"CVE-\d{4}-\d{4,}").fillna(False)
    if invalid.any():
        raise ValueError(f"Found {int(invalid.sum())} malformed {column} values: "
                         f"{frame.loc[invalid, column].head(3).tolist()}")


def parse_dates(frame: pd.DataFrame, column: str, date_format: str) -> None:
    raw = frame[column].astype("string").str.strip()
    dates = pd.to_datetime(raw, format=date_format, errors="coerce")
    invalid = dates.isna()
    if invalid.any():
        raise ValueError(f"{column}: {int(invalid.sum())} missing/invalid dates; "
                         f"examples: {raw.loc[invalid].head(3).tolist()}")
    frame[column] = dates.dt.strftime(date_format)


def validate_numeric(frame: pd.DataFrame, include_in_kev: bool) -> None:
    epss_input = frame["epss"].astype("string").str.strip()
    epss_values = pd.to_numeric(epss_input, errors="coerce")
    invalid_epss = epss_input.notna() & epss_input.ne("") & epss_values.isna()
    out_of_range = epss_values.notna() & ~epss_values.between(0, 1)
    if invalid_epss.any() or out_of_range.any():
        raise ValueError("Found invalid EPSS values (must be blank or between 0 and 1)")
    # Keep missing EPSS missing, not 0: missing is not the same as low risk.
    frame["epss"] = epss_values.astype("Float32")
    for col in (["in_kev"] if include_in_kev else []) + FLAGS:
        raw = frame[col].astype("string").str.strip()
        values = pd.to_numeric(raw, errors="coerce")
        invalid = (raw.notna() & raw.ne("") & values.isna()) | (values.notna() & ~values.isin([0, 1]))
        if invalid.any():
            raise ValueError(f"{col}: found {int(invalid.sum())} invalid values; expected 0 or 1")
        # Keep missing binary values missing so that no data is invented.
        frame[col] = values.astype("Int8")


def unique_keys(frame: pd.DataFrame, keys: list[str], filename: str) -> tuple[pd.DataFrame, int]:
    before = len(frame)
    frame = frame.drop_duplicates().copy()  # remove identical duplicates only
    removed = before - len(frame)
    conflicting = frame.duplicated(subset=keys, keep=False)
    if conflicting.any():
        examples = frame.loc[conflicting, keys].head(4).to_dict("records")
        raise ValueError(f"{filename}: conflicting duplicate keys {keys}: {examples}. Review manually.")
    return frame, removed


def save(frame: pd.DataFrame, output_dir: Path, filename: str, removed: int, report: dict) -> None:
    path = output_dir / filename
    frame.to_csv(path, index=False, date_format="%Y-%m-%d")
    missing = {c: int(frame[c].isna().sum()) for c in frame.columns if frame[c].isna().any()}
    report[filename] = {"rows": len(frame), "exact_duplicate_rows_removed": removed,
                        "missing_values_by_column": missing}
    print(f"  {filename}: {len(frame):,} rows, {removed:,} exact duplicates removed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean IBM datathon CSV files without overwriting originals")
    parser.add_argument("--data-dir", default="data", help="Directory containing original CSVs")
    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = data_dir / "cleaned"
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "notes": [
            "Original files have not been modified.",
            "No target/label column has been created in this cleaning step.",
            "Never use current NIST metadata as an historical feature without checking availability dates.",
            "in_kev is for checking snapshot eligibility, not a feature on the 2026 scoreboard.",
            "EPSS model changed in March 2025; February 2025 historical data has a known gap.",
            "June 2026 scoreboard vulnerabilities intentionally use July snapshot_month.",
        ]
    }

    print("[1/6] Vulnerability publication dates (only safe core columns for baseline model)")
    pub = read_columns(data_dir / "vulnerabilities.csv", ["cve_id", "published"])
    normalize_ids(pub)
    parse_dates(pub, "published", "%Y-%m-%d")
    pub, ndup = unique_keys(pub, ["cve_id"], "vulnerabilities.csv")
    publication_lookup = pub.set_index("cve_id")["published"]
    pub_ids = set(pub["cve_id"].tolist())
    save(pub, output_dir, "vulnerabilities_publication_clean.csv", ndup, report)

    print("[2/6] Historical confirmed KEV additions")
    kev = read_columns(data_dir / "known_exploited.csv", ["cve_id", "date_added"], all_columns=True)
    normalize_ids(kev)
    parse_dates(kev, "date_added", "%Y-%m-%d")
    if "due_date" in kev.columns:
        # Due date isn't an input feature, so preserve blanks rather than raising an error.
        due = pd.to_datetime(kev["due_date"], format="%Y-%m-%d", errors="coerce")
        kev["due_date"] = due.dt.strftime("%Y-%m-%d")
    kev, ndup = unique_keys(kev, ["cve_id"], "known_exploited.csv")
    add_dates = pd.to_datetime(kev["date_added"])
    pub_for_kev = pd.to_datetime(kev["cve_id"].map(publication_lookup), errors="coerce")
    early = (pub_for_kev.notna() & (add_dates < pub_for_kev))
    report["known_exploited_caveats"] = {
        "kev_added_before_nist_published": int(early.sum()),
        "kev_cves_without_publication_match": int(pub_for_kev.isna().sum()),
        "max_kev_date": kev["date_added"].max(),
    }
    save(kev, output_dir, "known_exploited_clean.csv", ndup, report)

    print("[3/6] Historical exploit signals; processing years separately to limit memory")
    for year in (2024, 2025):
        filename = f"exploit_signals_{year}.csv"
        signals = read_columns(data_dir / filename, SIGNAL_COLS)
        normalize_ids(signals)
        parse_dates(signals, "month", "%Y-%m")
        if not signals["month"].str.startswith(str(year)).all():
            raise ValueError(f"{filename} contains snapshot months outside {year}")
        validate_numeric(signals, include_in_kev=True)
        signals, ndup = unique_keys(signals, ["month", "cve_id"], filename)
        report[f"exploit_signals_{year}_caveats"] = {
            "unique_cves_not_in_vulnerabilities_csv": int(signals.loc[~signals["cve_id"].isin(pub_ids), "cve_id"].nunique()),
            "snapshot_months_present": sorted(signals["month"].unique().tolist()),
        }
        save(signals, output_dir, f"exploit_signals_{year}_clean.csv", ndup, report)
        del signals

    print("[4/6] 2026 scoreboard: keep every eligible candidate")
    board = read_columns(data_dir / "scoreboard_2026.csv", SCOREBOARD_COLS)
    normalize_ids(board)
    parse_dates(board, "published", "%Y-%m-%d")
    parse_dates(board, "snapshot_month", "%Y-%m")
    validate_numeric(board, include_in_kev=False)
    board, ndup = unique_keys(board, ["cve_id"], "scoreboard_2026.csv")
    published_month = board["published"].str.slice(0, 7)
    if (board["snapshot_month"] < published_month).any():
        raise ValueError("Some scoreboard snapshot_month values precede publication; inspect source")
    report["scoreboard_caveats"] = {
        "june_published_with_july_snapshot": int(((published_month == "2026-06") & (board["snapshot_month"] == "2026-07")).sum()),
        "number_of_candidates": len(board),
    }
    save(board, output_dir, "scoreboard_2026_clean.csv", ndup, report)
    if len(board) != 72_670:
        raise ValueError(
            f"Expected exactly 72,670 scoreboard candidates; found {len(board):,}. "
            "Do not train or submit with an incomplete scoreboard."
        )

    print("[5/6] Optional weakness types lookup")
    weakness_path = data_dir / "weakness_types.csv"
    if weakness_path.exists():
        weakness = read_columns(weakness_path, ["cwe_id", "name", "kind", "abstraction", "status", "description"], all_columns=True)
        weakness["cwe_id"] = weakness["cwe_id"].astype("string").str.strip()
        if weakness["cwe_id"].isna().any():
            raise ValueError("weakness_types.csv has empty CWE IDs")
        weakness, ndup = unique_keys(weakness, ["cwe_id"], "weakness_types.csv")
        save(weakness, output_dir, "weakness_types_clean.csv", ndup, report)
    else:
        print("  Skipped (not needed for initial LightGBM baseline)")

    print("[6/6] Data-quality report")
    (output_dir / "cleaning_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Done. See {output_dir} and cleaning_report.json")
    print("Next step: construct historical, time-safe training labels and features.")


if __name__ == "__main__":
    main()
