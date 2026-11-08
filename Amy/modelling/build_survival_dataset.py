"""Build time-to-KEV training, validation, and prediction tables.

IBM Before the Breach / Sheridan Datathon 2026

PREREQUISITES (your existing scripts):
    python clean_ibm_data.py --data-dir data
    python build_ibm_training_yearend.py --data-dir data

RUN:
    python build_survival_dataset.py --data-dir data

INPUTS:
    data/cleaned/known_exploited_clean.csv
    data/processed/backtest_train_yearend_2024.csv
    data/processed/backtest_validation_yearend_2025.csv
    data/processed/scoreboard_features_2026_yearend.csv

OUTPUTS (data/survival/):
    survival_train_2024.csv             2024 events/censoring by 2024-12-31
    survival_validation_2025.csv       2025 events/censoring by 2025-12-31,
                                       PLUS separate Nov-04 labels and durations
    survival_scoreboard_2026.csv       Unlabeled 2026 CVEs, with time until Nov 04
    survival_data_report.json

The 2024 cohort is the ONLY data used to fit the validation models. Historical
KEV dates are only used for survival outcomes, never as predictive features.
2026 KEV outcomes are not available in the supplied data and are NOT imputed.
All intervals begin at the END of the provided snapshot_month (not publication).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FLAGS = ("exploitdb", "metasploit", "nuclei", "poc_github")
FEATURES = ("epss", *FLAGS)
REQUIRED = ("cve_id", "published", "snapshot_month", *FEATURES)


def read_table(path: Path, required: tuple[str, ...]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find {path}. Run the original two pipeline scripts first.")
    table = pd.read_csv(path, low_memory=False)
    missing = sorted(set(required).difference(table.columns))
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    table["cve_id"] = table["cve_id"].astype("string").str.strip().str.upper()
    if table["cve_id"].isna().any() or table["cve_id"].duplicated().any():
        raise ValueError(f"{path.name} has null or duplicate CVE IDs")
    return table


def add_snapshot_dates(table: pd.DataFrame, name: str) -> pd.DataFrame:
    table = table.copy()
    snap = pd.to_datetime(table["snapshot_month"].astype(str) + "-01", errors="coerce")
    if snap.isna().any():
        raise ValueError(f"{name}: invalid snapshot_month values")
    table["snapshot_end"] = (snap + pd.offsets.MonthEnd(0)).dt.normalize()
    published = pd.to_datetime(table["published"], errors="coerce")
    if published.isna().any() or (table["snapshot_end"] < published).any():
        raise ValueError(f"{name}: invalid publication or snapshot dates")
    return table


def add_outcomes(table: pd.DataFrame, cutoff: str, suffix: str = "") -> pd.DataFrame:
    """Right-censor CVEs absent from KEV by the chosen calendar cutoff."""
    deadline = pd.Timestamp(cutoff)
    if (table["snapshot_end"] >= deadline).any():
        raise ValueError(f"Snapshots must precede the cutoff {cutoff}")

    observed = table["kev_date"].notna() & (table["kev_date"] > table["snapshot_end"]) & (
        table["kev_date"] <= deadline
    )
    end_date = table["kev_date"].where(observed, deadline)
    duration = (end_date - table["snapshot_end"]).dt.days
    if (duration <= 0).any():
        raise ValueError("Survival duration must always be positive")
    table[f"event{suffix}"] = observed.astype("int8")
    table[f"duration_days{suffix}"] = duration.astype("int32")
    return table


def attach_historical_outcomes(raw: pd.DataFrame, kev: pd.DataFrame, year: int) -> pd.DataFrame:
    table = add_snapshot_dates(raw, str(year))
    table = table.merge(kev, on="cve_id", how="left", validate="one_to_one")
    if not (table["snapshot_end"].dt.year == year).all():
        raise ValueError(f"Historical cohort {year} contains snapshots from another year")
    # The original builder has already excluded records in KEV at the snapshot.
    unexpected_prior = table["kev_date"].notna() & (table["kev_date"] <= table["snapshot_end"])
    if unexpected_prior.any():
        raise ValueError(f"{year}: {unexpected_prior.sum()} CVEs were already in KEV at snapshot")
    table = add_outcomes(table, f"{year}-12-31")
    if "target" not in table:
        raise ValueError(f"{year}: historical table must contain the year-end target")
    old_target = pd.to_numeric(table["target"], errors="raise").astype(int)
    if not (old_target == table["event"]).all():
        raise ValueError(f"{year}: survival outcomes disagree with the existing year-end target")
    if year == 2025:
        table = add_outcomes(table, "2025-11-04", suffix="_nov04")
    table = table.drop(columns=["target", "kev_date"])
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Create KEV time-to-event datasets")
    parser.add_argument("--data-dir", default="data", help="Folder containing cleaned/ and processed/")
    args = parser.parse_args()

    root = Path(args.data_dir).resolve()
    processed, cleaned = root / "processed", root / "cleaned"
    output = root / "survival"
    output.mkdir(parents=True, exist_ok=True)

    kev = read_table(cleaned / "known_exploited_clean.csv", ("cve_id", "date_added"))[
        ["cve_id", "date_added"]
    ]
    kev["kev_date"] = pd.to_datetime(kev.pop("date_added"), errors="raise")

    print("[1/3] Building 2024 training survival records...")
    train_raw = read_table(processed / "backtest_train_yearend_2024.csv", (*REQUIRED, "target"))
    train = attach_historical_outcomes(train_raw, kev, 2024)
    train.to_csv(output / "survival_train_2024.csv", index=False, date_format="%Y-%m-%d")

    print("[2/3] Building 2025 validation survival records...")
    val_raw = read_table(processed / "backtest_validation_yearend_2025.csv", (*REQUIRED, "target"))
    val = attach_historical_outcomes(val_raw, kev, 2025)
    val.to_csv(output / "survival_validation_2025.csv", index=False, date_format="%Y-%m-%d")

    print("[3/3] Preparing unlabeled 2026 scoreboard...")
    score_raw = read_table(processed / "scoreboard_features_2026_yearend.csv", REQUIRED)
    if len(score_raw) != 72_670:
        raise ValueError(f"Expected 72,670 2026 candidates, got {len(score_raw):,}")
    scoreboard = add_snapshot_dates(score_raw, "2026 scoreboard")
    scoreboard["horizon_days"] = (
        pd.Timestamp("2026-11-04") - scoreboard["snapshot_end"]
    ).dt.days.astype("int32")
    if (scoreboard["horizon_days"] <= 0).any():
        raise ValueError("2026 scoreboard has snapshots after the evaluation deadline")
    scoreboard.to_csv(output / "survival_scoreboard_2026.csv", index=False, date_format="%Y-%m-%d")

    report = {
        "training": {
            "candidates": int(len(train)),
            "observed_kev_events_to_dec31": int(train["event"].sum()),
            "censored": int((train["event"] == 0).sum()),
            "cutoff": "2024-12-31",
        },
        "validation": {
            "candidates": int(len(val)),
            "observed_kev_events_to_nov04": int(val["event_nov04"].sum()),
            "observed_kev_events_to_dec31": int(val["event"].sum()),
            "cutoffs": ["2025-11-04", "2025-12-31"],
        },
        "scoreboard": {
            "candidates": int(len(scoreboard)),
            "prediction_deadline": "2026-11-04",
            "minimum_horizon_days": int(scoreboard["horizon_days"].min()),
            "maximum_horizon_days": int(scoreboard["horizon_days"].max()),
        },
        "source_features_only": list(FEATURES),
        "notes": [
            "2024 training data contains no KEV additions after 2024-12-31.",
            "2025 holdout labels are never used as training features.",
            "Right-censored means not observed in KEV by cutoff, NOT never exploited.",
            "The end of snapshot_month is time zero; June 2026 CVEs use July snapshots.",
            "Do not pass event/duration/label columns to predictive feature engineering.",
        ],
    }
    (output / "survival_data_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for period, table in (("2024 train", train), ("2025 validate", val)):
        print(f"  {period}: {len(table):,} CVEs, {int(table['event'].sum())} year-end KEV events")
    print(f"  2025 KEV events by Nov 4: {int(val['event_nov04'].sum())}")
    print(f"  2026 scoreboard: {len(scoreboard):,} CVEs")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
