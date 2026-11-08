"""Build a read-only historical blind-spot dataset for Chen's 3-model evaluation.

Save as Amy/feature/build_blind_spots.py and run from the project root:
    python Amy/feature/build_blind_spots.py

Requires pandas + pyarrow and the original IBM vulnerabilities.csv and
known_exploited.csv files. No files outside Amy/feature/output are written.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

FEATURE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FEATURE_DIR.parent.parent
RESULTS = PROJECT_ROOT / "chen" / "results"
OUTPUT = FEATURE_DIR / "output"


def find_data_dir(explicit: str | None) -> Path:
    candidates = ([Path(explicit).expanduser()] if explicit else []) + [
        PROJECT_ROOT / "IBM" / "data",
        PROJECT_ROOT / "IBM" / "data" / "data",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "data" / "data",
        PROJECT_ROOT / "chen" / "IBM" / "data",
        PROJECT_ROOT / "chen" / "IBM" / "data" / "data",
    ]
    for directory in candidates:
        if (directory / "vulnerabilities.csv").is_file() and (
            directory / "known_exploited.csv"
        ).is_file():
            return directory
    raise FileNotFoundError(
        "Cannot find IBM's vulnerabilities.csv and known_exploited.csv. "
        "Use --data-dir to specify the directory containing both CSV files."
    )


def require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="Folder containing the original IBM CSVs")
    args = parser.parse_args()

    lr_path = RESULTS / "cv_oof_lr.parquet"
    lora_path = RESULTS / "cv_oof_lora.parquet"
    for path in (lr_path, lora_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing model scores: {path}")
    data_dir = find_data_dir(args.data_dir)

    # Use Chen's historical, out-of-fold scores; do not train or edit a model.
    lr = pd.read_parquet(lr_path)
    lora = pd.read_parquet(lora_path)
    require_columns(lr, {"cve_id", "lr_full_llmx_noemb", "lgbm_luis_only"}, lr_path.name)
    require_columns(lora, {"cve_id", "lora_control"}, lora_path.name)
    for name, frame in (("LR", lr), ("LoRA", lora)):
        if frame["cve_id"].isna().any() or frame["cve_id"].duplicated().any():
            raise ValueError(f"Duplicate or empty CVE IDs in {name} scores")

    pred = lr[["cve_id", "lr_full_llmx_noemb", "lgbm_luis_only"]].merge(
        lora[["cve_id", "lora_control"]], on="cve_id", how="inner", validate="one_to_one"
    )
    if len(pred) != len(lr) or len(pred) != len(lora):
        raise ValueError("LR and LoRA historical score files do not contain the same CVEs")

    # Reproduce the rank averaging from chen/scripts/analysis/eval_all.py.
    voters = ["lr_full_llmx_noemb", "lora_control", "lgbm_luis_only"]
    if pred[voters].isna().any().any():
        raise ValueError("Missing scores in at least one model voter")
    pred["ensemble_score"] = sum(pred[c].rank(pct=True) for c in voters) / 3

    # We only need the published date and the historical KEV addition date.
    nvd = pd.read_csv(
        data_dir / "vulnerabilities.csv", usecols=["cve_id", "published"],
        dtype={"cve_id": "string"}
    )
    nvd["published"] = pd.to_datetime(nvd["published"], errors="coerce")
    if nvd["cve_id"].duplicated().any():
        raise ValueError("Duplicate CVE IDs in vulnerabilities.csv")

    kev = pd.read_csv(
        data_dir / "known_exploited.csv", usecols=["cve_id", "date_added"],
        dtype={"cve_id": "string"}
    )
    kev["date_added"] = pd.to_datetime(kev["date_added"], errors="coerce")
    if kev["cve_id"].duplicated().any():
        raise ValueError("Duplicate CVE IDs in known_exploited.csv")

    pred = pred.merge(nvd, on="cve_id", how="left", validate="one_to_one")
    pred = pred.merge(kev.rename(columns={"date_added": "kev_date_added"}),
                      on="cve_id", how="left", validate="one_to_one")
    if pred["published"].isna().any():
        raise ValueError("Historical model CVEs are missing published dates in IBM data")

    # Same November 4, *publication-year* outcome used by Chen. Rows in
    # historical OOF scores are already filtered for snapshot eligibility.
    cutoff = pd.to_datetime(pred["published"].dt.year.astype(str) + "-11-04")
    pred["kev_by_nov4"] = (
        pred["kev_date_added"].notna() &
        (pred["kev_date_added"] <= cutoff)
    ).astype(int)
    pred["snapshot_month"] = pred["published"].dt.strftime("%Y-%m")

    # Fixed tie-breaker makes CSV order reproducible. Actual ranks in the
    # released Chen evaluation have very few ties; compare hits to verify.
    pred = pred.sort_values(["ensemble_score", "cve_id"],
                            ascending=[False, True]).reset_index(drop=True)
    pred["rank"] = pred.index + 1
    k = round(500 * len(pred) / 72670)
    pred["selected_top_k"] = pred["rank"] <= k
    positives = pred[pred["kev_by_nov4"] == 1]
    hits = int((positives["rank"] <= k).sum())
    missed = positives[positives["rank"] > k].copy()

    summary = {
        "evaluation": "Historical 5-fold purged out-of-fold scores (not 2026 ground truth)",
        "historical_cves": int(len(pred)),
        "top_k": int(k),
        "kev_positives_by_nov4": int(len(positives)),
        "caught_in_top_k": hits,
        "missed_outside_top_k": int(len(missed)),
        "recall_at_k": round(hits / len(positives), 4) if len(positives) else None,
        "models": voters,
        "note": "A miss means outside the top-K review budget, not a prediction of zero risk.",
    }

    # Halt rather than silently show different statistics than Chen's report.
    if (len(pred), k, len(positives), hits) != (75148, 517, 75, 32):
        print(json.dumps(summary, indent=2))
        raise RuntimeError(
            "Results do not match Chen's documented 32/75 evaluation. "
            "Check dataset versions and scoring files before generating dashboard outputs."
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    columns = ["cve_id", "published", "snapshot_month", "kev_date_added",
               "kev_by_nov4", "rank", "ensemble_score", "selected_top_k"] + voters
    pred[columns].to_csv(OUTPUT / "historical_rankings.csv", index=False)
    missed[columns].to_csv(OUTPUT / "blind_spots.csv", index=False)
    with (OUTPUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nFiles saved ONLY to: {OUTPUT}")


if __name__ == "__main__":
    main()
