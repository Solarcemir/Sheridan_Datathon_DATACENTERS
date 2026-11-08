"""Test whether model disagreement identifies historical KEV misses.

Run from the repository root:

    python Amy/feature/analyze_blind_spots.py

Reads Amy/feature/output historical rankings and investigation cases, Chen's
fold file (read-only), and the original monthly exploit-signal CSVs. Writes
only the four new files in Amy/feature/output. Does not retrain a model, call
watsonx.ai, or modify Chen's, Luis's, or existing feature files.

The watchlist rules below were fixed before the hit counts were interpreted.
They are evaluated on the same historical out-of-fold population used for the
original 32/75 result. That is a pre-specified retrospective test, not a
second independent trial and not a license to pick the best-looking rule.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


FEATURE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FEATURE_DIR.parent.parent
OUTPUT_DIR = FEATURE_DIR / "output"
FOLDS_PATH = PROJECT_ROOT / "chen" / "processed" / "cv_folds.parquet"

VOTERS = (
    ("lr_full_llmx_noemb", "logistic_regression"),
    ("lora_control", "granite_lora"),
    ("lgbm_luis_only", "lightgbm"),
)
VOTER_COLUMNS = [column for column, _name in VOTERS]
DISPLAY_OF = {column: name for column, name in VOTERS}
SIGNAL_FLAGS = ("exploitdb", "metasploit", "nuclei", "poc_github")
BUDGETS = (50, 100, 250, 500)
# Confirmatory selection rule, fixed before any watchlist hit count was computed:
# outside the ensemble top K, keep CVEs that at least one model placed inside its
# own top K, ordered by how far that model sits above the three-model mean.
# Other rules are baselines or secondary checks.
PRIMARY_STRATEGY = "single_model_dissent"
DISSENT_RANK_GATE = 517
NEIGHBOR_HALF_WIDTH = 250
# Communication thresholds fixed in advance. They are not decision cutoffs
# for the watchlist and were not scanned against the labels.
MATERIAL_RANGE = 0.10
STRONG_RANGE = 0.25
EPSS_V4_FIRST_MONTH = "2025-03"
MOTIVATING_CVE = "CVE-2025-6554"


def find_data_dir() -> Path:
    needed = ("exploit_signals_2024.csv", "exploit_signals_2025.csv")
    candidates = [
        PROJECT_ROOT / "IBM" / "data",
        PROJECT_ROOT / "IBM" / "data" / "data",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "data" / "data",
        PROJECT_ROOT / "chen" / "IBM" / "data",
        PROJECT_ROOT / "chen" / "IBM" / "data" / "data",
    ]
    for directory in candidates:
        if all((directory / name).is_file() for name in needed):
            return directory
    raise FileNotFoundError(
        "Could not find exploit_signals_2024.csv and exploit_signals_2025.csv. "
        "Expected them under data/ or IBM/data/."
    )


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    if value is None or isinstance(value, (str, int)):
        return value
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(clean(payload), indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054):
    if trials <= 0:
        return None, None
    proportion = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (proportion + z2 / (2.0 * trials)) / denominator
    margin = z * math.sqrt(proportion * (1.0 - proportion) / trials + z2 / (4.0 * trials * trials)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def fmt_p(p_value) -> str:
    if p_value is None or (isinstance(p_value, float) and not math.isfinite(p_value)):
        return "n/a"
    if p_value < 0.001:
        return f"{p_value:.2e}"
    return f"{p_value:.3f}"


def fmt_int(value: int) -> str:
    return f"{int(value):,}"


def fmt_pct(value) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def quantiles(series: pd.Series) -> dict:
    data = series.dropna().to_numpy(dtype=float)
    if len(data) == 0:
        return {"n": 0}
    points = [0.5, 0.9, 0.99]
    values = np.quantile(data, points)
    return {
        "n": int(len(data)),
        "min": float(np.min(data)),
        "p50": float(values[0]),
        "p90": float(values[1]),
        "p99": float(values[2]),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
    }


def load_rankings() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    summary = json.loads((OUTPUT_DIR / "summary.json").read_text(encoding="utf-8"))
    rankings = pd.read_csv(
        OUTPUT_DIR / "historical_rankings.csv",
        dtype={"cve_id": "string", "snapshot_month": "string"},
        low_memory=False,
    )
    misses = pd.read_csv(OUTPUT_DIR / "blind_spots.csv", dtype={"cve_id": "string"})
    required = {
        "cve_id", "published", "snapshot_month", "kev_date_added", "kev_by_nov4",
        "rank", "ensemble_score", "selected_top_k", *VOTER_COLUMNS,
    }
    missing = sorted(required - set(rankings.columns))
    if missing:
        raise ValueError(f"historical_rankings.csv is missing columns: {missing}")
    if rankings["cve_id"].isna().any() or rankings["cve_id"].duplicated().any():
        raise ValueError("historical_rankings.csv has missing or duplicate CVE IDs")
    rankings["rank"] = pd.to_numeric(rankings["rank"], errors="raise").astype(int)
    rankings["kev_by_nov4"] = pd.to_numeric(rankings["kev_by_nov4"], errors="raise").astype(int)
    rankings["ensemble_score"] = pd.to_numeric(rankings["ensemble_score"], errors="raise")
    for column in VOTER_COLUMNS:
        rankings[column] = pd.to_numeric(rankings[column], errors="raise")
        if rankings[column].isna().any():
            raise ValueError(f"{column} contains missing scores")
    rankings["published"] = pd.to_datetime(rankings["published"], errors="raise")
    rankings["kev_date_added"] = pd.to_datetime(rankings["kev_date_added"], errors="coerce")
    k = int(summary["top_k"])
    if tuple(rankings["rank"].agg(["min", "max", "nunique"])) != (1, len(rankings), len(rankings)):
        raise ValueError("Historical ranks are not a unique permutation of 1..N")
    official_order = rankings.sort_values(["ensemble_score", "cve_id"], ascending=[False, True])
    if official_order["cve_id"].tolist() != rankings.sort_values("rank")["cve_id"].tolist():
        raise ValueError("Stored ranks do not follow ensemble_score desc, cve_id asc")
    if int((rankings["rank"] <= k).sum()) != k:
        raise ValueError("Top-K cutoff does not select exactly K rows")
    positive = rankings["kev_by_nov4"] == 1
    caught_n = int((positive & (rankings["rank"] <= k)).sum())
    missed_ids = set(rankings.loc[positive & (rankings["rank"] > k), "cve_id"])
    if (
        len(rankings) != summary["historical_cves"]
        or int(positive.sum()) != summary["kev_positives_by_nov4"]
        or caught_n != summary["caught_in_top_k"]
        or len(missed_ids) != summary["missed_outside_top_k"]
        or missed_ids != set(misses["cve_id"])
        or (len(rankings), k, int(positive.sum()), caught_n) != (75148, 517, 75, 32)
    ):
        raise ValueError("Rankings do not reproduce the documented 75,148 / 75 / 32 / 43 evaluation")

    recomputed = sum(rankings[column].rank(method="average", pct=True) for column in VOTER_COLUMNS) / 3.0
    score_gap = float((recomputed - rankings["ensemble_score"]).abs().max())
    if score_gap > 1e-8:
        raise ValueError(f"Recomputed ensemble percentile mean differs by up to {score_gap}")
    for column, _name in VOTERS:
        rankings[f"pct_{column}"] = rankings[column].rank(method="average", pct=True)
        rankings[f"rank_{column}"] = rankings[column].rank(ascending=False, method="min").astype(int)
    percent_columns = [f"pct_{column}" for column in VOTER_COLUMNS]
    rank_columns = [f"rank_{column}" for column in VOTER_COLUMNS]
    by_rank = rankings.sort_values("rank")
    boundary_scores = by_rank["ensemble_score"].to_numpy()
    tie_info = {
        "ensemble_score_tied_rows": int(rankings["ensemble_score"].duplicated(keep=False).sum()),
        "ensemble_tie_across_k_boundary": bool(boundary_scores[k - 1] == boundary_scores[k]),
        "rows_sharing_rank_k_score": int((rankings["ensemble_score"] == boundary_scores[k - 1]).sum()),
        "voter_tied_rows": {
            DISPLAY_OF[column]: int(rankings[column].duplicated(keep=False).sum()) for column in VOTER_COLUMNS
        },
        "voter_largest_tie_block": {
            DISPLAY_OF[column]: int(rankings[column].value_counts().iloc[0]) for column in VOTER_COLUMNS
        },
        "note": (
            "Percentile ranks use average ties, as in Chen's ensemble. Displayed competition ranks use "
            "minimum ties, matching investigation_cases.json, so a CVE in a large tie block shows the best "
            "rank of that block. Selection rules use average-tie percentiles, except the dissent gate."
        ),
    }
    percent_matrix = rankings[percent_columns].to_numpy(dtype=float)
    rank_matrix = rankings[rank_columns].to_numpy(dtype=int)
    order = np.argsort(-percent_matrix, axis=1)
    row_index = np.arange(len(rankings))
    best_pos = order[:, 0]
    second_pos = order[:, 1]
    tied = percent_matrix[row_index, best_pos] == percent_matrix[row_index, second_pos]
    rankings["best_pct"] = percent_matrix.max(axis=1)
    rankings["worst_pct"] = percent_matrix.min(axis=1)
    rankings["pct_range"] = rankings["best_pct"] - rankings["worst_pct"]
    rankings["pct_std"] = rankings[percent_columns].std(axis=1, ddof=0)
    rankings["mean_pct"] = percent_matrix.mean(axis=1)
    rankings["upside"] = rankings["best_pct"] - rankings["mean_pct"]
    rankings["best_rank"] = rank_matrix.min(axis=1).astype(int)
    rankings["best_model"] = np.where(tied, "tie", [VOTER_COLUMNS[i] for i in best_pos])
    rankings["epss_era"] = np.where(
        rankings["snapshot_month"] >= EPSS_V4_FIRST_MONTH, "epss_v4", "epss_v3"
    )
    folds = pd.read_parquet(FOLDS_PATH)
    folds["cve_id"] = folds["cve_id"].astype("string")
    if folds["cve_id"].duplicated().any():
        raise ValueError("Duplicate CVE IDs in cv_folds.parquet")
    rankings = rankings.merge(folds[["cve_id", "fold", "group"]], on="cve_id", how="left", validate="one_to_one")
    if rankings["fold"].isna().any():
        raise ValueError("Some historical CVEs are missing a purged-CV fold id")
    rankings["fold"] = rankings["fold"].astype(int)
    rankings["kev_group"] = rankings["group"].where(rankings["group"].notna(), None)
    rankings = rankings.drop(columns=["group"]).sort_values("rank").reset_index(drop=True)
    if rankings["rank"].tolist() != list(range(1, len(rankings) + 1)):
        raise ValueError("Rank frame is not in rank order after the fold join")
    meta = {
        "score_gap": score_gap,
        "ties": tie_info,
        "top_k": k,
        "summary": summary,
        "percent_columns": percent_columns,
        "rank_columns": rank_columns,
    }
    return rankings, meta, misses


def load_investigation_ranks() -> dict[str, dict]:
    payload = json.loads((OUTPUT_DIR / "investigation_cases.json").read_text(encoding="utf-8"))
    cases = payload["cases"]
    if len(cases) != 75:
        raise ValueError("investigation_cases.json does not contain 75 cases")
    return {case["cve_id"]: case for case in cases}


def verify_case_ranks(rankings: pd.DataFrame, cases: dict[str, dict]) -> None:
    positives = rankings[rankings["kev_by_nov4"] == 1]
    if set(positives["cve_id"]) != set(cases):
        raise ValueError("Investigation cases and ranking positives are different CVE sets")
    for row in positives.itertuples(index=False):
        case = cases[row.cve_id]
        if int(case["model_evaluation"]["historical_rank"]) != int(row.rank):
            raise ValueError(f"Ensemble rank mismatch for {row.cve_id}")
        voters = case["model_evaluation"]["voters"]
        for column in VOTER_COLUMNS:
            recorded = int(voters[column]["historical_rank"])
            computed = int(getattr(row, f"rank_{column}"))
            if recorded != computed:
                raise ValueError(
                    f"Individual rank mismatch for {row.cve_id} {column}: file {recorded}, recomputed {computed}"
                )


def load_snapshot_signals(rankings: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    keys = set(rankings["cve_id"] + "|" + rankings["snapshot_month"])
    months = set(rankings["snapshot_month"].astype(str))
    month_parts: dict[str, list[np.ndarray]] = {month: [] for month in months}
    matched: dict[str, dict] = {}
    usecols = ["month", "cve_id", "epss", *SIGNAL_FLAGS]
    for year in (2024, 2025):
        path = data_dir / f"exploit_signals_{year}.csv"
        for chunk in pd.read_csv(
            path,
            usecols=usecols,
            dtype={"month": "string", "cve_id": "string"},
            chunksize=300_000,
        ):
            chunk = chunk[chunk["month"].isin(months)]
            if chunk.empty:
                continue
            epss = pd.to_numeric(chunk["epss"], errors="coerce")
            epss = epss.where(epss.between(0, 1))
            for month, values in epss.groupby(chunk["month"], sort=False):
                array = values.dropna().to_numpy(dtype=float)
                if len(array):
                    month_parts[str(month)].append(array)
            chunk = chunk.copy()
            chunk["key"] = chunk["cve_id"] + "|" + chunk["month"]
            hit = chunk[chunk["key"].isin(keys)]
            for record in hit.itertuples(index=False):
                key = str(record.key)
                if key in matched:
                    raise ValueError(f"Duplicate snapshot signal row for {key}")
                flags = {flag: int(getattr(record, flag) == 1) for flag in SIGNAL_FLAGS}
                raw_epss = getattr(record, "epss")
                parsed = pd.to_numeric(raw_epss, errors="coerce")
                parsed = float(parsed) if pd.notna(parsed) and 0.0 <= float(parsed) <= 1.0 else None
                matched[key] = {
                    "record_present": True,
                    "epss": parsed,
                    "any_recorded_code_flag": any(flags.values()),
                    **{f"flag_{flag}": flags[flag] for flag in SIGNAL_FLAGS},
                }
    month_arrays = {
        month: np.sort(np.concatenate(parts)) if parts else np.array([], dtype=float)
        for month, parts in month_parts.items()
    }
    rows = []
    for cve_id, month in zip(rankings["cve_id"], rankings["snapshot_month"]):
        key = f"{cve_id}|{month}"
        info = matched.get(key)
        if info is None:
            info = {
                "record_present": False,
                "epss": None,
                "any_recorded_code_flag": False,
                **{f"flag_{flag}": None for flag in SIGNAL_FLAGS},
            }
        distribution = month_arrays.get(str(month))
        epss_value = info["epss"]
        if epss_value is not None and distribution is not None and len(distribution):
            percentile = 100.0 * np.searchsorted(distribution, epss_value, side="right") / len(distribution)
            percentile = round(float(percentile), 2)
        else:
            percentile = None
        rows.append({
            "cve_id": cve_id,
            "signal_record_present": info["record_present"],
            "epss": info["epss"],
            "epss_percentile_within_month": percentile,
            "any_recorded_code_flag": bool(info["any_recorded_code_flag"]) if info["record_present"] else False,
            **{f"flag_{flag}": info[f"flag_{flag}"] for flag in SIGNAL_FLAGS},
        })
    signals = pd.DataFrame(rows)
    if signals["cve_id"].duplicated().any():
        raise ValueError("Signal join produced duplicate CVEs")
    return signals


def verify_positive_signals(rankings: pd.DataFrame, cases: dict[str, dict]) -> None:
    positives = rankings[rankings["kev_by_nov4"] == 1]
    caught_percentiles = []
    missed_percentiles = []
    caught_flags = missed_flags = 0
    for row in positives.itertuples(index=False):
        case = cases[row.cve_id]
        signals = case["snapshot_signals"]
        if bool(signals["record_present"]) != bool(row.signal_record_present):
            raise ValueError(f"Signal-row presence mismatch for {row.cve_id}")
        if bool(signals["any_recorded_code_flag"]) != bool(row.any_recorded_code_flag):
            raise ValueError(f"Exploit-code flag mismatch for {row.cve_id}")
        recorded_epss = signals["epss"]
        if recorded_epss is None:
            if not pd.isna(row.epss):
                raise ValueError(f"EPSS missingness mismatch for {row.cve_id}")
        elif abs(float(recorded_epss) - float(row.epss)) > 1e-8:
            raise ValueError(f"EPSS value mismatch for {row.cve_id}")
        recorded_pct = signals["epss_percentile_within_observed_month"]
        computed_pct = None if pd.isna(row.epss_percentile_within_month) else row.epss_percentile_within_month
        if recorded_pct is None or computed_pct is None:
            if recorded_pct is not None or computed_pct is not None:
                raise ValueError(f"EPSS percentile missingness mismatch for {row.cve_id}")
        elif abs(float(recorded_pct) - float(computed_pct)) > 0.011:
            raise ValueError(
                f"EPSS percentile mismatch for {row.cve_id}: file {recorded_pct}, recomputed {computed_pct}"
            )
        bucket = caught_percentiles if row.rank <= 517 else missed_percentiles
        if computed_pct is not None:
            bucket.append(float(computed_pct))
        if row.any_recorded_code_flag:
            if row.rank <= 517:
                caught_flags += 1
            else:
                missed_flags += 1
    if (caught_flags, missed_flags) != (27, 4):
        raise ValueError(f"Flag counts {caught_flags}/{missed_flags} do not match the preliminary 27/4 split")
    if abs(float(np.median(caught_percentiles)) - 55.17) > 0.02:
        raise ValueError("Caught median EPSS percentile does not match 55.17")
    if abs(float(np.median(missed_percentiles)) - 13.28) > 0.02:
        raise ValueError("Missed median EPSS percentile does not match 13.28")


def cohort_name(row_rank: int, positive: int, k: int) -> str:
    if positive == 1 and row_rank <= k:
        return "caught_positive"
    if positive == 1:
        return "missed_positive"
    return "not_confirmed_by_cutoff"


def disagreement_summary(rankings: pd.DataFrame, k: int) -> dict:
    frame = rankings.sort_values("rank").reset_index(drop=True)
    if not np.array_equal(frame["rank"].to_numpy(), np.arange(1, len(frame) + 1)):
        raise ValueError("Neighbor windows require the frame to be sorted by unique ensemble rank")
    outside = frame[frame["rank"] > k]
    missed = frame[(frame["kev_by_nov4"] == 1) & (frame["rank"] > k)].copy()
    caught = frame[(frame["kev_by_nov4"] == 1) & (frame["rank"] <= k)]
    not_confirmed = frame[frame["kev_by_nov4"] == 0]
    outside_negative = outside[outside["kev_by_nov4"] == 0]

    def band(rank: int) -> str:
        if rank <= 1000:
            return "near_miss_518_to_1000"
        if rank <= 5000:
            return "mid_miss_1001_to_5000"
        return "deep_miss_over_5000"

    missed["miss_band"] = missed["rank"].map(band)
    cohorts = {
        "caught_positive": caught,
        "missed_positive": missed,
        "not_confirmed_by_cutoff": not_confirmed,
        "outside_top_k_not_confirmed": outside_negative,
    }
    cohort_stats = {}
    for name, subset in cohorts.items():
        cohort_stats[name] = {
            "n": int(len(subset)),
            "percentile_range": quantiles(subset["pct_range"]),
            "percentile_std": quantiles(subset["pct_std"]),
            "upside_gap": quantiles(subset["upside"]),
            "best_individual_rank": quantiles(subset["best_rank"]),
            "share_range_at_least_0_10": float((subset["pct_range"] >= MATERIAL_RANGE).mean()) if len(subset) else None,
            "share_range_at_least_0_25": float((subset["pct_range"] >= STRONG_RANGE).mean()) if len(subset) else None,
            "share_best_rank_inside_top_k": float((subset["best_rank"] <= k).mean()) if len(subset) else None,
            "n_best_rank_inside_top_k": int((subset["best_rank"] <= k).sum()),
            "best_model_counts": subset["best_model"].value_counts().to_dict(),
        }

    def safe_mannwhitney(left: pd.Series, right: pd.Series) -> dict:
        result = stats.mannwhitneyu(left, right, alternative="two-sided", method="auto")
        return {"u": float(result.statistic), "p_value": float(result.pvalue), "n_left": int(len(left)), "n_right": int(len(right))}

    range_values = frame["pct_range"].to_numpy(dtype=float)
    upside_values = frame["upside"].to_numpy(dtype=float)
    best_values = frame["best_pct"].to_numpy(dtype=float)
    neighbor_rows = []
    for index, row in missed.iterrows():
        lo = max(k, int(index) - NEIGHBOR_HALF_WIDTH)
        hi = min(len(frame) - 1, int(index) + NEIGHBOR_HALF_WIDTH)
        neighbor_index = np.r_[np.arange(lo, int(index)), np.arange(int(index) + 1, hi + 1)]
        if len(neighbor_index) == 0:
            raise ValueError(f"No rank neighbors for {row.cve_id}")
        neighbor_rows.append({
            "cve_id": row.cve_id,
            "n_neighbors": int(len(neighbor_index)),
            "range_share_neighbors_lower": float(np.mean(range_values[neighbor_index] < row.pct_range)),
            "upside_share_neighbors_lower": float(np.mean(upside_values[neighbor_index] < row.upside)),
            "best_pct_share_neighbors_lower": float(np.mean(best_values[neighbor_index] < row.best_pct)),
        })
    neighbors = pd.DataFrame(neighbor_rows)

    def wilcoxon_against_half(shares: pd.Series) -> dict:
        deltas = shares.to_numpy(dtype=float) - 0.5
        if np.allclose(deltas, 0):
            return {"statistic": None, "p_value": 1.0, "median_share": float(shares.median())}
        result = stats.wilcoxon(deltas, alternative="two-sided", zero_method="wilcox")
        return {
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "median_share": float(shares.median()),
            "n": int(len(shares)),
        }

    outside_spearman_range = stats.spearmanr(outside["rank"], outside["pct_range"])
    outside_spearman_upside = stats.spearmanr(outside["rank"], outside["upside"])
    model_only_hits = {}
    positive = frame["kev_by_nov4"] == 1
    for column, display in VOTERS:
        selected = frame[f"rank_{column}"] <= k
        model_only_hits[display] = {
            "column": column,
            "positives_inside_own_top_k": int((positive & selected).sum()),
            "misses_with_this_model_inside_top_k": int(((frame["rank"] > k) & positive & selected).sum()),
        }
    band_rows = []
    for band_name, subset in missed.groupby("miss_band"):
        band_rows.append({
            "miss_band": band_name,
            "n": int(len(subset)),
            "n_best_rank_inside_top_k": int((subset["best_rank"] <= k).sum()),
            "n_best_rank_at_most_2k": int((subset["best_rank"] <= 2 * k).sum()),
            "n_dilution_at_least_1000_places": int((subset["rank"] - subset["best_rank"] >= 1000).sum()),
            "median_percentile_range": float(subset["pct_range"].median()),
            "median_upside": float(subset["upside"].median()),
            "median_best_rank": float(subset["best_rank"].median()),
            "median_ensemble_rank": float(subset["rank"].median()),
            "best_model_counts": subset["best_model"].value_counts().to_dict(),
        })
    winner_counts = missed.loc[missed["best_model"] != "tie", "best_model"].value_counts().to_dict()
    return {
        "cohorts": cohort_stats,
        "unstratified_tests": {
            "note": (
                "These compare missed positives with all unconfirmed CVEs outside the top K. "
                "Misses sit closer to the cutoff than a typical unconfirmed CVE, so a difference "
                "here can be ensemble rank in disguise. The neighbor test is the rank-localized check."
            ),
            "percentile_range_missed_vs_outside_unconfirmed": safe_mannwhitney(
                missed["pct_range"], outside_negative["pct_range"]
            ),
            "upside_missed_vs_outside_unconfirmed": safe_mannwhitney(
                missed["upside"], outside_negative["upside"]
            ),
        },
        "neighbor_test": {
            "definition": (
                f"For each missed positive, compare it with CVEs ranked within {NEIGHBOR_HALF_WIDTH} "
                "places on either side, never crossing back into the original top K. "
                "The share is the fraction of those neighbors with a smaller value. "
                "A median share of 0.5 means the miss looks ordinary for its ensemble rank."
            ),
            "percentile_range": wilcoxon_against_half(neighbors["range_share_neighbors_lower"]),
            "upside_gap": wilcoxon_against_half(neighbors["upside_share_neighbors_lower"]),
            "best_voter_percentile": wilcoxon_against_half(neighbors["best_pct_share_neighbors_lower"]),
            "per_miss": neighbors.to_dict("records"),
        },
        "outside_rank_correlations": {
            "spearman_rank_vs_percentile_range": {
                "rho": float(outside_spearman_range.statistic),
                "p_value": float(outside_spearman_range.pvalue),
            },
            "spearman_rank_vs_upside": {
                "rho": float(outside_spearman_upside.statistic),
                "p_value": float(outside_spearman_upside.pvalue),
            },
        },
        "single_model_top_k": model_only_hits,
        "miss_bands": band_rows,
        "missed_best_model_counts_excluding_ties": {DISPLAY_OF.get(key, key): int(value) for key, value in winner_counts.items()},
        "missed_best_model_ties": int((missed["best_model"] == "tie").sum()),
        "n_missed_best_rank_inside_top_k": int((missed["best_rank"] <= k).sum()),
        "n_outside_unconfirmed_best_rank_inside_top_k": int((outside_negative["best_rank"] <= k).sum()),
        "n_missed_dilution_at_least_1000": int((missed["rank"] - missed["best_rank"] >= 1000).sum()),
        "neighbors": neighbors,
        "missed_frame": missed,
    }


def order_outside(outside: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy == "expanded_ensemble":
        return outside.sort_values(["rank", "cve_id"], ascending=[True, True], kind="mergesort")
    if strategy == "best_voter_percentile":
        return outside.sort_values(["best_pct", "rank", "cve_id"], ascending=[False, True, True], kind="mergesort")
    if strategy == "single_model_dissent":
        pool = outside[outside["best_rank"] <= DISSENT_RANK_GATE]
        return pool.sort_values(["upside", "rank", "cve_id"], ascending=[False, True, True], kind="mergesort")
    if strategy == "percentile_range":
        return outside.sort_values(
            ["pct_range", "best_pct", "rank", "cve_id"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
    if strategy == "upside_gap":
        return outside.sort_values(
            ["upside", "best_pct", "rank", "cve_id"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
    if strategy == "sparse_or_missing_best_voter":
        pool = outside[~outside["any_recorded_code_flag"]]
        return pool.sort_values(["best_pct", "rank", "cve_id"], ascending=[False, True, True], kind="mergesort")
    if strategy == "recorded_zero_flags_best_voter":
        pool = outside[outside["signal_record_present"] & ~outside["any_recorded_code_flag"]]
        return pool.sort_values(["best_pct", "rank", "cve_id"], ascending=[False, True, True], kind="mergesort")
    if strategy.startswith("individual_"):
        display = strategy.replace("individual_", "")
        column = next(column for column, name in VOTERS if name == display)
        return outside.sort_values(
            [f"pct_{column}", "rank", "cve_id"],
            ascending=[False, True, True],
            kind="mergesort",
        )
    raise KeyError(strategy)


STRATEGY_CATALOG = (
    ("expanded_ensemble", "baseline", "Next B CVEs in the existing ensemble order."),
    ("single_model_dissent", "primary", "Outside the top K, among CVEs at least one model ranked inside its own top 517, review the B with the largest gap between the leading model's percentile and the three-model mean."),
    ("best_voter_percentile", "baseline", "Outside the top K, review the B CVEs with the highest single-model percentile rank."),
    ("individual_logistic_regression", "baseline", "Outside the top K, review the B CVEs logistic regression ranks highest."),
    ("individual_granite_lora", "baseline", "Outside the top K, review the B CVEs Granite LoRA ranks highest."),
    ("individual_lightgbm", "baseline", "Outside the top K, review the B CVEs LightGBM ranks highest."),
    ("upside_gap", "secondary", "Outside the top K, review the B CVEs whose leading model exceeds the three-model mean by the most."),
    ("percentile_range", "secondary", "Outside the top K, review the B CVEs with the largest percentile-rank range."),
    ("sparse_or_missing_best_voter", "secondary", "Same best-voter rule, restricted to CVEs with no recorded public exploit-code flag, including missing monthly rows."),
    ("recorded_zero_flags_best_voter", "secondary", "Same best-voter rule, restricted to monthly rows that are present and have all four code flags equal to zero."),
)
PRIMARY_SORT_KEY = {
    "expanded_ensemble": "rank",
    "single_model_dissent": "upside",
    "best_voter_percentile": "best_pct",
    "individual_logistic_regression": "pct_lr_full_llmx_noemb",
    "individual_granite_lora": "pct_lora_control",
    "individual_lightgbm": "pct_lgbm_luis_only",
    "upside_gap": "upside",
    "percentile_range": "pct_range",
    "sparse_or_missing_best_voter": "best_pct",
    "recorded_zero_flags_best_voter": "best_pct",
}


def tie_audit(ordered: pd.DataFrame, strategy: str, budget: int) -> dict:
    key = PRIMARY_SORT_KEY[strategy]
    if budget >= len(ordered):
        return {"boundary_tie": False, "pool_size": int(len(ordered)), "exhausted_pool": True}
    cutoff = ordered.iloc[budget - 1][key]
    nxt = ordered.iloc[budget][key]
    tied = bool(cutoff == nxt)
    block = ordered[ordered[key] == cutoff]
    return {
        "boundary_tie": tied,
        "pool_size": int(len(ordered)),
        "exhausted_pool": False,
        "tied_primary_key_count": int(len(block)),
        "positives_in_tied_primary_key": int(block["kev_by_nov4"].sum()),
        "tie_break": "ensemble rank ascending, then CVE ID ascending",
    }


def mcnemar_exact(only_left: int, only_right: int) -> float:
    discordant = only_left + only_right
    if discordant == 0:
        return 1.0
    return float(stats.binomtest(only_left, discordant, 0.5, alternative="two-sided").pvalue)


def evaluate_watchlists(rankings: pd.DataFrame, k: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    outside = rankings[rankings["rank"] > k].copy()
    if outside["cve_id"].isin(rankings.loc[rankings["rank"] <= k, "cve_id"]).any():
        raise ValueError("Watchlist pool overlaps the original top K")
    n_outside = len(outside)
    positives_outside = int(outside["kev_by_nov4"].sum())
    if positives_outside != 43 or n_outside != len(rankings) - k:
        raise ValueError("Outside-top-K pool does not contain exactly the 43 misses")
    orders = {name: order_outside(outside, name) for name, _role, _note in STRATEGY_CATALOG}
    for name, ordered in orders.items():
        if ordered["cve_id"].duplicated().any():
            raise ValueError(f"{name} produced duplicate CVEs")
        if ordered["rank"].min() <= k:
            raise ValueError(f"{name} included a CVE already inside the top K")
    positions = {
        name: pd.Series(np.arange(1, len(ordered) + 1), index=ordered["cve_id"].to_numpy())
        for name, ordered in orders.items()
    }
    expanded_pos = rankings.loc[rankings["rank"] > k, "rank"] - k
    expanded_pos.index = rankings.loc[rankings["rank"] > k, "cve_id"]
    if not positions["expanded_ensemble"].reindex(expanded_pos.index).astype(int).equals(expanded_pos.astype(int)):
        raise ValueError("Expanded-ensemble positions do not equal rank minus K")

    rows = []
    selected_ids: dict[tuple[str, int], list[str]] = {}
    for budget in BUDGETS:
        chosen = {}
        for name, ordered in orders.items():
            if len(ordered) < budget:
                raise ValueError(f"{name} pool has {len(ordered)} CVEs, fewer than budget {budget}")
            ids = ordered["cve_id"].iloc[:budget].astype(str).tolist()
            if len(ids) != budget or len(set(ids)) != budget:
                raise ValueError(f"{name} at B={budget} did not select {budget} unique CVEs")
            selected_ids[(name, budget)] = ids
            chosen[name] = set(ids)
        expanded_ids = chosen["expanded_ensemble"]
        primary_ids = chosen[PRIMARY_STRATEGY]
        for name, role, note in STRATEGY_CATALOG:
            ids = selected_ids[(name, budget)]
            id_set = set(ids)
            labels = outside.set_index("cve_id").loc[ids, "kev_by_nov4"].astype(int)
            additional = int(labels.sum())
            if additional != len(id_set & set(outside.loc[outside["kev_by_nov4"] == 1, "cve_id"])):
                raise ValueError(f"{name} at B={budget}: label sum disagrees with positive-ID intersection")
            positive_ids = [cve_id for cve_id, label in zip(ids, labels) if int(label) == 1]
            only_this = sorted(id_set & set(positive_ids) - expanded_ids) if name != "expanded_ensemble" else []
            # Positives recovered by this strategy but not by expanded ensemble.
            only_this_pos = sorted(set(positive_ids) - expanded_ids)
            only_expanded_pos = sorted((expanded_ids & set(outside.loc[outside["kev_by_nov4"] == 1, "cve_id"])) - set(positive_ids))
            overlap_pos = sorted(set(positive_ids) & expanded_ids)
            low, high = wilson_interval(additional, budget)
            hyper_p = float(stats.hypergeom.sf(additional - 1, n_outside, positives_outside, budget)) if additional else float(
                stats.hypergeom.sf(-1, n_outside, positives_outside, budget)
            )
            folds = outside.set_index("cve_id").loc[positive_ids, "fold"] if positive_ids else pd.Series(dtype=int)
            early = int((folds <= 2).sum()) if positive_ids else 0
            late = int((folds >= 3).sum()) if positive_ids else 0
            audit = tie_audit(orders[name], name, budget)
            rows.append({
                "strategy": name,
                "role": role,
                "additional_budget_B": budget,
                "extra_cves_reviewed": budget,
                "total_cves_reviewed": k + budget,
                "base_positives_in_top_517": 32,
                "eligible_pool_size": int(len(orders[name])),
                "additional_positives": additional,
                "additional_not_confirmed_by_cutoff": budget - additional,
                "total_positives_recovered": 32 + additional,
                "total_not_confirmed_in_review_set": (k - 32) + (budget - additional),
                "precision_additional": additional / budget,
                "precision_wilson_low": low,
                "precision_wilson_high": high,
                "recall_at_total_budget": (32 + additional) / 75,
                "additional_recall_of_43_misses": additional / 43,
                "additional_positive_ids": ";".join(positive_ids),
                "overlap_positive_ids_with_expanded_ensemble": ";".join(overlap_pos),
                "only_this_strategy_positive_ids_vs_expanded": ";".join(only_this_pos),
                "only_expanded_positive_ids_vs_this": ";".join(only_expanded_pos),
                "mcnemar_only_this": len(only_this_pos),
                "mcnemar_only_expanded": len(only_expanded_pos),
                "mcnemar_exact_p_vs_expanded": 1.0 if name == "expanded_ensemble" else mcnemar_exact(len(only_this_pos), len(only_expanded_pos)),
                "candidate_set_overlap_with_expanded": len(id_set & expanded_ids),
                "candidate_set_jaccard_with_expanded": len(id_set & expanded_ids) / len(id_set | expanded_ids),
                "candidate_set_overlap_with_primary": len(id_set & primary_ids),
                "early_fold_0_to_2_additional_positives": early,
                "late_fold_3_to_4_additional_positives": late,
                "expected_additional_positives_if_random": positives_outside * budget / n_outside,
                "hypergeom_p_at_least_observed": hyper_p,
                "boundary_tie": audit["boundary_tie"],
                "definition": note,
            })
        # Nested-budget prefix check happens after all budgets exist.
    table = pd.DataFrame(rows)
    for name, _role, _note in STRATEGY_CATALOG:
        for smaller, larger in zip(BUDGETS, BUDGETS[1:]):
            small_ids = selected_ids[(name, smaller)]
            large_ids = selected_ids[(name, larger)]
            if large_ids[:smaller] != small_ids:
                raise ValueError(f"{name} is not nested: B={smaller} is not a prefix of B={larger}")
    return table, positions


def full_pool_check(rankings: pd.DataFrame, k: int) -> dict:
    """Post hoc: whole dissent pool versus the same number of next ensemble ranks.

    Added after the pre-specified budgets were scored, so it is exploratory only.
    """
    outside = rankings[rankings["rank"] > k]
    pool = outside[outside["best_rank"] <= DISSENT_RANK_GATE]
    size = int(len(pool))
    expanded = outside.sort_values("rank").iloc[:size]
    pool_pos = set(pool.loc[pool["kev_by_nov4"] == 1, "cve_id"].astype(str))
    expanded_pos = set(expanded.loc[expanded["kev_by_nov4"] == 1, "cve_id"].astype(str))
    only_pool = sorted(pool_pos - expanded_pos)
    only_expanded = sorted(expanded_pos - pool_pos)
    by_fold = {}
    for fold in sorted(outside["fold"].unique()):
        p = pool[pool["fold"] == fold]
        by_fold[int(fold)] = {"pool_n": int(len(p)), "pool_positives": int(p["kev_by_nov4"].sum())}
    return {
        "status": "exploratory_post_hoc",
        "why_post_hoc": (
            "Noticed after the pre-specified budgets were scored: the upside-gap ordering placed most pool "
            "positives late. Comparing the whole pool with an equal-size ensemble extension was not planned."
        ),
        "pool_size": size,
        "pool_positives": len(pool_pos),
        "expanded_same_size_positives": len(expanded_pos),
        "expanded_same_size_last_rank": int(expanded["rank"].max()),
        "shared_positive_ids": sorted(pool_pos & expanded_pos),
        "only_pool_positive_ids": only_pool,
        "only_expanded_positive_ids": only_expanded,
        "mcnemar_exact_p": mcnemar_exact(len(only_pool), len(only_expanded)),
        "candidate_overlap": int(len(set(pool["cve_id"]) & set(expanded["cve_id"]))),
        "pool_by_fold": by_fold,
    }


def allocate_exact(sizes: list[int], budget: int) -> list[int]:
    """Split an integer budget across blocks, largest fractional remainder first."""
    total = sum(sizes)
    if total <= 0 or budget < 0:
        raise ValueError("Cannot allocate a block budget")
    raw = [budget * size / total for size in sizes]
    floors = [int(math.floor(value)) for value in raw]
    remainder = budget - sum(floors)
    order = sorted(range(len(sizes)), key=lambda i: (raw[i] - floors[i], sizes[i]), reverse=True)
    for index in order[:remainder]:
        floors[index] += 1
    if sum(floors) != budget:
        raise ValueError("Block allocation does not sum to the budget")
    return floors


def block_sensitivity(rankings: pd.DataFrame, k: int) -> dict:
    """Split each extra budget across time blocks, using only CVEs outside the global top K.

    This does not retune a threshold and does not replace the pooled top-517 result.
    It checks whether the primary comparison depends on early and late CVEs competing
    in one global extra-review list.
    """
    outside = rankings[rankings["rank"] > k]
    folds = sorted(int(value) for value in outside["fold"].unique())
    blocks = {fold: outside[outside["fold"] == fold] for fold in folds}
    sizes = [len(blocks[fold]) for fold in folds]
    results = []
    summary = []
    for budget in BUDGETS:
        allocation = dict(zip(folds, allocate_exact(sizes, budget)))
        expanded_hits = 0
        primary_hits = 0
        for fold in folds:
            rest = blocks[fold]
            b_fold = allocation[fold]
            fold_rows = {"fold": fold, "budget_B_global": budget, "block_extra": b_fold, "block_outside_n": int(len(rest))}
            for strategy in ("expanded_ensemble", PRIMARY_STRATEGY):
                picked = order_outside(rest, strategy).iloc[:b_fold]
                if len(picked) != b_fold:
                    raise ValueError("Block budget could not be filled")
                if (picked["rank"] <= k).any():
                    raise ValueError("Block watchlist included an original top-K CVE")
                hits = int(picked["kev_by_nov4"].sum())
                fold_rows[f"{strategy}_additional_positives"] = hits
                fold_rows[f"{strategy}_ids"] = ";".join(picked.loc[picked["kev_by_nov4"] == 1, "cve_id"].astype(str))
                if strategy == "expanded_ensemble":
                    expanded_hits += hits
                else:
                    primary_hits += hits
            results.append(fold_rows)
        summary.append({
            "additional_budget_target": budget,
            "actual_extra_cves_summed_across_blocks": int(sum(allocation.values())),
            "budgets_matched_between_strategies": True,
            "expanded_ensemble_additional_positives": expanded_hits,
            "primary_additional_positives": primary_hits,
            "note": (
                "Extra slots are split across publish-date blocks in proportion to how many "
                "of the block's CVEs sit outside the original global top 517. The original 32 "
                "are not recounted. This is a sensitivity, not a new selected model."
            ),
        })
    return {"by_fold": results, "summed": summary}


def signal_summary(rankings: pd.DataFrame, k: int) -> dict:
    frame = rankings.copy()
    frame["cohort"] = [
        cohort_name(rank, positive, k)
        for rank, positive in zip(frame["rank"], frame["kev_by_nov4"])
    ]
    frame["no_recorded_code"] = ~frame["any_recorded_code_flag"]

    def pack(subset: pd.DataFrame) -> dict:
        present = subset[subset["signal_record_present"]]
        epss_known = present[present["epss"].notna()]
        return {
            "n": int(len(subset)),
            "signal_row_present": int(subset["signal_record_present"].sum()),
            "signal_row_absent": int((~subset["signal_record_present"]).sum()),
            "any_recorded_code_flag": int(subset["any_recorded_code_flag"].sum()),
            "record_present_all_flags_zero": int((present["any_recorded_code_flag"] == False).sum()) if len(present) else 0,
            "share_any_recorded_code_flag": float(subset["any_recorded_code_flag"].mean()) if len(subset) else None,
            "epss_present": int(epss_known.shape[0]),
            "median_epss_when_present": None if epss_known.empty else float(epss_known["epss"].median()),
            "share_epss_exactly_zero_when_present": None if epss_known.empty else float((epss_known["epss"] == 0).mean()),
            "median_within_month_epss_percentile": (
                None if epss_known.empty else float(epss_known["epss_percentile_within_month"].median())
            ),
        }

    by_cohort = {name: pack(subset) for name, subset in frame.groupby("cohort")}
    by_cohort["top_k_all_labels"] = pack(frame[frame["rank"] <= k])
    by_cohort["outside_top_k_all_labels"] = pack(frame[frame["rank"] > k])
    by_cohort["full_population"] = pack(frame)
    era = {}
    for era_name, era_frame in frame.groupby("epss_era"):
        era[str(era_name)] = {
            cohort: pack(subset)
            for cohort, subset in era_frame.groupby("cohort")
        }
        era[str(era_name)]["full_population"] = pack(era_frame)
    frame["signal_state"] = np.select(
        [~frame["signal_record_present"], frame["any_recorded_code_flag"]],
        ["monthly_row_absent", "recorded_code_flag"],
        default="row_present_all_flags_zero",
    )
    disagreement_by_signal = {}
    outside_frame = frame[frame["rank"] > k]
    for state, state_frame in frame.groupby("signal_state"):
        entry = {}
        for cohort, subset in state_frame.groupby("cohort"):
            if cohort == "not_confirmed_by_cutoff":
                subset = subset[subset["rank"] > k]
                cohort = "outside_top_k_not_confirmed"
            entry[cohort] = {
                "n": int(len(subset)),
                "median_percentile_range": None if subset.empty else float(subset["pct_range"].median()),
                "median_upside_gap": None if subset.empty else float(subset["upside"].median()),
                "n_best_rank_inside_top_k": int((subset["best_rank"] <= k).sum()),
            }
        pool = outside_frame[(outside_frame["signal_state"] == state) & (outside_frame["best_rank"] <= k)]
        everything = outside_frame[outside_frame["signal_state"] == state]
        entry["outside_top_k_dissent_pool"] = {
            "n": int(len(pool)),
            "positives": int(pool["kev_by_nov4"].sum()),
            "positive_rate": None if pool.empty else float(pool["kev_by_nov4"].mean()),
        }
        entry["outside_top_k_all"] = {
            "n": int(len(everything)),
            "positives": int(everything["kev_by_nov4"].sum()),
            "positive_rate": None if everything.empty else float(everything["kev_by_nov4"].mean()),
        }
        disagreement_by_signal[str(state)] = entry
    missed = frame[frame["cohort"] == "missed_positive"]
    sparse_misses = missed[~missed["any_recorded_code_flag"]]
    flagged_misses = missed[missed["any_recorded_code_flag"]]
    return {
        "epss_v4_assumption": (
            "EPSS v4 replaced v3 on 17 March 2025. Snapshot months through 2025-02 are treated as v3. "
            "Snapshot month 2025-03 and later are treated as v4 because the monthly file is a month-end "
            "view and the cutover was 17 March. Raw EPSS is not compared across this boundary. "
            "Within-month percentiles are the EPSS summary used in every pooled comparison."
        ),
        "by_cohort": by_cohort,
        "by_era_and_cohort": era,
        "disagreement_by_signal_state": {
            "note": (
                "Exploratory, not a pre-specified test. Signal states are mutually exclusive: monthly row absent "
                "(flags and EPSS unknown), row present with all four code flags zero, or at least one recorded flag. "
                "The dissent pool is CVEs outside the ensemble top K that at least one model ranked inside its own top K. "
                "Positive rates are counts against the November 4 label, not exploitation probabilities."
            ),
            "states": disagreement_by_signal,
        },
        "missed_without_recorded_code": {
            "n": int(len(sparse_misses)),
            "n_signal_row_absent": int((~sparse_misses["signal_record_present"]).sum()),
            "n_best_rank_inside_top_k": int((sparse_misses["best_rank"] <= k).sum()),
            "median_best_rank": float(sparse_misses["best_rank"].median()) if len(sparse_misses) else None,
            "best_model_counts": sparse_misses["best_model"].map(lambda value: DISPLAY_OF.get(value, value)).value_counts().to_dict(),
        },
        "missed_with_recorded_code": {
            "n": int(len(flagged_misses)),
            "cve_ids": flagged_misses["cve_id"].astype(str).tolist(),
            "n_best_rank_inside_top_k": int((flagged_misses["best_rank"] <= k).sum()),
            "median_ensemble_rank": float(flagged_misses["rank"].median()) if len(flagged_misses) else None,
        },
    }


def compact_misses(rankings: pd.DataFrame, positions: dict[str, pd.Series], neighbors: pd.DataFrame, k: int) -> list[dict]:
    missed = rankings[(rankings["kev_by_nov4"] == 1) & (rankings["rank"] > k)].copy()
    neighbor_map = neighbors.set_index("cve_id")
    records = []
    for row in missed.itertuples(index=False):
        cve_id = str(row.cve_id)
        records.append({
            "cve_id": cve_id,
            "fold": int(row.fold),
            "kev_group": None if pd.isna(row.kev_group) else str(row.kev_group),
            "published": pd.Timestamp(row.published).strftime("%Y-%m-%d"),
            "snapshot_month": str(row.snapshot_month),
            "epss_era": str(row.epss_era),
            "kev_date_added": None if pd.isna(row.kev_date_added) else pd.Timestamp(row.kev_date_added).strftime("%Y-%m-%d"),
            "ensemble_rank": int(row.rank),
            "ensemble_score": float(row.ensemble_score),
            "places_outside_cutoff": int(row.rank - k),
            "individual_ranks": {
                DISPLAY_OF[column]: {
                    "column": column,
                    "rank": int(getattr(row, f"rank_{column}")),
                    "percentile_rank": float(getattr(row, f"pct_{column}")),
                    "inside_original_top_k": bool(getattr(row, f"rank_{column}") <= k),
                }
                for column in VOTER_COLUMNS
            },
            "best_model": DISPLAY_OF.get(row.best_model, row.best_model),
            "best_rank": int(row.best_rank),
            "percentile_range": float(row.pct_range),
            "percentile_std": float(row.pct_std),
            "upside_gap": float(row.upside),
            "review_position_if_extra_slots_follow": {
                name: int(series.loc[cve_id]) for name, series in positions.items() if cve_id in series.index
            },
            "signal_record_present": bool(row.signal_record_present),
            "epss": None if pd.isna(row.epss) else float(row.epss),
            "epss_percentile_within_month": None if pd.isna(row.epss_percentile_within_month) else float(row.epss_percentile_within_month),
            "any_recorded_code_flag": bool(row.any_recorded_code_flag),
            "code_flags": {
                flag: None if pd.isna(getattr(row, f"flag_{flag}")) else int(getattr(row, f"flag_{flag}"))
                for flag in SIGNAL_FLAGS
            },
            "neighbor_range_share_lower": float(neighbor_map.loc[cve_id, "range_share_neighbors_lower"]),
            "neighbor_upside_share_lower": float(neighbor_map.loc[cve_id, "upside_share_neighbors_lower"]),
        })
    return records


def build_case_studies(cases: dict[str, dict], compact: list[dict], k: int) -> dict:
    by_id = {row["cve_id"]: row for row in compact}
    frame = pd.DataFrame(compact)
    frame["improvement_best_voter"] = (
        frame["review_position_if_extra_slots_follow"].map(
            lambda item: item["expanded_ensemble"] - item.get(PRIMARY_STRATEGY, 10**9)
        )
    )
    selected: list[tuple[str, str]] = []

    def add(cve_id: str, reason: str) -> None:
        if cve_id not in by_id:
            return
        selected.append((cve_id, reason))

    nearest = frame.sort_values(["ensemble_rank", "cve_id"]).iloc[0]
    add(str(nearest["cve_id"]), "Closest missed positive to the original top-517 cutoff.")
    frame["pos_best"] = frame["review_position_if_extra_slots_follow"].map(lambda item: item.get(PRIMARY_STRATEGY, 10**9))
    frame["pos_expanded"] = frame["review_position_if_extra_slots_follow"].map(lambda item: item["expanded_ensemble"])
    best_case = frame.sort_values(["pos_best", "cve_id"]).iloc[0]
    if int(best_case["pos_best"]) < 10**9:
        add(str(best_case["cve_id"]), "Missed positive that the pre-specified single-model-dissent rule would reach first.")
    improved = frame.sort_values(["improvement_best_voter", "cve_id"], ascending=[False, True]).iloc[0]
    if int(improved["improvement_best_voter"]) > 0:
        add(
            str(improved["cve_id"]),
            "Missed positive whose extra-review position improves the most under the single-model-dissent rule instead of the ensemble order.",
        )
    consensus = frame.sort_values(["best_rank", "ensemble_rank"], ascending=[False, False]).iloc[0]
    add(str(consensus["cve_id"]), "Consensus failure: even the most favorable individual model ranked this missed positive poorly.")
    deepest = frame.sort_values(["ensemble_rank", "cve_id"], ascending=[False, True]).iloc[0]
    add(str(deepest["cve_id"]), "Deepest missed positive by ensemble rank.")
    if MOTIVATING_CVE in by_id:
        add(MOTIVATING_CVE, "Motivating example supplied with the research question: one model is much closer to the cutoff than the ensemble.")
    sparse = frame[~frame["any_recorded_code_flag"]].sort_values(["best_rank", "cve_id"])
    if len(sparse):
        add(str(sparse.iloc[0]["cve_id"]), "Missed positive with no recorded public exploit-code flag and the best individual rank in that sparse-signal group.")
    flagged = frame[frame["any_recorded_code_flag"]].sort_values(["ensemble_rank", "cve_id"])
    if len(flagged):
        add(str(flagged.iloc[0]["cve_id"]), "Missed positive that did have a recorded public exploit-code flag and was still outside the top 517.")

    grouped: dict[str, list[str]] = {}
    for cve_id, reason in selected:
        grouped.setdefault(cve_id, [])
        if reason not in grouped[cve_id]:
            grouped[cve_id].append(reason)
    # Keep the deterministic first-seen order and cap the gallery.
    ordered_ids = []
    for cve_id, _reason in selected:
        if cve_id not in ordered_ids:
            ordered_ids.append(cve_id)
    ordered_ids = ordered_ids[:8]

    demo_ids = []
    for cve_id in (
        str(improved["cve_id"]) if int(improved["improvement_best_voter"]) > 0 else str(best_case["cve_id"]),
        str(nearest["cve_id"]),
        str(consensus["cve_id"]),
    ):
        if cve_id not in demo_ids:
            demo_ids.append(cve_id)
    if len(demo_ids) < 3 and MOTIVATING_CVE in by_id and MOTIVATING_CVE not in demo_ids:
        demo_ids.append(MOTIVATING_CVE)
    demo_ids = demo_ids[:3]

    case_payloads = []
    for cve_id in ordered_ids:
        stats_row = by_id[cve_id]
        case = cases[cve_id]
        outcome = case["confirmed_outcome"]
        signals = case["snapshot_signals"]
        nvd = case["current_nvd_context"]
        positions_row = stats_row["review_position_if_extra_slots_follow"]
        budgets = {}
        for budget in BUDGETS:
            budgets[str(budget)] = {
                strategy: bool(positions_row.get(strategy, 10**9) <= budget)
                for strategy in (
                    "expanded_ensemble",
                    "single_model_dissent",
                    "best_voter_percentile",
                    "upside_gap",
                    "percentile_range",
                    "individual_logistic_regression",
                    "individual_granite_lora",
                    "individual_lightgbm",
                )
                if strategy in positions_row
            }
        notes = [
            (
                f"Ensemble rank {stats_row['ensemble_rank']:,} is {stats_row['places_outside_cutoff']:,} "
                f"places outside the top {k}. Following the existing order would review it after "
                f"{positions_row['expanded_ensemble'] - 1:,} other extra CVEs."
            ),
            (
                "Individual competition ranks, with rank 1 best: "
                + "; ".join(
                    f"{name} {info['rank']:,} (percentile {info['percentile_rank']:.4f}"
                    f"{', inside top 517' if info['inside_original_top_k'] else ''})"
                    for name, info in stats_row["individual_ranks"].items()
                )
                + f". Percentile range {stats_row['percentile_range']:.4f}; "
                f"upside of the leading model over the mean {stats_row['upside_gap']:.4f}."
            ),
            (
                (
                    f"Single-model-dissent extra-review position {positions_row[PRIMARY_STRATEGY]:,}. "
                    if PRIMARY_STRATEGY in positions_row
                    else f"Not eligible for the single-model-dissent pool: no model ranked it inside its own top {DISSENT_RANK_GATE}. "
                )
                + f"Best-voter extra-review position {positions_row.get('best_voter_percentile'):,}. "
                f"Upside-gap position {positions_row.get('upside_gap'):,}. "
                f"Percentile-range position {positions_row.get('percentile_range'):,}."
            ),
        ]
        if signals["record_present"]:
            flag_text = ", ".join(f"{flag}={signals['code_flags'][flag]}" for flag in SIGNAL_FLAGS)
            epss_text = "missing" if signals["epss"] is None else f"{signals['epss']}"
            pct_text = (
                "missing" if signals["epss_percentile_within_observed_month"] is None
                else str(signals["epss_percentile_within_observed_month"])
            )
            notes.append(
                f"Snapshot {signals['snapshot_month']} ({stats_row['epss_era']}) has a monthly signal row. "
                f"EPSS {epss_text}; within-month EPSS percentile {pct_text}. Code flags: {flag_text}. "
                "A zero flag means that source had no recorded code, not that exploitation was impossible."
            )
        else:
            notes.append(
                f"Snapshot {stats_row['snapshot_month']} ({stats_row['epss_era']}) has no monthly signal row. "
                "EPSS and the code flags are missing. The benchmark convention stores flags as zero in that case; "
                "that default is not evidence that exploit code was absent."
            )
        notes.append(
            f"Historical KEV addition date {outcome['kev_added']}, {outcome['days_published_to_kev']} days after publication. "
            "That date is an outcome. It was not a model input."
        )
        if stats_row["kev_group"]:
            notes.append(
                f"This CVE is in hand-built related group '{stats_row['kev_group']}'. "
                "Hits in the same group are not independent events."
            )
        case_payloads.append({
            "cve_id": cve_id,
            "demonstration_priority": demo_ids.index(cve_id) + 1 if cve_id in demo_ids else None,
            "tags": grouped[cve_id],
            "ensemble_rank": stats_row["ensemble_rank"],
            "top_k": k,
            "individual_model_ranks": stats_row["individual_ranks"],
            "disagreement": {
                "best_model": stats_row["best_model"],
                "best_rank": stats_row["best_rank"],
                "percentile_range": stats_row["percentile_range"],
                "percentile_std_ddof0": stats_row["percentile_std"],
                "upside_gap": stats_row["upside_gap"],
                "neighbor_share_with_smaller_range": stats_row["neighbor_range_share_lower"],
                "neighbor_share_with_smaller_upside": stats_row["neighbor_upside_share_lower"],
            },
            "extra_review_positions": positions_row,
            "included_under_additional_budget": budgets,
            "snapshot_signals": {
                "snapshot_month": signals["snapshot_month"],
                "epss_era": stats_row["epss_era"],
                "record_present": signals["record_present"],
                "epss": signals["epss"],
                "epss_percentile_within_observed_month": signals["epss_percentile_within_observed_month"],
                "code_flags": signals["code_flags"],
                "any_recorded_code_flag": signals["any_recorded_code_flag"],
                "missingness_note": signals["interpretation_note"],
            },
            "retrospective_nvd_context": nvd,
            "retrospective_nvd_warning": case["current_nvd_context_warning"],
            "confirmed_historical_outcome": {
                "published": outcome["published"],
                "kev_added": outcome["kev_added"],
                "days_published_to_kev": outcome["days_published_to_kev"],
                "catalog": outcome.get("kev_catalog_context"),
                "label": "Historical KEV addition under the November 4 publication-year cutoff. Not a 2026 outcome.",
            },
            "fold": stats_row["fold"],
            "kev_group": stats_row["kev_group"],
            "evidence_backed_notes": notes,
            "uncertainties": [
                "A rank gap is an observed difference among three scores. It does not identify which feature caused a model to score the CVE higher.",
                "Current NIST description and CVSS fields may have been edited after the snapshot.",
                "Absence of a monthly signal row, or a zero exploit-code flag, is not proof that no exploit existed.",
                "Not being in KEV by the cutoff is not claimed for the comparison CVEs; this case is one of the 43 confirmed misses.",
                "Whether a secondary rule would have queued this CVE is a historical recount, not a prospective trial.",
            ],
        })
    return {
        "metadata": {
            "selection": "Deterministic rules fixed in the script. Cases may carry more than one tag.",
            "strongest_demonstration_ids": demo_ids,
            "demonstration_rule": (
                "The three demonstrations are the miss the single-model-dissent rule moves up the most "
                "(or the miss it reaches first, if it moves none up), the miss closest to the ensemble cutoff, "
                "and the consensus failure with the worst best-individual rank."
            ),
        },
        "cases": case_payloads,
    }


def decision_from_table(table: pd.DataFrame) -> dict:
    primary = table[table["strategy"] == PRIMARY_STRATEGY].sort_values("additional_budget_B")
    expanded = table[table["strategy"] == "expanded_ensemble"].sort_values("additional_budget_B")
    merged = primary.merge(
        expanded,
        on="additional_budget_B",
        suffixes=("_primary", "_expanded"),
    )
    deltas = {
        int(row.additional_budget_B): int(row.additional_positives_primary - row.additional_positives_expanded)
        for row in merged.itertuples(index=False)
    }
    values = list(deltas.values())
    if max(values) <= 0:
        code = "does_not_beat_ensemble"
    elif min(values) >= 1 and deltas[500] >= 2:
        code = "limited_support"
    else:
        code = "fragile_or_inconsistent"
    baseline_deltas = {}
    for name in ("best_voter_percentile", "upside_gap", "percentile_range"):
        other = table[table["strategy"] == name].set_index("additional_budget_B")["additional_positives"]
        base = expanded.set_index("additional_budget_B")["additional_positives"]
        baseline_deltas[name] = {int(b): int(other[b] - base[b]) for b in base.index}
    return {
        "primary_strategy": PRIMARY_STRATEGY,
        "deltas_primary_minus_expanded": deltas,
        "code": code,
        "other_rules_minus_expanded_not_confirmatory": baseline_deltas,
    }


def markdown_report(
    rankings: pd.DataFrame,
    disagreement: dict,
    table: pd.DataFrame,
    signals: dict,
    studies: dict,
    sensitivity: dict,
    decision: dict,
    k: int,
    pool_check: dict,
) -> str:
    n = len(rankings)
    missed = disagreement["missed_frame"]
    neighbor = disagreement["neighbor_test"]
    cohorts = disagreement["cohorts"]
    caught_flags = signals["by_cohort"]["caught_positive"]
    missed_flags = signals["by_cohort"]["missed_positive"]
    population = signals["by_cohort"]["full_population"]
    top = signals["by_cohort"]["top_k_all_labels"]
    outside = signals["by_cohort"]["outside_top_k_all_labels"]

    def strategy_line(name: str, budget: int) -> pd.Series:
        hit = table[(table["strategy"] == name) & (table["additional_budget_B"] == budget)]
        return hit.iloc[0]

    lines = []
    lines.append("# Blind-spot investigation: does model disagreement recover missed KEV cases?")
    lines.append("")
    lines.append("Historical out-of-fold evaluation for 2024–2025. These are not 2026 challenge outcomes.")
    lines.append("")
    lines.append("## 1. Research questions and methodology")
    lines.append("")
    lines.append(
        "The team's ensemble is the mean of three percentile ranks: logistic regression "
        "(`lr_full_llmx_noemb`), Granite LoRA (`lora_control`), and LightGBM (`lgbm_luis_only`). "
        f"On {fmt_int(n)} historical CVEs it places {fmt_int(k)} vulnerabilities in the review set and "
        "recovers 32 of 75 KEV additions that fall under the November 4 publication-year cutoff. "
        "The other 43 confirmed additions are the historical misses. A miss means outside that review "
        "budget. It does not mean the model assigned the vulnerability zero risk."
    )
    lines.append("")
    lines.append("Two questions are separated on purpose.")
    lines.append("")
    lines.append(
        "1. **Distinctiveness.** Among all 75,148 CVEs, is disagreement between the three models "
        "unusual for the 43 misses, once their ensemble rank is taken into account?"
    )
    lines.append(
        "2. **Usefulness.** If analysts review B extra CVEs, does a pre-specified disagreement rule "
        "recover more of those 43 than simply continuing down the ensemble list?"
    )
    lines.append("")
    lines.append("Percentile ranks are pandas average ranks divided by 75,148. Higher means the model scored the CVE higher. Raw scores are never compared, because the three models do not share a scale. For a CVE with percentile ranks p1, p2, and p3:")
    lines.append("")
    lines.append("- Ensemble score = (p1 + p2 + p3) / 3. Recomputed values match the stored ensemble score.")
    lines.append("- Percentile range = max(p) − min(p).")
    lines.append("- Percentile standard deviation uses the three ranks as the whole set (divisor 3, not 2).")
    lines.append("- Upside gap = max(p) − mean(p). This is the amount by which the leading model sits above the ensemble mean.")
    lines.append("- Individual competition ranks use minimum-rank ties. Rank 1 is the highest score. The stored ensemble order, including its CVE-ID tie break, is left unchanged.")
    lines.append("")
    lines.append(
        f"The confirmatory watchlist rule is `{PRIMARY_STRATEGY}`: after the original top {k}, the eligible pool is "
        f"every remaining CVE that at least one model ranked inside its own top {DISSENT_RANK_GATE} (a CVE one model "
        "would have flagged on its own but the average did not). The B extra slots go to the pool members with the "
        "largest upside gap, meaning the leading model sits furthest above the three-model mean. "
        "The rule, the gate, and the budgets were fixed before any watchlist hit count was computed. "
        "Ties break toward the better ensemble rank, then toward CVE ID, so tie breaks do not favor the new rule. "
        "Budgets B = 50, 100, 250, and 500 were fixed in advance. Expanding the ensemble is the main baseline. "
        "The best single-model percentile (`best_voter_percentile`) and each model alone are additional baselines. "
        "The ungated upside gap, the raw percentile range, and two sparse-signal restrictions are secondary checks. "
        "Rules were not searched against the labels. Because every larger budget "
        "contains the smaller one, the four budgets are nested looks at one ordering, not four independent experiments."
    )
    lines.append("")
    lines.append(
        "The same out-of-fold scores and the same 75 labels are used. This recount does not create a new "
        "holdout. A chronological block split is reported only as a sensitivity: Chen's five purged folds are "
        "consecutive publish-date blocks, and the extra budget is split across them in proportion to how many of "
        "each block's CVEs sit outside the global top 517. Thresholds are not refit. "
        "Related KEV groups from the purged CV design are reported when a selected case belongs to one; "
        "hits were not dropped, because the published 32/75 count does not drop them either."
    )
    lines.append("")
    lines.append("## 2. Validated descriptive findings")
    lines.append("")
    lines.append(
        f"Integrity checks passed. The file contains {fmt_int(n)} unique CVEs, 75 historical positives, "
        f"32 inside rank {k}, and 43 outside it. Stored ensemble scores match the mean of the three "
        "recomputed percentile ranks. Individual ranks for all 75 positives match `investigation_cases.json`. "
        "Snapshot EPSS values, within-month percentiles, and exploit-code flags for those 75 match a fresh "
        "join to the monthly signal files, including the preliminary medians 55.17 and 13.28 and the 27 versus 4 flag split."
    )
    lines.append("")
    caught_range = cohorts["caught_positive"]["percentile_range"]["p50"]
    missed_range = cohorts["missed_positive"]["percentile_range"]["p50"]
    other_range = cohorts["not_confirmed_by_cutoff"]["percentile_range"]["p50"]
    outside_range = cohorts["outside_top_k_not_confirmed"]["percentile_range"]["p50"]
    lines.append(
        f"Median percentile range is {caught_range:.3f} for the 32 caught positives, {missed_range:.3f} for the "
        f"43 misses, {outside_range:.3f} for unconfirmed CVEs outside the top {k}, and {other_range:.3f} for all "
        "CVEs that were not confirmed in KEV by the cutoff. "
        f"A range of at least {MATERIAL_RANGE:.2f} occurs for "
        f"{cohorts['missed_positive']['share_range_at_least_0_10']:.1%} of misses and "
        f"{cohorts['outside_top_k_not_confirmed']['share_range_at_least_0_10']:.1%} of unconfirmed CVEs outside the cutoff. "
        f"A range of at least {STRONG_RANGE:.2f} occurs for "
        f"{cohorts['missed_positive']['share_range_at_least_0_25']:.1%} of misses and "
        f"{cohorts['outside_top_k_not_confirmed']['share_range_at_least_0_25']:.1%} of that unconfirmed group. "
        "Disagreement at those pre-set sizes is common, not a rare signature of the misses."
    )
    lines.append("")
    upside_test = neighbor["upside_gap"]
    range_test = neighbor["percentile_range"]
    lines.append(
        f"The rank-localized check asks whether a miss has a larger upside gap than CVEs within "
        f"{NEIGHBOR_HALF_WIDTH} ensemble places of it, staying outside the original top {k}. "
        f"The median miss has a larger upside than {upside_test['median_share']:.1%} of those neighbors "
        f"(Wilcoxon signed-rank p = {fmt_p(upside_test['p_value'])} against a 50% share). "
        f"For percentile range, the median share is {range_test['median_share']:.1%} "
        f"(p = {fmt_p(range_test['p_value'])}). "
        "A share of one half would mean the misses look ordinary for where the ensemble already placed them. "
        + (
            "The observed shares are above one half: compared with CVEs the ensemble ranked about the same, "
            "misses more often had one model sitting well above the other two. This is a real, rank-adjusted "
            "association in this file. Whether it is useful depends on how many non-confirmed CVEs share it, "
            "which the watchlist test below measures directly."
            if upside_test["median_share"] > 0.6 and upside_test["p_value"] < 0.05
            else "The observed shares do not show a clear rank-adjusted disagreement signature."
        )
    )
    lines.append("")
    unstrat = disagreement["unstratified_tests"]["percentile_range_missed_vs_outside_unconfirmed"]
    lines.append(
        f"Without that rank adjustment the picture reverses: misses have a smaller median range "
        f"({missed_range:.3f}) than unconfirmed CVEs outside the cutoff ({outside_range:.3f}; Mann-Whitney "
        f"p = {fmt_p(unstrat['p_value'])}), because misses sit nearer the top of the list, where the three models "
        "agree more. Both statements are true; only the rank-adjusted one bears on the blind-spot question."
    )
    lines.append("")
    rho = disagreement["outside_rank_correlations"]["spearman_rank_vs_percentile_range"]["rho"]
    rho_up = disagreement["outside_rank_correlations"]["spearman_rank_vs_upside"]["rho"]
    lines.append(
        f"Outside the top {k}, Spearman correlation between ensemble rank and percentile range is {rho:.3f}, "
        f"and between ensemble rank and upside gap is {rho_up:.3f}. Both are close to zero: large disagreement "
        "is spread across the whole list rather than concentrated near the cutoff. That is why a global "
        "'largest disagreement' queue mostly selects CVEs far from the cutoff, where positives are rarest."
    )
    lines.append("")
    n_inside = disagreement["n_missed_best_rank_inside_top_k"]
    n_neg_inside = disagreement["n_outside_unconfirmed_best_rank_inside_top_k"]
    lines.append(
        f"{n_inside} of the 43 misses have at least one individual model inside the original top {k}. "
        f"{fmt_int(n_neg_inside)} unconfirmed CVEs outside the ensemble top {k} also have at least one model inside that top {k}. "
        "One model clearing the cutoff is therefore a property of hundreds or thousands of non-confirmed CVEs, "
        "not a short list of misses. "
        f"{disagreement['n_missed_dilution_at_least_1000']} misses are at least 1,000 places better in their "
        "best individual rank than in the ensemble. That gap is an observed reordering. It is not, by itself, "
        "a statement about why the models differed."
    )
    lines.append("")
    winner_bits = ", ".join(
        f"{name} {count}" for name, count in sorted(disagreement["missed_best_model_counts_excluding_ties"].items())
    )
    lines.append(
        f"Among misses with a unique leading model, the counts are {winner_bits}. "
        f"Exact leading-model ties: {disagreement['missed_best_model_ties']}. "
        "Leading the other two models on a miss does not mean that model found a cause the others missed."
    )
    lines.append("")
    lines.append("Single-model top-517 counts, using each model's own ranking of the same scores:")
    lines.append("")
    lines.append("| Model | Positives inside its own top 517 | Of which are ensemble misses |")
    lines.append("|---|---:|---:|")
    for _column, display in VOTERS:
        info = disagreement["single_model_top_k"][display]
        lines.append(
            f"| {display} | {info['positives_inside_own_top_k']} | {info['misses_with_this_model_inside_top_k']} |"
        )
    lines.append("")
    lines.append("Misses by distance from the cutoff:")
    lines.append("")
    lines.append("| Band | Misses | Best model inside top 517 | Best rank ≤ 1,034 | Median best rank | Median range |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in sorted(disagreement["miss_bands"], key=lambda item: item["miss_band"]):
        lines.append(
            f"| {row['miss_band']} | {row['n']} | {row['n_best_rank_inside_top_k']} | "
            f"{row['n_best_rank_at_most_2k']} | {row['median_best_rank']:.0f} | {row['median_percentile_range']:.3f} |"
        )
    lines.append("")
    lines.append(
        f"Recorded public exploit-code flags are present for {caught_flags['any_recorded_code_flag']} of 32 caught "
        f"positives and {missed_flags['any_recorded_code_flag']} of 43 misses. "
        f"In the full historical population the same flags are present for {population['any_recorded_code_flag']:,} of "
        f"{population['n']:,} CVEs ({population['share_any_recorded_code_flag']:.1%}). "
        f"Inside the ensemble top {k} the flag rate is {top['share_any_recorded_code_flag']:.1%} "
        f"({top['any_recorded_code_flag']:,} of {top['n']:,}). "
        f"Outside it, the rate is {outside['share_any_recorded_code_flag']:.1%}. "
        "The caught-versus-missed flag gap is real in this file. It is also what a model that uses those flags "
        "as inputs is expected to produce: flagged CVEs are pushed into the review set, and the residual misses "
        "are mostly unflagged. Relative to the whole population, lacking a recorded flag is the common state, "
        "not a rare pattern that identifies the 43."
    )
    lines.append("")
    lines.append(
        f"Monthly signal rows are missing for {missed_flags['signal_row_absent']} misses and for "
        f"{population['signal_row_absent']:,} historical CVEs overall. "
        f"Among rows with a numeric EPSS, the median raw score is {caught_flags['median_epss_when_present']} "
        f"for caught positives and {missed_flags['median_epss_when_present']} for misses; "
        f"the share of exact zeros is {caught_flags['share_epss_exactly_zero_when_present']:.1%} and "
        f"{missed_flags['share_epss_exactly_zero_when_present']:.1%}. "
        "Raw EPSS is reported only for orientation; the preliminary comparison used the within-month percentile, "
        "which is reproduced (55.17 caught, 13.28 missed). "
        f"{signals['epss_v4_assumption']}"
    )
    lines.append("")
    sparse = signals["missed_without_recorded_code"]
    lines.append(
        f"Of the {sparse['n']} misses with no recorded code flag, {sparse['n_best_rank_inside_top_k']} "
        f"still have an individual model inside the top {k}. Their median best individual rank is "
        f"{sparse['median_best_rank']:.0f}. "
        f"The {signals['missed_with_recorded_code']['n']} misses that do have a recorded flag are "
        + ", ".join(signals["missed_with_recorded_code"]["cve_ids"])
        + ". Flag presence did not put them inside the ensemble budget. That is an observed exception, not a cause."
    )
    lines.append("")
    states = signals["disagreement_by_signal_state"]["states"]
    lines.append(
        "Disagreement by recorded-signal state (exploratory). States are mutually exclusive. The dissent pool is "
        f"CVEs outside the ensemble top {k} that at least one model ranked inside its own top {k}."
    )
    lines.append("")
    lines.append("| Signal state | Misses | Median range, misses | Median range, unconfirmed outside | Outside top K: positives / CVEs | Dissent pool: positives / CVEs |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for state in ("recorded_code_flag", "row_present_all_flags_zero", "monthly_row_absent"):
        entry = states.get(state, {})
        miss = entry.get("missed_positive", {"n": 0, "median_percentile_range": None})
        neg = entry.get("outside_top_k_not_confirmed", {"median_percentile_range": None})
        allx = entry["outside_top_k_all"]
        pool = entry["outside_top_k_dissent_pool"]
        mm = "n/a" if miss["median_percentile_range"] is None else f"{miss['median_percentile_range']:.3f}"
        nm = "n/a" if neg["median_percentile_range"] is None else f"{neg['median_percentile_range']:.3f}"
        lines.append(
            f"| {state} | {miss['n']} | {mm} | {nm} | {allx['positives']} / {fmt_int(allx['n'])} | "
            f"{pool['positives']} / {fmt_int(pool['n'])} |"
        )
    lines.append("")
    expanded_500 = strategy_line("expanded_ensemble", 500)
    rate_bits = "; ".join(
        f"{state}: pool {states[state]['outside_top_k_dissent_pool']['positive_rate']:.2%} versus stratum "
        f"{states[state]['outside_top_k_all']['positive_rate']:.2%}"
        for state in ("recorded_code_flag", "row_present_all_flags_zero", "monthly_row_absent")
        if states.get(state, {}).get("outside_top_k_dissent_pool", {}).get("positive_rate") is not None
    )
    lines.append(
        f"Positive rates: {rate_bits}. For comparison, the next 500 ensemble ranks have "
        f"{expanded_500.precision_additional:.2%}. The dissent pool is richer than its stratum in each state, "
        "but it is not clearly richer than simply continuing down the ensemble list; any state that edges above "
        "that rate does so on one or a few positives. Dissent mostly tracks being near the top of the list, "
        "which the ensemble already encodes. Sparse recorded signals are the normal state: "
        f"{1 - outside['share_any_recorded_code_flag']:.1%} of CVEs outside the top {k} have no recorded flag, "
        "so 'no flag' alone cannot single out the misses."
    )
    lines.append("")
    lines.append("## 3. Watchlist evaluation")
    lines.append("")
    lines.append(
        f"Every strategy below starts from the same 32 positives in the top {k} and then reviews exactly B more CVEs. "
        f"Those B CVEs are drawn only from ranks {k + 1} and worse. "
        f"A random draw of B CVEs from the {fmt_int(n - k)} outside the cutoff would be expected to contain "
        f"about {43 * 50 / (n - k):.2f}, {43 * 100 / (n - k):.2f}, {43 * 250 / (n - k):.2f}, and "
        f"{43 * 500 / (n - k):.2f} positives at B = 50, 100, 250, and 500. "
        "An unconfirmed CVE is an observed non-positive at the cutoff, not proof it was never exploited. "
        "Precision below is the share of the extra B that are historical positives. Wilson intervals are wide because the counts are small."
    )
    lines.append("")
    lines.append("| Strategy | Role | B | Extra positives | Total positives | Precision of extra set | Recall at 517+B | Only this rule vs expanded | Only expanded | McNemar p | Extra-set overlap with expanded |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in table.sort_values(["additional_budget_B", "role", "strategy"]).itertuples(index=False):
        lines.append(
            f"| {row.strategy} | {row.role} | {row.additional_budget_B} | {row.additional_positives} | "
            f"{row.total_positives_recovered} | {row.precision_additional:.3f} "
            f"({row.precision_wilson_low:.3f}–{row.precision_wilson_high:.3f}) | "
            f"{row.recall_at_total_budget:.3f} | {row.mcnemar_only_this} | {row.mcnemar_only_expanded} | "
            f"{fmt_p(row.mcnemar_exact_p_vs_expanded)} | {row.candidate_set_overlap_with_expanded} |"
        )
    lines.append("")
    lines.append("Positive CVE IDs recovered beyond the original 32, in review order:")
    lines.append("")
    for budget in BUDGETS:
        for name in ("expanded_ensemble", PRIMARY_STRATEGY, "best_voter_percentile", "upside_gap", "percentile_range"):
            row = strategy_line(name, budget)
            found = row.additional_positive_ids or "(none)"
            lines.append(f"- B={budget}, `{name}`: {found}")
    lines.append("")
    lines.append(
        "Block sensitivity, not the headline metric: the same extra budget is split across the five "
        "publish-date blocks, and each block may spend its share only on CVEs outside the original "
        "global top 517. The original 32 stay put. Thresholds are not refit."
    )
    lines.append("")
    lines.append("| Target B | Actual extra CVEs across blocks | Expanded extra positives | Single-model-dissent extra positives |")
    lines.append("|---:|---:|---:|---:|")
    for row in sensitivity["summed"]:
        lines.append(
            f"| {row['additional_budget_target']} | {row['actual_extra_cves_summed_across_blocks']} | "
            f"{row['expanded_ensemble_additional_positives']} | {row['primary_additional_positives']} |"
        )
    lines.append("")
    lines.append("## 4. Does disagreement add value?")
    lines.append("")
    delta_text = ", ".join(
        f"B={budget}: {delta:+d}" for budget, delta in sorted(decision["deltas_primary_minus_expanded"].items())
    )
    lines.append(
        f"Primary rule minus expanded ensemble, in additional positives: {delta_text}. "
        "Positive means the single-model-dissent rule recovered more historical misses at that same extra budget."
    )
    lines.append("")
    code = decision["code"]
    if code == "does_not_beat_ensemble":
        lines.append(
            "The pre-specified disagreement watchlist does not beat extending the ensemble. "
            "At every tested budget it recovers the same number of additional positives or fewer. "
            "Individual ranks still move some misses, but they do not move enough misses ahead of the "
            "unconfirmed CVEs that also received a high score from one model. "
            "The raw percentile-range queue is a negative control: large spread is common deep in the list, "
            "and selecting on spread alone is not a review strategy supported by these counts."
        )
    elif code == "limited_support":
        lines.append(
            "The primary rule recovers more misses than the expanded ensemble at every tested budget, "
            "including at least two additional positives at B = 500. That is a real difference on this "
            "historical file. It is still one retrospective pass over 43 misses, the budgets are nested, "
            "and the absolute precision of the extra slots remains low. Treat it as limited support for "
            "an optional queue, with the measured precision shown beside it, not as a confirmed new detector."
        )
    else:
        lines.append(
            "Any gain for the primary rule is small or inconsistent across budgets. A difference of one "
            "or two nested positives, or a gain at one budget that disappears at another, is not enough "
            "to change the review queue. The result is inconclusive as an operational improvement and "
            "should not be presented as evidence that the watchlist works."
        )
    lines.append("")
    lines.append(
        "Secondary rules, including each model alone and the sparse-signal restrictions, were not used to "
        "choose the recommendation. If one of them looks better in the table, that comparison was made "
        "after seeing these labels and would need a later time split before anyone relied on it. "
        "No threshold was tuned. The largest nominal gain of any non-primary rule over the expanded ensemble is "
        + ", ".join(
            f"`{name}` {max(d.values()):+d}" for name, d in decision["other_rules_minus_expanded_not_confirmatory"].items()
        )
        + " additional positives at its best budget; the sparse-signal variants are in the table. "
        "Differences of one CVE are within noise (McNemar p values in the table)."
    )
    lines.append("")
    lines.append(
        "Exploratory, post hoc check (not pre-specified). The whole dissent pool contains "
        f"{pool_check['pool_positives']} positives among {fmt_int(pool_check['pool_size'])} CVEs. "
        f"The next {fmt_int(pool_check['pool_size'])} ensemble ranks (through rank {fmt_int(pool_check['expanded_same_size_last_rank'])}) "
        f"contain {pool_check['expanded_same_size_positives']}. "
        f"Positives only in the pool: {', '.join(pool_check['only_pool_positive_ids']) or 'none'}. "
        f"Only in the ensemble extension: {', '.join(pool_check['only_expanded_positive_ids']) or 'none'}. "
        f"Exact McNemar p = {fmt_p(pool_check['mcnemar_exact_p'])}. "
        "Pool membership may carry a little information the upside-gap ordering wasted, but this budget size was "
        "chosen after seeing the result and the difference is a handful of CVEs. It is a hypothesis for a future "
        "time window, not a finding."
    )
    lines.append("")
    lines.append("## 5. Strongest examples")
    lines.append("")
    lines.append(
        "The three demonstration cases were chosen by a fixed rule: the miss the single-model-dissent ordering "
        "promotes the most, the miss closest to the ensemble cutoff, and the miss that every model ranked poorly. "
        "They illustrate the evaluation. They are not the three CVEs a dashboard should call the model's biggest mistakes "
        "unless the positions below say the alternative queue would actually have reached them."
    )
    lines.append("")
    demo_ids = studies["metadata"]["strongest_demonstration_ids"]
    by_case = {case["cve_id"]: case for case in studies["cases"]}
    for number, cve_id in enumerate(demo_ids, start=1):
        case = by_case[cve_id]
        description = case["retrospective_nvd_context"].get("description") or ""
        description = " ".join(description.split())
        if len(description) > 280:
            description = description[:277] + "..."
        vendor = (case["confirmed_historical_outcome"].get("catalog") or {}).get("vendor")
        product = (case["confirmed_historical_outcome"].get("catalog") or {}).get("product")
        lines.append(f"### {number}. {cve_id}")
        lines.append("")
        lines.append(" ".join(case["tags"]))
        lines.append("")
        for note in case["evidence_backed_notes"]:
            lines.append(note)
            lines.append("")
        lines.append(
            f"Retrospective NIST description, not snapshot-verified: {description}"
        )
        lines.append("")
        if vendor or product:
            lines.append(
                f"KEV catalog context, which is an outcome record: {vendor or 'unknown vendor'}, {product or 'unknown product'}."
            )
            lines.append("")
    lines.append("Other tagged cases are in `blind_spot_case_studies.json` with the same rank and signal fields.")
    lines.append("")
    lines.append("## 6. Statistical limitations")
    lines.append("")
    lines.append(
        "- Forty-three misses cannot support a precise estimate. Wilson intervals on extra-slot precision are wide, and a McNemar test on one or two discordant CVEs will not look decisive. That uncertainty is the result."
    )
    lines.append(
        "- The four budgets reuse one sorted list. A win that appears only after peeking at all four should not be counted four times."
    )
    lines.append(
        "- Labels and scores are the same historical out-of-fold evaluation that produced 32/75. The rules were pre-specified, but the population is not a fresh future window. The block table is a sensitivity, not an independent trial."
    )
    lines.append(
        "- Five-fold scores for earlier blocks were produced by models that also saw later blocks during the original training rotation. Chen's forward-only check found little inflation for the main model. This script does not repeat that training."
    )
    lines.append(
        "- Hand-built KEV groups mean some positives are related. Counting them as separate hits matches the published ensemble metric and overstates the number of independent events."
    )
    lines.append(
        "- Exploit flags and EPSS are inputs the models were built to use. A caught-versus-missed gap on those fields is partly mechanical. It is not a newly discovered unused signal."
    )
    lines.append(
        "- Missing monthly rows and zero code flags are missing records, not proof of no exploitation. Current CVSS and descriptions are retrospective. EPSS v3 and v4 are split at snapshot month 2025-03 under the 17 March 2025 cutover assumption; March's monthly file is not tagged with a model version."
    )
    lines.append(
        "- CVEs outside KEV at the cutoff can still have been exploited. False-alert counts are workload against this label, not a claim that the analyst would have been wasting time in the world."
    )
    lines.append(
        "- No 2026 hidden outcome is used or inferred. Rank gaps are not causal explanations of attacker behavior or of a model's internal features."
    )
    lines.append("")
    lines.append("## 7. Recommendation for the Blind Spot Investigator")
    lines.append("")
    if code == "does_not_beat_ensemble":
        lines.append(
            "Do not add a disagreement watchlist as a second review queue. On this historical evaluation it does "
            "not recover more confirmed KEV additions than spending the same extra budget further down the ensemble list. "
            "The investigator should still open a selected CVE and show the three ranks, the percentile range, and the "
            "upside gap, because those numbers explain how the ensemble average was formed. The display should say that "
            "a large rank gap was common and did not, by itself, justify jumping the queue."
        )
        lines.append("")
        lines.append(
            "The evidence-based secondary review is the next slice of the existing ensemble ranking. The historical "
            "precision of that slice is the expanded-ensemble row in the table, and it should be shown as historical "
            "precision, not as a probability for 2026. Beside a miss, show whether a snapshot signal row existed, "
            "whether any of the four code flags was recorded, and the within-month EPSS percentile, with missing values "
            "left missing. State that weak recorded signals are associated with being outside the top 517 because the "
            "models use those signals, and that most CVEs in the population also lack recorded flags."
        )
    elif code == "limited_support":
        lines.append(
            "The historical recount gives limited support for an optional single-model-dissent queue: after the top 517, "
            "additional slots drawn from CVEs one model placed in its own top 517, ordered by upside gap, recovered more of the 43 misses "
            "than the ensemble order at every pre-set budget. Show the measured extra-slot precision and the "
            "Wilson interval in the dashboard. Do not describe the queue as validated on a new time period. "
            "Keep the expanded ensemble visible as the baseline the rule beat on this file only."
        )
        lines.append("")
        lines.append(
            "Also show the three ranks for the CVE an analyst opens. Do not narrate a cause for the disagreement. "
            "Keep snapshot-signal missingness explicit, and do not treat a 2026 CVE's absence from KEV as a result."
        )
    else:
        lines.append(
            "Do not change the review order to follow model disagreement. The primary rule's advantage over the "
            "next B ensemble CVEs is too small or too inconsistent to be an operational finding. "
            "Use the investigator as an explanation panel: individual ranks, percentile range, upside gap, "
            "snapshot-signal presence, and the historical fact that extending the ensemble is the better-supported "
            "way to spend extra review time. Label every number as historical evaluation, not a 2026 prediction."
        )
    lines.append("")
    lines.append(
        "Generated by `Amy/feature/analyze_blind_spots.py` from `historical_rankings.csv`, "
        "`investigation_cases.json`, `chen/processed/cv_folds.parquet`, and the monthly exploit-signal CSVs. "
        "Machine-readable companions: `disagreement_analysis.json`, `watchlist_evaluation.csv`, "
        "and `blind_spot_case_studies.json`."
    )
    lines.append("")
    return "\n".join(lines)


def notable_observations(disagreement: dict, decision: dict, signals: dict) -> list[str]:
    notes = []
    upside_share = disagreement["neighbor_test"]["upside_gap"]["median_share"]
    if 0.4 <= upside_share <= 0.6:
        notes.append(
            "After conditioning on ensemble rank, missed positives have a typical upside gap relative to their neighbors."
        )
    elif upside_share > 0.6:
        notes.append(
            "After conditioning on ensemble rank, missed positives tend to have a larger upside gap than their neighbors. "
            "The watchlist table is the check on whether that shift is large enough to spend review budget on."
        )
    else:
        notes.append(
            "After conditioning on ensemble rank, missed positives tend to have a smaller upside gap than their neighbors."
        )
    if decision["code"] == "does_not_beat_ensemble":
        notes.append(
            "At every pre-specified extra budget, ordering by the best individual percentile recovered no more "
            "historical positives than continuing in ensemble order."
        )
    notes.append(
        f"{disagreement['n_missed_best_rank_inside_top_k']} missed positives were inside at least one model's own top 517, "
        f"alongside {disagreement['n_outside_unconfirmed_best_rank_inside_top_k']} unconfirmed CVEs."
    )
    population_rate = signals["by_cohort"]["full_population"]["share_any_recorded_code_flag"]
    missed_rate = signals["by_cohort"]["missed_positive"]["share_any_recorded_code_flag"]
    notes.append(
        f"Recorded exploit-code flags occur for {missed_rate:.1%} of misses and {population_rate:.1%} of all historical CVEs."
    )
    return notes


def main() -> None:
    rankings, meta, _misses = load_rankings()
    cases = load_investigation_ranks()
    verify_case_ranks(rankings, cases)
    k = meta["top_k"]
    data_dir = find_data_dir()
    signals_frame = load_snapshot_signals(rankings, data_dir)
    rankings = rankings.merge(signals_frame, on="cve_id", how="left", validate="one_to_one")
    verify_positive_signals(rankings, cases)
    if rankings["signal_record_present"].isna().any():
        raise ValueError("Signal join left historical CVEs without a missingness flag")

    disagreement = disagreement_summary(rankings, k)
    table, positions = evaluate_watchlists(rankings, k)
    sensitivity = block_sensitivity(rankings, k)
    pool_check = full_pool_check(rankings, k)
    signal_info = signal_summary(rankings, k)
    compact = compact_misses(rankings, positions, disagreement["neighbors"], k)
    studies = build_case_studies(cases, compact, k)
    decision = decision_from_table(table)

    # Confirm every reported budget matches the selection count already enforced.
    if not (table["extra_cves_reviewed"] == table["additional_budget_B"]).all():
        raise ValueError("A strategy row's review count does not match its budget")
    if not (table["total_cves_reviewed"] == k + table["additional_budget_B"]).all():
        raise ValueError("Total review budget does not equal 517 plus B")

    observations = notable_observations(disagreement, decision, signal_info)
    analysis = {
        "evaluation": "Historical 5-fold purged out-of-fold scores, November 4 publication-year labels. Not 2026 outcomes.",
        "population": int(len(rankings)),
        "positives": 75,
        "top_k": k,
        "caught": 32,
        "missed": 43,
        "definitions": {
            "percentile_rank": "pandas Series.rank(method='average', pct=True); higher is a higher model score",
            "ensemble_score": "mean of the three percentile ranks; matches historical_rankings.csv",
            "individual_competition_rank": "pandas rank(ascending=False, method='min'); 1 is best",
            "percentile_range": "max percentile minus min percentile",
            "percentile_std": "population standard deviation of the three percentile ranks (ddof=0)",
            "upside_gap": "max percentile minus mean percentile",
            "primary_watchlist": PRIMARY_STRATEGY,
            "tie_break": "better stored ensemble rank, then CVE ID; does not favor the challenger",
            "material_range_threshold": MATERIAL_RANGE,
            "strong_range_threshold": STRONG_RANGE,
            "neighbor_half_width": NEIGHBOR_HALF_WIDTH,
            "label_caveat": "A CVE not in KEV by the cutoff is an observed non-positive, not proof of no exploitation.",
            "causal_caveat": "Rank differences are descriptive. They are not explanations of why a model scored a CVE as it did.",
        },
        "validation": {
            "max_abs_ensemble_score_gap": meta["score_gap"],
            "ties": meta["ties"],
            "watchlist_candidates_exclude_original_top_k": True,
            "every_strategy_selects_exactly_B_unique_cves": True,
            "individual_ranks_match_investigation_cases": True,
            "snapshot_signals_match_investigation_cases_for_75_positives": True,
            "caught_and_missed_reproduced": True,
            "duplicate_cve_ids": 0,
        },
        "cohorts": disagreement["cohorts"],
        "unstratified_tests_confounded_by_rank": disagreement["unstratified_tests"],
        "neighbor_test": {
            "definition": disagreement["neighbor_test"]["definition"],
            "percentile_range": disagreement["neighbor_test"]["percentile_range"],
            "upside_gap": disagreement["neighbor_test"]["upside_gap"],
            "best_voter_percentile": disagreement["neighbor_test"]["best_voter_percentile"],
        },
        "outside_rank_correlations": disagreement["outside_rank_correlations"],
        "single_model_top_k": disagreement["single_model_top_k"],
        "miss_bands": disagreement["miss_bands"],
        "missed_best_model_counts_excluding_ties": disagreement["missed_best_model_counts_excluding_ties"],
        "missed_best_model_ties": disagreement["missed_best_model_ties"],
        "n_missed_best_rank_inside_top_k": disagreement["n_missed_best_rank_inside_top_k"],
        "n_outside_unconfirmed_best_rank_inside_top_k": disagreement["n_outside_unconfirmed_best_rank_inside_top_k"],
        "n_missed_dilution_at_least_1000_places": disagreement["n_missed_dilution_at_least_1000"],
        "signal_patterns": signal_info,
        "block_sensitivity": {
            "summed": sensitivity["summed"],
            "by_fold": sensitivity["by_fold"],
        },
        "primary_versus_expanded_ensemble": decision,
        "exploratory_full_dissent_pool_check": pool_check,
        "notable_observations": observations,
        "missed_cases": compact,
    }
    # Drop the non-serializable frames if any slipped in.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "disagreement_analysis.json", analysis)
    table.to_csv(OUTPUT_DIR / "watchlist_evaluation.csv", index=False)
    write_json(OUTPUT_DIR / "blind_spot_case_studies.json", studies)
    report = markdown_report(rankings, disagreement, table, signal_info, studies, sensitivity, decision, k, pool_check)
    (OUTPUT_DIR / "research_findings.md").write_text(report, encoding="utf-8")

    print("VALIDATION OK")
    print(f"population={len(rankings)} positives=75 caught=32 missed=43 K={k}")
    print("primary minus expanded:", decision)
    focus = table[table["strategy"].isin(["expanded_ensemble", PRIMARY_STRATEGY, "best_voter_percentile", "percentile_range", "upside_gap"])][
        ["strategy", "additional_budget_B", "additional_positives", "precision_additional", "total_positives_recovered"]
    ]
    print(focus.to_string(index=False))
    print("demonstrations:", studies["metadata"]["strongest_demonstration_ids"])
    print("neighbor upside median share:", disagreement["neighbor_test"]["upside_gap"]["median_share"])
    print("misses with a model inside top K:", disagreement["n_missed_best_rank_inside_top_k"])
    print("unconfirmed outside with a model inside top K:", disagreement["n_outside_unconfirmed_best_rank_inside_top_k"])


if __name__ == "__main__":
    main()
