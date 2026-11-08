"""Train and backtest TWO survival models on IBM Before the Breach data.

Models:
  1. Ridge-regularized Cox proportional hazards (scipy implementation).
  2. L2-regularized discrete-time logistic hazard model (scikit-learn).

Data split: fit on 2024 snapshots + KEV outcomes through Dec 31, 2024;
            score 2025 snapshots and evaluate on 2025 Nov 04 and Dec 31.
This uses zero future metadata, and the 2025 labels do NOT train the models.

PREREQUISITES:
    pip install numpy pandas scipy scikit-learn
    python build_survival_dataset.py --data-dir data

BACKTEST (run this first):
    python train_survival_models.py --data-dir data

OPTIONAL, after checking validation scores:
    python train_survival_models.py --data-dir data --score-2026 --model discrete
    python train_survival_models.py --data-dir data --score-2026 --model cox

OUTPUTS in data/survival/results/:
    backtest_2025_rankings.csv            complete scored historical cohort
    backtest_metrics.json                 reproducible Hits@100 and Hits@500
    scored_2026_survival.csv              only with --score-2026
    top500_survival_<model>.csv           only with --score-2026

Why the 2026 command refits: Once you finish choosing a model using the
2024->2025 validation, both 2024 and 2025 year-end observations can train
that final model (they were all historically available before 2026).

Caveat: 14-day discrete-time intervals are an approximation. A final partially
observed interval for a *censored* CVE is dropped from fitting, and prediction
for fractional horizons assumes constant event intensity within each interval.
Cox baseline hazard is stepped and flat beyond the last observed event time.
Neither model is expected to be well calibrated with just 33 training events.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FLAGS = ("exploitdb", "metasploit", "nuclei", "poc_github")
FEATURE_NAMES = (
    "epss_log10", "epss_raw", "epss_missing", *FLAGS, "public_flag_count",
)


def read_csv_checked(path: Path, extra: tuple[str, ...] = ()) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run build_survival_dataset.py first.")
    df = pd.read_csv(path, low_memory=False)
    required = {"cve_id", "epss", *FLAGS, *extra}
    absent = sorted(required - set(df.columns))
    if absent:
        raise ValueError(f"{path.name}: missing columns {absent}")
    if df["cve_id"].duplicated().any():
        raise ValueError(f"Duplicate CVEs in {path.name}")
    return df


def features(df: pd.DataFrame) -> np.ndarray:
    """Only historical snapshot-safe model inputs; no event/target/KEV dates."""
    epss = pd.to_numeric(df["epss"], errors="coerce").to_numpy(dtype=float)
    if np.any((~np.isnan(epss)) & ((epss < 0) | (epss > 1))):
        raise ValueError("EPSS must be between 0 and 1 or missing")
    missing = np.isnan(epss).astype(float)
    safe_epss = np.where(np.isnan(epss), 0.0, epss)
    log_epss = np.log10(np.clip(safe_epss, 1e-6, 1.0))

    flags = df[list(FLAGS)].apply(pd.to_numeric, errors="raise").fillna(0).to_numpy(dtype=float)
    if not np.isin(flags, [0, 1]).all():
        raise ValueError("Public exploit flags must be 0 or 1")
    count = flags.sum(axis=1)
    x = np.column_stack([log_epss, safe_epss, missing, flags, count])
    assert x.shape[1] == len(FEATURE_NAMES)
    return x.astype(np.float64, copy=False)


class RidgeCox:
    """Cox PH partial likelihood with L2 penalty and Breslow ties/baseline.

    The optimization is written out to avoid an extra dependency on lifelines.
    This is intentionally a small, inspectable regularized baseline model.
    """

    def __init__(self, penalty: float = 10.0):
        self.penalty = penalty
        self.scaler = StandardScaler()

    def fit(self, raw_x: np.ndarray, durations: np.ndarray, events: np.ndarray):
        x = self.scaler.fit_transform(raw_x)
        durations = np.asarray(durations, dtype=np.int32)
        events = np.asarray(events, dtype=np.int8)
        if not ((durations > 0).all() and np.isin(events, [0, 1]).all()):
            raise ValueError("Cox fit requires positive durations and binary event indicators")
        if int(events.sum()) < 2:
            raise ValueError("Not enough observed KEV events for Cox model")

        order = np.argsort(-durations, kind="mergesort")
        sorted_d = durations[order]
        xx = x[order]
        event_times, counts = np.unique(durations[events == 1], return_counts=True)
        endpoints = np.searchsorted(-sorted_d, -event_times, side="right") - 1
        sum_event_x = x[events == 1].sum(axis=0)
        n_events = int(events.sum())

        def objective(beta):
            eta = xx @ beta
            shift = float(np.max(eta))
            weights = np.exp(eta - shift)
            risks = np.cumsum(weights)
            risk_x = np.cumsum(weights[:, None] * xx, axis=0)
            denominators = risks[endpoints]

            log_likelihood = (
                float(sum_event_x @ beta)
                - float(np.dot(counts, np.log(denominators) + shift))
            )
            grad_ll = sum_event_x - (
                counts[:, None] * (risk_x[endpoints] / denominators[:, None])
            ).sum(axis=0)

            # Divide by observed-event count so the regularization strength
            # remains interpretable when combining the 2024 and 2025 cohorts.
            nll = -log_likelihood / n_events + 0.5 * self.penalty * np.dot(beta, beta)
            grad = -grad_ll / n_events + self.penalty * beta
            return nll, grad

        res = minimize(
            objective,
            x0=np.zeros(x.shape[1]),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 350, "ftol": 1e-11},
        )
        if not res.success:
            raise RuntimeError(f"Cox fitting failed: {res.message}")
        self.coef_ = res.x
        eta = xx @ self.coef_
        shift = float(eta.max())
        risk_sum = np.cumsum(np.exp(eta - shift))[endpoints]
        hazard_increments = counts * np.exp(-shift) / risk_sum
        self.event_times_ = event_times
        self.baseline_cumulative_ = np.cumsum(hazard_increments)
        return self

    def predict_probability(self, raw_x: np.ndarray, horizons: np.ndarray) -> np.ndarray:
        horizons = np.asarray(horizons, dtype=np.int32)
        if (horizons < 0).any():
            raise ValueError("Prediction horizons must be nonnegative")
        x = self.scaler.transform(raw_x)
        eta = np.clip(x @ self.coef_, -40, 40)
        at_idx = np.searchsorted(self.event_times_, horizons, side="right") - 1
        baseline = np.zeros(len(horizons), dtype=float)
        valid = at_idx >= 0
        baseline[valid] = self.baseline_cumulative_[at_idx[valid]]
        integrated_hazard = baseline * np.exp(eta)
        return np.clip(-np.expm1(-integrated_hazard), 0.0, 1.0)


class DiscreteHazard:
    """L2 logistic model: probability of an event in each 14-day age interval.

    Models P(event in interval k | survived previous intervals, x, k).
    A partial censor-only final interval is omitted from fitting. Partial
    prediction horizons use a constant-within-interval hazard approximation.
    """

    def __init__(self, interval_days: int = 14, c: float = 0.1):
        if interval_days <= 0 or c <= 0:
            raise ValueError("interval_days and regularization C must be positive")
        self.interval_days = interval_days
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=c, solver="lbfgs", max_iter=400)

    def fit(self, raw_x: np.ndarray, durations: np.ndarray, events: np.ndarray):
        durations = np.asarray(durations, dtype=np.int32)
        events = np.asarray(events, dtype=np.int8)
        if (durations <= 0).any() or not np.isin(events, [0, 1]).all():
            raise ValueError("Invalid survival duration or event value")
        interval = self.interval_days
        # Positive event: include all complete previous intervals and the
        # final interval where KEV occurred (even if the event was early).
        # Censored: include only fully observed intervals (not a short last one).
        repeats = np.where(events == 1, (durations - 1) // interval + 1, durations // interval)
        index = np.repeat(np.arange(len(durations), dtype=np.int32), repeats)
        if len(index) == 0:
            raise ValueError("No survival person-period observations")
        offsets = np.cumsum(repeats) - repeats
        interval_idx = np.arange(len(index), dtype=np.int32) - np.repeat(offsets, repeats)
        # Using a smooth log(age) trend avoids 20+ unconstrained parameters
        # when we only have 33 event observations in the training year.
        log_age = np.log1p((interval_idx + 0.5) * interval).astype(float)
        expanded = np.column_stack([raw_x[index], log_age])
        outcome = np.zeros(len(index), dtype=np.int8)
        pos_end_positions = np.cumsum(repeats)[events == 1] - 1
        outcome[pos_end_positions] = 1
        if outcome.sum() != events.sum():
            raise AssertionError("Survival expansion lost KEV events")

        expanded = self.scaler.fit_transform(expanded)
        self.model.fit(expanded, outcome)
        self.training_intervals_ = len(outcome)
        self.training_events_ = int(outcome.sum())
        return self

    def predict_probability(self, raw_x: np.ndarray, horizons: np.ndarray) -> np.ndarray:
        horizons = np.asarray(horizons, dtype=np.int32)
        if (horizons < 0).any():
            raise ValueError("Prediction horizons must be nonnegative")
        interval = self.interval_days
        log_survival = np.zeros(len(horizons), dtype=float)
        if len(horizons) == 0:
            return log_survival
        for k in range(int(np.ceil(horizons.max() / interval))):
            remaining = horizons - (k * interval)
            active = remaining > 0
            if not active.any():
                break
            within = np.minimum(remaining[active], interval) / interval
            log_age = np.full(active.sum(), np.log1p((k + 0.5) * interval))
            x = self.scaler.transform(np.column_stack([raw_x[active], log_age]))
            p = self.model.predict_proba(x)[:, 1]
            # Constant within-interval force-of-risk approximation for any
            # fractional final interval (rather than counting a full 14 days).
            log_survival[active] += within * np.log1p(-np.clip(p, 0, 1 - 1e-12))
        return np.clip(-np.expm1(log_survival), 0.0, 1.0)


def fit_two_models(train: pd.DataFrame, interval_days: int, cox_penalty: float, logit_c: float):
    x = features(train)
    t = train["duration_days"].to_numpy(dtype=np.int32)
    y = train["event"].to_numpy(dtype=np.int8)
    print(f"  Fitting Cox ridge on {len(train):,} CVEs with {int(y.sum())} events...", flush=True)
    cox = RidgeCox(penalty=cox_penalty).fit(x, t, y)
    print(f"  Fitting discrete hazard ({interval_days}-day intervals)...", flush=True)
    discrete = DiscreteHazard(interval_days=interval_days, c=logit_c).fit(x, t, y)
    print(f"  Expanded discrete-time observations: {discrete.training_intervals_:,}")
    return cox, discrete


def rank_frame(df: pd.DataFrame, score: str) -> pd.DataFrame:
    return df.sort_values([score, "cve_id"], ascending=[False, True], kind="stable")


def hits(df: pd.DataFrame, score: str, label: str, k: int) -> tuple[int, list[str]]:
    selected = rank_frame(df, score).head(k)
    correct = selected.loc[selected[label] == 1, "cve_id"].tolist()
    return len(correct), correct


def evaluate_predictions(val: pd.DataFrame, result_dir: Path) -> dict:
    val = val.copy()
    val["epss_sort"] = pd.to_numeric(val["epss"], errors="coerce").fillna(-1.0)
    flag_sum = val[list(FLAGS)].apply(pd.to_numeric, errors="raise").fillna(0).sum(axis=1)
    val["public_flag"] = (flag_sum > 0).astype(int)
    # Explicit two-key baseline (has public exploit flag first, then EPSS).
    baseline_order = val.sort_values(
        ["public_flag", "epss_sort", "cve_id"],
        ascending=[False, False, True], kind="stable",
    )
    results = {}
    comparison = []
    for cutoff_name, label in (("nov04", "event_nov04"), ("yearend", "event")):
        base_epss = rank_frame(val, "epss_sort")
        rankings = {
            "epss_only": base_epss,
            "public_flag_then_epss": baseline_order,
            "cox": rank_frame(val, f"cox_prob_{cutoff_name}"),
            "discrete": rank_frame(val, f"discrete_prob_{cutoff_name}"),
        }
        baseline_correct = set(
            baseline_order.head(500).loc[baseline_order.head(500)[label] == 1, "cve_id"]
        )
        results[cutoff_name] = {
            "positives": int(val[label].sum()), "candidates": len(val), "models": {}
        }
        for name, ranked in rankings.items():
            model_hits = {}
            for k in (100, 500):
                correct = ranked.head(k).loc[ranked.head(k)[label] == 1, "cve_id"].tolist()
                model_hits[f"hits_at_{k}"] = len(correct)
            correct_500 = set(
                ranked.head(500).loc[ranked.head(500)[label] == 1, "cve_id"]
            )
            model_hits["unique_kev_hits_vs_flag_baseline"] = len(correct_500 - baseline_correct)
            model_hits["top500_cve_overlap_with_flag_baseline"] = len(
                set(ranked.head(500)["cve_id"]) & set(baseline_order.head(500)["cve_id"])
            )
            results[cutoff_name]["models"][name] = model_hits
            comparison.append({"cutoff": cutoff_name, "model": name, **model_hits})

    val = val.drop(columns=["epss_sort", "public_flag"])
    val.to_csv(result_dir / "backtest_2025_rankings.csv", index=False)
    (result_dir / "backtest_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n2025 validation results (2024-trained models; not 2026 scores):")
    print(pd.DataFrame(comparison)[[
        "cutoff", "model", "hits_at_100", "hits_at_500", "unique_kev_hits_vs_flag_baseline"
    ]].to_string(index=False))
    return results


def main():
    parser = argparse.ArgumentParser(description="Regularized Cox + discrete-time KEV hazard models")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--interval-days", type=int, default=14)
    parser.add_argument("--cox-penalty", type=float, default=10.0)
    parser.add_argument("--logit-c", type=float, default=0.1)
    parser.add_argument("--score-2026", action="store_true", help="Refit on 2024+2025 and score all 2026 CVEs")
    parser.add_argument("--model", choices=("discrete", "cox"), default="discrete",
                        help="Which model to export as the final top500 when --score-2026 is used")
    args = parser.parse_args()
    if args.cox_penalty < 0:
        parser.error("--cox-penalty must be nonnegative")
    if args.logit_c <= 0 or args.interval_days <= 0:
        parser.error("--logit-c and --interval-days must be positive")

    source = Path(args.data_dir).resolve() / "survival"
    out = source / "results"
    out.mkdir(parents=True, exist_ok=True)
    train = read_csv_checked(source / "survival_train_2024.csv", ("event", "duration_days"))
    val = read_csv_checked(source / "survival_validation_2025.csv", (
        "event", "duration_days", "event_nov04", "duration_days_nov04"
    ))

    print("[1/3] Fit 2024-only models (no 2025 labels)...", flush=True)
    cox, discrete = fit_two_models(train, args.interval_days, args.cox_penalty, args.logit_c)

    print("[2/3] Predict historical 2025 KEV probabilities...", flush=True)
    val_x = features(val)
    for cutoff_name, col in (("nov04", "duration_days_nov04"), ("yearend", "duration_days")):
        # Important: the duration to predict is the available window until
        # the deadline, not the realized event/censoring duration! An event
        # could occur early; that outcome must NEVER become a model feature.
        deadline = pd.Timestamp("2025-11-04" if cutoff_name == "nov04" else "2025-12-31")
        snapshot = pd.to_datetime(val["snapshot_end"], errors="raise")
        horizon = (deadline - snapshot).dt.days.to_numpy(dtype=np.int32)
        if (horizon <= 0).any():
            raise ValueError("Nonpositive validation horizon")
        val[f"cox_prob_{cutoff_name}"] = cox.predict_probability(val_x, horizon)
        val[f"discrete_prob_{cutoff_name}"] = discrete.predict_probability(val_x, horizon)

    metrics = evaluate_predictions(val, out)
    print(f"\nSaved 2025 rankings and metrics in {out}")

    if args.score_2026:
        print("\n[3/3] Refit on 2024+2025 year-end outcomes; score 2026...", flush=True)
        combined = pd.concat([train, val], ignore_index=True)
        if len(combined) != len(train) + len(val):
            raise AssertionError("Unexpected combined cohort length")
        final_cox, final_discrete = fit_two_models(
            combined, args.interval_days, args.cox_penalty, args.logit_c
        )
        score = read_csv_checked(source / "survival_scoreboard_2026.csv", ("horizon_days",))
        if len(score) != 72_670:
            raise ValueError("Expected exactly 72,670 2026 candidates")
        horizons = score["horizon_days"].to_numpy(dtype=np.int32)
        score_x = features(score)
        score["cox_prob_nov04"] = final_cox.predict_probability(score_x, horizons)
        score["discrete_prob_nov04"] = final_discrete.predict_probability(score_x, horizons)
        target_col = f"{args.model}_prob_nov04"
        ranked = rank_frame(score, target_col)
        ranked.to_csv(out / "scored_2026_survival.csv", index=False)
        submission = ranked[["cve_id"]].head(500)
        if len(submission) != 500 or submission["cve_id"].duplicated().any():
            raise ValueError("Invalid top500 submission")
        path = out / f"top500_survival_{args.model}.csv"
        submission.to_csv(path, index=False)
        print(f"  Saved full 2026 ranking: {out / 'scored_2026_survival.csv'}")
        print(f"  Saved 500 CVEs: {path}")
    else:
        print("\n[3/3] 2026 scoring skipped. Inspect 2025 results first;")
        print("      then add --score-2026 --model discrete (or cox) when ready.")
    print("DONE.")


if __name__ == "__main__":
    main()
