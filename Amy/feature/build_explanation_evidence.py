"""Build feature-level evidence explaining why each historical KEV positive was caught or missed.

Run from the repository root:

    python Amy/feature/build_explanation_evidence.py

What it does, read-only:
  1. Rebuilds the 100 logistic-regression features for the 75,148 historical CVEs by calling
     Chen's own feature functions (chen/scripts/stack.py and cv.py) on a base table rebuilt in
     memory from the IBM CSVs, exactly as chen/scripts/build_historical.py builds it.
  2. Applies Chen's saved final LR weights (chen/results/final_lr_coefficients.csv, per standard
     deviation) so each feature's contribution is weight x standardized value. That is an exact
     decomposition of the final linear model's score. No model is trained or refit.
  3. Checks the reconstruction against Chen's out-of-fold LR scores before trusting it.
  4. For each of the 75 positives, compares its contributions with the average CVE in the
     ensemble's top 517, which is the "why was this below the review line" question.

Writes only Amy/feature/output/explanation_evidence.json.

Limits, stated in the output too: the attribution covers the linear voter only. Granite LoRA
and LightGBM have no saved per-feature attribution, so only their ranks are reported. The final
LR weights were fit on all historical CVEs, while the evaluated ranks came from out-of-fold
models, so the attribution describes how the team's linear model weighs features, not the exact
fold model that produced each rank.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FEATURE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FEATURE_DIR.parent.parent
OUTPUT_DIR = FEATURE_DIR / "output"
CHEN = PROJECT_ROOT / "chen"
DATA_DIR = PROJECT_ROOT / "data"

sys.dont_write_bytecode = True  # never leave __pycache__ inside chen/
sys.path.insert(0, str(CHEN / "scripts"))
import cv  # noqa: E402
import stack  # noqa: E402

K = 517
FLAGS = ["exploitdb", "metasploit", "nuclei", "poc_github"]
VULN_COLS = [
    "cve_id", "assigner", "cvss3_score", "cvss4_score", "attack_vector", "attack_complexity",
    "privileges_required", "user_interaction", "scope",
]
VOTERS = {"lr_full_llmx_noemb": "logistic_regression", "lora_control": "granite_lora",
          "lgbm_luis_only": "lightgbm"}
N_HOLDING_BACK = 6
N_HELPING = 4
MIN_SPEARMAN = 0.80

LABELS = {
    "exploitdb": "Exploit-DB entry recorded at snapshot",
    "metasploit": "Metasploit module recorded at snapshot",
    "nuclei": "Nuclei template recorded at snapshot",
    "poc_github": "GitHub proof-of-concept recorded at snapshot",
    "any_code": "any public exploit code recorded at snapshot",
    "epss_pct": "EPSS percentile within the snapshot month",
    "epss_missing": "EPSS missing at snapshot",
    "cvss": "CVSS base score (v3, else v4)",
    "cvss_missing": "CVSS missing",
    "av_network": "attack vector is network",
    "ac_low": "attack complexity is low",
    "pr_none": "no privileges required",
    "ui_none": "no user interaction required",
    "scope_changed": "CVSS scope changed",
    "wordpress_cna": "assigned by a WordPress-focused CNA",
    "log_assigner_kev_rate": "assigner's historical KEV rate",
    "log_assigner_kev": "assigner's prior KEV count",
    "log_days_since_assigner_kev": "days since the assigner's last KEV entry",
    "log_cwe_kev_rate": "historical KEV rate of this weakness type (CWE)",
    "log_batch_size": "size of the same-day release batch from this assigner",
    "out_of_band": "released outside the vendor's normal patch cycle",
    "log1p_twins": "number of near-identical sibling CVEs",
    "sibling_has_code": "a sibling CVE had public exploit code",
    "old_id": "CVE ID reserved a year or more before publication",
    "patch_tuesday": "published on Microsoft Patch Tuesday",
    "eol_text": "description mentions end-of-life",
    "unauth_text": "description says unauthenticated",
    "cvss4_urgent": "CVSS 4 urgency Red or Amber",
    "cvss4_automatable": "CVSS 4 automatable",
    "log1p_vendor_name_kev": "vendor's prior KEV count",
    "log1p_product_name_kev": "product's prior KEV count",
    "log1p_named_ransomware_kev": "prior ransomware-linked KEV entries for this vendor/product",
    "log1p_assigner_ransomware_kev": "assigner's prior ransomware-linked KEV entries",
    "sibling_max_epss_pct": "highest EPSS percentile among sibling CVEs",
    "rce_text": "description mentions remote code execution",
    "auth_bypass_text": "description mentions authentication bypass",
    "preauth_text": "description mentions pre-authentication",
    "log1p_n_versions": "number of affected versions listed",
    "cvss4_at_none": "CVSS 4 attack requirements none",
    "cvss4_subsequent_high": "CVSS 4 high impact on subsequent systems",
    "cwe_unknown": "weakness type unknown",
    "cwe_broad": "weakness type is a broad catch-all CWE",
    "text_kev_lr_logit": "description text resembles past KEV entries (text classifier)",
    "text_kev_sim": "similarity of description to past KEV descriptions",
    "log1p_adjacent_ids": "number of adjacent CVE IDs published together",
    "llm_vendor_kev": "Granite-identified vendor has prior KEV entries",
    "llm_product_kev": "Granite-identified product has prior KEV entries",
    "llm_product_ransomware_kev": "Granite-identified product has ransomware-linked KEV entries",
    "llm_internet_exposed": "Granite judged the product internet-exposed",
    "llm_install_base": "Granite's install-base rating",
    "llm_attacker_appeal": "Granite's attacker-appeal rating",
    "llm_end_of_life": "Granite judged the product end-of-life",
}
LLM_ANSWER_FIELDS = ["vendor", "product", "product_type", "internet_exposed", "install_base", "used_by",
                     "precondition", "impact", "end_of_life", "fix_status", "typical_attacker",
                     "exploitation_style", "attacker_appeal"]


def label(feature: str) -> str:
    if feature in LABELS:
        return LABELS[feature]
    if feature.startswith("llm_") and "=" in feature:
        field, value = feature[4:].split("=", 1)
        return f"Granite read the description as {field.replace('_', ' ')} = {value.replace('_', ' ')}"
    return feature


def readable(feature: str, value: float):
    if value is None or not math.isfinite(value):
        return None
    if feature.startswith("log1p_"):
        return {"model_value": round(value, 4), "approx_count": round(math.expm1(value), 2)}
    if feature.startswith("log_"):
        return {"model_value": round(value, 4), "approx_original": round(math.exp(value), 6)}
    return round(value, 4)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if value is None or isinstance(value, (str, int)):
        return value
    if pd.isna(value):
        return None
    return str(value)


def base_table(rankings: pd.DataFrame) -> pd.DataFrame:
    vuln = pd.read_csv(DATA_DIR / "vulnerabilities.csv", usecols=VULN_COLS, dtype={"cve_id": "string"})
    vuln = vuln[vuln["cve_id"].isin(set(rankings["cve_id"]))]
    if vuln["cve_id"].duplicated().any():
        raise ValueError("Duplicate CVE IDs in vulnerabilities.csv")
    keys = set(rankings["cve_id"] + "|" + rankings["snapshot_month"])
    months = set(rankings["snapshot_month"])
    parts = []
    for year in (2024, 2025):
        for chunk in pd.read_csv(DATA_DIR / f"exploit_signals_{year}.csv", usecols=["month", "cve_id", "epss", *FLAGS],
                                 dtype={"month": "string", "cve_id": "string"}, chunksize=400_000):
            chunk = chunk[chunk["month"].isin(months)]
            chunk = chunk[(chunk["cve_id"] + "|" + chunk["month"]).isin(keys)]
            if len(chunk):
                parts.append(chunk.rename(columns={"month": "snapshot_month"}))
    sig = pd.concat(parts, ignore_index=True)
    if sig.duplicated(["cve_id", "snapshot_month"]).any():
        raise ValueError("Duplicate monthly signal rows")
    df = rankings[["cve_id", "snapshot_month"]].merge(vuln, on="cve_id", how="left", validate="one_to_one")
    df = df.merge(sig, on=["cve_id", "snapshot_month"], how="left", validate="one_to_one", indicator=True)
    df["signal_row"] = df.pop("_merge") == "both"
    df[FLAGS] = df[FLAGS].fillna(0).astype("int8")
    if df["assigner"].isna().any():
        raise ValueError("Some historical CVEs are missing from vulnerabilities.csv")
    return df


def peer_stats(ctx: pd.DataFrame, column: str, value, cve_id: str, what: str) -> dict | None:
    if value is None or pd.isna(value) or str(value) == "":
        return None
    group = ctx[(ctx[column] == value) & (ctx["cve_id"] != cve_id)]
    kev = group[group["kev_by_nov4"] == 1]
    return {
        "group": what,
        "value": value,
        "other_historical_cves": int(len(group)),
        "other_kev_additions_by_cutoff": int(len(kev)),
        "of_those_inside_top_517": int((kev["rank"] <= K).sum()),
        "kev_rate_per_1000": round(1000 * len(kev) / len(group), 2) if len(group) else None,
        "share_of_group_inside_top_517": round(float((group["rank"] <= K).mean()), 4) if len(group) else None,
    }


PRODUCT_TYPE_RULES = (
    (re.compile(r"\b(chrome|chromium|firefox|safari|webkit|v8)\b", re.I), "browser"),
    (re.compile(r"\b(android|ios|ipados)\b", re.I), "mobile_os_or_app"),
    (re.compile(r"\bwordpress\b.*\b(plugin|theme)\b", re.I), "wordpress_plugin_or_theme"),
)


def information_gaps(i: int, df: pd.DataFrame, X: pd.DataFrame, days_after_snapshot,
                     description: str, product_type) -> list[dict]:
    gaps = []
    if not bool(df.at[i, "signal_row"]):
        gaps.append({"code": "no_signal_row", "fact": (
            "No monthly exploit-signal row existed at snapshot, so EPSS and all four exploit-code flags were "
            "unknown. The model stores unknown flags as zero, so 'no exploit code' counted against this CVE "
            "even though exploit-code status was never observed.")})
    else:
        if pd.isna(df.at[i, "epss"]):
            gaps.append({"code": "epss_missing", "fact": "A monthly signal row existed but EPSS was blank."})
        if X.at[i, "any_code"] == 0:
            gaps.append({"code": "no_recorded_exploit_code", "fact": "None of Exploit-DB, Metasploit, Nuclei or GitHub PoC had recorded code at snapshot."})
    if X.at[i, "cvss_missing"] == 1:
        gaps.append({"code": "cvss_missing", "fact": "No CVSS v3 or v4 score was available; the model used the median."})
    if X.at[i, "cwe_unknown"] == 1:
        gaps.append({"code": "cwe_unknown", "fact": "The weakness type (CWE) was unknown."})
    elif X.at[i, "cwe_broad"] == 1:
        gaps.append({"code": "cwe_broad", "fact": "The weakness type was a broad catch-all CWE, which carries little signal."})
    if X.at[i, "llm_vendor_kev"] == 0:
        gaps.append({"code": "vendor_no_prior_kev", "fact": "Granite's vendor match found no prior KEV entries for this vendor, so vendor history gave no lift."})
    if days_after_snapshot is not None and days_after_snapshot <= 31:
        gaps.append({"code": "kev_soon_after_snapshot", "fact": f"KEV addition came {days_after_snapshot} days after the snapshot (outcome timing, not a model input): little time for public signals to accumulate before exploitation was confirmed."})
    if product_type and not pd.isna(product_type):
        for pattern, expected in PRODUCT_TYPE_RULES:
            match = pattern.search(description or "")
            if match and product_type != expected:
                gaps.append({"code": "granite_product_type_possible_mismatch", "fact": (
                    f"Granite classified the product as '{product_type}', but the description mentions "
                    f"'{match.group(0)}', which a keyword rule associates with '{expected}'. The product-type "
                    "features the model used may be wrong for this CVE. Keyword rule on the retrospective "
                    "description; a possible issue, not confirmed.")})
                break
    return gaps


def main() -> None:
    rankings = pd.read_csv(OUTPUT_DIR / "historical_rankings.csv", dtype={"cve_id": "string", "snapshot_month": "string"},
                           low_memory=False).sort_values("rank").reset_index(drop=True)
    if len(rankings) != 75148 or int(rankings["kev_by_nov4"].sum()) != 75:
        raise ValueError("historical_rankings.csv does not match the 75,148 / 75 evaluation")
    coef = pd.read_csv(CHEN / "results" / "final_lr_coefficients.csv", index_col=0)["coef_per_sd"]

    df = base_table(rankings)
    X = pd.concat([stack.features(df), stack.meta_features(df), cv.team_feats(df), cv.llm_feats(df)], axis=1)
    missing_cols = sorted(set(coef.index) - set(X.columns))
    if missing_cols:
        raise ValueError(f"Rebuilt features lack coefficient columns: {missing_cols}")
    unused_cols = sorted(set(X.columns) - set(coef.index))
    X = X[coef.index]
    nan_counts = X.isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    inf_counts = np.isinf(X.to_numpy()).sum()
    if inf_counts:
        raise ValueError(f"{inf_counts} infinite feature values; check log transforms")
    X = X.fillna(0.0)

    mean, sd = X.mean(), X.std(ddof=0).replace(0, 1.0)
    Z = (X - mean) / sd
    contrib = Z * coef
    linear = contrib.sum(axis=1)

    rho = float(stats.spearmanr(linear, rankings["lr_full_llmx_noemb"]).statistic)
    linear_rank = linear.rank(ascending=False, method="min").astype(int)
    oof_rank = rankings["lr_full_llmx_noemb"].rank(ascending=False, method="min").astype(int)
    top_overlap = int(((linear_rank <= K) & (oof_rank <= K)).sum())
    positive = rankings["kev_by_nov4"] == 1
    if rho < MIN_SPEARMAN:
        raise RuntimeError(f"Reconstructed linear score only matches OOF LR at Spearman {rho:.3f}; not trusting attribution")
    linear_hits = int((positive & (linear_rank <= K)).sum())

    top = rankings["rank"] <= K
    ref = contrib[top].mean()
    ref_raw = X[top].mean()
    linear_cutoff = float(np.sort(linear.to_numpy())[::-1][K - 1])

    extract = pd.read_parquet(CHEN / "processed" / "llm_extract.parquet").drop_duplicates("cve_id").set_index("cve_id")
    disagreement = json.loads((OUTPUT_DIR / "disagreement_analysis.json").read_text(encoding="utf-8"))
    miss_info = {c["cve_id"]: c for c in disagreement["missed_cases"]}
    cases_json = json.loads((OUTPUT_DIR / "investigation_cases.json").read_text(encoding="utf-8"))
    case_info = {c["cve_id"]: c for c in cases_json["cases"]}
    studies = json.loads((OUTPUT_DIR / "blind_spot_case_studies.json").read_text(encoding="utf-8"))
    study_tags = {c["cve_id"]: c["tags"] for c in studies["cases"]}

    ctx = rankings[["cve_id", "rank", "kev_by_nov4"]].copy()
    ctx["product_type"] = ctx["cve_id"].map(extract["product_type"])
    ctx["vendor"] = ctx["cve_id"].map(extract["vendor"].astype("string").str.strip().str.lower())
    ctx["signal_state"] = np.select(
        [~df["signal_row"].to_numpy(), (df[FLAGS].max(axis=1) > 0).to_numpy()],
        ["monthly_row_absent", "recorded_code_flag"], default="row_present_all_flags_zero")

    def driver(i: int, feature: str) -> dict:
        return {
            "feature": feature,
            "meaning": label(feature),
            "this_cve_value": readable(feature, float(X.at[i, feature])),
            "typical_top_517_value": readable(feature, float(ref_raw[feature])),
            "weight_per_sd": round(float(coef[feature]), 4),
            "contribution_vs_top_517": round(float(contrib.at[i, feature] - ref[feature]), 4),
        }

    cases = []
    gaps_by_class = {"caught": [], "missed": []}
    for i in np.flatnonzero(positive.to_numpy()):
        row = rankings.iloc[i]
        cve = str(row["cve_id"])
        cls = "caught" if row["rank"] <= K else "missed"
        gap = contrib.iloc[i] - ref
        gaps_by_class[cls].append(gap)
        order = gap.sort_values()
        holding = [driver(i, f) for f in order.index[:N_HOLDING_BACK] if order[f] < 0]
        helping = [driver(i, f) for f in order.index[::-1][:N_HELPING] if order[f] > 0]
        oof_pct = float(rankings["lr_full_llmx_noemb"].rank(pct=True).iloc[i])
        lin_pct = float(linear.rank(pct=True).iloc[i])
        granite = extract.loc[cve, LLM_ANSWER_FIELDS].to_dict() if cve in extract.index else None
        inv = case_info[cve]
        miss = miss_info.get(cve)
        cases.append({
            "cve_id": cve,
            "classification": cls,
            "ranking": {
                "ensemble_rank": int(row["rank"]),
                "review_cutoff": K,
                "places_outside_cutoff": max(0, int(row["rank"]) - K),
                "voter_ranks": {name: int(inv["model_evaluation"]["voters"][col]["historical_rank"])
                                for col, name in VOTERS.items()},
                "miss_band": inv.get("miss_band"),
            },
            "linear_model_attribution": {
                "score_gap_vs_typical_top_517": round(float(gap.sum()), 4),
                "gap_to_linear_cutoff": round(float(linear.iloc[i] - linear_cutoff), 4),
                "reconstructed_linear_rank": int(linear_rank.iloc[i]),
                "out_of_fold_lr_rank": int(oof_rank.iloc[i]),
                "reconstruction_percentile_difference": round(abs(lin_pct - oof_pct), 4),
                "attribution_reliable_for_this_cve": abs(lin_pct - oof_pct) <= 0.05,
                "features_holding_it_back": holding,
                "features_in_its_favor": helping,
            },
            "granite_description_reading": granite,
            "snapshot_signals": inv["snapshot_signals"],
            "retrospective_nvd_context": {
                "warning": inv["current_nvd_context_warning"],
                "description": (inv["current_nvd_context"].get("description") or "")[:700],
                "cvss3_score": inv["current_nvd_context"].get("cvss3_score"),
                "cwe_ids": inv["current_nvd_context"].get("cwe_ids"),
            },
            "outcome_not_a_model_input": {
                "kev_added": inv["confirmed_outcome"]["kev_added"],
                "days_published_to_kev": inv["confirmed_outcome"]["days_published_to_kev"],
                "vendor": (inv["confirmed_outcome"].get("kev_catalog_context") or {}).get("vendor"),
                "product": (inv["confirmed_outcome"].get("kev_catalog_context") or {}).get("product"),
            },
            "information_gaps": information_gaps(i, df, X, inv["confirmed_outcome"].get("days_after_snapshot_to_kev"),
                                                 inv["current_nvd_context"].get("description") or "",
                                                 ctx.at[i, "product_type"]),
            "peer_context": {
                "note": ("Historical outcomes for similar CVEs in the same 2024-2025 evaluation, excluding this CVE. "
                         "Outcome context for lessons; not information the model had at snapshot. CVEs already in KEV "
                         "by the end of their publication month were excluded from the evaluation, so groups whose "
                         "bugs are confirmed exploited very quickly (for example browser zero-days) are undercounted."),
                "same_product_type": peer_stats(ctx, "product_type", ctx.at[i, "product_type"], cve, "Granite product type"),
                "same_vendor": peer_stats(ctx, "vendor", ctx.at[i, "vendor"], cve, "Granite-extracted vendor"),
                "same_signal_state": peer_stats(ctx, "signal_state", ctx.at[i, "signal_state"], cve, "exploit-signal state at snapshot"),
            },
            "blind_spot_analysis": None if miss is None else {
                "best_single_model": miss["best_model"],
                "best_single_model_rank": miss["best_rank"],
                "percentile_range_across_models": round(miss["percentile_range"], 4),
                "case_study_tags": study_tags.get(cve, []),
            },
        })

    caught_mean = pd.DataFrame(gaps_by_class["caught"]).mean()
    missed_mean = pd.DataFrame(gaps_by_class["missed"]).mean()
    separation = (missed_mean - caught_mean).sort_values()
    cohort = {
        "description": (
            "Average contribution of each feature, relative to the typical top-517 CVE, for the 43 missed "
            "versus the 32 caught KEV positives. Negative 'missed_minus_caught' means the feature pulled "
            "the misses down more than the caught cases. Linear voter only."
        ),
        "mean_score_gap_vs_top_517": {"caught": round(float(caught_mean.sum()), 4),
                                      "missed": round(float(missed_mean.sum()), 4)},
        "features_separating_missed_from_caught": [
            {"feature": f, "meaning": label(f), "missed_minus_caught": round(float(separation[f]), 4),
             "caught_mean_value": round(float(X.loc[positive & top, f].mean()), 4),
             "missed_mean_value": round(float(X.loc[positive & ~top, f].mean()), 4),
             "all_cves_mean_value": round(float(mean[f]), 4)}
            for f in list(separation.index[:10]) + list(separation.index[::-1][:3])
        ],
        "miss_bands": {band: int(sum(c["ranking"]["miss_band"] == band for c in cases))
                       for band in ("near_miss_518_to_1000", "mid_miss_1001_to_5000", "deep_miss_over_5000")},
        "blind_spot_findings": {
            "disagreement_watchlist_beats_expanded_ensemble": False,
            "deltas_vs_expanded_ensemble": disagreement["primary_versus_expanded_ensemble"]["deltas_primary_minus_expanded"],
            "misses_with_a_single_model_inside_top_517": disagreement["n_missed_best_rank_inside_top_k"],
            "non_confirmed_cves_with_a_single_model_inside_top_517": disagreement["n_outside_unconfirmed_best_rank_inside_top_k"],
        },
    }
    payload = {
        "metadata": {
            "population": 75148, "positives": 75, "caught": 32, "missed": 43, "top_k": K,
            "evaluation": "Historical 2024-2025 out-of-fold evaluation with November 4 labels. Not 2026 outcomes.",
            "method": (
                "Contribution = final LR weight per SD x standardized feature value (standardized over the 75,148 "
                "historical CVEs). 'vs top 517' subtracts the average contribution among the ensemble's top 517, so "
                "the sum over features is how far the CVE's linear score sits below or above a typical selected CVE."
            ),
            "validation": {
                "spearman_reconstructed_vs_out_of_fold_lr": round(rho, 4),
                "top_517_overlap_reconstructed_vs_out_of_fold_lr": top_overlap,
                "kev_hits_in_reconstructed_linear_top_517": linear_hits,
                "out_of_fold_lr_hits_reported_by_chen": 26,
                "features_rebuilt": int(X.shape[1]),
                "rebuilt_columns_not_in_final_model": unused_cols,
                "features_with_missing_values_filled_with_0": {k: int(v) for k, v in nan_counts.items()},
                "cases_where_attribution_unreliable": [c["cve_id"] for c in cases
                                                       if not c["linear_model_attribution"]["attribution_reliable_for_this_cve"]],
            },
            "caveats": [
                "Attribution covers the logistic-regression voter only. Granite LoRA and LightGBM have no saved per-feature attribution; only their ranks are given.",
                "The final LR weights were fit on all historical CVEs, including these positives. The evaluated ranks came from out-of-fold models. Attribution shows how the team's linear model weighs features, not the exact fold model.",
                "A feature 'holding a CVE back' is a statement about the model's arithmetic, not about whether the CVE was dangerous.",
                "Granite description readings and text features use the description text available to the model; NVD fields labelled retrospective may have been revised after the snapshot.",
                "KEV dates and catalog names are outcomes, not model inputs.",
                "Missing monthly signal rows and zero exploit flags are missing records, not proof of no exploitation.",
            ],
        },
        "cohort_contrast": cohort,
        "cases": cases,
    }
    out = OUTPUT_DIR / "explanation_evidence.json"
    out.write_text(json.dumps(clean(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Spearman reconstructed vs out-of-fold LR: {rho:.4f}; top-517 overlap {top_overlap}; linear hits {linear_hits}")
    print(f"Unreliable attributions: {payload['metadata']['validation']['cases_where_attribution_unreliable']}")
    print("Top features separating misses from caught:")
    for item in cohort["features_separating_missed_from_caught"][:8]:
        print(f"  {item['missed_minus_caught']:+.3f}  {item['feature']}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
