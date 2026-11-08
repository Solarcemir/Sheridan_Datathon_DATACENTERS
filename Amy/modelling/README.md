# Survival-model experiment

**In short:** this folder tests whether modelling *how long until a CVE is added to KEV* (survival analysis) ranks CVEs better than a simple "public exploit code first, then EPSS" rule. It did not: the best survival model found 15 of 48 confirmed exploits in its top 500, against 17 for the simple rule. This was an exploratory side project. It is not the team's final model, and the Blind Spot Investigator dashboard does not use it.

## Overview

This folder contains an independent data-preparation and predictive-modeling workstream for IBM's **Before the Breach** challenge. The challenge is to rank **72,670 vulnerabilities (CVEs)** published from January through September 2026 by the likelihood that they would subsequently enter CISA's **Known Exploited Vulnerabilities (KEV)** catalog by **November 4, 2026**.

The work here focused on preparing trustworthy historical data, constructing reproducible training and validation cohorts, and evaluating whether **time-to-event (survival) models** could improve on simple vulnerability-ranking baselines. These were **exploratory models**, not the team's selected final submission model.

### 1. Data cleaning and validation

`clean_ibm_data.py` processes the original IBM CSVs into `data/cleaned/` without modifying the source files. It:

- Standardizes and validates CVE identifiers, publication dates, and snapshot months.
- Checks for duplicate records and conflicting primary keys.
- Validates EPSS scores (0–1) and binary exploit indicators.
- Preserves missing EPSS values rather than replacing them with zero.
- Checks KEV dates against publication dates and reports data-quality exceptions.
- Ensures the cleaned 2026 scoreboard contains exactly **72,670** unique CVEs.
- Produces `cleaning_report.json` with record counts, missing values, and quality checks.

The cleaning pipeline covers vulnerability publication dates, 2024–2025 exploit signals, known KEV entries, the 2026 scoreboard, and the optional CWE lookup. Original `affected_products.csv` and `references.csv` are retained as reference data but are **not used as model features** in this workstream.

### 2. Historical training and validation cohorts

`build_ibm_training_yearend.py` reconstructs historical predictions using the exploit signals available at each candidate's snapshot month, excluding CVEs already recorded as KEV at that snapshot.

- **Training:** January–September 2024 publications, labeled using subsequent KEV additions.
- **Validation:** January–September 2025 publications, held out from model fitting.
- **Prediction input:** All 72,670 unlabeled 2026 scoreboard candidates.

The default historical outcome is a KEV addition **after the snapshot and on or before December 31** of the relevant year. Separate datasets were also generated with a **November 4 cutoff**, as well as an exploratory **June-publication/July-snapshot** variant to mirror the missing June 2026 signals in the official benchmark.

| Cohort | Eligible CVEs | Confirmed KEV additions by Dec. 31 | By Nov. 4 |
| --- | ---: | ---: | ---: |
| 2024 training | 28,839 | 33 | 27 |
| 2025 validation | 35,280 | 52 | 48 |
| 2026 scoreboard | 72,670 | Unknown | Unknown |

The historical features were limited to `epss`, `exploitdb`, `metasploit`, `nuclei`, and `poc_github`. KEV dates and `in_kev` were used for eligibility and outcome construction, **not as model inputs**.

### 3. Survival-data preparation

`build_survival_dataset.py` converts the year-end historical cohorts into time-to-event records in `data/survival/`:

- `event`: whether KEV addition was observed within the relevant period.
- `duration_days`: days from the end of the prediction snapshot to KEV addition or the cutoff date.
- `event_nov04` and `duration_days_nov04`: additional November 4 evaluation fields for the 2025 validation set.
- `horizon_days`: days remaining from each 2026 snapshot to November 4, 2026.

Candidates with no recorded KEV addition by the cutoff are **right-censored**. They are not assumed to be permanently safe or never exploited.

### 4. Model training and evaluation

`train_survival_models.py` implements two independent time-to-event approaches:

1. **Ridge-regularized Cox proportional hazards model**, using a proportional-hazards formulation.
2. **L2-regularized discrete-time hazard model**, using logistic regression over 14-day risk intervals.

Both models were trained using **2024 data only** for the historical backtest and evaluated against the held-out **2025** candidates. Model inputs included transformed and raw EPSS, an EPSS-missing indicator, the four public exploit-code flags, and the number of flags present. Outcome dates and labels were excluded from predictive features.

The experiment also evaluated two simple references: **EPSS-only** and **public exploit flag first, then EPSS**. The main ranking metric was **Hits@500** (the number of confirmed KEV additions found among the 500 highest-ranked CVEs); Hits@100, overlap, and unique correct selections were also recorded.

## Results

### 2025 backtest — November 4 cutoff

**35,280 candidates · 48 confirmed KEV additions**

| Method | Hits@100 | Hits@500 | Additional KEV hits vs. flag baseline | Top-500 overlap with flag baseline |
| --- | ---: | ---: | ---: | ---: |
| EPSS only | 4 | 8 | 0 | 144 |
| Public exploit flag → EPSS | 8 | **17** | — | 500 |
| Cox survival | 7 | 15 | 1 | 376 |
| Discrete-time survival | 8 | 12 | 1 | 358 |

### 2025 backtest — December 31 cutoff

**35,280 candidates · 52 confirmed KEV additions**

| Method | Hits@100 | Hits@500 | Additional KEV hits vs. flag baseline | Top-500 overlap with flag baseline |
| --- | ---: | ---: | ---: | ---: |
| EPSS only | 4 | 8 | 0 | 144 |
| Public exploit flag → EPSS | 8 | **18** | — | 500 |
| Cox survival | 7 | 15 | 1 | 379 |
| Discrete-time survival | 8 | 14 | 1 | 355 |

**Conclusion:** Neither survival model outperformed the public-exploit-flag baseline on Hits@500. However, each recovered **one additional confirmed KEV case** outside that baseline's top 500, indicating some complementary ranking behavior. That gain did not offset the baseline hits displaced by either model. These results were therefore retained as experimental evidence rather than promoted as the team's final ranking.

## Key findings and limitations

- **Severe class imbalance:** Only 33 positive examples occur in the 2024 training cohort and 52 in the 2025 year-end validation cohort. Ranking results may be unstable with so few events.
- **Public exploit evidence is predictive but incomplete:** The November 4, 2025 validation cohort included 48 KEV additions; 27 had no public exploit-code flag at their snapshot. The survival models did not recover these unflagged cases in their top 500.
- **Limited differentiation:** Many CVEs share very similar EPSS and exploit-code values. Changing the model family cannot compensate for missing predictive features.
- **Historical leakage:** Current NIST status, CVSS/CWE enrichment, reference links, and affected-product records can reflect updates made after a vulnerability's snapshot. They were not introduced into the historical models without proof of contemporaneous availability.
- **Missing and shifting signals:** February 2025 has a documented historical-signal gap, EPSS changed its scoring model in March 2025, and the official 2026 dataset uses July snapshots for June-published CVEs.
- **Evaluation alignment:** Historical November 4 and year-end evaluations answer different questions. Comparisons to other teammates' results require the same candidates, labels, cutoff, and ranking metric.
- **Validation reuse:** The 2025 set has informed several experiments and should not be represented as a pristine untouched test set for later model selection.

A CVE absent from KEV is **not proof** that exploitation never occurred; KEV is a catalog of confirmed, recorded exploitation.

## Folder structure

```text
Amy/modelling/
├── clean_ibm_data.py                # Step 1: cleaning and data-quality checks
├── build_ibm_training_yearend.py    # Step 2: historical cohorts and cutoff variants
├── build_survival_dataset.py        # Step 3: event-time and censoring data
├── train_survival_models.py         # Step 4: Cox, discrete-time and baseline evaluation
└── data/
    ├── cleaned/                     # Cleaned IBM files + cleaning_report.json
    ├── processed/
    │   ├── backtest_train_yearend_2024.csv
    │   ├── backtest_validation_yearend_2025.csv
    │   ├── scoreboard_features_2026_yearend.csv
    │   └── training_report_{yearend,nov04,yearend_junejuly}.json
    └── survival/
        ├── survival_train_2024.csv
        ├── survival_validation_2025.csv
        ├── survival_scoreboard_2026.csv
        ├── survival_data_report.json
        └── results/
            ├── backtest_2025_rankings.csv   # Every 2025 CVE with each method's rank
            └── backtest_metrics.json        # The Hits@100 and Hits@500 numbers above
```

`processed/` keeps only the **year-end** cohort files, because those are the ones the survival models read. The November 4 results above don't need separate files: the 2025 validation set already carries `event_nov04` and `duration_days_nov04`. The CSVs for the November 4 and June-to-July variants, and the combined `all_labeled_*.csv` files, were deleted because nothing reads them. Their `training_report_*.json` summaries are kept, and step 2 can rebuild the CSVs (see below).

The original IBM CSVs are not stored here. They live in the repository's top-level `data/` folder and are never modified.

## How to reproduce

Requires Python 3.10+ with `pandas`, `numpy`, `scipy` and `scikit-learn`. Run from `Amy/modelling/`:

```powershell
cd Amy/modelling

# Step 1 reads the original CSVs from --data-dir and writes data/cleaned/.
# Copy the IBM CSVs into Amy/modelling/data/ first, so nothing is written outside Amy/.
python clean_ibm_data.py --data-dir data

python build_ibm_training_yearend.py --data-dir data      # year-end cohorts (the default)
python build_survival_dataset.py --data-dir data
python train_survival_models.py --data-dir data           # 2025 backtest -> data/survival/results/
```

Optional variants:

```powershell
python build_ibm_training_yearend.py --data-dir data --cutoff nov04          # November 4 labels
python build_ibm_training_yearend.py --data-dir data --june-use-july         # July snapshots for June CVEs
python train_survival_models.py --data-dir data --score-2026 --model cox     # refit on 2024+2025 and score 2026
```

The cleaned files and results are already in place, so you only need to rerun these to check or change the experiment.

## Final output — Individual workstream

This workstream produced **a reproducible, time-aware modeling pipeline and an independent survival-model comparison**. Its purpose was to test whether incorporating the *time remaining until KEV addition* improves prediction beyond a simple exploit-signal ranking—not to duplicate the team's other modeling work or claim responsibility for its final submission.

**Outputs delivered:**

- **Validated data pipeline:** Cleaned historical and scoreboard inputs with an audit report, while leaving the original CSVs unchanged.
- **Comparable historical cohorts:** Publication-month training and validation tables with cutoff-specific KEV labels, plus 72,670 unlabeled candidates ready for 2026 scoring.
- **Survival-ready datasets:** Event indicators, observed/censored durations, and prediction horizons to evaluate time-to-event approaches.
- **Two trained experimental model families:** Ridge-regularized Cox proportional hazards and L2-regularized discrete-time hazard modeling, evaluated against EPSS-only and public-exploit-flag baselines.
- **Reproducible backtest evidence:** CVE-level rankings, Hits@100 and Hits@500 results, top-500 overlap, and correct detections unique to each method.

**Best survival result:** The Cox model found **15 of 48** future KEV additions by the November 4, 2025 cutoff and **15 of 52** by December 31, 2025. The simpler public-exploit-flag baseline achieved **17 of 48** and **18 of 52**, respectively. Both survival methods identified **one positive CVE outside the baseline's top 500** in each evaluation, but neither achieved a higher overall Hits@500.

**Outcome:** The experiment showed that accounting explicitly for event timing and censoring was feasible, but did **not** improve top-500 accuracy with these five historical exploit-signal features. The survival results remain useful as an independently evaluated alternative and as evidence about which confirmed cases different approaches select. **These findings are exploratory and are not presented as the team's final selected prediction method.**
