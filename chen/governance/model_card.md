# Model card: CVE → CISA KEV prioritization (linear model of the final ensemble)

## Purpose and intended use

- **What it does:** ranks newly published CVEs by how likely CISA is to add them to the Known Exploited Vulnerabilities catalog in the following months, so a security team can patch the most likely-to-be-exploited first.
- **Who it's for:** vulnerability-management teams, as a prioritization aid next to vendor guidance.
- **Not for:** deciding that a CVE is safe to ignore. About 40% of exploited CVEs look ordinary at publish time (see Limitations).

## Model

- **Algorithm:** L2-regularized logistic regression on 100 standardized features, with balanced class weights. C = 1e-4, chosen by grouped cross-validation inside the training data.
- **Final ranking (team build, `luis/analysis/final_build.py`):** an equal-weight rank average of three voters: a tree model (XGBoost, 40 seeds), a linear voter (this model averaged with Luis's logistic regression), and a LoRA-fine-tuned **IBM Granite 4.2 8B** reading each CVE. CVEs caught by Granite 30B demotion filters are pushed down, then at most 3 CVEs per assigner per month.
- **IBM technology in the features:**
  - **Granite 4.2 30B** (served with vLLM) read every CVE description and answered fixed questions: product type, internet exposure, install base, precondition, impact, end-of-life, typical attacker, attacker appeal. Its answers became 55 of the 100 features.
  - **Granite Embedding r2** finds the most similar past exploited CVE, used for explanations.

## Data

- **Training:** 75,148 CVEs published Jan 2024–Sep 2025, each with what was known at the end of its publish month: EPSS percentile, public exploit code (Exploit-DB, Metasploit, Nuclei, GitHub PoC), CVSS, CWE, assigner, description.
- **Labels:** 119 later added to KEV by 2025-12-31 (75 by Nov 4 of their publish year, the evaluation window).
- **Scored data:** the 72,670 CVEs published Jan–Sep 2026 (the IBM datathon scoreboard).

## Leakage controls

- **Excluded:** fields NIST fills in after exploitation (analysis status, product lists, reference counts and links, CVSS source, last-modified), and description phrases such as "exploited in the wild" (descriptions may be edited later).
- **KEV history cutoff:** history features use KEV entries only up to the end of the previous year, matching what is knowable for 2026.
- **Memorization check:** Granite 4.2 30B could not identify any of 122 exploited CVEs from their ID alone.

## Evaluation

- **Protocol:** 5-fold cross-validation over time blocks, with related CVEs (exploit chains, same-product batches) kept in the same block. Also a forward-only check: each block scored only by a model trained on earlier blocks.
- **Metric:** hits in the top 517, the share the judges use (top 500 of 72,670).

| Ranking | Hits (KEV by Nov 4) | vs baseline, 95% CI |
|---|---|---|
| Exploit code first, then EPSS (standard practice) | 20 | – |
| This linear model alone | 26 | −1 to +12 |
| Final ensemble (linear + Granite LoRA + LightGBM) | 32 | +5 to +19 |

**Forward-only:** the linear model scores 23 vs 18 for the baseline, the same as cross-validation on those blocks (24). So the result is not inflated by training on later data.

## Limitations

- **Invisible zero-days:** targeted zero-days on phones, browsers and messaging apps, and bugs that are only dangerous when chained, have no public exploit code at publish time. They are hard for any model.
- **Few positives:** with 75–119 positives, differences under about ±4 hits are noise. Results are reported with bootstrap confidence intervals.
- **Drift:** 2026 inputs differ from training. KEV history has grown and release sizes doubled (see `drift_table.csv`). Rankings within 2026 are more reliable than the probability values.
- **Label gaps:** KEV is a conservative label. Some exploited CVEs never reach KEV, so the model's "false positives" include real attacker favourites.
