# Project Context · Regularized Logistic Regression & Factor Mining

## 0. TL;DR
- **Model:** L2-Regularized Logistic Regression tuned via 3-Fold Stratified CV on 2024 data (optimal C = 0.003).
- **2025 Test Set Performance:** Achieved an **Average Precision (PR-AUC) of 0.2432** on the held-out 2025 pool (48,101 CVEs, 116 positives).
- **Benchmark Discovery:** Evaluated the **3-per-assigner-month cap**, which filters vendor patch-dump flooding and boosts Top-500 hit concentration.
- **Leakage Compliance:** 100% compliant with competition rules. Dropped `nvd_status`, `last_modified`, `n_affected_cpes`, `n_references`, `cvss3_source`, `affected_products.csv`, and `references.csv`.

## 1. Data Views & Row Counts
- **Training Set (2024):** 39,913 CVEs, 112 confirmed post-publish KEV positives.
- **Test Set (2025):** 48,101 CVEs, 116 confirmed post-publish KEV positives.
- **Scoreboard (2026):** 72,670 CVEs ranked for final submission.

## 2. Feature Engineering & Statistical Lifts (24 Features)
1. **Assigner Behavioral Signals:**
   - `is_out_of_band`: Flaw was the single release from vendor that day (5.5x empirical lift).
   - `is_patch_dump`: >= 20 CVEs dumped on same day (0.3x lift, negative filter).
   - `is_wordpress_cna`: Wordfence/Patchstack/WPScan (0 KEV positives, strong negative filter).
   - `is_vulncheck`: High-risk security assigner (14x lift).
2. **Text Regex Signals (Description):**
   - `txt_auth_bypass` (9.5x lift), `txt_unauth` (5.5x lift), `txt_rce` (4.4x lift), `txt_eol` (13x lift).
3. **Behavioral & Interaction Features:**
   - `deadly_triad`: Remote Network + Unauthenticated + Zero User Interaction.
   - `has_any_exploit`: Flag for any public PoC code (GitHub, Metasploit, Nuclei, Exploit-DB).
   - `is_cwe_unknown`: NVD-CWE-noinfo / Other (0 positives in 3,551, negative filter).

## 3. Evaluation Protocol
- **Split:** Strictly temporal (2024 Train, 2025 Test).
- **GridSearch:** Evaluated C in [1e-4, 1e-2] using Stratified 3-Fold CV on 2024. Optimal C = 0.003.
- **PR-AUC:** 0.2432 (vs 0.0024 baseline rate, representing a 100x precision lift).

## 4. Integration with Team Pipeline
This linear model's calibrated probability predictions are designed to be rank-averaged with the LoRA fine-tuned Granite 8B and LightGBM models for the final ensemble submission.