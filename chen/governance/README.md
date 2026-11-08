# watsonx.governance package: final KEV-prediction model (Chen)

For the teammate with the IBM Cloud account. Everything needed to (1) register the model with an **AI factsheet** and (2) run a **drift evaluation** comparing 2024–25 training data with the 2026 scoreboard. Built by `scripts/governance_pack.py`. It contains only derived numeric features and scores: no IBM descriptions or raw data files.

| File | Use in watsonx.governance |
|---|---|
| `model_card.md` | factsheet text: purpose, data, features, evaluation, limitations (paste into the use case / model entry) |
| `feature_schema.csv` | input schema: 100 features, group, type, why each is known at the snapshot |
| `training_sample.csv` | training data reference for drift: 10,000 rows (all 119 positives + random negatives), features, labels, out-of-fold LR score |
| `scoreboard_sample.csv` | payload / scoring data: 10,000 random 2026 scoreboard rows, same features, final LR score |
| `drift_table.csv` | our own per-feature drift check (means, standardized difference, PSI), to compare with the platform's drift result |

## Steps (outline; check current IBM docs, the UI changes between versions)

1. In watsonx.governance, create an **AI use case** ("Prioritize newly published CVEs by likelihood of CISA KEV addition").
2. Add the model as an **external (third-party) model**: tabular binary classification, label column `label_kev_by_nov4`, prediction/score column `lr_score_out_of_fold`. Use `feature_schema.csv` for the input schema.
3. Upload `training_sample.csv` as the **training data** reference.
4. Upload `scoreboard_sample.csv` as **payload** data (score column `lr_score_final`) and run the **drift** evaluation.
5. Fill the factsheet from `model_card.md`. Attach the drift result and our `drift_table.csv`.
6. Screenshot the factsheet, the drift result and the lifecycle view for the slides.

## What the drift evaluation should show (our own numbers, `drift_table.csv`)

6 of 100 features have PSI > 0.25 (major shift), 11 are between 0.1 and 0.25. The large ones are KEV-history features (assigner KEV rate PSI 1.23, days since last KEV 0.66, assigner KEV count 0.64). KEV grows every year, so every 2026 CVE sits closer to past KEV activity. Also: release sizes doubled in 2026 (sibling max EPSS 0.43), and the 2026 TF-IDF model saw more KEV examples (0.24). The shift is mostly the same for every 2026 CVE, so the *ranking* within 2026 is less affected than the probability values.

**Not in the repo:** `training_sample.csv` and `scoreboard_sample.csv` contain copies of the IBM data columns (exploit flags, EPSS), which CONTRIBUTING keeps out of git. Regenerate them on a machine with the data: `python chen/scripts/governance_pack.py`.
