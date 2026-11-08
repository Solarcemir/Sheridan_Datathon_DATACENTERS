# chen/submissions

Scores for the 72,670 scoreboard CVEs, consumed by `luis/analysis/final_build.py` (the team's `top500_final.csv` is built there).

- `lora8b_scoreboard_2026.parquet`: `cve_id, lora_score`, Granite 4.2 8B LoRA, average of 4 seeds (mean percentile rank of logP(Yes) - logP(No)).
- `lr_best_scoreboard_2026.parquet`: `cve_id, lr_score`, the best logistic regression (100 features).

Both were trained on all CVEs of 2024-01..2025-09; higher = more likely. Details: `../context_Chen.md` section 8.4.
- `lr_unified_scoreboard_2026.parquet`: `cve_id, lr_score`, unified LR v1 (118 features; ours + Luis-only columns), adopted by the 14.16 rule; see `../context_Chen.md` section 8.6.
