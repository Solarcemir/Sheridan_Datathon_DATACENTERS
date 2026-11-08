# chen/submissions

Scores for the 72,670 scoreboard CVEs, consumed by `luis/analysis/final_build.py` (the team's `top500_final.csv` is built there).

- `lora8b_scoreboard_2026.parquet`: `cve_id, lora_score`, Granite 4.2 8B LoRA, average of 4 seeds (mean percentile rank of logP(Yes) - logP(No)).
- `lr_best_scoreboard_2026.parquet`: `cve_id, lr_score`, the best logistic regression (100 features).

Both were trained on all CVEs of 2024-01..2025-09; higher = more likely. Details: `../context_Chen.md` section 8.4.
- `lr_unified_scoreboard_2026.parquet`: `cve_id, lr_score`, unified LR v1 (118 features; ours + Luis-only columns), adopted by the 14.16 rule; see `../context_Chen.md` section 8.6.

## final_model_all_cves.parquet (repo-scanner table)

The final Top-500 model applied to every CVE in the IBM snapshot (`chen/scripts/all_cves.py`): the 72,670 scoreboard
CVEs keep their real final scores; the other 328,058 are scored like a 2026 bug (signals only where the IBM files have
them, KEV history up to 2025-12-31). Columns: `score` (percentile vs the 2026 scoreboard), `top500` (would rank inside
our Top 500, ignoring the vendor cap), the three voters' percentiles (`tree`, `linear`, `llm`), `granite_demoted`, and
Granite 30B's vendor / product / product_type / impact / typical_attacker / attacker_appeal. CVEs already on KEV at the
end of their publish month aren't scored. `chen/frontend/scan_data/` is the same table in per-year JSON for the
static scanner page (`chen/frontend/scan.html`).
