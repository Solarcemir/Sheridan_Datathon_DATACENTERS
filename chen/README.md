# chen/: GPU-box workstream (purged CV, Granite 4.2 LoRA + world-knowledge extraction)

Start with **`context_Chen.md`**: results, every experiment (including the failed ones), leakage rules, run order and file index.

## Folder map

| Path | What |
|---|---|
| `context_Chen.md` | full handoff |
| `scripts/` | pipeline code (see the run order in `context_Chen.md` §12) |
| `scripts/analysis/` | one-off analyses that produced `reports/*.txt` |
| `processed/` | derived feature tables keyed by `cve_id`, KEV groups, CV folds, Granite 30B extraction |
| `results/` | out-of-fold CV scores, LoRA fold scores, LLM-scorer runs, memorization probe |
| `reports/` | result tables as text |
| `submissions/` | (final `top500_final.csv` pending: `context_Chen.md` §10) |

## Setup

```
cd chen
mkdir -p IBM && ln -s ../../IBM-20261107T174321Z-1-001/IBM/data/data IBM/data   # IBM data (not committed)
uv run python scripts/build_historical.py                                       # rebuild the data views
```

GPU scripts need vLLM 0.32 / PEFT (see `context_Chen.md` §12). The most useful parts for others need no GPU: `processed/llm_extract.parquet` (Granite 30B answers for 147,818 CVEs), `processed/*_feat_*.parquet`, `results/cv_oof_*.parquet`.
