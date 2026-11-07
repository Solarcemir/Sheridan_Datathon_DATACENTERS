# Before the Breach · IBM track, Sheridan Datathon 2026

Predict which 2026 CVEs CISA will add to its Known Exploited Vulnerabilities list, and explain why.

**Start here: [`context_Luis.md`](context_Luis.md)**. It holds the full handoff: method, every experiment with numbers, conclusions, and what to do next. Every number in it points to a file in this repo (see its §13 Evidence index).

Team hub (rules, data map, shared notes, Top 500 table): https://claude.ai/artifact/Np2SfMguMKySJnBfvNfnjR

## What's in each folder

| Folder | What it holds | Who writes it |
|---|---|---|
| `context_Luis.md` | Handoff document: findings, A/B backtests, conclusions, next steps | — |
| `analysis/` | All code, in run order (see below) | — |
| `reports/` | **Evidence for the findings**: result tables (`*.csv`) and charts (`figures/`) | `train_baseline.py`, `sibling_boost.py`, `granite_*.py` |
| `submissions/` | Candidate `top500` files for 2026 (one column, `cve_id`) | `train_baseline.py`, `granite_instruct.py` |
| `derived/` | Feature dictionaries, full 2026 rankings with reasons, row-by-row backtest rankings, Granite Instruct answers | `build_features.py`, `train_baseline.py`, `sibling_boost.py`, `granite_*.py` |
| `site/` | Source of the team hub page | — |

Big or regenerable files are not committed (see `.gitignore`): the raw data, the feature tables, and the Granite embedding matrix. Rebuild them with the commands below.

## Reproduce

Unzip the hackathon data so the CSVs are in `IBM-20261107T174321Z-1-001/IBM/data/data/`, then from the repo root:

```
python -E -P analysis/backtest_2025.py        # 2025 backtest (52 positives)
python -E -P analysis/build_features.py       # 2024/2025 replicas + 2026 features (leak-free)
python -E -P analysis/granite_features.py     # IBM Granite Embedding features + clusters (GPU recommended)
python -E -P analysis/train_baseline.py       # baselines and LightGBM, A/B evaluation, charts, top500_v1
python -E -P analysis/sibling_boost.py        # sibling boost + per-assigner cap tests, backtest rankings
python -E -P analysis/granite_instruct.py     # IBM Granite Instruct re-ranking + per-CVE reasons, top500_v2
python -E -P analysis/export_top500_json.py   # data file for the hub's Top 500 table
```

Requirements and known environment issues are in `context_Luis.md` §12.
