# luis/ · Luis's workstream

**Start with [`context_Luis.md`](context_Luis.md)**: method, every experiment with numbers, conclusions, the merge plan with Chen, and next steps. Its §13 maps every number to a file in this folder.

Team hub (Top 500 table with Granite reasons, rules, data map, shared notes): https://claude.ai/artifact/Np2SfMguMKySJnBfvNfnjR

| Folder | What it holds |
|---|---|
| `analysis/` | All code, in run order (below) |
| `reports/` | Evidence for the findings: result tables (`*.csv`, `density_check.txt`) and charts (`figures/`) |
| `submissions/` | Candidate 2026 `top500` files (one column, `cve_id`). Current pick: `top500_v2.csv` |
| `derived/` | Feature dictionaries, full 2026 rankings with reasons, row-by-row A/B backtest rankings, Granite Instruct answers |
| `site/` | Source of the team hub page and its `top500.json` |
| `received/` | Snapshots of teammates' docs this work refers to (Chen's GPU-box context) |
| `derived/for_chen/` | Our features as Parquet for Jan 2024–Dec 2025 + the 2026 scoreboard, keyed by `cve_id`, to merge into Chen's CV (`build_features.py --all-months`) |

Not committed (big or regenerated in minutes): `derived/features_*.csv`, `derived/leak_only_*.csv`, `derived/backtest_cohorts_2025.csv`, `derived/granite/emb_granite30m.npy`, `derived/granite/granite_features_*.csv`.

## Reproduce

Keep the IBM data unzipped at the **repo root** (`IBM-20261107T174321Z-1-001/IBM/data/data/`; git ignores it). From the repo root:

```
python -E -P luis/analysis/backtest_2025.py        # 2025 backtest (52 positives)
python -E -P luis/analysis/build_features.py       # 2024/2025 replicas + 2026 features (leak-free)
python -E -P luis/analysis/granite_features.py     # IBM Granite Embedding features + clusters (GPU recommended)
python -E -P luis/analysis/train_baseline.py       # baselines and LightGBM, A/B evaluation, charts, top500_v1
python -E -P luis/analysis/sibling_boost.py        # sibling boost + per-assigner cap tests, backtest rankings
python -E -P luis/analysis/granite_instruct.py     # IBM Granite Instruct re-ranking + per-CVE reasons, top500_v2
python -E -P luis/analysis/export_top500_json.py   # site/top500.json for the hub's Top 500 table
python -E -P luis/analysis/density_check.py        # numbers at scoreboard density, comparable with Chen's CV
```

Environment notes and known install issues: `context_Luis.md` §12.
