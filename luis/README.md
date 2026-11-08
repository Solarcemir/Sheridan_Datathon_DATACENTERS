# luis/ · Luis's workstream

**Start with [`context_Luis.md`](context_Luis.md)**: method, every experiment with numbers, conclusions, the merge plan with Chen, and next steps. Its §13 maps every number to a file in this folder.

Team hub (Top 500 table with Granite reasons, rules, data map, shared notes): https://claude.ai/artifact/Np2SfMguMKySJnBfvNfnjR

| Folder | What it holds |
|---|---|
| `analysis/` | All code, in run order (below) |
| `reports/` | Evidence for the findings: result tables (`*.csv`, `density_check.txt`) and charts (`figures/`) |
| `submissions/` | Intermediate candidate rankings from the modeling workstream. **The official final submission is the root [`top500.csv`](../top500.csv)** (XGBoost + Luis & Chen LRs + Chen's Granite LoRA, Granite filters, cap 3). |
| `derived/` | Feature dictionaries, full 2026 rankings with reasons, row-by-row A/B backtest rankings, Granite Instruct answers |
| `site/` | Source of the team hub page and its `top500.json` |
| `frontend/` | **Breach Radar**, the story page (pipeline network, weights, what worked, insights before/after the model, the 500) *and* `repo-scan.html`, the repo-scanner UI. Live story page: https://claude.ai/artifact/TQYCiUtkqQbqtyZi7yvvNh · docs in `frontend/README.md` |
| `repo_scanner/` | Working prototype: scans a real `package-lock.json`/`requirements.txt`, matches CVEs via OSV.dev, ranks the non-KEV ones with our exported model. Plan: `proposal_repo_scanner.md`. Run it: `repo_scanner/README.md` |
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
python -E -P luis/analysis/chen_insights.py        # Chen's features/models on our A/B, bootstrap, missed positives
python -E -P luis/analysis/chen_insights.py --score-2026   # submissions/top500_v3.csv (LightGBM + LR)
python -E -P luis/analysis/model_zoo.py            # XGBoost / CatBoost / LambdaRank / big linear + ensembles (~11 min, GPU)
python -E -P luis/analysis/model_zoo.py --score-2026       # current pick: submissions/top500_v4.csv (XGBoost big + LR)
python -E -P luis/analysis/team_merge.py           # Luis + Chen (Granite 30B features, LoRA) on the forward A/B test
python -E -P luis/analysis/team_merge.py --score-2026      # v4 explained by Granite 30B; v5 = v4 + Chen LoRA if his file exists
python -E -P luis/analysis/export_top500_json.py "<label>" reports/top500_v4_explained.csv "<note>"   # hub table data
python -E -P luis/analysis/static_prior.py         # more training years (2022-24) without signals: rejected, see context §14.9
python -E -P luis/analysis/postprocess.py          # Granite-30B demotion filters + vendor caps on v4 (A/B test)
python -E -P luis/analysis/postprocess.py --score-2026     # current pick: submissions/top500_v4f.csv
python -E -P luis/analysis/team_merge.py --score-2026 --ranking derived/scored_2026_v4f.csv rank_v4f v4f   # explain v4f
python -E -P luis/analysis/stats_review.py         # seed variability, Holm tests, Romano-Wolf set, power, 2026 projection (~7 min)
python -E -P luis/analysis/nested_tuning.py        # grid search inside training only (~4 min): doesn't help, keep defaults
python -E -P luis/analysis/final_build.py          # THE FINAL: root top500.csv (Chen's LoRA + LR voters)
python -E -P luis/analysis/chen_round3.py          # Chen's 3rd push on our A/B: final recipe, LoRA 30B / llmx, sensitivity (~3 min)

python -E -P luis/repo_scanner/train_export.py     # repo scanner: train + export the servable model (once, ~15s)
python -E -P luis/repo_scanner/server.py           # repo scanner: API + frontend on http://127.0.0.1:8900/repo-scan.html
```

Environment notes and known install issues: `context_Luis.md` §12.

**Repo scanner (product demo):** full install + run steps from a fresh clone are in [`repo_scanner/README.md`](repo_scanner/README.md#install-and-run-it-from-scratch).
