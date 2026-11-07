# How to contribute (for teammates and their AI assistants)

Repo: https://github.com/Solarcemir/Sheridan_Datathon_DATACENTERS · **one branch: `main`**.

## 1. Your folder

- All your work goes in a folder at the repo root named after you, lowercase first name: `luis/`, `chen/`, `<yourname>/`. Create it if it doesn't exist.
- Never edit another teammate's folder or the root files (`README.md`, `CONTRIBUTING.md`, `.gitignore`). If something there needs to change, say so in the team chat.

## 2. Never commit

- **The IBM hackathon data. Everyone already has it.** That means the `IBM-*/` folder, `data.zip`, and these files anywhere in the repo: `vulnerabilities.csv`, `affected_products.csv`, `references.csv`, `known_exploited.csv`, `exploit_signals_2024.csv`, `exploit_signals_2025.csv`, `weakness_types.csv`, `scoreboard_2026.csv`. The root `.gitignore` blocks them; never bypass it with `git add -f`.
- Any file over 50 MB (GitHub rejects anything over 100 MB). Shrink it (Parquet, float16, PCA), split it, or leave it out and say where it lives.
- Model weights, LoRA adapters and checkpoints (`*.safetensors`, `*.pt`, `*.pth`, `*.gguf`, `*.ckpt`). Describe them in your context file instead.
- Secrets: API keys, Hugging Face tokens, passwords, `.env` files.
- Virtual environments and caches (`venv*/`, `.venv/`, `__pycache__/`).

## 3. Do commit (inside your folder)

- The code that reproduces your results.
- **Any derived dataset others can reuse**, keyed by `cve_id`, Parquet preferred (CSV is fine): feature tables, out-of-fold scores, LLM extraction outputs, small embeddings, CV folds, KEV group files, backtest rankings.
- Result tables and charts.
- Candidate rankings: `<yourname>/submissions/top500_<short_description>.csv`, exactly one column `cve_id`, 500 rows.
- `<yourname>/README.md` with a folder map and the run order.
- **`<yourname>/context_<Yourname>.md`** (see section 4). This is the most important file.

## 4. What `context_<Yourname>.md` must contain

Write it for a teammate who never saw your session. Be very specific, use tables, and make every number point to a file in your folder.

1. **TL;DR:** best result, the ranking file you'd submit, what you'd do next.
2. **Data views:** how you built each cohort (which CVEs, snapshot month, how signals were joined, what was dropped and why) and row counts.
3. **Labels:** exact definition and cutoff date, e.g. "added to KEV after the end of the snapshot month and on or before Nov 4 of the publish year", plus positive counts.
4. **Leakage rules:** the exact columns you used, used with care, and excluded, each with a reason, plus the KEV-history cutoff you applied.
5. **Evaluation protocol:** splits or CV folds; whether any test period is scored by a model that saw later data; the K you used; metrics; seeds; confidence intervals. **Always also report hits at scoreboard density** (K = 500 × test rows / 72,670) with judge-aligned labels (KEV by Nov 4), so numbers compare across folders.
6. **Features:** every feature with its definition and why it is known at the snapshot.
7. **Models:** algorithms, hyperparameters, and how they were tuned (only inside training folds?).
8. **Every experiment with its numbers,** in tables, including the ones that failed and the baselines on the same split (CVSS first, EPSS first, exploit code then EPSS).
9. **IBM technology:** model names and versions, how they were served, and what each one added, measured.
10. **Findings and insights** with numbers (lifts, KEV rates, counts).
11. **Agreements and disagreements** with `luis/context_Luis.md`, `chen/`, and others, citing numbers.
12. **Environment:** hardware, Python and package versions, install gotchas, exact run order.
13. **File index:** claim → file.

## 5. Git workflow

```
git checkout main
git pull --rebase origin main
# work only inside <yourname>/
git status                      # check: nothing from IBM, nothing huge, only your folder
git add <yourname>/
git commit -m "<yourname>: <what changed>"
git pull --rebase origin main
git push origin main
```

No new branches and no force pushes. If a push is rejected, `git pull --rebase origin main` and push again. Since everyone writes only to their own folder, conflicts shouldn't happen.

## 6. Shared conventions

- IBM data path, relative to the repo root: `IBM-20261107T174321Z-1-001/IBM/data/data/`. Build your paths from the script's location so everyone's code runs on everyone's machine.
- `cve_id` is the key in every shared table.
- Don't use columns that leak the answer: `nvd_status`, `n_references`, `n_affected_cpes`, `last_modified`, `cvss3_source`, `affected_products.csv` presence, news/threat-intel links (see `luis/context_Luis.md` §3).
