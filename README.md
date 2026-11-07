# Before the Breach · IBM track, Sheridan Datathon 2026

Team repo. Goal: rank the 72,670 CVEs in `scoreboard_2026.csv` so the top 500 contains as many CVEs as possible that CISA adds to its Known Exploited Vulnerabilities list by Nov 4, 2026, and explain why.

**One branch (`main`). Each teammate works only inside a folder with their name.** Rules for people and their AI assistants: [`CONTRIBUTING.md`](CONTRIBUTING.md).

| Folder | Owner | Start with |
|---|---|---|
| [`luis/`](luis/) | Luis | [`luis/context_Luis.md`](luis/context_Luis.md) |
| `chen/` | Chen (GPU box: Granite 30B/8B, LoRA, purged CV) | `chen/context_Chen.md` (to be pushed) |
| `<name>/` | other teammates | `<name>/context_<Name>.md` |

- **Team hub** (Top 500 table, rules, data map, shared notes): https://claude.ai/artifact/Np2SfMguMKySJnBfvNfnjR
- **Current fallback submission:** `luis/submissions/top500_v2.csv`, until the merged pipeline in `luis/context_Luis.md` §14 produces the final `top500.csv`.

## Data

Everyone already has the IBM data. Keep it unzipped at the repo root in `IBM-20261107T174321Z-1-001/IBM/data/data/` (git ignores it) so everyone's scripts find it at the same relative path. **Never commit it.**
