# repo_scanner · the Top 500 model, pointed at your dependencies

Working implementation of the MVP in [`luis/proposal_repo_scanner.md`](../proposal_repo_scanner.md): paste or
upload a `package-lock.json` / `requirements.txt`, we match its exact dependency versions against known CVEs
(via [OSV.dev](https://osv.dev)), and rank every match that isn't already on the CISA KEV list with the same
feature definitions as `luis/analysis/common.py` — not raw CVSS.

**Run it (one command, after the one-time step below):**
```
python -E -P luis/repo_scanner/train_export.py   # once: trains + exports the servable model (~15s)
python -E -P luis/repo_scanner/server.py         # starts the API + serves the frontend
```
Then open **http://127.0.0.1:8900/repo-scan.html**. The IBM data must be unzipped at the repo root, same as
every other script here (`IBM-20261107T174321Z-1-001/IBM/data/data/`).

## What it actually does (MVP scope — see the proposal's §10 for what's deferred)

| Step | File | How |
|---|---|---|
| Parse the lockfile | `lockfile_parse.py` | npm (`package-lock.json`, v1-v3) and PyPI (`requirements.txt`, pinned versions). Nothing else yet — that's the scope cut, not an accident. |
| Match dependencies to CVEs | `live_sources.py: osv_batch` | Live [OSV.dev](https://osv.dev) batch API. Resolves each OSV/GHSA id to its CVE alias. |
| CVE metadata | `common_live.py: local_cve_meta` | The team's local NVD-style snapshot (`vulnerabilities.csv`, 1988–2026, ~401k rows) — covers nearly any real-world CVE instantly, no live NVD call needed. Falls back to a live NVD API lookup only for ids too new to be in the snapshot. |
| "Already exploited" + history features | `common_live.py: HistoryTables` | The local CISA KEV snapshot, extended with one live fetch of the current feed for freshness (`live_sources.py: fetch_kev_live`). Same assigner/vendor/product KEV-rate formulas as `build_features.py`'s `add_history`, computed over all history to date instead of "before a fixed training year" — correct for a live tool, which should use the freshest history it has. |
| EPSS | `live_sources.py: epss_batch` | Live [FIRST.org EPSS API](https://www.first.org/epss/api). |
| Exploit-source flags | `live_sources.py: exploit_flags` | Best-effort live checks: Exploit-DB (cached CSV mirror), Nuclei templates (raw-path probe), GitHub PoC (via the `Trickest/cve` mirror). **Metasploit is always `False` in this build** — the module-metadata file is tens of MB, too slow to fetch per scan; see the proposal. |
| The text-vs-KEV features | `train_export.py` + `common_live.py` | Our own TF-IDF-vs-past-KEV-descriptions model, retrained once on all current KEV entries and exported (`model/tfidf_kev.joblib`), then scored live on each matched CVE's description. |
| Scoring | `scan.py` | The exported **LR A** (the same 56-feature logistic regression used as half of the linear voter in `final_build.py`), retrained on the 2024+2025 replicas and exported by `train_export.py`. **Not included yet**: the XGBoost tree voter, LR B (Chen's 100-feature model), and the Granite 8B LoRA — see the proposal's §10 for why. |
| Demotion | `scan.py: demoted` | A **regex stand-in** for 2 of the 4 Granite-30B demotion rules (denial-of-service-only phrasing, WordPress-plugin phrasing). The other two (`typical_attacker=unlikely_target`, `attacker_appeal=low`) need a live Granite call and aren't approximated — flagged honestly in the UI, not faked. |

## A real finding from building this

Scoring real, old dependency CVEs (Django, Flask, PyYAML, requests, ... from 2016-2019) surfaces a genuine
distribution-shift case the proposal predicted but didn't have a concrete example for yet: many pre-2017 CVEs
were filed under the generic `cve@mitre.org` CNA, before most projects had their own delegated CNA. That
assigner alone carries an enormous historical KEV count (hundreds of entries — it's the catch-all for a lot of
infamous old vulnerabilities), so `assigner_kev_all`/`assigner_kev_3y` read as very high for *any* CVE filed
that way, regardless of how exploitable that specific CVE actually is. The model was trained on 2024-2025
CVEs, where per-project CNAs are now the norm, so it never had to learn that "assigner = MITRE" is close to
an uninformative default for old software, not a strong vendor-history signal. Worth saying in the
write-up/demo rather than hiding.

## Files

| File | Role |
|---|---|
| `common_live.py` | Live ports of `common.py`'s feature logic; the local-data loaders; `build_feature_row()`, the live equivalent of a training row. |
| `live_sources.py` | Every external API call (OSV, EPSS, CISA KEV, NVD fallback, exploit-source checks). All fail soft. |
| `lockfile_parse.py` | npm / PyPI lockfile parsing. |
| `train_export.py` | Run once (or after a `git pull` that touches the IBM data) to produce `model/`. |
| `scan.py` | Orchestrates one scan; also runnable as a CLI (`python scan.py path/to/lockfile`) for testing without the server. |
| `server.py` | stdlib-only HTTP server: `POST /api/scan` + serves `luis/frontend/` statically. No new runtime dependency to install. |
| `model/` | Exported artifacts from `train_export.py`. Git-ignored (regenerate, don't commit — same convention as `luis/derived/features_*.csv`). |
| `test_requirements.txt` | A real, old `requirements.txt` with known CVEs, used to test end to end. |
| `demo_repo/` | A small, fake two-service app (Python backend + Node frontend) built for demos: real, old, known-vulnerable dependency versions, with source code that plausibly uses each one. See `demo_repo/README.md` for what to expect. |

## Known limitations (say these out loud in the demo)

- **One voter.** Tree + LR B + LoRA are not wired in. On our own backtest, one voter alone scores meaningfully
  lower than the full ensemble — treat this build's ranking as a demonstration of the *approach*, not the
  project's best number.
- **Two of four demotion rules are regex approximations**, not Granite 30B.
- **npm and PyPI only.** No Maven, Go, Cargo, RubyGems, NuGet, Composer.
- **No GitHub URL input.** Paste or upload a lockfile directly.
- **Metasploit flag is always false.**
- See `proposal_repo_scanner.md` §11 for the rest (SCA matching isn't perfect; this hasn't been validated on a
  "dependency CVEs only" cohort the way the Top 500 pipeline was validated on its replicas).
