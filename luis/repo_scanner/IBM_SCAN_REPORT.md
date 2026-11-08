# Repo scanner on four IBM open-source repos (Nov 8, 2026)

Scanned through `server.py`'s `/api/scan-github` (depth-1 clone of the default branch, every
`package-lock.json` / `requirements.txt` found, clone deleted afterwards). Raw results: `cache/gh_*.json`
(git-ignored; re-run to regenerate). Each scan took 6–10 s.

| Repo | Status | Deps scanned | CVEs matched | Already on KEV | Top action |
|---|---|---|---|---|---|
| `IBM/MAX-Image-Resolution-Enhancer` | **archived**, ~1k stars | 5 | 285 | **1** | Pillow 6.2.2 → 13.0.0 |
| `IBM/data-prep-kit` (2 example lockfiles) | active | 12 | 11 | 0 | langchain_community 0.3.1 → 0.3.27 |
| `IBM/openapi-validator` | active (pushed Nov 4) | 839 | 5 (+1 rejected id skipped) | 0 | replace `pkg` (no fix exists) |
| `IBM/node-sdk-core` | active (pushed Nov 6) | 910 | 3 (+1 rejected id skipped) | 0 | glob 13.0.6 → 14.0.2 |

## Findings and recommendations

**1. `IBM/MAX-Image-Resolution-Enhancer` — the only urgent one.**
- `Pillow==6.2.2` is affected by **CVE-2023-4863** (libwebp heap overflow), which is **on the CISA KEV list**:
  it has been exploited in the wild. Nothing to predict here.
- The model's top non-KEV pick is **CVE-2023-50447** (arbitrary code execution through `PIL.ImageMath.eval`,
  public PoC, p = 0.97).
- `tensorflow==2.6.0` carries 214 CVEs, mostly input-validation crashes in `tf.raw_ops`; top pick CVE-2022-29206 (p = 0.87).
- **Recommendation:** the repo is archived, so IBM won't patch it. Anyone still deploying this model server should
  fork it and upgrade **Pillow to ≥ 13.0.0** and **TensorFlow to ≥ 2.12.1** (the versions that clear every known
  CVE for each), or stop exposing it. If only one change is possible, Pillow first: it's the one with an exploited CVE.

**2. `IBM/data-prep-kit` — medium, examples only.** The CVEs come from `examples/agentic` and `examples/rag-pdf-1`,
not the core library.
- Top pick **CVE-2025-6984** (`langchain_community` XXE in the EverNote loader, p = 0.83) → upgrade to 0.3.27.
- **CVE-2024-47081** (`requests` can leak `.netrc` credentials to a crafted URL, p = 0.57) → upgrade to ≥ 2.32.4
  (2.33.0 clears both requests CVEs).
- The other four (black, langchain, langgraph, langchain-openai) score low (p ≤ 0.25); fold them into the next
  routine dependency bump. Note `black`'s CVE only matters if the repo uses Black's GitHub Action.

**3. `IBM/openapi-validator` — one item without a fix.**
- **CVE-2024-24828** in `pkg@5.8.1` (native code written to a shared `/tmp/pkg/` directory: local privilege
  escalation, p = 0.79). There is **no fixed version**: `pkg` is deprecated. **Recommendation:** replace it (e.g.
  Node's single-executable applications or another bundler), or confine it to build machines where no other user
  shares `/tmp`.
- glob, minimatch, ip-address: low score (p < 0.1), patch versions available — routine bump.

**4. `IBM/node-sdk-core` — clean.** 910 dependencies, 3 CVEs, all low score with patch releases available
(glob → 14.0.2, ip-address → 10.7.4). Routine bump; nothing urgent.

## What this run fixed in the scanner itself

Testing on real repos surfaced four bugs, all fixed before the numbers above:
1. **Every old CVE scored 1.0.** Cause: `age_days` is 0–30 in training but ~2,300 for a 2020 CVE; the linear model
   extrapolated. Live features are now clipped to the training range (`model/feature_bounds.json`).
2. **Wrong upgrade target.** `fixed_in` returned the last range listed in the advisory (Django 1.11.0 → "1.8.19").
   It now returns the smallest fix above the installed version, for that package only.
3. **Rejected CVE ids ranked near the top** (CVE-2026-93753, p = 0.69). Rejected ids are now skipped and counted.
4. **Crash on CVEs newer than the local NVD snapshot** (date came back as text from the live NVD fallback). Fixed.

## Caveats to state with these results

- The model was trained to answer "will a *newly published* CVE reach KEV?". Most dependency CVEs are years old,
  so their scores are an extrapolation that we clip but haven't validated. Treat the ranking as triage order,
  not a calibrated probability.
- One voter only (LR A). Two of four Granite demotion rules are regex stand-ins. Metasploit flag is always off.
- Only lockfiles on the default branch are seen, up to 8 per repo, npm and PyPI only.

## Second sweep (Nov 8): 18 archived IBM repos, picked by stars, for demo candidates

The 5 most impressive are now one-click examples in `repo-scan.html`. 9 of the 18 had no `package-lock.json`/`requirements.txt` on the default branch (yarn, no lockfile, etc.) and are skipped.

| Repo | Deps | CVEs | On KEV | p ≥ 0.7 | Top action |
|---|---|---|---|---|---|
| `IBM/MAX-Object-Detector` | 3 | 266 | **1** (Pillow 8.3.2) | 18 | Pillow → 13.0.0 (52 CVEs), tensorflow → 2.12.1 |
| `IBM/ipfs-social-proof` | 1,506 | 270 | 0 | 116 | tar-fs 1.16.3 → 1.16.6 (CVE-2024-12905, p 0.97), qs → 6.16.0 |
| `IBM/watson-speech-translator` | 730 | 145 | 0 | 49 | routine npm bump, qs/underscore first |
| `IBM/watson-banking-chatbot` | 333 | 102 | 0 | 22 | qs 6.7.0 → 6.16.1 (CVE-2022-24999), underscore → 1.13.8 |
| `IBM/watson-vehicle-damage-analyzer` | 324 | 78 | 0 | 26 | npm bump |
| `IBM/watson-stock-advisor` | 347 | 68 | 0 | 29 | qs, underscore, lodash → 4.17.21 |
| `IBM/MAX-Object-Detector-Web-App` | 232 | 56 | 0 | 16 | npm bump |
| `IBM/pytorch-seq2seq` | 0 | 0 | 0 | 0 | requirements.txt has no pinned versions |

All are archived, so these are "if you still deploy it, fork and upgrade" findings, not open IBM vulnerabilities.

## Public Sheridan Datathon 2026 repos (Nov 8)

Searched GitHub for "sheridan datathon", scanned the 11 most recent. Most are fresh 2026 projects with current
dependencies (0 CVEs) or no lockfile. One stands out and is now a one-click example:

- **`seanyzlee/sheridan-datathon`**: `web/package-lock.json`, 227 deps, 23 CVEs, **1 on CISA KEV**: `vite@6.0.3` has
  CVE-2025-31125 (dev-server arbitrary file read; fix in 6.0.13). Top non-KEV picks are also vite (CVE-2025-30208 p 0.98).
  Upgrade plan: vite → 6.4.4, react-router → 7.18.0. Only matters if the vite dev server is exposed to a network.
- `Pranvos/GraniteWall---Sheridan-Datathon-2026`: 285 deps, 0 CVEs, the "clean" example.
- Our own repo (`Solarcemir/Sheridan_Datathon_DATACENTERS`) is also an example: it scans `luis/repo_scanner/demo_repo`.
