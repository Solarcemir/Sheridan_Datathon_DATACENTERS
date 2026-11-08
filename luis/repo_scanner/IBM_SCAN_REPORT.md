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
