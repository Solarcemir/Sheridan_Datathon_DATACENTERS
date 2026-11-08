# Proposal: turn our model into a repo-scanning tool

**Status:** proposal, not started. For the team to review and decide on before anyone builds it.
**Author:** Luis. **Companion docs:** `luis/context_Luis.md` (the model this reuses), `chen/context_Chen.md` (Chen's pieces this needs), `luis/frontend/breach-story.html` (the UI this extends).

## 1. The idea in one paragraph

Right now our pipeline ranks the 72,670 CVEs in the 2026 scoreboard. The same pipeline can rank **any list of CVEs**, including the ones affecting a real codebase's dependencies. So: a developer points us at their repo (really, just its lockfiles), we find which known CVEs affect their exact dependency versions, and instead of showing them in raw CVSS order (what every other tool does), we run them through our trained ensemble and hand back "patch these first, and here's why" — the same `rank`, `reason`, and Granite explanation columns already in `top500_final_explained.csv`, just for their stack instead of the scoreboard.

This is a repositioning, not a rewrite: the ranking engine already exists (`luis/analysis/final_build.py`). What's new is everything in front of it — getting from "a repo" to "a table of CVE rows with our 156 feature columns filled in."

## 2. Why this is worth building

- **The problem is real and well known**: SCA tools (Dependabot, Snyk, `npm audit`, Trivy) find the CVEs in your dependencies, but rank them by CVSS, and CVSS is a *severity* score, not a *will-this-get-exploited* score. Our whole project's headline finding is that CVSS-first ranking gets 5 hits on our practice exam; our ensemble gets 32. A tool that re-ranks a real dependency list by the same signal is a direct, demonstrable use of that finding.
- **It's a natural demo**: "here's our model finding the 500 CVEs worth watching in 2026" is abstract to a judge. "Here's our model looking at *numpy*'s own CVE history and telling you which one actually mattered" is concrete.
- **Almost everything it needs, we already built**: the feature definitions, the trained models, the demotion filters, the explanation generator, and now the dark pipeline UI to show it happening.

## 3. What it is not

- Not a static analyzer. It never reads source code, only dependency manifests/lockfiles.
- Not a new CVE database. It uses existing ones (below) to find *which* CVEs apply; our job starts after that, at *ranking* them.
- Not a real-time scanner for brand-new, undisclosed vulnerabilities. It only works on CVEs that are already public, same as the whole project.

## 4. User flow

1. User pastes a repo URL, or uploads just their lockfile(s) (`package-lock.json`, `requirements.txt` / `poetry.lock`, `go.sum`, `Cargo.lock`, `pom.xml`, `Gemfile.lock`, ...). **We never need the source code**, which is also the easiest privacy story to tell in a demo: say this out loud before anyone uploads anything.
2. We resolve "package X at version Y" → "these CVE/GHSA IDs affect it" (section 6).
3. For every matched CVE that **isn't already on the CISA KEV list**, we build the same feature row our training pipeline builds, and score it with the trained ensemble (section 7).
4. We apply the same Granite-30B demotion rules and the same per-assigner-month cap logic (reused as-is from `final_build.py`, reinterpreted as "per-vendor" since a repo's CVE count is small).
5. We render it in the existing `breach-story.html` visual language: the same pipeline-as-hero page, except the "CVE snapshot" node now reads "your repo: 340 dependencies, 58 CVEs found" instead of "72,670 CVEs · 2026", and the "book" modal is *their* ranked list with *their* recommended upgrades, not the scoreboard's.

## 5. Architecture

```mermaid
flowchart LR
    A[Lockfiles<br/>package-lock.json, requirements.txt, go.sum, ...] --> B[osv-scanner<br/>package@version -> CVE/GHSA IDs]
    B --> C[CVE enrichment<br/>NVD + EPSS + KEV feed + exploit-source lookups]
    C --> D[Granite 30B extraction<br/>cached by CVE id]
    D --> E[Feature row<br/>same 156 columns as training]
    E --> F[Trained ensemble<br/>XGBoost + LR A + LR B + LoRA 8B]
    F --> G[Demotion filters + vendor cap]
    G --> H[Ranked, explained list<br/>+ upgrade target per row]
```

Nothing in boxes A–D exists in our repo yet. Everything from E onward is `luis/analysis/final_build.py`, unmodified in logic, just fed a different input table.

## 6. Finding the CVEs: don't write a dependency parser

Every ecosystem's lockfile format is its own small nightmare (npm's three lockfile versions, Python's four packaging tools, Maven's transitive resolution...). Don't reinvent this. Use **[OSV-Scanner](https://github.com/google/osv-scanner)** (Google, Apache-2.0, a single static binary): point it at a repo or a lockfile, it outputs JSON with every `(package, version, ecosystem)` → `(OSV/CVE/GHSA id, affected range, fixed version)` match, across npm, PyPI, Maven, Go, Cargo, RubyGems, NuGet, Packagist, and more. It's also what it reads that matters for the privacy pitch: manifests and lockfiles only, never source.

Its underlying data source is **[OSV.dev](https://osv.dev)** (free, no key, has a batch query endpoint `POST /v1/querybatch` if we ever want to call it directly instead of shelling out to the binary). OSV entries carry a GHSA id and, for most, the matching CVE id — we need the CVE id specifically, since that's what our model and the KEV list key on.

## 7. Filling in our 156 columns for a live CVE

This is the real work. Walking our own feature groups (`luis/analysis/common.py`, `chen/scripts/stack.py`) and where each comes from live:

| Group (as in the pipeline UI) | Columns | Live source | Notes |
|---|---|---|---|
| Exploit signals | `epss`, `epss_pct`, `exploitdb`, `metasploit`, `nuclei`, `poc_github`, `any_flag` | **EPSS**: [FIRST.org EPSS API](https://www.first.org/epss/api) (`api.first.org/data/v1/epss?cve=...`), free, no key. **Exploit-DB**: [exploitdb CSV mirror](https://gitlab.com/exploit-database/exploitdb) grep by CVE ref. **Metasploit**: grep `metasploit-framework`'s module metadata for the CVE ref. **Nuclei**: grep [nuclei-templates](https://github.com/projectdiscovery/nuclei-templates) for the CVE id in template metadata. **GitHub PoC**: reuse [`Trickest/cve`](https://github.com/trickest/cve), which already maintains CVE → PoC-repo mappings, instead of re-deriving it via GitHub code search. | `epss_pct` needs a percentile within *some* cohort — for a single repo scan there's no natural cohort, so rank within "all CVEs scanned in this run," or skip the percentile and use the raw EPSS score, which our model treats as the primary signal anyway. |
| Description text | `txt_*` regex flags, `text_kev_lr`, `text_kev_sim` | NVD's `description` field for the regex flags (trivial, same regexes already in `feature_dictionary.csv`). `text_kev_lr`/`text_kev_sim` need the **same TF-IDF model we trained on past KEV descriptions** — export it (it's small, scikit-learn `TfidfVectorizer` + `LogisticRegression`, pickle it) and run it on the live description. | No new modeling, just serialize what's already trained. |
| Release batch | `n_same_day`, `sib_flags`, `n_adjacent_ids`, `twins`, `log_batch_size`, ... | These describe how the CVE was *published* (same-day siblings from the same assigner). Computable from NVD's bulk CVE feed for that assigner/day — a live lookup, not something the repo tells us. | Lower priority: these features matter for *batch* vulnerabilities (e.g. Oracle's 1,097-CVE day), rare for a single dependency; fine to zero-fill in v1 and note it. |
| Vendor history | `assigner_kev_3y`, `kev_product_hist`, `kev_vendor_hist`, ... | **CISA KEV feed**: `cisa.gov/.../known_exploited_vulnerabilities.json`, free, no auth, refreshed daily. Cross-reference by vendor/product/CWE/assigner exactly like `add_chen_lite` / `meta_features` already do — same code, live feed instead of the static training-year CSV. | Direct reuse of existing logic. |
| Granite 30B reading (55 cols) | `llm_*` one-hots (product_type, impact, typical_attacker, attacker_appeal, ...) | Call the same prompt schema used to build `chen/processed/llm_extract.parquet`, live, on the CVE's description. **Cache by CVE id** — the description never changes, so a CVE scored once is scored forever. | The one real new *infra* dependency: needs Chen's prompt + a reachable Granite 4.2 30B endpoint. See §8. |
| Granite Embedding | clusters, nearest-KEV similarity | Embed the live description with the same embedding model, compare against the precomputed KEV-description vector index. | Also cacheable by CVE id; the KEV side of the index only grows, re-embed it on each CISA KEV update (daily cron, cheap). |
| CVSS, CWE & dates | `cvss`, `cwe_unknown`, `days_to_cutoff`, ... | Straight from NVD's CVE record. `days_to_cutoff` doesn't apply to a live tool (there's no fixed judge deadline) — drop it or repurpose as "days since publish." | |

**Net read**: nothing here needs new machine learning. It needs API calls, two small serialized artifacts (the TF-IDF model, the embedding index), and porting existing Python functions from "read a training CSV" to "look up one CVE live." That's a real but bounded amount of plumbing, not a research problem.

## 8. Serving the trained models

What has to be exported from training to run outside a notebook, concretely:

| Model | What to export | Where it lives today |
|---|---|---|
| XGBoost (tree voter) | `model.save_model("xgb.json")` — one call, standard XGBoost serialization | trained fresh each run in `final_build.py`; needs to be trained once and saved |
| LR A (Luis, 56 features) | scaler means/scales + coefficient vector + intercept (a few hundred floats; dump as JSON or `joblib`) | `luis/analysis/common.py: m_lr` |
| LR B (Chen, 100 features) | same, plus the exact column order/transform code in `chen/scripts/stack.py` | Chen's box |
| Granite 4.2 8B LoRA | needs the adapter weights hosted somewhere callable (Chen's GPU box, or wherever it's deployed) — this is the heaviest serving dependency | Chen's box; **flag as v2, not v1** (see §10) |
| Granite 4.2 30B extraction | needs a reachable inference endpoint (watsonx.ai or Chen's box) + the exact prompt | Chen's prompt/schema |
| Demotion filters + cap | just the 4 threshold rules + the cap value — copy as constants, no export needed | `luis/analysis/postprocess.py` |

**We should ask Chen directly**: can the 8B LoRA and the 30B extraction be called from a small always-on endpoint (even a slow one, for a live demo), or do they only exist as one-off batch jobs on the GPU box? That answer decides whether v1 includes the LLM voter at all.

## 9. API contract (sketch)

```
POST /scan
  { "lockfiles": [{ "filename": "package-lock.json", "content": "<raw text>" }] }
  -> { "scan_id": "..." , "status": "running" }

GET /scan/{scan_id}
  -> {
       "status": "done",
       "summary": { "dependencies": 340, "cves_found": 58, "already_in_kev": 3, "scored": 55 },
       "rows": [
         {
           "rank": 1,
           "cve_id": "CVE-2023-xxxxx",
           "package": "left-pad@1.1.2",
           "fixed_in": "1.3.1",
           "epss": 0.41,
           "flags": { "exploitdb": false, "metasploit": false, "nuclei": true, "poc_github": true },
           "text_kev_lr": 0.81,
           "nearest_kev": "CVE-2020-xxxxx",
           "kev_sim": 0.92,
           "granite": { "product_type": "library_or_framework", "impact": "remote_code_execution", "attacker_appeal": "high" },
           "reason": "Public exploit code (Nuclei, GitHub PoC); resembles KEV entry CVE-2020-xxxxx (similarity 0.92); ...",
           "recommendation": "Upgrade to 1.3.1"
         }
       ]
     }
```

Same row shape as `reports/top500_final_explained.csv` plus two new fields that only make sense per-repo: `package` (which dependency, at which installed version) and `fixed_in` (straight from OSV-Scanner's output — it already reports the fixed version, we don't compute this).

## 10. MVP vs. stretch (be honest about what fits before the deadline)

**MVP (buildable in a focused day or two, one person):**
- OSV-Scanner wired up, lockfile upload only (no live GitHub clone — less surface area, faster demo).
- Feature rows limited to: exploit flags (EPSS + the 4 flags), CVSS/CWE/description regex flags, KEV vendor/product history. Skip `llm_*` and the LoRA voter.
- Scoring with **tree + LR A only** (two voters, no LLM) — defensible, because our own `stats_review.py` already showed the LLM voter's gain is the one still inside noise on some checks, and because it removes the one serving dependency we don't control (Chen's endpoints).
- Demotion filters applied using a **rule-based stand-in** for the 4 Granite flags (e.g. regex for "denial of service" in the description) instead of a live 30B call — note in the demo that this is an approximation.
- Output: the existing `breach-story.html` "book" modal, repointed at the scan's own rows instead of `story_data.json`.

**v2 / stretch, only if MVP works and there's time:**
- Live Granite 30B extraction (needs Chen's endpoint) → real demotion filters, real `llm_*` features, real LR B.
- The 8B LoRA as the third voter.
- GitHub URL input (clone, locate lockfiles automatically) instead of manual upload.
- Caching layer (a real cache of Granite answers + embeddings by CVE id, so a second scan of a similar repo is instant).

## 11. Risks and things to say honestly in the write-up/demo

- **SCA matching is never perfect.** OSV-Scanner can miss transitive dependencies in some ecosystems and occasionally over-matches version ranges. We inherit whatever error rate it has; we don't improve it.
- **Distribution shift.** Our model was trained and validated on *all* NVD CVEs published in a given month, not specifically on CVEs that happen to affect open-source dependencies. The patterns it learned (assigner history, batch releases, description phrasing) should transfer reasonably, but we have not validated hit-rate on a "dependency CVEs only" cohort — say this plainly rather than imply the 32/75 backtest number carries over unchanged.
- **EPSS/KEV staleness.** Unlike the training setup (frozen snapshots, carefully leak-checked), a live tool's EPSS and KEV lookups are *whatever they are today* — which is correct and intended for a live tool, but worth stating so nobody confuses it with the backtest methodology.
- **Granite coverage gaps.** Not every CVE will have been through Chen's 30B extraction yet; cache misses need a sane fallback (score without the `llm_*` block, same as "Granite coverage" gaps we already measured during training).

## 12. Rough build plan

| Step | Owner (suggested) | Depends on |
|---|---|---|
| Export XGBoost + LR A as servable artifacts | Luis | none |
| Wire up OSV-Scanner, parse its JSON into a dependency→CVE table | Luis | none |
| Live NVD + EPSS + KEV-feed lookups, port `add_chen_lite`/`meta_features` to live mode | Luis | none |
| Rule-based stand-in for the demotion filters | Luis | none |
| Scoring endpoint (tree + LR A, filters, cap) | Luis | the above |
| Frontend: new "scan" entry point reusing `breach-story.html`'s pipeline + modal UI | Luis | scoring endpoint |
| Can the 30B extraction / 8B LoRA be called live? | **ask Chen** | — |
| LR B + real Granite filters + LLM voter (if Chen's answer is yes) | Chen + Luis | Chen's endpoint |

## 13. Open questions for the team

1. Do we actually build this before the deadline, or pitch it as "future work" in the write-up/video with just this plan and maybe a static mockup?
2. If we build it: MVP scope from §10, or something smaller still (e.g. a CLI that prints a ranked table, no web UI at all)?
3. Can Chen's 30B/8B endpoints take live, one-off calls during a demo, or are they only set up for batch jobs on his box?
4. Do we scan a few real, recognizable open-source repos for the demo (more convincing) or let the judges paste their own (riskier live-demo, more impressive if it works)?
