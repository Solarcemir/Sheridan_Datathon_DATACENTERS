# Project context · IBM "Before the Breach" (Sheridan Datathon 2026)

Luis's handoff document (folder `luis/`; paths below are relative to it unless they start with `luis/`). Last updated Nov 7, 2026, ~5:15 pm. Written so a teammate with a stronger machine can pick up the work without re-deriving anything. Every number below comes from a script in `luis/analysis/` and is backed by a file listed in **§13 Evidence index**.

- **Team hub (web):** https://claude.ai/artifact/Np2SfMguMKySJnBfvNfnjR (ranked Top 500 table with Granite reasons, rules, CSV/column explorer, data map, shared notes; code in `site/breach-hub.html`)
- **Best ranking right now:** `submissions/top500_v2.csv`, with explanations in `reports/top500_v2_explained.csv` (see §9)

---

## 0. TL;DR

1. **Task:** pick 500 of 72,670 CVEs (published Jan–Sep 2026). Score = how many CISA adds to KEV by Nov 4, 2026. Official baselines: 6–8 hits.
2. **We built a local exam** that copies the benchmark on 2024 and 2025 data (the "replicas"), because the 2026 answer is secret. Everything is measured as hits in the top 500 on those replicas.
3. **Two directions:** A = train 2024 → test 2025 (realistic, past predicts future). B = train 2025 → test 2024 (robustness only, uses future data). Report A as the headline, A+B to pick between methods.
4. **Best clean result on this side:** LightGBM on leak-free features with a TF-IDF text model, then **at most 5 CVEs per assigner per month**: **A = 23, B = 12** in the top 500 of each year (vs exploit+EPSS 17 / 8, CVSS 6 / 1). **At the real scoreboard density** (top 243 of 35k, top 198 of 29k) it's **A 16 + B 7 = 23 vs 20** for exploit+EPSS, and pooled like Chen's CV it's 23 vs 19. **Chen's GPU-box pipeline (LR + LoRA Granite 8B) gets 28–30 on a comparable set, so it's the stronger core.** See §14.
5. **Noise is ±3 hits.** The same model gave 20 or 23 in A just by changing CPU thread count. Don't trust gaps under ~4.
6. **Key insights:** the description text is the strongest single signal (AUC 0.91); 58% of exploited CVEs have no public exploit signal; NIST status columns leak the answer; a single vendor patch batch (Oracle, 1,097 CVEs on one day) can flood the top 500, and capping per assigner fixes it.
7. **IBM Granite:** embeddings give great clusters (which technologies get attacked) but don't beat TF-IDF for ranking. Granite Instruct's yes/no answers carry signal: internet-facing 1.75× and enterprise 1.85× the KEV rate, and **0 of 67 "niche product" CVEs reached KEV**. Re-ranking with them adds +2 without the cap, but nothing once the cap is on (34 vs 35). Granite also writes the product, category and reason for every CVE in our top 500.
8. **The top-500 boundary is unstable:** two runs of the same method (`top500_v2` vs `top500_v1_cap5`) share only 409 of 500 CVEs, from LightGBM randomness alone.
9. **Next on a strong PC:** average many runs/seeds to stabilize the boundary, bigger Granite models and more questions on more candidates, bootstrap CIs, and a smarter cap (exempt multi-vendor CNAs like MITRE/VulnCheck/GitHub). See §11.

---

## 1. The challenge

- **Input:** `scoreboard_2026.csv`, 72,670 CVEs published Jan–Sep 2026 that were **not** on KEV at the end of their `snapshot_month`. Columns: `cve_id, published, snapshot_month, epss, exploitdb, metasploit, nuclei, poc_github` (signals as of the end of the snapshot month).
- **Output:** `top500.csv`, one column `cve_id`, 500 rows.
- **Score:** number of those 500 added to KEV through **Nov 4, 2026**. Only the judges know it.
- **Rule:** use only what was known at the end of each CVE's `snapshot_month`. Use ≥1 IBM technology. State the columns used in the write-up.
- **Judging:** Discovery 35% · Data Use 25% · Communication 25% · Impact 15%.
- **Devpost:** 2–3 min video, repo, short write-up (with columns used), links to outside data, `top500.csv`.

| Official baseline (README) | Hits in the top 500 |
|---|---|
| Highest CVSS 3 first | 7 |
| EPSS first | 6 |
| Public exploit code first, then EPSS | 8 |

"CVSS found 7" means: sort all 72,670 by CVSS, take the top 500, and 7 of those reached KEV. Nobody knows which 7. It's a bar to beat, not a hint.

## 2. The data (repo root: `IBM-20261107T174321Z-1-001/IBM/data/data/`, ~850 MB, never committed)

| File | Rows | Grain | Use |
|---|---|---|---|
| `scoreboard_2026.csv` | 72,670 | 1 CVE | What we rank |
| `vulnerabilities.csv` | 401,082 | 1 CVE (1988 → Nov 6, 2026) | Description, assigner, CVSS, CWE, NIST status |
| `affected_products.csv` | 3,155,934 | 1 product/version range per CVE | Vendor/product (only ~50% of the scoreboard has rows, and having rows leaks) |
| `references.csv` | 1,360,487 | 1 link per CVE | Tags (Patch, Exploit, Advisory…). News/threat-intel links leak |
| `known_exploited.csv` | 1,484 | 1 KEV entry (Nov 2021 → Dec 31, 2025) | **Labels** via `date_added` |
| `exploit_signals_2024.csv` | 3,023,284 | 1 CVE per month, 2024 | EPSS + flags at each month-end |
| `exploit_signals_2025.csv` | 3,440,450 | 1 CVE per month, 2025 | Same; same EPSS model as 2026 |
| `weakness_types.csv` | 1,374 | 1 CWE | CWE names |

Joins: everything on `cve_id`; `weakness_types.cwe_id` ↔ `vulnerabilities.cwe_ids` (split on `;`); signals on `cve_id + month`.

**Data facts we measured:**
- KEV additions by year: 2021 311 · 2022 555 · 2023 187 · 2024 186 · 2025 245. 334 with known ransomware use. Top vendors: Microsoft 350, Apple 86, Cisco 82, Adobe 76, Google 67, Oracle 42, Apache 38, Ivanti 30, VMware 26, D-Link 25. KEV is 72% NETWORK attack vector, 71% no privileges, 73% no user interaction; median CVSS 8.8.
- 2026 scoreboard: EPSS present for 71,239; median **0.0018**, p99 0.0155; only **66** with EPSS > 0.1 and **6** > 0.5. Flags: GitHub PoC 1,527 · Nuclei 167 · Exploit-DB 37 · Metasploit 15 · any flag 1,645. CVSS 3 present 91.3%, CVSS 4 31.5%.
- 2026 `nvd_status`: Analyzed 31,265 · Deferred 27,207 · Modified 5,118 · Awaiting Analysis 5,108 · Received 3,614 · Undergoing 358. Only 50.4% have products.
- 2026 assigners with most CVEs: GitHub 10,894 · (UUID) 7,147 · VulnCheck 6,792 · VulDB 5,355 · Patchstack 4,072 · MITRE 3,330 · Wordfence 3,114 · Oracle 3,075 · Microsoft 2,941 · Chrome 2,928.

## 3. Leakage (what we can and cannot use)

The NIST files are **today's** versions. NIST analyzes exploited CVEs first, and adds references/products after attacks.

| Light | Columns | Why |
|---|---|---|
| Safe | `epss`, `exploitdb`, `metasploit`, `nuclei`, `poc_github`, `published`, `snapshot_month`, `description`, `assigner`, 2024–25 signals, `weakness_types` | Known at publication or at the snapshot |
| Careful | CVSS 3/4 and its parts, `cwe_ids`, `affected_products`, `references` tags/domains | Might have been filled in by NIST later. Declare and test with/without |
| Leak | `nvd_status`, `n_references`, `n_affected_cpes`, `last_modified`, `cvss3_source`, blank scores, news/threat-intel links | Give the answer away |

**Evidence of leakage (2025 backtest):** 100% of the 52 positives have `nvd_status = Analyzed` vs 50.1% of negatives; 100% of positives have products vs 64.9% of negatives. `last_modified` is mid-2026 or later on every row.

**How much leakage inflates our model:** LightGBM with the leak columns gets A 22–24 / B 10–11 vs 20–23 / 7–9 without. Only ~2–4 hits. Good news for the write-up: the clean model loses little.

The main feature files (`derived/features_{year}.csv`) are **leak-free by construction**; leak columns live in `derived/leak_only_{year}.csv` and are only merged for the demo row.

Other traps: EPSS changed models in March 2025 (use the percentile within each month, never raw values across years); Feb 2025 is missing ~14,500 CVEs in the signals; ~11,800 signal IDs are rejected CVEs not in `vulnerabilities.csv`; 62 CVEs joined KEV before their publish date; the ID year isn't the publish year (CVE-2024-13159 was published in 2025); June 2026 has no signals so June CVEs use July as `snapshot_month`.

## 4. Evaluation methodology (read this before changing anything)

### 4.1 Replicas: how we rebuild the exam for past years

For year Y (2024, 2025) and each month m:
1. **Cohort:** CVEs published in month m (`vulnerabilities.published`) that were **not on KEV at the end of m** (`known_exploited.date_added <= month_end` → excluded). This is exactly how the scoreboard was built.
2. **Signals:** EPSS + 4 flags from `exploit_signals_Y.csv` for month m (blank EPSS → 0, plus an `epss_missing` flag).
3. **Labels:**
   - `y` = joined KEV after the end of m and on or before **Dec 31 of Y** (used for training).
   - `y_nov4` = joined KEV after the end of m and on or before **Nov 4 of Y** (the judges' cutoff day; used for evaluation).
4. **History features** for year Y only use KEV entries added **before Jan 1 of Y**.

| Replica | Months | Rows | `y` positives | Test positives (Jan–Sep, `y_nov4`) |
|---|---|---|---|---|
| 2024 | Jan–Nov 2024 | 36,441 | 40 | 27 (in 28,839 rows) |
| 2025 | Jan–Nov 2025 | 42,563 | 55 | 48 (in 35,281 rows) |
| 2026 | scoreboard | 72,670 | unknown | unknown |

Why the label window matters: a January CVE has ~10 months to reach KEV; a September one ~5 weeks. The feature `days_to_cutoff` encodes this.

### 4.2 How hits@500 is computed

```
score every row of the test cohort (Jan–Sep only)
sort descending, take the first 500
hits@500 = number of those 500 with y_nov4 == 1
```
A CVE counts once: in the month it was published, with that month's signals. A CVE that shows up in later signal months is not a new candidate. "Missed" = positive ranked 501+.

### 4.3 A and B: two practice exams

- **A = train on the 2024 replica → test on 2025 Jan–Sep.** The realistic one: learn from the past, predict the future. **Headline number.**
- **B = train on the 2025 replica → test on 2024 Jan–Sep.** Uses future data to predict the past, so it is *not* a valid forecast. It only checks that a method isn't tuned to one year.
- We pick methods by **A+B**, break ties by A. With 48 and 27 positives, one year alone can mislead by luck.
- Note: 2024 EPSS uses the old model, and B has fewer positives, so B numbers are naturally lower.

### 4.4 Noise and selection bias (important)

- **Same model, different CPU thread count → A went from 23 to 20.** LightGBM's multithreaded histograms aren't bit-identical. We average 5 seeds, and it's still ±3. Treat gaps < 4 as ties.
- Every time we pick the best of many variants on A/B, the winner is optimistic. The hand heuristic was designed while looking at 2025, so its A score is unfair (only B is fair for it).
- The 2025 test has 35k rows and the real scoreboard has 72k. Taking the top 500 is stricter in 2026 (0.69% vs 1.4%), so absolute numbers won't transfer. **For numbers comparable with Chen's CV and with the real task, use K = 500 × rows / 72,670** (243 for the 2025 test, 198 for 2024). All A/B tables in §7–§8 use top 500 per year unless noted; density-matched numbers are in §14.1.

## 5. Features (`derived/feature_dictionary.csv`, built by `analysis/build_features.py`)

| Group | Risk | Features | How |
|---|---|---|---|
| signals | safe | `epss`, `epss_pct`, `epss_missing`, 4 flags, `n_flags`, `any_flag` | `epss_pct` = percentile within the snapshot month |
| time | safe | `age_days`, `days_to_cutoff` | days from publish to month-end; days from month-end to the label cutoff |
| text | safe | 18 regex flags: `txt_rce`, `txt_unauth`, `txt_in_wild`, `txt_remote`, `txt_wordpress`, `txt_plugin`, `txt_sqli`, `txt_xss`, `txt_traversal`, `txt_deser`, `txt_cmdi`, `txt_memory`, `txt_privesc`, `txt_edge`, `txt_ssrf`, `txt_upload`, `txt_dos`, `txt_kernel`; `desc_len` | case-insensitive regex on `description` |
| text | safe | `text_kev_lr`, `text_kev_sim` | TF-IDF (1–2 grams, 150k features) + logistic regression trained on KEV descriptions (published ≥2015, added before Jan 1 of Y: 925 / 1,105 / 1,343 positives) vs 40k random non-KEV; `text_kev_sim` = max cosine to those KEV descriptions |
| siblings | safe | `n_same_day`, `sib_flags`, `sib_max_epss_pct`, `n_adjacent_ids` | same assigner + same publish day within the cohort; adjacent = ID numbers within 3 |
| history | safe | `assigner_kev_3y`, `assigner_rate_3y`, `assigner_volume_3y`, `assigner_kev_all`, `assigner_kev_prev_year`, `kev_product_hist`, `kev_product_hits`, `kev_vendor_hist` | KEV before Jan 1 of Y; product/vendor = KEV names (≥3 entries) found as word n-grams in the description |
| kev_now | careful | `assigner_kev_ytd`, `sib_kev_known` | same-year CVEs from the same assigner already on KEV by the snapshot (see §6.2, the "gap") |
| cvss | careful | 8 CVSS 3 parts as ordinals, `cvss3`, `has_cvss4`, `cwe_kev_rate` | `cwe_kev_rate` = smoothed KEV rate of the CWE over the previous 3 years |
| granite | safe | `gr_kev_lr`, `gr_kev_sim`, `gr_kev_knn5`, `gr_cluster_rate`, `gr_hybrid_lr` | see §7 |
| leak | leak | `LEAK_nvd_analyzed`, `LEAK_n_references`, `LEAK_n_affected_cpes` | demo only, separate file |

## 6. Findings, with numbers

### 6.1 About the data and KEV
1. **Base rate is tiny:** 0.15% of a monthly cohort reaches KEV (52 of 35,281 in the 2025 backtest; ~1 in 680).
2. **58% of positives have no public exploit signal** (only 42% have any flag; only 7.7% have EPSS > 0.1). EPSS and flags alone can never find most of them.
3. **KEV adds fast:** of same-year KEV additions, 81% (2023), 74% (2024), 75% (2025) happened within 30 days of publication; median 3–4 days; 40–49 per year on or before publication day. Those are already excluded from the scoreboard, so **what's left are the slow, harder cases**.
4. **KEV rate by signal (2025 backtest, base 0.15%):**

| Signal | CVEs | Positives | KEV rate |
|---|---|---|---|
| Description says "in the wild" / "actively exploited" | 28 | 5 | 17.9% |
| Nuclei template | 109 | 9 | 8.3% |
| Metasploit module | 22 | 1 | 4.5% |
| GitHub PoC | 650 | 19 | 2.9% |
| Exploit-DB | 69 | 2 | 2.9% |
| Text mentions RCE | 2,054 | 18 | 0.88% |
| Text mentions auth bypass / unauthenticated | 2,574 | 19 | 0.74% |
| `privileges_required = NONE` | 16,527 | 39 | 0.24% |
| `user_interaction = NONE` / `REQUIRED` | 23,791 / 10,162 | 46 / 6 | 0.19% / 0.06% |
| No flag at all | — | — | 0.09% |

5. **WordPress is a safe negative filter:** 3,053 CVEs mention WordPress in 2025 and **0** reached KEV. The 2026 scoreboard has 4,820 (Wordfence 3,114, Patchstack 4,072, WPScan 1,446 assigned CVEs).
6. **Clusters of CVEs:** 13 of 52 positives (25%) come in groups of consecutive IDs from the same vendor, published together (Ivanti EPM CVE-2024-13159/60/61; CVE-2025-2746/47; 48927/28; 2775/76; Dassault 6204/05; Samsung 21042/43).
7. **Assigner matters** (2025 backtest): VulnCheck 6/402 (1.5%, 10× base), MITRE 11/3,784, GitHub 5/1,984, Samsung 2/184, ZDI 2/376, Microsoft 2/906. VulnCheck has 6,792 CVEs in 2026, so its rate will dilute. Mentioning a KEV vendor name in the text does **not** help (0.11% vs 0.17%).

### 6.2 The scoreboard "gap" (a legit signal)
72,794 CVEs were published Jan–Sep 2026; the scoreboard has 72,670. **The 124 missing ones are all "Analyzed" and come from Microsoft (21), Cisco (13), GitHub (8), Chrome (7), MITRE (6), Joomla, Check Point, VulnCheck, SonicWall, Adobe, Apple, Fortinet, Palo Alto, Ivanti…** They are the CVEs **already on KEV at the end of their month** (the README says the scoreboard excludes those). The replicas have the same: 81 (2024) and 96 (2025). KEV status was public at the time, so using it is legit, but **declare it**. We use it for `assigner_kev_ytd` (how many of this assigner's same-year CVEs were already on KEV at the snapshot) and `sib_kev_known` (a same-assigner CVE within 3 days already on KEV). In models it was neutral (±1).

### 6.3 What drives the model
- **Univariate AUC / lift in the top 5% (2024+2025 replicas):** `text_kev_lr` 0.912 / ×10.9 · `gr_kev_lr` 0.812 / ×6.6 · `assigner_rate_3y` 0.794 / ×3.2 · `cvss3` 0.756 / ×4.2 · `gr_cluster_rate` 0.755 / ×3.0 · `cwe_kev_rate` 0.751 / ×4.5 · `text_kev_sim` 0.749 / ×3.9 · `LEAK_n_affected_cpes` 0.738 / ×5.0 · `gr_kev_sim` 0.707 / ×5.2 · `any_flag` 0.694 / ×9.1 · `poc_github` 0.667 / ×7.5 · `txt_unauth` 0.66 / ×4.5 · `epss` 0.644 / ×5.7 · `nuclei` 0.589 / ×4.8.
- **Mean |SHAP| (v0 model):** `text_kev_lr` 0.58 · `assigner_rate_3y` 0.37 · `n_same_day` 0.31 · `desc_len` 0.24 · `sib_max_epss_pct` 0.23 · `n_adjacent_ids` 0.22 · `epss_pct` 0.21 · `epss` 0.17 · `age_days` 0.17 · `txt_unauth` 0.12.
- **Ablations (LightGBM, A+B total, base = 30):** − exploit signals → **14** · − TF-IDF text model → 25 · − assigner/product history → 26 · − siblings → 27 · − text keyword flags → 29 · − Granite → 30.
- Charts: `reports/figures/univariate_lift.png`, `feature_importance.png`, `feature_correlation.png`.

## 7. IBM Granite

### 7.1 Granite Embedding (`analysis/granite_features.py`)
- Model `ibm-granite/granite-embedding-30m-english` (384 dims, normalized). Embedded **193,236** descriptions (all cohort CVEs + all KEV since 2015 + 60k random non-KEV) on an RTX 4070 laptop in ~3 min. Cache: `derived/granite/emb_granite30m.npy` (not in git, 141 MB).
- Per year Y, using only KEV added before Jan 1 of Y: `gr_kev_lr` (logistic on embeddings), `gr_kev_sim` / `gr_kev_knn5` (max / top-5 mean cosine to past KEV), `gr_cluster_rate` (60 k-means clusters, smoothed KEV rate), `gr_hybrid_lr` (TF-IDF words + 64 PCA dims of Granite, weight 0.1, one logistic model).
- **Result:** Granite alone is weaker than TF-IDF (AUC 0.81 vs 0.91). Likely reason: KEV is driven by **exact product names** (Ivanti, FortiOS, Exchange), which TF-IDF keeps word by word; Granite captures the *kind* of bug. The hybrid is the best text-only model (A+B 17 vs 15 TF-IDF and 7 Granite) but doesn't improve LightGBM (26 vs 30).
- Hybrid weight sweep (text-only hits@500, AUC): 2024 test: W=0 → 6 (0.899), 0.02 → 6 (0.904), 0.05 → 7 (0.904), 0.1 → 8 (0.903). 2025 test: W=0 → 8 (0.909), 0.02 → 8 (0.912), 0.05 → 8 (0.909), 0.1 → 9 (0.907).
- **Where Granite shines: the clusters** (`reports/granite_clusters_2026.csv`, which technologies get exploited):

| Cluster theme (top KEV products) | KEV rate | 2026 CVEs in it |
|---|---|---|
| Windows, Win32k, Exchange Server, Active Directory | 15.0% | 1,094 |
| Windows, Exchange, Office, SharePoint | 9.6% | 470 |
| Chromium V8, Chromium, Blink, Firefox | 8.8% | 529 |
| Internet Explorer, Windows, Office, Edge | 8.0% | 72 |
| Windows, Office, Exchange, IE | 6.6% | 845 |
| Apple iOS / iPadOS / macOS | 6.3% | 729 |
| Edge appliances: Nagios XI, Accellion FTA, Zyxel, SonicWall SMA100 | 5.9% | 1,865 |
| Chromium (Intents, Mojo, V8) | 5.6% | 2,583 |
| Cisco IOS / IOS XE, small-business routers | 4.5% | 776 |
| Veritas, Juniper Junos, PAN-OS, ManageEngine | 4.0% | 3,272 |

### 7.2 Granite Instruct (`analysis/granite_instruct.py`)
- Model `ibm-granite/granite-3.3-2b-instruct` (2.5B params, bf16, ~5 GB VRAM).
- **Design:** stage 1 = the v1 LightGBM, scored out-of-sample (2024 rows by a 2025 model, 2025 by a 2024 model, 2026 by both). The top 5% of each year (2024: 1,823 · 2025: 2,129 · 2026: 3,634) are candidates. They contain 16/27 (2024) and 28/48 (2025) of the test positives (at 8%: 17/27 and 33/48). Granite answers yes/no questions per candidate; we read **log P(yes) − log P(no)** from one forward pass (no text generation). A logistic stage 2 (stage-1 percentile + answers), trained on the other year, re-orders the candidates. Options are evaluated with and without the per-assigner cap (§8.3).
- **Questions:** `q_internet` (usually exposed to the internet: VPN, firewall, gateway, mail/web server, file transfer, remote access?), `q_enterprise` (widely used by large companies or governments?), `q_niche` (small plugin, hobby or niche product?). Also tested but dropped for speed (they overlap with regex flags): `q_preauth`, `q_takeover`.
- Example answers (log-odds): Cisco IKEv2 bug → internet +6.3, enterprise +4.9; FortiOS heap overflow → internet +5.4, enterprise +3.9; Shenzhen Wi-Fi repeater → enterprise −12.5; SMM firmware bug → internet −12.2.
- Then Granite **writes product, category and a one-sentence reason** for each CVE in the final 2026 top 500 (`reports/top500_v2_explained.csv`), which feeds the hub's Top 500 table.
- **Speed lessons (laptop RTX 4070):** text generation ~0.6 s per CVE. Yes/no scoring ~4–6 min per question per 2,000 CVEs. Raw forward of 32×155 tokens takes 1.0 s (GPU does 36 TFLOPS bf16). Two fixes mattered: `logits_to_keep=1` (only the last position goes through the vocab head) and `OMP_WAIT_POLICY=PASSIVE` (idle LightGBM threads were spinning and starving GPU kernel launches: 118 s → 72 s).
- **Runtime on the laptop:** 2024 candidates ~15 min, 2025 ~10 min, 2026 ~15 min (3 questions each), 500 reasons ~4 min. Answers are cached in `derived/granite/instruct_{year}.csv` and reasons in `derived/granite/reasons_2026.csv`, so re-runs only score new CVEs (a cached re-run takes ~4 min).
- **What Granite's answers say** (candidate pool 2024+2025: 3,952 CVEs, KEV rate 1.42%; `reports/granite_instruct_lift.csv`):

| Question | Granite says yes | KEV among yes | KEV rate yes | KEV rate no | Lift |
|---|---|---|---|---|---|
| internet-facing | 1,246 | 25 | 2.01% | 1.15% | 1.75× |
| widely used by enterprises | 1,262 | 26 | 2.06% | 1.12% | 1.85× |
| niche / hobby product | 67 | **0** | 0% | 1.44% | 0 |

- **Re-ranking results** (`reports/granite_instruct_results.csv`; stage 1 here = the v1 LightGBM re-run in this script):

| Option | A | B | Total |
|---|---|---|---|
| stage 1 (v1 LightGBM) | 20 | 9 | 29 |
| stage 2 logistic [stage-1 rank only] (control) | 20 | 9 | 29 |
| stage 2 logistic [stage-1 rank + Granite answers] | 21 | 10 | 31 |
| stage 2 rank average [stage 1, Granite] | 20 | 10 | 30 |
| **stage 1 + cap 5** | **23** | **12** | **35** |
| control + cap 5 | 23 | 12 | 35 |
| stage 2 Granite + cap 5 | 23 | 11 | 34 |
| stage 2 rank average + cap 5 | 22 | 11 | 33 |

  Granite adds +1 in each direction without the cap (the control proves it's Granite, not the re-ordering), but the cap captures the same gain and more. The script picked **stage 1 + cap 5** for 2026 (`reports/final_ranking_choice.txt`).
- **Explanations:** for each 2026 top-500 CVE Granite wrote product, category and a reason (`reports/top500_v2_explained.csv`). Category mix: enterprise server software 282, network device 92, other 44, library/dev tool 19, web app/plugin 19, OS 19, desktop app 15, browser 7, IoT 3. Granite sometimes echoes the template instead of a product name (15 blank after cleanup) and once gave no reason. `analysis/export_top500_json.py` cleans this for the hub.

## 8. All experiments and results

### 8.1 v0: direction A only (train 2024 → test 2025)

| Ranking | Hits@500 (Nov 4) | Hits@500 (Dec 31) |
|---|---|---|
| CVSS 3 first | 6 | 6 |
| EPSS first | 8 | 8 |
| Exploit flag, then EPSS | 17 | 18 |
| Hand heuristic* | 21 | 23 |
| TF-IDF text model only | 9 | 10 |
| Logistic regression (safe) | 17 | 18 |
| **LightGBM (safe)** | **20** | **20** |
| LightGBM (safe + kev_now) | 20 | 20 |
| LightGBM (safe + kev_now + CVSS) | 17 | 17 |
| LightGBM + LEAK (demo) | 23 | 25 |

\* heuristic = `epss_pct + 2·any_flag + 3·nuclei + 5·txt_in_wild + txt_rce + txt_unauth − txt_wordpress`, designed while looking at 2025.
Honest 2024 check (models trained on 2025, 27 positives): CVSS 1 · EPSS 4 · exploit+EPSS 8 · heuristic 9 · LightGBM 9 · **LightGBM + heuristic rank average 11**.

### 8.2 v1: both directions, with Granite embeddings (`reports/model_results.csv`)

| Ranking | A | B | Total |
|---|---|---|---|
| CVSS 3 first | 6 | 1 | 7 |
| EPSS first | 8 | 4 | 12 |
| Exploit flag, then EPSS | 17 | 8 | 25 |
| Hand heuristic (fair on B only) | 21 | 9 | 30 |
| Text only: TF-IDF / Granite / hybrid | 9 / 4 / 9 | 6 / 3 / 8 | 15 / 7 / 17 |
| Logistic regression, safe + TF-IDF + Granite | 15 | 8 | 23 |
| **LightGBM, safe, TF-IDF text (v1)** | **23** | **7** | **30** |
| LightGBM, safe, Granite text | 17 | 8 | 25 |
| LightGBM, safe, hybrid text | 19 | 7 | 26 |
| LightGBM, safe, all text | 19 | 7 | 26 |
| LightGBM, hybrid + kev_now | 19 | 7 | 26 |
| LightGBM, hybrid + kev_now + CVSS | 21 | 6 | 27 |
| Ensemble LightGBM + exploit/EPSS (rank average) | 18 | 9 | 27 |
| LightGBM + LEAK (demo) | 22 | 10 | 32 |

LightGBM settings: `n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8, colsample_bytree=0.7, reg_lambda=5, scale_pos_weight=25`, average of 5 seeds. Not tuned.

### 8.3 Sibling boost and the per-assigner cap (`analysis/sibling_boost.py`)
Stage 1 here = the same v1 LightGBM, run with `n_jobs=4` (A = 20, B = 9: the ±3 noise in action).

**Sibling boost** (give a CVE part of its strongest sibling's rank; siblings = same assigner, same month, ID within K, published within 3 days): `final = s1_pct + λ·max(0, sib_max − s1_pct)` when `sib_max ≥ gate`.

| K | gate | λ | A | B | A+B |
|---|---|---|---|---|---|
| — | — | 0 (plain model) | 20 | 9 | 29 |
| 3 | 0.95 | 0.5 | 20 | 10 | 30 |
| 10 | 0.99 | 0.75 | 21 | 10 | 31 |
| 10 | 0.95 | 1.0 | 15 | 8 | 23 |
| 50 | 0.95 | 1.0 | 15 | 6 | 21 |

Conclusion: **no real gain**; full inheritance hurts. Retro check: CVEs outside the top 500 with a sibling inside it reach KEV at 0.2–0.65%, vs 0.05–0.08% for others (3–8× more), but the top 500 itself runs at 1.8–4.0%. Pulling siblings in displaces better candidates. The sibling features inside the model already capture what's there.

**Oracle problem:** the v1 2026 top 500 had **56 Oracle CVEs** (46 from July 2026 alone). All entered through `text_kev_lr` (median 0.914): Oracle writes every description from one template ("Easily exploitable vulnerability allows unauthenticated attacker with network access…") and some Oracle WebLogic/EBS CVEs are on KEV. Products: Coherence 18, WebLogic 5, Access Manager 4, Platform Security 3, Identity Manager 3, Workflow 3. Oracle published **1,097 CVEs on one day** in 2026. In the backtests Oracle had **0 KEV** out of 235 (2025) and 282 (2024) CVEs.

**Per-assigner-month cap** (keep at most C CVEs per assigner per snapshot month in the top 500, fill with the next best):

| Cap C | A | B | A+B | Distinct assigners in top 500 (A) |
|---|---|---|---|---|
| 1 | 17 | 8 | 25 | 191 |
| 2 | 19 | 11 | 30 | 135 |
| 3 | 21 | 11 | 32 | 105 |
| 4 | 23 | 11 | 34 | 97 |
| **5** | **23** | **12** | **35** | 93 |
| 6 | 21 | 11 | 32 | 91 |
| 8 | 21 | 11 | 32 | 90 |
| 10 | 21 | 10 | 31 | 89 |
| none | 20 | 9 | 29 | 83 |

Every cap from 3 to 8 beats no cap in **both** directions; that's a plateau, not a lucky spike. We use C = 5. Intuition: real exploitation is spread across many vendors, and a single patch batch shares the same text style and signals, so its errors are correlated. Without the cap, the plain top 500 has up to 29 (A) / 46 (B) CVEs from one assigner-month.

## 9. Rankings for 2026 (`submissions/`)

| File | Method | Notes |
|---|---|---|
| `top500_v0.csv` | LightGBM safe (incl. TF-IDF), trained on 2024+2025 | 212 with exploit flag, 1 WordPress, 62 Oracle |
| `top500_v1.csv` | Same, re-run with Granite features available but TF-IDF variant picked | 208 flags, 0 WordPress, 56 Oracle; 420 shared with v0 |
| `top500_v1_cap5.csv` | v1 + cap of 5 per assigner per month | 211 flags, 19 Oracle, 137 assigners, deepest v1 rank used = 595; 438 shared with v1 |
| **`top500_v2.csv`** | **Stage 1 (v1 LightGBM, re-trained in `granite_instruct.py`) + cap 5; Granite reasons** | **Current pick.** Best A+B (23 + 12). Shares 409 CVEs with `top500_v1_cap5` (same method, different random run) and 388 with v1 |

Full rankings: `derived/scored_2026_v0.csv`, `derived/scored_2026_v1.csv` (rank, score, assigner, EPSS, flags, top 3 SHAP reasons), and the explained v2 top 500 in `reports/top500_v2_explained.csv` (product, category, Granite probabilities, signals, reason). Monthly mix of `top500_v1_cap5`: Jan 48 · Feb 43 · Mar 69 · Apr 70 · May 58 · Jul 99 (includes June) · Aug 78 · Sep 35.

**Before submitting:** the 91 CVEs that differ between `top500_v2` and `top500_v1_cap5` sit near the boundary. A safer final file averages the scores of many runs (10+ seeds, both scripts) and then applies the cap once (§11.1).

## 10. Conclusions and insights (for the write-up and the video)

1. **Severity isn't likelihood.** CVSS-first finds 1–6 exploited CVEs in our backtests; our clean model finds ~20–23 (A).
2. **The easy ones are already gone.** 75–80% of KEV additions happen within 30 days, and those are excluded from the scoreboard by construction. The benchmark is about the slow cases.
3. **Most exploited CVEs have no public exploit signal (58%),** so EPSS/flag-based ranking has a hard ceiling. Text and context fill the gap.
4. **The description is the strongest single signal** (AUC 0.91 for a TF-IDF model trained on past KEV). Exact product names matter more than semantics.
5. **NIST metadata leaks the answer** (100% of positives "Analyzed"). We exclude it and show it inflates results by only ~2–4 hits.
6. **Diversify across vendors.** Capping 5 CVEs per assigner per month is the most consistent gain we found (+6 across A+B). One noisy vendor batch (Oracle, 1,097 CVEs in a day) can flood a naive model.
7. **Sibling CVEs carry real but weak signal** (3–8× base rate), already captured by features; a post-hoc boost doesn't help.
8. **IBM Granite** maps which technologies get exploited (Windows/Exchange 15%, Chromium 8.8%, Apple 6.3%, edge appliances 5.9%), gives a semantic text model, and reads candidates like an analyst: internet-facing (1.75×) and enterprise (1.85×) raise the KEV rate, and products it calls niche never reached KEV (0/67). It also writes the per-CVE reasons engineers see.
9. **Honest uncertainty:** ±3 hits is noise; we report A as the forecast-style test and A+B for robustness. Two random runs of the same method agree on only 409 of 500 CVEs.

## 11. What to do next on a strong lab PC (prioritized)

1. **Stabilize the final ranking (quick win).** Train the stage-1 LightGBM with 20–50 seeds (and `deterministic=True, force_row_wise=True` to remove the thread effect), average the scores, apply the cap of 5, and use that as the final `top500.csv`. Re-check A/B with the averaged score. Today two runs of the same method share only 409/500 CVEs.
2. **Extend Granite Instruct.**
   - Use all 5 questions, more candidates (top 10–15%), and a bigger model: `ibm-granite/granite-3.3-8b-instruct`, or Granite 4.0 if the installed `transformers` supports it.
   - On Linux, serve it with **vLLM** (Red Hat AI Inference Server is also an IBM option) for 10–50× throughput.
   - Use prefix caching: same description, 5 questions.
   - Add questions such as "Is a patch-gap/edge device likely targeted by ransomware groups?" or "Is this product typically deployed by governments?". Check each on A/B.
   - Use `q_niche` as a hard negative filter (0/67 reached KEV) and check it on A/B with the cap on.
3. **Stabilize the evaluation.**
   - 20+ seeds, bootstrap CIs on hits@500 (resample positives/cohorts), and rolling monthly origin (train on months < m, test on m) to get many more test points.
   - Then tune LightGBM (num_leaves, min_child_samples, scale_pos_weight, objective `lambdarank`) with **time-based** CV only.
4. **Smarter cap.** MITRE, VulnCheck, GitHub, VulDB and ZDI assign CVEs for many vendors, so per-assigner may be too coarse. Try a cap per **(vendor/product extracted by Granite, month)**, or exempt multi-vendor CNAs. Also try caps per publish day.
5. **Bigger / tuned embeddings.** Try `granite-embedding-english-r2` or the 278M multilingual model. Fine-tune contrastively on KEV vs non-KEV using only pre-Y data. Add sibling similarity (cosine between same-day CVEs).
6. **Product-level KEV history.** Use Granite to extract a normalized vendor/product for every CVE (not just KEV names found by n-grams), then count past KEV per product. TF-IDF suggests product identity is the key signal.
7. **Month quotas.** January CVEs had ~10 months to reach KEV, September ~5 weeks. Check whether the model under-ranks late months (top500_v1_cap5 has only 35 from September) and whether a quota helps on A/B.
8. **Freeze a final pipeline** (one command from raw data to `top500.csv`), and write the columns-used table with the traffic light for the write-up.

## 12. Environment and reproduction

**Machine used so far:** Windows 11, Python 3.12.7, RTX 4070 Laptop (8 GB), CUDA via torch 2.6.0+cu118.

**Packages installed during this work** (on top of an existing env with numpy 1.26.4, pandas 2.3.3, scikit-learn 1.8.0, matplotlib 3.10.8):
- `lightgbm 4.7.0`
- `sentence-transformers 3.4.1`, `transformers 4.49.0`, `huggingface_hub 0.29.3`, `tokenizers 0.21.4`
- Pin `numpy==1.26.4`. `shap` pulled numpy 2.x and broke OpenCV, so it was removed (LightGBM gives SHAP via `pred_contrib=True`). Newer `transformers 5.x` / `huggingface_hub 2.x` bumped `anyio`, `click` and `h11` and broke FastAPI/gTTS/httpcore on this machine, so we stayed on 4.49.
- No `accelerate`: models are loaded and moved with `.to("cuda")`.

**Gotchas:**
- Run scripts with `python -E -P` (with `-I`, a broken system copy of matplotlib loads).
- TensorFlow is installed, so set `USE_TF=0` (scripts set it).
- Set `OMP_WAIT_POLICY=PASSIVE` when LightGBM and the GPU share a process (scripts set it).
- On a fresh Linux lab PC: `pip install lightgbm sentence-transformers transformers torch pandas scikit-learn matplotlib` is enough; vLLM is optional.

**Run order (from the repo root; the IBM data stays unzipped at the repo root in `IBM-20261107T174321Z-1-001/IBM/data/data/` and is never committed):**
```
python -E -P luis/analysis/backtest_2025.py        # 2025 backtest, ~1 min        -> derived/backtest_cohorts_2025.csv
python -E -P luis/analysis/build_features.py       # replicas + 2026, ~2 min      -> derived/features_*.csv, leak_only_*.csv
python -E -P luis/analysis/granite_features.py     # Granite Embedding, ~3 min GPU -> derived/granite/*
python -E -P luis/analysis/train_baseline.py       # models A/B, charts, ~3 min   -> reports/*, submissions/top500_v1.csv
python -E -P luis/analysis/sibling_boost.py        # sibling boost + backtest rankings, ~2 min
python -E -P luis/analysis/granite_instruct.py     # Granite Instruct, ~1 h on the laptop -> submissions/top500_v2.csv
python -E -P luis/analysis/export_top500_json.py   # site/top500.json for the hub's Top 500 table
```
`python -E -P luis/analysis/granite_instruct.py --test` scores 96 candidates with all 5 questions and prints examples.

## 13. Evidence index (where each claim comes from)

| Claim / section | File |
|---|---|
| Baselines and the 52-positive backtest (§6.1, §8.1) | `analysis/backtest_2025.py`, `derived/backtest_cohorts_2025.csv` (regenerate) |
| Feature definitions and risk (§5) | `derived/feature_dictionary.csv`, `derived/granite/feature_dictionary_granite.csv` |
| A/B results, ablations (§8.2, §6.3) | `reports/model_results.csv` |
| Univariate AUC and lift (§6.3) | `reports/univariate_lift.csv`, `reports/figures/univariate_lift.png` |
| What the final model uses (§6.3) | `reports/feature_importance.csv`, `reports/figures/feature_importance.png` |
| Feature correlations | `reports/figures/feature_correlation.png` |
| Granite clusters (§7.1) | `reports/granite_clusters_2026.csv` |
| Sibling boost grid and retro check (§8.3) | `reports/sibling_boost_results.csv`, `reports/sibling_retro_check.csv` |
| Every backtest candidate with model rank, final rank, label (§4.2) | `derived/backtest_rankings_A.csv`, `derived/backtest_rankings_B.csv` |
| Granite Instruct answers (cache) and results (§7.2) | `derived/granite/instruct_{year}.csv`, `reports/granite_instruct_results.csv`, `reports/granite_instruct_lift.csv`, `reports/granite_instruct_coefficients.csv` (if a stage-2 option wins) |
| Which option the script picked for 2026 | `reports/final_ranking_choice.txt` |
| Explained 2026 top 500 (product, category, Granite probabilities, reason) | `reports/top500_v2_explained.csv`, `derived/granite/reasons_2026.csv`, `site/top500.json` (hub table) |
| 2026 rankings (§9) | `submissions/*.csv`, `derived/scored_2026_v*.csv` |
| Density-matched numbers, `txt_in_wild` ablation, vendor vs product matching (§14) | `analysis/density_check.py`, `reports/density_check.txt` |
| Chen's GPU-box workstream (CV, LR + LoRA, Granite 30B/8B, embeddings r2) | `received/CHEN_context_gpu_box_2026-11-07.md` |

Not in git (too big or regenerated in minutes): raw data, `derived/features_*.csv`, `derived/leak_only_*.csv`, `derived/backtest_cohorts_2025.csv`, `derived/granite/emb_granite30m.npy`, `derived/granite/granite_features_*.csv`.

## 14. Merge with Chen's GPU-box workstream (`received/CHEN_context_gpu_box_2026-11-07.md`)

Chen ran a parallel pipeline on a 4× H100 box: purged 5-fold time-blocked CV with bootstrap CIs, a strongly regularized logistic regression with rich metadata features, granite-embedding-english-r2, Granite 4.2 30B/8B via vLLM (zero-shot, many-shot, LoRA, structured extraction), and a memorization probe.

### 14.1 Same yardstick (checked on our replicas, `deterministic=True`, 10 seeds)

| Ranking | A @243 | B @198 | A+B | Pooled 2024+2025, top 441 of 64,120 (75 pos) |
|---|---|---|---|---|
| Exploit, then EPSS | 13 | 6 | 19 | 19 |
| Exploit, then EPSS + cap 5 | 14 | 6 | 20 | 19 |
| LightGBM safe | 15 | 6 | 21 | 23 |
| **LightGBM safe + cap 5** | **16** | **7** | **23** | 23 |
| LightGBM safe without `txt_in_wild` + cap 5 | 16 | 7 | 23 | 23 |

Chen, judge-aligned labels (75 positives, top 517 of 75,148): exploit+EPSS 20 · best LR 26 · LoRA 25 · **LR + LoRA rank average 28, with cap 3: 30**. The sets differ a little (his includes Oct–Dec 2024), but the gap (+5 to +7 over us) is bigger than the noise. **His LR + LoRA is the stronger core.** The cap helps both pipelines.

### 14.2 Disagreements, settled with data
- **`txt_in_wild`:** removing it changes nothing on our side (A+B 23 → 23 at density, pooled 23 → 23). **Agree with Chen: drop it.** It costs nothing and removes a leakage objection (description edits can't be dated). 9 of the 21 scoreboard CVEs with that text are in `top500_v2`; rerun without it before submitting.
- **"KEV vendor in text doesn't help" vs "3.8× for product":** no real conflict. Our numbers agree: product-level match lift 3.2× (`kev_product_hits`), vendor-level 1.8× (`kev_vendor_hist`); Chen: product 3.8×, vendor 1.4×. Our earlier "doesn't help" was about vendor names.
- **2024 positives 27 vs 67:** label window (Nov 4 of the publish year vs Dec 31, 2025) plus month ranges. Use `y_nov4` everywhere (Chen agrees).
- **One thing to flag back to Chen:** 5-fold time-blocked CV trains on *future* blocks when testing earlier ones (like our B). Report the forward-only folds (test block trained only on earlier blocks) next to the 5-fold numbers so the headline is a true forecast.

### 14.3 What each side brings to the final pipeline
- **From us → add to Chen's CV:** `text_kev_lr` / `text_kev_sim` (TF-IDF KEV text model, AUC 0.91; his embedding features don't replace exact product names), the per-assigner-month cap, `n_adjacent_ids` and sibling features, LightGBM settings (`scale_pos_weight=25`, 5+ seeds), `days_to_cutoff` and `age_days` (his §10.3–10.4 to-dos), the leak-inflation demo, the Granite 2B yes/no finding (niche → 0/67), and the explained top 500 / hub table.
- **From Chen → already beyond our §11:** Granite 30B structured extraction on all CVEs (§11.2, §11.6), purged CV with bootstrap CIs (§11.3), granite-embedding-english-r2 (§11.5), LoRA 8B, memorization probe (0/122), strong negative filters (unknown CWE 0/3,551; CVSS 4 E:P 0/2,669; WordPress CNAs), out-of-band advisory 5.5×, EOL text 13×, and the missed-positives analysis (22/52 invisible at publish time: 0% exploit code).

### 14.4 Proposed split of work (Nov 7–8)
- **Chen (GPU box):** finish extraction → add our TF-IDF + adjacent IDs + LightGBM to his CV with `y_nov4` → final LR + LightGBM + LoRA rank average → cap 3–5 → score the 2026 scoreboard once → `top500.csv`. Drop `txt_in_wild`.
- **Luis side:** the write-up (columns-used table with the traffic light, leak demo, A/B + CV methodology, insights from both docs), the video script, the hub's Top 500 table fed from the final ranking with Granite reasons (Chen's nearest-KEV matches make great explanations), and a sanity check of the final file (Oracle July batch, month mix, WordPress = 0).
- **Until Chen's ranking exists,** `submissions/top500_v2.csv` stays the team's fallback.

## 15. Research sources

- Jacobs et al., "Enhancing Vulnerability Prioritization: Data-Driven Exploit Predictions with Community-Driven Insights" (EPSS): [arXiv 2302.14172](https://arxiv.org/pdf/2302.14172). EPSS is an XGBoost model on ~1,400 features: exploit code, description keywords, CWE, vendor, references.
- Evaluation of EPSS on severe KEV CVEs: [arXiv 2411.02618](https://arxiv.org/html/2411.02618v1). CVEs later added to KEV often didn't have a high EPSS beforehand, consistent with our finding 3.
- EPSS feature overviews: [INCIBE](https://www.incibe.es/en/incibe-cert/blog/epss-advancing-vulnerability-prediction-and-management), [Phoenix Security](https://phoenix.security/what-is-epss).
