# Project context · Luis · IBM "Before the Breach" (Sheridan Datathon 2026)

Luis's handoff document (folder `luis/`; paths below are relative to it unless they start with `luis/`). Last updated **Nov 7, 2026, evening, after Chen's 3rd push (§14.15)**. Written so a teammate can pick up the work without re-deriving anything. Every number comes from a script in `luis/analysis/` and is backed by a file listed in **§13 Evidence index**. Companion: `chen/context_Chen.md` (Chen's GPU-box workstream).

- **Team hub (web):** https://claude.ai/artifact/Np2SfMguMKySJnBfvNfnjR (ranked Top 500 table explained by IBM Granite, rules, CSV/column explorer, data map, shared notes; code in `site/breach-hub.html`)
- **Current pick (team final): `submissions/top500_final.csv`**, built by one command, `python -E -P luis/analysis/final_build.py` (§14.14). **Chen's 2026 files are in**, so it is now the full 3-voter recipe: equal-weight rank average of **tree** (XGBoost big, 40 seeds) + **linear** (Luis's LR averaged with Chen's best LR) + **LLM** (Chen's Granite 4.2 8B LoRA), then 4 Granite-30B demotion filters and a cap of 3 per assigner-month. Explained in `reports/top500_final_explained.csv` and shown in the hub. Checked by both of us (§14.15, Chen §8.5); nothing Chen pushed on Nov 7 beats it.
- **Practice-exam headline** (2024 + 2025 Jan–Sep pooled, top 441 of 64,120 = the scoreboard's selection rate, 75 exploited CVEs): exploit-then-EPSS **19–20** · v4 **26–28** (+1 to +16; varies by run) · v4f **27–30** pooled / **31** at density (A 20 of 48 + B 11 of 27) · **final recipe ≈ 32** (+4 to +20 vs exploit-then-EPSS; −5 to +9 vs v4f; estimate, uses Chen's out-of-fold scores, §14.15). On Chen's 5-fold CV (75 positives, top 517) the same recipe gets 32 vs 20 for the baseline.

---

## 0. TL;DR

1. **Task:** pick 500 of 72,670 CVEs (published Jan–Sep 2026). Score = how many CISA adds to KEV by Nov 4, 2026. Official baselines: 6–8 hits.
2. **Local exam:** the 2026 answer is secret, so we rebuilt the benchmark on 2024 and 2025 ("replicas": same cohort rule, signals at the snapshot, labels = KEV by Nov 4 of that year). **A** = train 2024 → test 2025 (the realistic forward test). **B** = train 2025 → test 2024 (robustness only). Numbers are hits at the scoreboard's selection rate (A top 243, B top 198, pooled top 441).
3. **Results ladder (pooled, 75 positives):** CVSS 5 · EPSS 11 · exploit-then-EPSS 19 · LightGBM 23 · LightGBM + LR 26 · XGBoost + LR (v4) 28 · v4f (+ Granite filters) 27–30 · **final: XGBoost + both LRs + Chen's LoRA ≈ 32** (estimate). Random would get 0.5. Everything above 26 is statistically tied (§14.12).
4. **What works:** pairing one tree model with one linear model (they disagree on 2/3 of their picks); capping CVEs per assigner per month (plateau for caps 3–8); the TF-IDF model trained on past KEV descriptions (best single feature, AUC 0.91); Chen's Granite 8B LoRA as a third voter.
5. **What doesn't:** heavier models alone, CatBoost with the raw assigner (memorizes), TF-IDF trained on the cohort's own positives, more training years without exploit signals (§14.9), a post-hoc sibling boost, "assigner heat" features, Granite embeddings as a ranker, and leak columns as a ranker (2–5 hits).
6. **The ceiling:** across every model, only 35 of 75 positives ever reach a top 441. The other 40 had no public exploit code and an ordinary EPSS at the snapshot. More data and bigger models don't find them; they look ordinary when published.
7. **Noise:** ±3 hits is noise (same model, different thread count: 20 vs 23). Two random runs of the same method share only 409 of 500 picks → average many seeds before the final cut.
8. **IBM technology used:** Granite Embedding 30M (clusters of exploited technologies), Granite 3.3 2B Instruct (yes/no reads: niche products 0/67 reached KEV; per-CVE reasons), and from Chen: Granite 4.2 30B world-knowledge extraction (55 features, explanations), Granite 4.2 8B LoRA (+2 to +4 hits), Granite Embedding r2 (nearest past KEV for explanations), and a memorization probe (0/122).
9. **Status of every idea** (tested, rejected, pending) is in the ledger right below.

---

## Status ledger

### Tested

| Idea | Result (pooled top 441 unless noted) | Verdict | Where |
|---|---|---|---|
| Baselines: CVSS / EPSS / exploit code then EPSS | 5 / 11 / 19 | reference | §14.1, `reports/density_check.txt` |
| LightGBM on Luis's leak-free features | 23 | kept as tree option | §8, §14.7 |
| Cap per assigner-month (3–8) | baseline 19→22; LightGBM A+B 29→35 at top 500/yr; plateau 3–8 | **keep (3)** | §8.3 |
| Sibling features inside the model | ablation −3 | keep | §6.3 |
| Post-hoc sibling boost | +1 at best, −6 with full inheritance | rejected | §8.3 |
| Drop `txt_in_wild` (possible after-the-fact edits) | 23 → 23 | dropped | §14.2 |
| Leak columns (`nvd_status`, `n_references`, `n_affected_cpes`) | leak-only 2–5; clean + leak +2 to +4 | excluded (demo only) | §3, §14.6 |
| Scoreboard gap / "already on KEV this year" features | ±1 | neutral, not used | §6.2 |
| Granite Embedding 30M (text model, PCA) | AUC 0.81 vs TF-IDF 0.91; no ranking gain; PCA-64 hurts the LR | clusters / story only | §7.1, §14.5 |
| Granite 3.3 2B Instruct yes/no re-ranking | +2 without cap, 0 with cap; niche 0/67 | explanations + filter idea | §7.2 |
| Strongly regularized LR + Chen's lite features, averaged with LightGBM | 26 (+1 to +14) | v3 | §14.5 |
| XGBoost small / big, CatBoost, LambdaRank, big TF-IDF linear, 7-model average | 22 / 23 / 14 / 21 / 18 / 22–23 | XGBoost big kept; rest rejected | §14.7 |
| XGBoost big + LR (v4), cap 3 | 26–28 (+1 to +16), AP 0.062–0.063 | base of the current pick | §14.7 |
| "Assigner heat" (same assigner's earlier-month signals) | neutral for trees, worse for LR | rejected | §14.7 |
| Chen's full LR (Granite 4.2 30B extraction, 55 features) on our forward test | alone 27 (28 at density, best single model); + XGBoost 26 | strong, but no gain over v4 | §14.8 |
| Chen's LoRA 8B (his out-of-fold scores) added to ensembles | +2 to +4 on every ensemble; v4 + LoRA 30 (+4 to +18) | **adopt for the final** | §14.8 |
| Static prior: 2022–2024 cohorts, 162 positives, no exploit signals | alone 14–17; v4 + static 24–27 | rejected | §14.9 |
| Clean vs leak-only comparison | leak is a filter (non-analyzed → 0% KEV), not a ranker | write-up material | §14.6 |
| **Granite 30B demotion filters** (DoS only, unlikely target, low appeal, WordPress), each checked on the training year first | v4 29 → **31** at density (A 18→20, B 11→11); pooled 26→27, AP 0.062→0.065; vs v4 −1 to +4 | **adopted: v4f (current pick)** | §14.10 |
| "Small/niche install base" filter | failed the training-year check (0.36×) and costs hits (24) | rejected | §14.10 |
| Cap by Granite-normalized vendor-month, by assigner-day, or both caps | 27 / 27 / 30 at density vs 29 | no gain over the assigner-month cap | §14.10 |
| **Statistical review** (8 XGBoost runs, Holm-corrected McNemar + paired bootstrap, Romano–Wolf set, power, 2026 projection) | v4f 29.9 ± 0.6 over 8 runs (v4 27.9 ± 0.4); after Holm only v4f + LoRA beats exploit+EPSS (p = 0.03); every ML model is tied with the best; minimum detectable gap ≈ 5 hits between similar models (+2 would need ~444 positives); 2026 ≈ 12 (7–22) vs 8 | **stop chasing +1–2 hits** | §14.12 |
| Nested hyperparameter tuning (random search over XGBoost / LightGBM grids, ensemble weight and cap, all chosen inside the training year) | inner AP 0.026 → 0.103, but test 31 → 28–29 at density and AP 0.065 → 0.055; choices flip between A and B | rejected: keep defaults | §14.13 |
| Bagged final (40 XGBoost seeds averaged) | A/B: bagged v4f 29 (runs 29–31); 2026: shares 491 of 500 with v4f | **adopted in `top500_final.csv`** | §14.12, §14.14 |
| **Time-series model on the exploit signals** (Granite TimeSeries TTM / 1D RNN) | every CVE has exactly 1 signal month at prediction time (0 of 72,670 scoreboard and 0 of 87,215 2024–25 CVEs have earlier rows) | **not feasible per CVE**; aggregate version low value | §14.11 |
| **Final recipe with Chen's 2026 files** (XGBoost 40 seeds + avg(Luis LR, Chen LR) + Chen LoRA 8B, filters, cap 3) | our A/B ≈ 32 (+4 to +20 vs baseline; −5 to +9 vs v4f, P(better) 0.66); Chen's CV 32 vs his best 33 (tie) | **adopted: `top500_final.csv`** | §14.15 |
| Granite 4.2 **30B** LoRA as the LLM voter (Chen) | our A/B 28 vs 32 with the 8B; Chen's CV 29 vs 32 | rejected (8B stays) | §14.15 |
| 8B LoRA with Granite 30B answers written into the prompt ("llmx", Chen) | our A/B 28; Chen's CV 29 | rejected | §14.15 |
| LLM voter = average of LoRAs (8B + 30B, or + llmx) | 31 / 30 vs 32 | no gain; Chen has no 2026 30B scores anyway | §14.15 |
| 4 voters (add the 30B LoRA as its own voter) | 32 = tie | not worth a 30B scoring run | §14.15 |
| Chen's nested XGBoost tuning (60 settings, inner purged CV) | inner AP 0.05 → 0.08, held-out tie (24 hits each) | rejected, same as §14.13 | Chen §8.3 |
| Chen's check of our recipe on his CV (proxies) | ours 32–33 vs his 33; filters +1..+2 on XGBoost ensembles, catch 1 of 75 positives | agree | Chen §8.5, §14.15 |
| Voter-normalization sensitivity (how each voter is put on 0–1 before averaging) | same recipes move 27–32 just from that choice | write-up material: ±3 hits is implementation noise | §14.15 |

### Not tested yet / pending

| Idea | Status / plan | Owner |
|---|---|---|
| **One unified LR instead of two** (Luis + Chen features and heuristics in one model; replaces the avg-of-2-LR linear voter) | plan, evidence, every column and a fixed decision rule in §14.16; Chen's AI designs and trains it on the H100 box. `final_build.py` picks up `chen/submissions/lr_unified_scoreboard_2026.parquet` automatically if Chen adopts it | Chen → Luis |
| ~~LoRA seed averaging~~ | **done** (Chen `759c4cc`): 4 seeds averaged; single seeds share only 135–264 of their top 500. Final rebuilt: shares 388 with the seed-0 build (§14.14) | done |
| Aggregate time series with Granite TimeSeries TTM (monthly KEV additions per vendor, forecast into 2026) | optional IBM demo for the write-up only; history is 50 months and very sparse per vendor; expected ranking gain ≈ 0 (§14.11) | optional |
| Weighted ensembles | deliberately not done (selection bias) | – |
| Forward-only LoRA folds (makes the "+LoRA" A/B numbers strictly forward) | Chen to-do; his forward LR check showed no inflation, so the bias is probably small | Chen |
| Month quotas / time-left adjustment | Chen tested: neutral; not repeated here | – |
| Write-up (columns-used table with the traffic light, leak demo, A/B + CV methodology, insights), video script | not started | team |

Done since the last version: Chen's 2026 voters are in the final; the hub's Top 500 table shows the final; Chen's LR is in the linear voter (as a harmless hedge, not because of a proven 2026 CVSS shift, see §14.15).

### Recommendation (Nov 7, evening), based on the statistical review (§14.12–14.13) and Chen's 3rd push (§14.15)

1. **Stop optimizing for +1–2 hits.** With 75 positives the smallest detectable gap between similar models is ~5 hits; every ML ensemble is statistically tied with the best, and a proper nested grid search made things worse (Chen's nested search agrees). More tuning would fit noise.
2. **Submit `luis/submissions/top500_final.csv`.** It is the recipe both workstreams agreed on before seeing the last results (tree + linear + Granite LoRA, Granite filters, cap 3), so picking it is not a winner's-curse choice. It is also the best point estimate on our A/B (32) and within 1 hit of Chen's best on his CV.
3. **Spend the remaining time on the write-up and the video** (Discovery 35% + Communication 25%). The strongest material: the leak demo, the invisible positives and the 35/75 ceiling, "more data and bigger models don't find them" (now including a Granite 30B LoRA), the vendor cap and Granite filters, the memorization probe, and this power analysis.
4. **Expectation to state honestly:** about 12 hits on 2026 (80% interval 7–22) vs 8 for the best README baseline.

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

**How much leakage inflates our model:** LightGBM with the leak columns gets A 22–24 / B 10–11 vs 20–23 / 7–9 without. Only ~2–4 hits. Good news for the write-up: the clean model loses little. **A leak-only model is useless for ranking** (2–5 hits at density vs 19 for exploit+EPSS): the leak works as a filter (no non-analyzed CVE ever reached KEV), not as a ranker. On 2026 the filter would cut ~50% of the scoreboard vs 26% of our replicas, so our inflation numbers are a lower bound. Full table in §14.6.

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
| `top500_v2.csv` | Stage 1 (v1 LightGBM, re-trained in `granite_instruct.py`) + cap 5; Granite reasons | Previous pick. A+B 23 + 12 at top 500 per year. Shares 409 CVEs with `top500_v1_cap5` (same method, different random run) and 388 with v1 |
| **`top500_v4.csv`** | **Rank average of XGBoost big (all features incl. Chen's, 5 GPU seeds) and LR (Luis safe + Chen, C tuned inside training); cap 3; trained on all 2024+2025 months** | **Current pick.** Best in the model zoo: 29 at density, 28 pooled (+1 to +16 vs exploit+EPSS), AP 0.063 (§14.7). 237 flags, 10 Oracle, 0 WordPress, 147 assigners. Shares 406 CVEs with v3 |
| `top500_v3.csv` | Rank average of LightGBM (Luis safe features, 10 deterministic seeds) and LR (C = 0.001, chosen inside training) on Luis safe + Chen features; cap 3 per assigner-month; trained on all 2024+2025 months; `txt_in_wild` dropped | Statistically tied with v4. First option whose pooled paired bootstrap beat exploit+EPSS (§14.5). 230 with exploit flag, 11 Oracle, 0 WordPress CNAs, 4 blank-CWE, 143 assigners; month mix Jan 49 · Feb 43 · Mar 65 · Apr 56 · May 65 · Jul 91 · Aug 85 · Sep 46. Shares 298 CVEs with v2. No Granite reasons yet for the CVEs that are new vs v2 |

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

*Kept for history; the current plan is the **Status ledger** at the top.*

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
- `xgboost 3.4.1` (CUDA build) and `catboost 1.2.10` (+ `graphviz`, `plotly`), installed with `numpy==1.26.4 anyio==3.7.1 click==8.1.7 h11==0.14.0` pinned so nothing else moved
- `sentence-transformers 3.4.1`, `transformers 4.49.0`, `huggingface_hub 0.29.3`, `tokenizers 0.21.4`
- Pin `numpy==1.26.4`. `shap` pulled numpy 2.x and broke OpenCV, so it was removed (LightGBM gives SHAP via `pred_contrib=True`). Newer `transformers 5.x` / `huggingface_hub 2.x` bumped `anyio`, `click` and `h11` and broke FastAPI/gTTS/httpcore on this machine, so we stayed on 4.49.
- No `accelerate`: models are loaded and moved with `.to("cuda")`.

**Gotchas:**
- Run scripts with `python -E -P` (with `-I`, a broken system copy of matplotlib loads).
- TensorFlow is installed, so set `USE_TF=0` (scripts set it).
- Set `OMP_WAIT_POLICY=PASSIVE` when LightGBM and the GPU share a process (scripts set it).
- If XGBoost warns "No visible GPU is found, setting device to CPU" while `nvidia-smi` sees the card, `CUDA_VISIBLE_DEVICES` is set to an empty string in that shell (it was in the VS Code session on Nov 7 evening). Unset it (`Remove-Item Env:CUDA_VISIBLE_DEVICES`) or set it to `0`. On CPU the scripts still run, ~4 s per XGBoost fit instead of ~1 s.
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
| Chen's insights on our setup: new features, LR, ensembles, bootstrap, missed positives (§14.5) | `analysis/chen_insights.py`, `reports/chen_insights_results.csv`, `reports/chen_insights_bootstrap.csv`, `reports/chen_features_lift.csv`, `reports/chen_missed_positives.csv`, `reports/chen_insights_log.txt` |
| Clean vs leak-only vs clean+leak (§3, §14.6) | `python -E -P luis/analysis/chen_insights.py --leak-test` → `reports/leak_vs_clean.csv`, `reports/leak_vs_clean_log.txt` |
| Model zoo: XGBoost, CatBoost, LambdaRank, big linear, ensembles, diversity (§14.7) | `analysis/model_zoo.py`, `reports/model_zoo_results.csv`, `reports/model_zoo_bootstrap.csv`, `reports/model_zoo_correlation.csv`, `reports/model_zoo_catboost_importance.csv`, `reports/model_zoo_log.txt` |
| Statistical review: seed variability, Holm tests, Romano–Wolf set, power, 2026 projection (§14.12) | `analysis/stats_review.py`, `reports/stats_review_seed_variability.csv`, `reports/stats_review_paired_tests.csv`, `reports/stats_review_best_set.csv`, `reports/stats_review_projection_2026.csv`, `reports/stats_review_log.txt` |
| Nested hyperparameter tuning (§14.13) | `analysis/nested_tuning.py`, `reports/nested_tuning_choices.csv`, `reports/nested_tuning_results.csv`, `reports/nested_tuning_bootstrap.csv`, `reports/nested_tuning_log.txt` |
| Final build: voters, filters, cap, sanity checks (§14.14) | `analysis/final_build.py`, `derived/xgb_bag_2026.parquet`, `derived/scored_2026_final.csv`, `submissions/top500_final.csv`, `reports/final_build_info.txt`, `reports/final_build_log.txt`, `reports/top500_final_explained.csv` |
| Every column of both LRs: group, coefficient per SD, definition, closest equivalent (§14.16) | `reports/lr_columns_luis_chen.csv`; Chen's coefficients from `chen/results/final_lr_coefficients.csv`, Luis's from his LR refit on the 2024 + 2025 replicas (C = 1e-3) |
| Chen's 3rd push on our A/B: final recipe with his voters, LoRA 30B / llmx, normalization sensitivity (§14.15) | `analysis/chen_round3.py`, `reports/chen_round3.csv`, `reports/chen_round3_log.txt`, `derived/xgb_bag_ab.parquet` (cache); Chen's side: `chen/reports/lora_cv_compare.txt`, `chen/reports/xgb_compare.txt`, `chen/reports/luis_recipe_check.txt` |
| Static prior from 2022–2024 cohorts (§14.9) | `analysis/static_prior.py`, `derived/static/static_cohort_{2022,2023,2024}.parquet`, `reports/static_prior_results.csv`, `reports/static_prior_bootstrap.csv`, `reports/static_prior_invisible.csv`, `reports/static_prior_log.txt` |
| Granite 30B demotion filters, vendor caps, v4f (§14.10) | `analysis/postprocess.py`, `reports/postprocess_filters_check.csv`, `reports/postprocess_results.csv`, `reports/postprocess_bootstrap.csv`, `reports/postprocess_log.txt`, `derived/ab_scores_v4.parquet`, `derived/scored_2026_v4f.csv`, `submissions/top500_v4f.csv`, `reports/top500_v4f_explained.csv` |
| Luis + Chen combined on the forward test, LoRA estimate (§14.8) | `analysis/team_merge.py`, `analysis/common.py`, `reports/team_merge_results.csv`, `reports/team_merge_bootstrap.csv`, `reports/team_merge_log.txt` |
| v4 explained with Granite 30B + nearest KEV (hub table) | `python -E -P luis/analysis/team_merge.py --score-2026` → `reports/top500_v4_explained.csv`; `analysis/export_top500_json.py` → `site/top500.json` |
| `top500_v4` (§9, §14.7) | `python -E -P luis/analysis/model_zoo.py --score-2026` → `derived/scored_2026_v4.csv`, `submissions/top500_v4.csv` |
| `top500_v3` (§9, §14.5) | `python -E -P luis/analysis/chen_insights.py --score-2026` → `derived/scored_2026_v3.csv`, `submissions/top500_v3.csv` |
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
- **Ready-made for Chen's CV:** `derived/for_chen/luis_features_hist_2024_2025.parquet` (87,908 rows, Jan 2024–Dec 2025, same cohort rule as his `historical_2024_2025.parquet`, which has 87,907) and `derived/for_chen/luis_features_scoreboard_2026.parquet` (72,670). Keyed by `cve_id`, with all our leak-free features, `y_nov4` (76 positives; 75 through Sep 2025) and `luis_y_dec31_of_year`. Built by `python -E -P luis/analysis/build_features.py --all-months`.
- **From Chen → already beyond our §11:** Granite 30B structured extraction on all CVEs (§11.2, §11.6), purged CV with bootstrap CIs (§11.3), granite-embedding-english-r2 (§11.5), LoRA 8B, memorization probe (0/122), strong negative filters (unknown CWE 0/3,551; CVSS 4 E:P 0/2,669; WordPress CNAs), out-of-band advisory 5.5×, EOL text 13×, and the missed-positives analysis (22/52 invisible at publish time: 0% exploit code).

### 14.4 Proposed split of work (Nov 7–8)
- **Chen (GPU box):** finish extraction → add our TF-IDF + adjacent IDs + LightGBM to his CV with `y_nov4` → final LR + LightGBM + LoRA rank average → cap 3–5 → score the 2026 scoreboard once → `top500.csv`. Drop `txt_in_wild`.
- **Luis side:** the write-up (columns-used table with the traffic light, leak demo, A/B + CV methodology, insights from both docs), the video script, the hub's Top 500 table fed from the final ranking with Granite reasons (Chen's nearest-KEV matches make great explanations), and a sanity check of the final file (Oracle July batch, month mix, WordPress = 0).
- **Until Chen's ranking exists,** `submissions/top500_v3.csv` is the team's fallback.

### 14.5 Chen's insights tested on our setup (`analysis/chen_insights.py`)

Same A/B replicas, labels `y_nov4`, `txt_in_wild` dropped everywhere, LightGBM deterministic with 10 seeds, LR with C picked **inside the training year only** (months 1–6 → 7–11, average precision; it picked C = 1e-4 to 3e-3, close to Chen's 1e-4 to 3e-4, except 0.1 once for the PCA variant). PCA-64 of our Granite-30M embedding was fit on 38,129 pre-2024 non-KEV descriptions.

**New features from Chen, measured on our test replicas** (2024+2025 Jan–Sep, 64,120 CVEs, base rate 0.117%; `reports/chen_features_lift.csv`):

| Feature | CVEs | KEV | Lift | Chen's number |
|---|---|---|---|---|
| `txt_auth_bypass` | 588 | 8 | **11.6×** | 9.5× |
| `txt_eol` (end-of-life text) | 401 | 3 | 6.4× | 13× |
| `assigner_ransom_kev` > 0 | 18,718 | 47 | 2.2× | 4.3× |
| `is_patch_tuesday` | 6,203 | 15 | 2.1× | 2× |
| `twins` (same text, assigner and day) | 2,947 | 7 | 2.0× | 1.6× |
| `old_id` (ID reserved in an earlier year) | 14,346 | 10 | 0.6× | 0.4× |
| `wp_cna` (Wordfence/Patchstack/WPScan) | 14,136 | **0** | 0 | 0 |
| `txt_vuldb_disclosed` | 5,826 | **0** | 0 | 0.2× |
| `cwe_unknown` (blank `cwe_ids`) | 3,020 | **0** | 0 | 0 of 3,551 |
| `cv4_e_poc` (CVSS 4 E:P) | 2,645 | **0** | 0 | 0 of 2,669 |
| `cv4_urgent` (U:Red/Amber) | 188 | 0 | 0 | 8× |
| `cv4_auto` (AU:Y) | 180 | 0 | 0 | 3× |

Note: in this dataset NIST's `NVD-CWE-noinfo` / `NVD-CWE-Other` placeholders appear as a **blank** `cwe_ids` (3,556 rows in 2024-01..2025-09; Chen counts 3,551). CVSS 4 urgency/automatable don't replicate on our shorter labels (0 of ~180 each), so treat them as unproven.

**Hits at scoreboard density** (A top 243, B top 198; `reports/chen_insights_results.csv`):

| Model | No cap | Cap 3 | Cap 5 |
|---|---|---|---|
| Exploit code, then EPSS | 19 | 22 | 20 |
| LightGBM, Luis safe | 20 | 21 | 23 |
| LightGBM + Chen safe features | 19 | 21 | 22 |
| LightGBM + Chen safe + careful | 19 | 22 | 22 |
| LR, Luis safe | 19 | 21 | 21 |
| LR + Chen safe + careful | 20 | 24 | 22 |
| LR + Chen + Granite-30M PCA-64 | 18 | 21 | 20 |
| Rank avg LightGBM(+Chen) + LR(+Chen+PCA) | 21 | 23 | 23 |
| Rank avg LightGBM(Luis) + LR(+Chen+PCA) | 21 | 23 | 24 |
| **Rank avg LightGBM(Luis) + LR(+Chen)** | **25** | **27** | 26 |
| Rank avg of 3 (+ LR with PCA) | 24 | 25 | 25 |

**Pooled, Chen-style** (both test years, top 441 of 64,120, 75 positives, 1,000-rep paired bootstrap; `reports/chen_insights_bootstrap.csv`):

| Model | Hits@441 (95% CI) | vs exploit+EPSS (95% CI) | Avg precision |
|---|---|---|---|
| Exploit code, then EPSS | 19 (12–29) | – | 0.029 |
| Exploit code, then EPSS, cap 3 | 22 (14–31) | −2 to +8 | 0.034 |
| LightGBM, Luis safe | 23 (14–33) | −4 to +11 | 0.035 |
| LR + Chen safe + careful, cap 3 | 24 (15–33) | −4 to +12 | 0.039 |
| **Rank avg LightGBM(Luis) + LR(+Chen)** | **26 (16–37)** | **+1 to +14** | **0.058** |
| Rank avg LightGBM(Luis) + LR(+Chen), cap 3 | 26 (16–37) | +0 to +14 | 0.061 |
| Rank avg of 3 | 26 (17–36) | +0 to +14 | 0.050 |

**What this says:**
1. **Chen's main insight replicates:** two different model families averaged beat either one. LightGBM (trees on our features) plus a heavily regularized LR (linear on our + his features) doubles average precision vs the baseline (0.029 → 0.058–0.061), and it's the first result here whose bootstrap CI excludes zero.
2. **Chen's features help the LR but not LightGBM.** The trees already get the same information from `assigner_rate_3y`, `n_same_day` and the text model.
3. **His strong negative filters replicate exactly** (WordPress CNAs, VulDB "disclosed", blank CWE and CVSS 4 E:P: all 0 KEV on our data too). They're safe to use as features; as hard filters, check the July Oracle batch first.
4. **Our Granite-30M PCA hurts the LR** (18–21). Chen's r2 PCA-64 helped his. Use his embeddings, not ours.
5. **The cap helps the baseline too** (19 → 22 with cap 3), so part of "the cap helps" is just vendor diversification, whatever the model.
6. **Chen's "invisible positives" replicate** (`reports/chen_missed_positives.csv`). With the 3-model average, 44 of 75 positives fall outside the top 882 (2 × K). The missed ones had exploit code 18% of the time (vs 74% for found ones), median EPSS percentile 0.62 (vs 0.96), median CVSS 7.8 (vs 9.0), and needed user interaction 14% of the time (vs 3%). Examples: XWiki CVE-2025-24893, Windows CVE-2024-26169 and CVE-2024-21338, SonicWall CVE-2024-53704, SharePoint CVE-2024-38094, Kentico CVE-2025-2746/2747, TeleMessage CVE-2025-48928.

### 14.6 Clean model vs leak-only model (`chen_insights.py --leak-test`, `reports/leak_vs_clean.csv`)

Same splits and bootstrap as §14.5. "Leak" = `nvd_status` analyzed, `n_references`, `n_affected_cpes`.

| Model | A@243 + B@198 | Pooled hits@441 (95% CI) | vs exploit+EPSS | Avg precision |
|---|---|---|---|---|
| Exploit code, then EPSS | 19 | 19 (12–29) | – | 0.029 |
| **Clean: LightGBM + LR(+Chen) rank avg** | **25** | **26 (16–37)** | **+1 to +14** | 0.058 |
| Leak only: LightGBM on the 3 leak columns | 2 | 2 (0–5) | −27 to −10 | 0.002 |
| Leak only: LR on the 3 leak columns | 4 | 4 (1–8) | −24 to −8 | 0.003 |
| Leak only: sort by `n_references` | 5 | 4 (1–9) | −24 to −7 | 0.003 |
| Leak + exploit signals: LightGBM | 8 | 7 (3–12) | −20 to −5 | 0.011 |
| Clean + leak: LightGBM + LR rank avg | 29 | 28 (18–39) | +2 to +16 | 0.075 |

- **The leak is a filter, not a ranker.** 100% of test positives are NVD-analyzed, but so are 73.9% of all test CVEs. Not-analyzed CVEs reach KEV at 0.000% and analyzed ones at 0.158%. It removes a quarter of the haystack and can't pick the top 500 by itself.
- **Clean and leak models pick different CVEs:** their top 441 share 3 CVEs and 0 hits. Added to the clean model, the leak gives +2 to +4 hits. That's the "inflation" number for the write-up.
- **For 2026 the leak would be stronger than this shows.** Our 2024–25 replicas are now 74% analyzed; the 2026 scoreboard is only ~50% (Analyzed 31,265 + Modified 5,118 of 72,670). The filter would cut half the 2026 haystack, so these numbers are a **lower bound** on what the leak is worth. This is exactly why the README warns about it, and why we excluded it.
- **Write-up message:** "We beat the baselines without NIST's after-the-fact metadata. Those columns alone can't rank (2–5 hits), and adding them would add 2–4 hits on our replicas, more on 2026, for the wrong reason."

### 14.7 Model zoo: heavier and different models (`analysis/model_zoo.py`)

Same setup as §14.5 (A/B, `y_nov4`, scoreboard density, pooled paired bootstrap vs exploit+EPSS **and vs the current best**, LightGBM + LR(+Chen)). Features: Luis safe (no `txt_in_wild`) + Chen safe/careful + a new "assigner heat" group (Chen's to-do #5: count, exploit-code share and mean EPSS percentile of the same assigner's CVEs from earlier snapshot months, each at its own snapshot). XGBoost and CatBoost ran on the GPU. Full run ≈ 11 min on the laptop.

| Model | A@243 + B@198 (no cap / cap 3) | Pooled hits@441 (95% CI) | vs exploit+EPSS | vs current best | AP |
|---|---|---|---|---|---|
| Exploit code, then EPSS | 19 / 22 | 19 (12–29) | – | −14 to −1 | 0.029 |
| LightGBM, Luis safe | 20 / 21 | 23 (14–33) | −4 to +11 | −8 to +2 | 0.035 |
| LightGBM, all features | 23 / 22 | 22 (13–32) | −6 to +11 | −10 to +2 | 0.037 |
| LR, Luis safe + Chen | 20 / 24 | 22 (13–30) | −5 to +10 | −11 to +1 | 0.037 |
| LR, all features (+ heat) | 16 / 21 | 18 (10–26) | −10 to +5 | −16 to −2 | 0.031 |
| XGBoost small (depth 4, 400 trees) | 23 / 24 | 22 (14–32) | −5 to +11 | −10 to +2 | 0.041 |
| **XGBoost big (depth 6, 1,500 trees)** | 23 / **26** | 23 (15–34); cap 3: 26 | −3 to +13 | −8 to +4 | **0.053** |
| CatBoost (assigner + main CWE as categoricals) | 13 / 15 | **14** (7–22) | −15 to +3 | −21 to −4 | 0.023 |
| LightGBM LambdaRank (query = month) | 23 / 23 | 21 (13–31) | −6 to +10 | −10 to +1 | 0.041 |
| Big linear (TF-IDF of the cohort + tabular) | 16 / 21 | 18 (10–27) | −10 to +8 | −15 to −1 | 0.028 |
| Current best: LightGBM + LR(+Chen) | 25 / 27 | 26 (16–37) | +1 to +14 | – | 0.058 |
| LightGBM(all) + LR(all, +heat) | 25 / 25 | 26 (16–37) | −1 to +14 | −4 to +4 | 0.052 |
| XGBoost small + LR(+Chen) | 25 / 26 | 25–26 | −1 to +14 | −5 to +3 | 0.053 |
| **XGBoost big + LR(+Chen), cap 3** | **29** | **28 (18–38)** | **+1 to +16** | −2 to +5 | **0.063** |
| LambdaRank + LR(+Chen) | 26 / 26 | 24–25 | −2 to +13 | −6 to +2 | 0.055 |
| CatBoost + LR(+Chen) | 17 / 21 | 19–21 | −8 to +10 | −14 to +2 | 0.033 |
| LightGBM + LR(+Chen) + CatBoost | 19 / 21 | 21–24 | −6 to +14 | −11 to +4 | 0.037 |
| LightGBM + LR(+Chen) + big linear | 25 / 26 | 25 | −2 to +14 | −5 to +4 | 0.050 |
| Everything (7 models) | 20 / 24 | 22–23 | −5 to +12 | −10 to +4 | 0.044 |

**Prediction diversity** (Spearman of pooled scores; `reports/model_zoo_correlation.csv`): the tree models agree with each other at 0.85–0.96 (top-441 overlap ~275–365 CVEs). The LR agrees with trees at only 0.68–0.73 (overlap ~150–165). CatBoost is the odd one out (0.54–0.70) but mostly because it's worse.

**Insights:**
1. **Gains come from pairing one tree model with one linear model, not from heavier models.** Every tree + LR pair beats its parts; averaging 7 models is worse than the best pair because weak, correlated members dilute it.
2. **XGBoost big is the best single model** (AP 0.053 vs 0.035 for LightGBM), and **XGBoost big + LR(+Chen) with cap 3 is the best option overall** (29 at density, 28 pooled, AP 0.063). Against the current best its CI is −2 to +5: better point estimate, not a proven improvement.
3. **High-cardinality categoricals overfit.** CatBoost with the raw assigner gets 14 hits. Encoding the assigner as a smoothed historical KEV rate (`assigner_rate_3y`) works; letting the model memorize assigners doesn't.
4. **Learn text from history, not from the cohort.** A TF-IDF model trained on the cohort's own 40–55 positives (big linear) gets 18. `text_kev_lr`, trained on ~1,000 past KEV descriptions, is still the #1 feature everywhere (CatBoost importance: `text_kev_lr` 20.5, `n_same_day` 10.8, `desc_len` 6.7, `assigner_rate_3y` 6.0).
5. **Assigner heat (Chen's to-do #5) doesn't help:** neutral for trees, worse for the LR.
6. **Ceiling of the current features:** across all single models, only **35 of 75** positives ever make a top 441. The other 40 are Chen's "invisible" positives. More model capacity won't find them; new information might (Granite 30B extraction, product-level history).

**`submissions/top500_v4.csv`** = XGBoost big (all features, 5 GPU seeds) + LR(Luis safe + Chen, C tuned inside training), rank average, cap 3, trained on all 2024+2025 months (`python -E -P luis/analysis/model_zoo.py --score-2026`). 237 with exploit flag, 10 Oracle, 0 WordPress CNAs, 4 blank-CWE, 147 assigners; shares 406 CVEs with v3. Month mix: Jan 41 · Feb 37 · Mar 61 · Apr 53 · May 67 · Jul 110 · Aug 69 · Sep 62.

### 14.8 Combining Luis + Chen on the forward test (`analysis/team_merge.py`)

Chen pushed his features (`chen/processed/*.parquet`): 29 metadata features, 55 features from his **Granite 4.2 30B** world-knowledge extraction, Granite Embedding r2 nearest-KEV similarity, the raw extraction (`llm_extract.parquet`), and his out-of-fold **LoRA 8B** scores. We rebuilt his ~100-feature "best LR" (his 15 base features from our columns + his metadata + our TF-IDF as a logit + his Granite 30B features) and tested everything on **our A/B replicas**, the strictly forward test he doesn't have. LoRA scores come from his 5-fold CV (a different, partly non-forward regime), so rows marked "+LoRA" are an estimate. Shared modules: `analysis/common.py`.

| Model | A@243 + B@198 (no cap / cap 3) | Pooled hits@441 (95% CI) | vs exploit+EPSS | vs v4 | AP |
|---|---|---|---|---|---|
| Exploit code, then EPSS | 19 / 22 | 19 (12–29) | – | −16 to −1 | 0.029 |
| XGBoost big (Luis) | 23 / 26 | 23–26 | −1 to +15 | −6 to +3 | 0.053 |
| **LR Chen-style (base + meta + TF-IDF + Granite 30B)** | 25 / **28** | 25–27 | +0 to +15 (cap 3) | −8 to +7 | 0.043 |
| LR Chen-style + Luis safe features | 24 / 24 | 23–26 | | | 0.043 |
| XGBoost big with Chen's features | 25 / 26 | 26–28 | −1 to +16 | | 0.048 |
| LoRA 8B alone (Chen OOF) | 23 / 25 | 23–26 | | | 0.040 |
| **v4: XGBoost + LR(Luis + Chen-lite), cap 3** | 26 / **29** | **28 (18–38)** | **+1 to +16** | – | **0.063** |
| XGBoost + LR(Chen-style) | 25 / 26 | 26 | −1 to +14 | −7 to +2 | 0.052 |
| LightGBM + LR(Chen-style) | 25 / 24 | 26 | | | 0.052 |
| XGBoost + LightGBM + LR(Chen-style) | 24 / 25 | 25–26 | | | 0.057 |
| **+LoRA: v4 + LoRA** | **29** / 27 | **30 (20–41)** | **+4 to +18** | −3 to +9 | **0.069** |
| +LoRA: XGBoost + LR(Chen-style + Luis) + LoRA | 28 / 27 | 31 (20–43) | +5 to +20 | −2 to +10 | 0.062 |
| +LoRA: LR(Chen-style) + LoRA | 28 / 28 | 31 (21–42) | +5 to +20 | −3 to +12 | 0.054 |
| +LoRA: LightGBM + LR(Chen-style) + LoRA (Chen's recipe) | 28 / 27 | 30 | +4 to +19 | | 0.065 |
| Chen's own OOF 3-model average (his 5-fold) | 26 / 27 | 28–29 | +2 to +18 | | 0.052 |

**What this says:**
1. **Without LoRA, v4 is still the best combination on the forward test.** Chen's Granite 30B features make his LR the **best single model at density** (28 with cap 3), but adding it to XGBoost doesn't beat v4's lighter LR (26 vs 28 pooled).
2. **Chen's LoRA 8B is the piece that adds: +2 to +4 hits on top of every ensemble.** All "+LoRA" combinations have bootstrap CIs vs the baseline starting at **+3 to +5**, the strongest evidence in the project. Same direction as Chen's own CV (+4 to +6).
3. **Proposed final pipeline: v4 components (XGBoost big + LR) + Chen's LoRA 8B, rank average, cap 3.** Expected ≈ 30 of 75 on the practice exam (an estimate, see the LoRA caveat). It only needs one thing from Chen: LoRA 8B scores for the 72,670 scoreboard CVEs, trained on 2024-01..2025-09, as `chen/submissions/lora8b_scoreboard_2026.parquet` with columns `cve_id, lora_score`. Then `python -E -P luis/analysis/team_merge.py --score-2026` writes `submissions/top500_v5.csv` automatically.
4. **Explanations now come from Granite 30B.** `reports/top500_v4_explained.csv` (also the hub's Top 500 table) gives each CVE a product name and type from Chen's extraction plus a reason built from the actual signals: public exploit code, EPSS percentile, impact and login need, attacker appeal and typical attacker, end-of-life, the most similar past KEV entry (Granite Embedding r2), and the assigner's KEV history. Example, #4: *Progress ShareFile Storage Zones Controller: public exploit code (Nuclei, GitHub PoC); EPSS in the top 1% of its month; remote code execution with no login needed; high attacker appeal (ransomware or extortion groups); resembles KEV entry CVE-2023-24489 (similarity 0.93); 5 past KEV entries from this assigner.*

### 14.9 Static prior: more training years without exploit signals (`analysis/static_prior.py`)

**Why:** the A/B models train on one replica year (40–55 positives), and 40 of 75 test positives are invisible to every model. Exploit signals only exist for 2024–25, but text, assigner, CVSS, siblings and KEV history exist for older years too.

**How:** cohorts for 2022, 2023, 2024 built with the replica rules (published in month m, not on KEV by the end of m), features that need no signals (text flags, TF-IDF KEV model, assigner/product history, CVSS parts, sibling counts, Chen-lite), and labels = KEV added after the snapshot month **and before Jan 1 of the test year**, so both tests stay strictly forward. Cohorts before 2022 can't be used: the KEV catalog started in Nov 2021, so history features would be empty. Training sets: A (test 2025) 93,606 CVEs / **162 positives** (2022: 66, 2023: 56, 2024: 40); B (test 2024) 53,738 / 95. Cohort caches: `derived/static/static_cohort_{2022,2023,2024}.parquet`.

| Model | A@243 + B@198 (no cap / cap 3) | Pooled hits@441 | vs exploit+EPSS | vs v4 | No-code positives found (of 44) |
|---|---|---|---|---|---|
| Exploit code, then EPSS | 19 / 22 | 19 | – | −16 to −1 | 0 |
| Static prior LR | 16 / 17 | 14–15 | −14 to +5 | −21 to −4 | 7 |
| Static prior LightGBM | 17 / 16 | 17 | −13 to +7 | −19 to −2 | 7 |
| **v4 (reference)** | 26 / **29** | **28** | +1 to +16 | – | 5 |
| v4 + static LR | 22 / 22 | 24–25 | | −10 to +4 | 9 |
| v4 + static LightGBM | 24 / 26 | 25–27 | | −9 to +6 | 8 |
| v4 + both static | 25 / 23 | 24 | | −11 to +4 | 10 |

**Verdict: rejected.** Four times the positives, but without exploit signals the prior is weaker than the baseline, and adding it to v4 costs 1–4 hits. It does catch a few more no-code positives (7–10 of 44 vs 5 for v4), but at the price of many false positives; only 3 of its hits are new vs v4. **Insight for the write-up:** the invisible positives aren't a training-data problem. With 4× more history, a model still can't tell them apart from ordinary CVEs at publish time.

### 14.10 Post-processing: Granite 30B demotion filters and vendor caps (`analysis/postprocess.py`)

Cheap, interpretable clean-ups on top of v4's scores (cached for A/B in `derived/ab_scores_v4.parquet`). **To avoid picking filters on the test data, each filter is first checked on the training year of each split** (A: 2024, B: 2025) and used only if its KEV rate there is below 0.25× the base rate (`reports/postprocess_filters_check.csv`):

| Filter (Chen's Granite 4.2 30B extraction) | 2024: CVEs / KEV / rate vs base | 2025: CVEs / KEV / rate vs base | Used |
|---|---|---|---|
| impact = denial of service only | 5,935 / 0 / 0 | 5,639 / 0 / 0 | yes |
| typical attacker = unlikely target | 6,505 / 0 / 0 | 8,032 / 0 / 0 | yes |
| attacker appeal = low | 12,797 / 1 / 0.07× | 13,495 / 1 / 0.06× | yes |
| product = WordPress plugin/theme | 9,120 / 0 / 0 | 9,922 / 0 / 0 | yes |
| install base = small/niche | 10,101 / 4 / 0.36× | 12,875 / 6 / 0.36× | **no** |

Filtered CVEs are pushed below everything else (score − 50), then the cap is applied (`reports/postprocess_results.csv`, `reports/postprocess_bootstrap.csv`):

| Variant on v4 | A@243 | B@198 | A+B | Pooled hits@441 | vs v4 + cap 3 | AP |
|---|---|---|---|---|---|---|
| v4, no cap | 18 | 8 | 26 | 26 | −4 to +3 | 0.060 |
| **v4 + cap 3 per assigner-month (reference)** | 18 | 11 | 29 | 26 | – | 0.062 |
| v4 + cap 3 per Granite vendor-month | 18 | 9 | 27 | 27 | −3 to +3 | 0.062 |
| v4 + cap 3 per assigner-day | 18 | 9 | 27 | 27 | −4 to +3 | 0.062 |
| v4 + both caps | 19 | 11 | 30 | 26 | +0 to +0 | 0.062 |
| v4 + Granite filters | 18 | 10 | 28 | 27 | −3 to +4 | 0.063 |
| **v4 + Granite filters + cap 3 (v4f)** | **20** | **11** | **31** | **27** | **−1 to +4** | **0.065** |
| v4 + cap 3 + only the DoS filter | 20 | 11 | 31 | 26 | −1 to +3 | 0.063 |
| v4 + cap 3 + only "unlikely target" | 19 | 11 | 30 | 26 | +0 to +3 | 0.063 |
| v4 + cap 3 + only "low appeal" | 19 | 11 | 30 | 26 | −1 to +3 | 0.064 |
| v4 + cap 3 + only WordPress | 19 | 11 | 30 | 27 | +0 to +3 | 0.063 |
| v4 + cap 3 + "small/niche" (failed the check) | 17 | 11 | 28 | 24 | −6 to +2 | 0.055 |

**Verdict:** the filters give a small but consistent gain (+2 at density, +1 pooled, higher AP), they're validated on training data only, and they're easy to explain ("we push down what Granite 30B says no attacker goes after"). Adopted as **v4f**. Note v4 itself scored 26 pooled in this run vs 28 in §14.7: XGBoost GPU randomness, another reminder that ±2–3 is noise.

**2026 (`python -E -P luis/analysis/postprocess.py --score-2026` → `submissions/top500_v4f.csv`, `derived/scored_2026_v4f.csv`):** 62 of v4's 500 picks are pushed out: unlikely target 27, **WordPress plugins 17** (assigned by CNAs other than Wordfence/Patchstack/WPScan, so our `wp_cna` flag missed them), low appeal 10, DoS only 8 (e.g. two Palo Alto PAN-OS DoS bugs). v4f: 224 with exploit code, 10 Oracle, 145 assigners; 438 shared with v4. Explained version: `python -E -P luis/analysis/team_merge.py --score-2026 --ranking derived/scored_2026_v4f.csv rank_v4f v4f` → `reports/top500_v4f_explained.csv` (hub table).

### 14.11 Can we use a time-series model (IBM Granite TimeSeries TTM, or a 1D RNN) on the signals?

**Per CVE: no.** At prediction time every CVE has exactly one signal snapshot, its own month: **0 of 72,670** scoreboard CVEs and **0 of 87,215** CVEs published in 2024–25 have any signal row before their publish month. A sequence model needs a sequence, and here each sequence has length 1. Later months exist for 2024–25 CVEs, but using them would be future information.

**Aggregate (per vendor / assigner / CWE): possible, low value.** The KEV daily series runs 1,517 days (Nov 3, 2021 → Dec 29, 2025; 50 months, median 18 additions a month, launch month 291). Per vendor it's very sparse: Microsoft 350, Apple 86, Cisco 82, Adobe 76, Google 67, Oracle 42, Apache 38, Ivanti 30 entries in 50 months, most months zero. A TTM forecast of "vendor KEV pressure" in 2026 would mostly reproduce `assigner_kev_prev_year` / `assigner_rate_3y`, and the aggregate signals we tried were neutral ("assigner heat" §14.7; Chen's prior signals and hidden-KEV features). **If we want IBM TimeSeries in the story,** the honest use is a small descriptive forecast (e.g. expected KEV additions per month) in the write-up, labelled as context, not as part of the ranking.

### 14.12 Statistical review: which differences are real? (`analysis/stats_review.py`)

All on the pooled practice exam (2024 + 2025 Jan–Sep, top 441 of 64,120, 75 positives). Files: `reports/stats_review_*.csv`, `reports/stats_review_log.txt`.

**1. Run-to-run variability.** XGBoost on GPU isn't bit-reproducible, so v4 was re-run 8 times (5 seeds each):

| | Mean of 8 runs | SD | Range | All 40 seeds averaged |
|---|---|---|---|---|
| v4 (cap 3) | 27.9 | 0.35 | 27–28 | 28 |
| **v4f (Granite filters + cap 3)** | **29.9** | 0.64 | 29–31 | 29 |

The filters add +1 to +3 in every one of the 8 runs.

**2. Paired tests vs exploit-then-EPSS (19 hits), Holm-corrected across the 10 comparisons:**

| Model | Hits | Positives only it finds / only baseline finds | McNemar p (Holm) | Bootstrap 95% CI of the gap | Bootstrap p (Holm) |
|---|---|---|---|---|---|
| v3: LightGBM + LR | 26 | 10 / 3 | 0.46 | +0 to +14 | 0.34 |
| v4 bagged | 28 | 12 / 3 | 0.21 | +1 to +16 | 0.18 |
| v4f bagged | 29 | 13 / 3 | 0.19 | +3 to +19 | 0.10 |
| v4f + LightGBM (3 models) | 29 | 13 / 3 | 0.19 | +2 to +17 | 0.12 |
| Chen's own OOF 3-model average | 29 | 13 / 3 | 0.19 | +2 to +18 | 0.11 |
| **v4f + Chen's LoRA (estimate)** | **30** | 13 / 2 | 0.07 | **+4 to +19** | **0.03** |

Against v4f, no model differs significantly (all Holm p ≥ 0.56). v4 vs v4f: the two top 441s differ by a single positive.

**3. Romano–Wolf set of models not significantly worse than the best** (single-step max-t, FWER 5%, best = v4f + LoRA, critical value 2.47): **every ML model is in the set** (v4f, Chen's average, v4, v3, XGBoost alone, the LR alone, even LightGBM alone and exploit+EPSS with a cap). **Only plain exploit-then-EPSS is out.** The data can say "ML beats the raw baseline" but can't rank our ensembles.

**4. Power** (80% power, two-sided 5%; from the SD of paired bootstrap differences):

| Comparison | SD of the paired gap | Smallest detectable gap | Positives needed to detect +2 |
|---|---|---|---|
| v4f vs exploit+EPSS | 4.1 | 11.4 hits | ~2,450 |
| v4f vs v4 (similar models) | 1.7 | 4.9 hits | ~440 |
| v4f + LoRA vs v4f | 3.1 | 8.7 hits | ~1,410 |

**5. 2026 projection** (projected = README 2026 anchor × model hits / anchor hits on our replicas; Gamma(h + 0.5) posteriors for every count):

| Model | Anchor: exploit-then-EPSS (8 in 2026, 19 here) | Anchor: EPSS-first (6 / 11) | Anchor: CVSS-first (7 / 5) |
|---|---|---|---|
| Exploit code, then EPSS | 8.0 (4–15) | 10 (5–21) | 27 (12–60) |
| **v4f** | **12 (7–22)** | 16 (8–31) | 40 (19–89) |
| v4f + Chen's LoRA | 13 (7–22) | 17 (8–32) | 42 (20–91) |

The exploit-then-EPSS anchor is the most comparable (same kind of signal as our strongest features), so **~12 hits (7–22)** is the honest expectation. The CVSS anchor is unstable because CVSS finds only 5 on our replicas. **But it reveals a shift:** in 2026 CVSS-first (7) beats EPSS-first (6), while on our replicas it's the opposite (5 vs 11). 2026 seems to reward severity more, and v4 uses no CVSS, so a CVSS-aware voter is a sensible hedge.

### 14.13 Nested hyperparameter tuning (`analysis/nested_tuning.py`)

To avoid tuning on the exam, the search runs **inside the training year only**: fit on months 01–06, validate on 07–11 (average precision), 30 random configurations each for XGBoost and LightGBM, then the XGBoost-vs-LR weight and the cap value; the winners are refit on the whole training year and evaluated **once** on the test year. Inner validation had only 19 (A) and 16 (B) positives.

| | A@243 | B@198 | A+B | Pooled | vs v4f default | AP |
|---|---|---|---|---|---|---|
| **v4f default** | **20** | **11** | **31** | **27** | – | **0.065** |
| v4f, tuned XGBoost | 19 | 10 | 29 | 26 | −5 to +3 | 0.055 |
| v4f, tuned XGBoost + weight + cap | 19 | 9 | 28 | 26 | −5 to +3 | 0.055 |
| LightGBM default + LR | 18 | 9 | 27 | 26 | −5 to +3 | 0.064 |
| LightGBM tuned + LR | 17 | 9 | 26 | 26 | −6 to +3 | 0.068 |

- **Inner gains didn't transfer:** XGBoost inner AP jumped 0.026 → 0.103 (A) and 0.049 → 0.083 (B), yet test hits dropped. That's overfitting to ~20 inner positives.
- **Choices flip between splits:** XGBoost depth 6 vs 8, learning rate 0.01 vs 0.03, positive weight 5 vs 100; ensemble weight 0.6 vs 0.7. The cap stayed at 3 in both.
- **Verdict: keep the defaults.** Write-up line: "we tuned honestly (nested, inside training only) and it didn't help, which is what the power analysis predicts."

### 14.14 The final build (`analysis/final_build.py`) and what Chen needs to send

**One command:** `python -E -P luis/analysis/final_build.py`. Equal-weight rank average of three voters (tree + linear + LLM, the pattern that kept winning on A/B):

| Voter | What | Status |
|---|---|---|
| tree | XGBoost big on Luis's features, **40 seeds averaged** (8 GPU runs × 5; cached in `derived/xgb_bag_2026.parquet`, ~7 min to rebuild) | in |
| linear | Luis's LR (safe + Chen-lite features) **averaged with Chen's best LR** (`chen/submissions/lr_best_scoreboard_2026.parquet`, `lr_full_llmx_noemb`, 100 features incl. CVSS and Granite 30B) | **in** (Nov 7, Chen's push `7d78aa0`) |
| LLM | Chen's Granite 4.2 8B LoRA (`chen/submissions/lora8b_scoreboard_2026.parquet`, `cv_control` recipe trained on all 75,148 CVEs of 2024-01..2025-09) | **in** |

Then the 4 Granite-30B demotion filters and a cap of 3 per assigner-month. Outputs: `submissions/top500_final.csv`, `derived/scored_2026_final.csv`, `reports/final_build_info.txt`, `reports/top500_final_explained.csv`, and the hub's `site/top500.json` (republish the hub after running).

**Current final build (Nov 7, ~8:40 pm; LLM voter = Chen's 4-seed LoRA average, pushed in `759c4cc`; `reports/final_build_info.txt`):**
- 500 unique CVEs, all in `scoreboard_2026.csv`, one column `cve_id`.
- **259 with public exploit code** (222 before any LoRA): the LoRA leans on exploit code.
- 19 Oracle; 0 Granite-demoted; 116 assigners; max 3 per assigner-month; median EPSS 0.004.
- Month mix: Jan 36 · Feb 31 · Mar 46 · Apr 51 · May 61 · Jul 113 · Aug 91 · Sep 71 (June CVEs are scored in July's snapshot).
- Top assigners: Apache 24, Microsoft 24, VulnCheck 23, MITRE 21, GitHub 21, Chrome 20, Oracle 19, Fortinet 16.

**Seed averaging mattered:**
- The earlier build with the single seed-0 LoRA shares **388 of 500** with this one.
- Chen's report (`chen/reports/lora_seed_stability.txt`): two single LoRA seeds share only 135–264 of their top 500, so the 4-seed average is the stable choice.
- Shares 290 with `top500_v4f.csv` and 271 with a Chen-style ranking (his LR + LoRA + our tree, no filters, no cap).

**Voter correlations:** the LoRA is the least correlated voter (Spearman 0.56 with the tree, 0.48 with the linear voter; tree vs linear 0.75).
- No NaN in any voter; Chen's files have 72,670 unique rows each.
- Expected on 2026: ~12 hits (80% 7–22) vs 8 for the best README baseline (§14.12). The previous build (no Chen voters) is reproduced by moving the two Chen files away and re-running.

**Where to train the LoRA: Chen's box.** Luis's laptop has an 8 GB RTX 4070; a Granite 4.2 8B LoRA needs ~16 GB in bf16 (4-bit QLoRA with bitsandbytes on Windows is fragile and much slower), and Granite 4.2 needs transformers 5.x, which broke Luis's environment. Chen has 4× H100, a tested recipe (~11 min per training) and vLLM scoring.

**Message for Chen's AI** (copy as is):

```
Context: Luis ran a statistical review of both workstreams on his strictly forward A/B test
(luis/context_Luis.md §14.8-14.13; pooled 2024+2025 Jan-Sep, top 441 of 64,120, 75 positives).
- v4f (Luis: XGBoost + LR, rank average, 4 Granite-30B demotion filters, cap 3) = 29-30 hits.
- v4f + Chen's LoRA 8B (your out-of-fold scores) = 30: the ONLY combination that beats
  exploit-then-EPSS (19) after Holm correction (p = 0.03, paired gap +4 to +19).
- Every ML ensemble is statistically tied with the best (Romano-Wolf); nested tuning made things
  worse. We stop tuning and build the final now.
- 2026 shift: in 2026 CVSS-first (7) beats EPSS-first (6), the opposite of our replicas. Your LR
  uses CVSS + Granite 30B, so it joins the final as a hedge voter.
Train on Chen's GPU box (Luis's laptop GPU has 8 GB; an 8B LoRA needs ~16 GB in bf16).

Please push exactly these two files to main:
1. chen/submissions/lora8b_scoreboard_2026.parquet, columns cve_id (str), lora_score (float).
   One Granite 4.2 8B LoRA, same recipe as cv_control (r16 / alpha 32, 20 random negatives per
   positive per epoch, 3 epochs, final epoch, no epoch picking), trained on ALL CVEs published
   2024-01..2025-09 (label = KEV by 2025-12-31). Score all 72,670 scoreboard_2026 rows with vLLM:
   logit(Yes) - logit(No). Higher = more likely.
2. chen/submissions/lr_best_scoreboard_2026.parquet, columns cve_id (str), lr_score (float).
   Your best LR (lr_full_llmx_noemb, ~100 features), C from your inner CV on the same rows,
   scored on all 72,670 rows.
Checks: 72,670 rows each, unique cve_id, no NaN, higher = more likely. Don't commit adapters or
weights. If the 30B LoRA beats the 8B in your CV, also push lora30b_scoreboard_2026.parquet (same
columns) and say so in chen/context_Chen.md.
Commit as "chen: 2026 scoreboard scores for the final", git pull --rebase origin main,
git push origin main, and tell Luis. He then runs `python -E -P luis/analysis/final_build.py`
(tree XGBoost 40 seeds + linear Luis LR & your LR + LLM your LoRA, equal weights, Granite filters,
cap 3) -> luis/submissions/top500_final.csv with explanations, and updates the team hub.
```

Chen delivered both files on Nov 7 (`7d78aa0`); no 30B file, because his 30B LoRA didn't beat the 8B (§14.15).

### 14.15 Chen's 3rd push (Nov 7, ~8 pm): anything better? (`analysis/chen_round3.py`)

**What came in** (`f4fbd4c`, `7d78aa0`, `87801ac`):
- the two 2026 score files for the final (§14.14);
- out-of-fold scores for a Granite 4.2 **30B** LoRA (`cv_lora30b_f*`) and for an 8B LoRA with Granite 30B's answers written into the prompt (`cv_llmx8b_f*`);
- the OOF of his final LR (`cv_oof_lr.lr_full_llmx_noemb`);
- a nested XGBoost tuning (`cv_xgb.parquet`, `xgb_tune_trials.csv`);
- a review of our final recipe on his CV (`reports/luis_recipe_check.txt`);
- a governance package (`chen/governance/`: model card, feature schema, drift table), useful for the write-up.

**Chen's own conclusions** (his 5-fold purged CV, 75 Nov-4 positives, top 517):
- the 30B LoRA alone gets 19 hits vs 25 for the 8B;
- in his 3-model average, the 30B gives 29 vs 32 with the 8B; it has the best AP when paired with the LR (0.101) but fewer hits in the top;
- llmx8b gives 29;
- tuned XGBoost ties fixed XGBoost and LightGBM (24 each), even though inner AP roughly doubled.

**Re-checked on our forward A/B** (pooled 2024 + 2025 Jan–Sep, top 441 of 64,120, 75 positives, KEV by Nov 4; Granite filters + cap 3 unless stated):
- Luis's voters are forward: XGBoost big with 20 seeds (CPU this time, see §12) and the LR, both trained on the other year.
- Chen's voters are his 5-fold OOF, so not strictly forward (his pushback 2), which makes them estimates.
- Normalization follows `stats_review.py`: Luis's voters are ranked within each split, Chen's OOF over the pooled rows, then the sum is re-ranked within the snapshot month.

| Recipe | Hits | 95% CI | vs exploit+EPSS | vs v4f | P(better than v4f) | AP |
|---|---|---|---|---|---|---|
| Exploit code, then EPSS (no cap) | 20 | 12–29 | – | −18 to −3 | 0.00 | 0.038 |
| v4f (XGBoost + Luis LR) = the final without Chen's files | 30 | 20–41 | +3 to +18 | – | – | 0.070 |
| **Final: XGBoost + avg(Luis LR, Chen LR) + LoRA 8B** | **32** | 21–43 | **+4 to +20** | −5 to +9 | **0.66** | 0.073 |
| same, no Granite filters | 31 | 21–42 | +3 to +19 | −6 to +8 | 0.56 | 0.070 |
| same, Luis LR only as linear | 30 | 20–42 | +3 to +19 | −5 to +7 | 0.52 | 0.071 |
| same, Chen LR only as linear | 30 | 20–41 | +3 to +18 | −6 to +6 | 0.47 | 0.062 |
| XGBoost + avg LR, no LLM | 28 | 19–39 | +2 to +15 | −6 to +2 | 0.10 | 0.077 |
| LLM voter = LoRA 30B | 28 | 19–39 | +1 to +15 | −8 to +5 | 0.25 | 0.067 |
| LLM voter = LoRA 8B + Granite answers (llmx) | 28 | 19–39 | +2 to +15 | −8 to +4 | 0.22 | 0.076 |
| LLM voter = avg(8B, 30B) | 31 | 20–41 | +3 to +18 | −6 to +7 | 0.49 | 0.074 |
| LLM voter = avg(8B, 30B, llmx) | 30 | 20–41 | +3 to +17 | −6 to +6 | 0.40 | 0.082 |
| 4 voters: + LoRA 30B as its own voter | 32 | 21–43 | +4 to +20 | −5 to +9 | 0.63 | 0.074 |
| **Chen's recipe: LightGBM + Chen LR + LoRA 8B, no filters** | **32** | 21–43 | +5 to +19 | −5 to +9 | 0.62 | 0.068 |
| Chen's recipe + Granite filters | 32 | 22–44 | +5 to +20 | −5 to +9 | 0.69 | 0.070 |
| LightGBM + avg LR + LoRA 8B (filters) | 32 | 21–44 | +4 to +20 | −5 to +9 | 0.67 | 0.078 |
| LightGBM + Luis LR + LoRA 8B (filters) | 31 | 20–42 | +3 to +19 | −6 to +7 | 0.54 | 0.072 |
| Chen's *saved* OOF average `avg_lrteam_lora_lgbm` (older "lrteam" LR + LightGBM trained in his CV; **not** his final recipe) | 29 | 19–39 | +1 to +16 | −9 to +6 | 0.36 | 0.055 |

*Correction:* an earlier version of this table labeled the last row "Chen's own OOF LR + LoRA + LightGBM". That was wrong. It's an older average saved in `cv_oof_avg.parquet`. **Chen's actual recipe scores 32 on our A/B too,** the same as our final. On his CV the two recipes are 33 (his) vs 32 (ours). Each wins by about 1 on its own exam, which is noise.

**What it says:**
- **The final recipe is the best point estimate (32) and needs nothing new.** It ties the 4-voter version. No single change is significant (every "vs v4f" interval contains 0), as the power analysis predicts (minimum detectable gap ≈ 5).
- **Bigger LLM, worse voter.** The 30B LoRA ranks like the 8B overall (Spearman 0.69) but finds fewer positives in the top. This agrees with Chen's CV and with our finding that heavier models don't reach the invisible positives (§10). Write-up line: "a 30B fine-tune didn't beat an 8B one".
- **The LoRA earns its place only next to both LRs:**
  - XGBoost + avg LR without the LoRA = 28;
  - with the LoRA = 32;
  - with Luis's LR alone, the LoRA adds 0 (30 → 30).
  So Chen's LR and his LoRA work as a pair, which matches his CV (LR + LoRA was his best 2-model pair).
- **Granite filters:** +1 here (31 → 32). Chen's CV: +0..+2, catching 1 of 75 positives (CVE-2024-23692, Rejetto HFS). Kept.
- **Sensitivity check, important for the write-up.** The same recipes were re-scored with the other two ways of putting voters on 0–1 (all per split; all within month). Final 32 / 31 / 27; v4f 30 / 30 / 29; 4 voters 32 / 32 / 30. Two or three hits move just from that implementation detail, CPU vs GPU XGBoost, or the seed count. That's also why our earlier "v4f + LoRA = 30, Holm p = 0.03" (§14.12) should be read as "consistently a few hits above the baseline", not as a precise number.

**Chen's pushbacks (his §8.5), and our answers:**
1. *"2026 rewards CVSS" isn't evidence of a shift* (7 vs 6 hits is noise). **Agreed.** The write-up must not call it a finding. His LR is in the final because it costs nothing and pairs well with the LoRA (above), not because of a shift.
2. *The "+LoRA" A/B numbers are slightly optimistic* (OOF, not strictly forward). **Agreed;** they're marked "est." everywhere. His forward-only LR check found no inflation.
3. *The LLM voter is a single run.* Valid, and now fixed: Chen pushed a 4-seed average and the final was rebuilt with it (§14.14).
4. *Check the final's composition before submitting.* Done above: Oracle 17 (cap holds, at most 3 per month), WordPress and other Granite-demoted 0, month mix spans Jan–Sep, 120 assigners.

**Decision:** submit `top500_final.csv` as built. Nothing in Chen's 3rd push beats it, and it's the recipe both of us signed off on before seeing these numbers.

### 14.16 Next: ONE linear model instead of two (plan, handed to Chen's GPU box)

**Goal.** Today the linear voter is the average of two logistic regressions: Luis's (56 features) and Chen's (100). Replace them with **one LR that uses every feature and heuristic that proved useful in either workstream**.

**Why:**
- One model to explain, so the write-up gets one coefficient table for "what drives exploitation".
- One model to maintain and one 2026 file.
- The two LRs agree only moderately (Spearman 0.59), so each sees something the other misses.

**The bar:** inside the final ensemble, the single LR must not lose to the 2-LR average.

**Status:** not trained yet. Chen's AI decides how (prompt at the end of this section).

#### The two LRs today

| | Luis's LR (`common.m_lr`, features `SAFE + CHEN_LITE`) | Chen's LR (`lr_full_llmx_noemb`, `chen/scripts/cv.py`, `final.py`) |
|---|---|---|
| Features | **56** = 43 "safe" + 13 "Chen-lite" (rebuilt by Luis from Chen's ideas) | **100** = 15 base + 27 meta + 3 team (from Luis's columns) + 55 Granite 4.2 30B extraction |
| Preprocessing | median imputation, StandardScaler | log/logit transforms in code, Granite NaN → 0, StandardScaler |
| Model | sklearn LogisticRegression, L2, `class_weight="balanced"`, max_iter 5000 | same (L2, balanced), max_iter 2000 |
| C | chosen inside training: fit months 01–06, validate 07–11 by average precision, grid 1e-4…1e-1. Final fit (2024 + 2025 replicas): **C = 1e-3** | inner grouped 3-fold CV (all positives + 15k sampled negatives) by AP, grid 1e-4…1e-2. Final: **C = 1e-4, the lower edge of the grid**, so the optimum may be stronger regularization |
| Training rows | A/B: one replica year. 2026: 2024 + 2025 replicas (y = KEV by Dec 31 of that year) | all 75,148 CVEs of 2024-01..2025-09 (label = KEV after the snapshot month through 2025-12-31; 119 positives) |
| Largest coefficients (per SD) | `text_kev_lr` +0.83 · `n_same_day` −0.73 · `n_adjacent_ids` −0.70 · `wp_cna` −0.50 · `txt_dos` −0.45 · `txt_unauth` +0.40 · `assigner_volume_3y` −0.36 · `assigner_ransom_kev` +0.35 · `txt_plugin` −0.34 · `cwe_unknown` −0.34 · `txt_kernel` +0.32 · `epss_missing` +0.31 · `days_to_cutoff` +0.30 · `id_year_lag` −0.28 | `text_kev_lr_logit` +0.43 · `log_batch_size` −0.22 · `llm_impact=denial_of_service` −0.20 · `llm_attacker_appeal` +0.19 · `llm_product_type=wordpress_plugin_or_theme` −0.14 · `eol_text` +0.14 · `any_code` +0.14 · `poc_github` +0.13 · `llm_install_base` +0.13 · `llm_product_type=enterprise_server_application` +0.13 · `out_of_band` +0.13 |

**Every column, with group, coefficient per SD, definition and the closest column in the other LR:** `reports/lr_columns_luis_chen.csv` (156 rows). The regexes behind Luis's text flags are in `derived/feature_dictionary.csv` and `analysis/common.py` (`add_chen_lite`).

**Luis's LR, all 56 columns:**
- **Signals (9):** `epss`, `epss_pct` (percentile within snapshot month), `epss_missing`, `exploitdb`, `metasploit`, `nuclei`, `poc_github`, `n_flags`, `any_flag`.
- **Time (2):** `age_days` (publication → end of snapshot month), `days_to_cutoff` (end of snapshot month → label cutoff).
- **Text (20):**
  - `desc_len`;
  - regex flags `txt_rce`, `txt_unauth`, `txt_remote`, `txt_wordpress`, `txt_plugin`, `txt_sqli`, `txt_xss`, `txt_traversal`, `txt_deser`, `txt_cmdi`, `txt_memory`, `txt_privesc`, `txt_edge`, `txt_ssrf`, `txt_upload`, `txt_dos`, `txt_kernel`;
  - `text_kev_lr` (TF-IDF logistic model trained only on past KEV descriptions; best single feature, AUC 0.91);
  - `text_kev_sim` (max cosine similarity to a past KEV description).
- **Siblings (4):** `n_same_day` (same assigner, same day, in the cohort), `sib_flags`, `sib_max_epss_pct`, `n_adjacent_ids` (same-day siblings within 3 ID numbers).
- **History, all before the CVE's year (8):** `assigner_kev_3y`, `assigner_rate_3y` (smoothed), `assigner_volume_3y`, `assigner_kev_all`, `assigner_kev_prev_year`, `kev_product_hist`, `kev_product_hits`, `kev_vendor_hist` (KEV product/vendor names found in the description).
- **Chen-lite (13):** `id_year_lag`, `old_id`, `wp_cna` (Wordfence/Patchstack/WPScan), `txt_eol`, `txt_vuldb_disclosed` (VulDB "exploit disclosed to the public" boilerplate), `txt_auth_bypass`, `assigner_ransom_kev`, `is_patch_tuesday`, `twins` (same assigner + day + description), `cwe_unknown`, `cv4_urgent`, `cv4_auto`, `cv4_e_poc` (CVSS 4 supplemental/threat metrics).

**Chen's LR, all 100 columns:**
- **Base (15):** `exploitdb`, `metasploit`, `nuclei`, `poc_github`, `any_code`, `epss_pct`, `epss_missing`, `cvss` (v3, else v4; median-filled), `cvss_missing`, `av_network`, `ac_low`, `pr_none`, `ui_none`, `scope_changed`, `wordpress_cna`.
- **Meta (27):**
  - assigner: `log_assigner_kev_rate`, `log_assigner_kev`, `log_days_since_assigner_kev`, `log1p_assigner_ransomware_kev`;
  - CWE: `log_cwe_kev_rate`, `cwe_unknown`, `cwe_broad`;
  - release batch: `log_batch_size`, `out_of_band`, `log1p_twins`, `patch_tuesday`, `old_id`;
  - siblings: `sibling_has_code`, `sibling_max_epss_pct`;
  - text: `eol_text`, `unauth_text`, `rce_text`, `auth_bypass_text`, `preauth_text`;
  - CVSS 4: `cvss4_urgent`, `cvss4_automatable`, `cvss4_at_none`, `cvss4_subsequent_high`;
  - KEV name history: `log1p_vendor_name_kev`, `log1p_product_name_kev`, `log1p_named_ransomware_kev`;
  - `log1p_n_versions`.
- **Team (3, from Luis's columns):** `text_kev_lr_logit`, `text_kev_sim`, `log1p_adjacent_ids`.
- **Granite 4.2 30B world-knowledge extraction (55):**
  - 7 ordinal/binary: `llm_vendor_kev`, `llm_product_kev`, `llm_product_ransomware_kev`, `llm_internet_exposed`, `llm_install_base`, `llm_attacker_appeal`, `llm_end_of_life`;
  - one-hots: `product_type` (16), `used_by` (5), `precondition` (7), `impact` (8), `fix_status` (4), `typical_attacker` (5), `exploitation_style` (3).

**Overlap:**
- 36 of Luis's 56 columns have a close counterpart in Chen's LR. Examples: `any_flag` ↔ `any_code`, `text_kev_lr` ↔ `text_kev_lr_logit`, `n_adjacent_ids` ↔ `log1p_adjacent_ids`, `sib_max_epss_pct` ↔ `sibling_max_epss_pct`, `wp_cna` ↔ `wordpress_cna`, `twins` ↔ `log1p_twins`, `txt_eol` ↔ `eol_text`, `assigner_ransom_kev` ↔ `log1p_assigner_ransomware_kev`. Several are marked "approx." in the CSV and need checking.
- **20 are Luis-only:** `epss` (raw), `n_flags`, `age_days`, `desc_len`, `txt_remote`, `txt_sqli`, `txt_xss`, `txt_traversal`, `txt_deser`, `txt_cmdi`, `txt_memory`, `txt_privesc`, `txt_ssrf`, `txt_upload`, `txt_kernel`, `assigner_kev_3y`, `assigner_volume_3y`, `days_to_cutoff`, `txt_vuldb_disclosed`, `cv4_e_poc`.
- **Chen-only:** CVSS base metrics, `log_cwe_kev_rate`, `cwe_broad`, `out_of_band`, `preauth_text`, `log1p_n_versions`, `cvss4_at_none`, `cvss4_subsequent_high`, `log1p_named_ransomware_kev`, and all 55 Granite features.

#### What we already know about merging them (evidence; read before designing)

| Evaluation | Chen LR | Luis LR | Naive union (Chen LR + every Luis column) | Avg of the 2 LRs as the linear voter |
|---|---|---|---|---|
| Chen 5-fold purged CV, alone (K 517, 75 Nov-4 positives): hits / cap 3 / AP | 26 / 28 / 0.070 | 24 / 25 / 0.037 (`lr_luis_only`) | **28 / 30** / 0.052 (`lr_best_luis_all`); with only Luis's 4 sibling columns: 26 / 29 / 0.071 (`lr_best_luis_sib`) | – |
| Chen forward-only (blocks 1–4, K 414, 66 positives): hits / cap 3 / AP | 23 / 23 / 0.062 | – | 23 / 23 / **0.075** | – |
| Our A/B at density (A@243 + B@198, cap 3; `team_merge.py`) | 28 | – | **24** (Chen-style + Luis safe columns) | – |
| Final ensemble on Chen's CV (tree + linear + LoRA, filters, cap 3; `chen/reports/luis_recipe_check.txt`) | 33 (his LR + LoRA + XGB) | 32 | not tested | 32 |
| Final ensemble on our A/B (pooled, §14.15) | 30 | 30 | not tested | **32** |

Reading:
- **A naive union is mixed.** On Chen's CV it gains 2 hits but loses AP. On his forward-only blocks it ties on hits and gains AP. On our A/B it loses 4.
- **Chen's naive union was really naive.** It appended every column of `luis/derived/for_chen/*.parquet` except `LUIS_DROP`. That includes columns that are **not** in Luis's LR: ordinal CVSS codes used as numbers (`cv_attack_vector` … `cv_availability_impact`), `cvss3`, `has_cvss4`, `cwe_kev_rate`, plus ~36 near-duplicates of his own columns. Anything above 10 got `log1p`. All of that went into one L2 penalty with C at the edge of the grid.
- **Averaging the two LRs works better than either one** inside the ensemble on our A/B (32 vs 30 / 30). Averaging two differently regularized models is itself a form of regularization. A single LR has to reproduce that, for example with a curated union, a better C, or per-block scaling.

#### Heuristics and rules we found useful (keep them; decide whether each belongs inside the LR or stays outside)

- **Outside the LR, as post-processing (validated, keep as is):**
  - **Granite 30B demotion** (impact = DoS only, typical attacker = unlikely target, attacker appeal = low, product = WordPress plugin/theme). Pushes 49–56% of CVEs down and catches 1 of 75 positives. Worth +0..+2 even though the LR already has these as one-hots.
  - **Cap 3 per assigner per snapshot month** (insurance for vendor batches such as Oracle's 1,097-CVE July).
  - **Rank average with the tree and the LoRA.**
- **Signals that carry the LR:** exploit code flags, EPSS percentile within month, the TF-IDF-vs-past-KEV text score (`text_kev_lr`), assigner and product/vendor KEV history, and **batch-release negatives** (`n_same_day`, `n_adjacent_ids`, `log_batch_size`, `twins`: big same-day releases rarely reach KEV). Also WordPress/plugin negatives, DoS negatives, end-of-life positives, and Granite's attacker appeal and install base.
- **Tested and NOT useful in an LR (don't add back without a reason):**
  - "assigner heat" features (worse for the LR, §14.7);
  - Granite Embedding PCA-64 (hurts the LR; `noemb` was Chen's best);
  - the Granite zero-shot score;
  - borrowed prior signals (assigner/vendor p90 EPSS, Nuclei/Metasploit share: 40 → 36–38 on Chen's CV);
  - 2022–23 static-prior training rows (§14.9);
  - month quotas;
  - CatBoost-style raw assigner IDs.
- **Leakage rules (hard):**
  - never use `nvd_status`, `n_references`, `n_affected_cpes`, `last_modified`, `cvss3_source`, `txt_in_wild`, `assigner_kev_ytd`, `sib_kev_known`;
  - every history count must be as of before the CVE's year (or before the snapshot), and every signal as of the end of its `snapshot_month`;
  - `days_to_cutoff` must use the label cutoff of the row's own setup: in Chen's CV, 2025-12-31 minus the end of the snapshot month; for 2026 scoring, 2026-11-04 minus the end of the snapshot month.

#### Decision rule (fixed before the run, to avoid picking the luckiest variant)

1. Choose the unified LR's settings **inside training only** (inner CV). Test at most **3 pre-specified versions** and report all of them, not only the best.
2. Compare as the **linear voter of the final ensemble**: tree + unified LR + LoRA 8B, Granite filters, cap 3, against tree + avg(Chen LR, Luis LR) + LoRA 8B, same post-processing. Use Chen's proxies from `luis_recipe_check.py` (`xgb_fixed_luis`, `lr_luis_only`) on his 5-fold purged CV **and** forward-only blocks. Report Nov-4 hits at K 517, cap 3, AP, and paired bootstrap CIs. Also report the LR alone.
3. **Adopt it if it is within 1 hit of the 2-LR version, or better, on both the 5-fold and forward-only checks, and AP is not clearly lower.** Ties go to the single LR (simpler, and it's what Luis wants). If it loses by 2 or more on either check, keep the 2-LR average.
4. Luis re-checks the out-of-fold scores on his A/B (`chen_round3.py`, new column) before the final is rebuilt.

#### How it plugs in

`final_build.py` uses `chen/submissions/lr_unified_scoreboard_2026.parquet` (`cve_id, lr_score`) as the **only** linear voter as soon as that file exists. Otherwise it keeps the average of the two LRs. So Chen pushes that file **only if the rule above adopts it**.

#### Prompt for Chen's AI (copy as is)

```
You are working in the team repo (branch main, only write inside chen/). Read luis/context_Luis.md
section 14.16 first (and 14.14-14.15 for the final recipe), plus luis/reports/lr_columns_luis_chen.csv
(all 156 columns of both LRs with coefficients, definitions and the closest equivalent in the other LR).

Task: Luis wants ONE logistic regression to replace the current linear voter, which is the average of
two LRs (Luis's 56 features and your lr_full_llmx_noemb with 100). It should use every feature and
heuristic that proved useful in either workstream. Reason about the design yourself; the context gives
the evidence, not a recipe. In particular:
- Your naive union (lr_best_luis_all) is mixed: +2 hits but lower AP on your 5-fold CV, tie on
  forward-only, -4 on Luis's A/B. It also took in ordinal CVSS codes as numbers, cvss3/has_cvss4/
  cwe_kev_rate, and ~36 near-duplicates of your own columns, under one L2 penalty with C = 1e-4 at the
  edge of your grid. Decide what a principled union looks like (dedup with checks on training data,
  consistent transforms, regularization range, per-block scaling or elastic net, which Luis-only
  columns to keep, the days_to_cutoff definition for your label horizon). Explain your choices.
- Keep the hard leakage rules in 14.16. Keep the Granite demotion filters and the cap of 3 per
  assigner-month as post-processing, outside the LR.
- Use the decision rule in 14.16, fixed in advance: at most 3 pre-specified versions, settings chosen
  inside training, compared as the linear voter of the final ensemble (tree + LR + LoRA 8B, filters,
  cap 3) against the current avg(your LR, Luis LR), on your 5-fold purged CV AND the forward-only
  blocks, reporting all versions. Adopt only if within 1 hit or better on both checks with AP not
  clearly lower; ties go to the single LR.

Deliverables (push to main, commit "chen: unified LR"; git pull --rebase origin main first):
1. Out-of-fold scores as new columns in chen/results/cv_oof_lr.parquet and cv_fwd_lr.parquet
   (lr_unified, or lr_unified_v1..v3).
2. chen/results/lr_unified_coefficients.csv (feature, coef_per_sd) and the exact column list.
3. chen/reports/lr_unified_compare.txt with all the comparisons above.
4. ONLY if adopted: chen/submissions/lr_unified_scoreboard_2026.parquet, columns cve_id (str),
   lr_score (float), trained on all 2024-01..2025-09 CVEs, scoring all 72,670 scoreboard rows; checks:
   72,670 rows, unique cve_id, no NaN, higher = more likely. Luis's final_build.py then uses it as the
   only linear voter automatically.
5. A section in chen/context_Chen.md: what you built, why, the results, and the verdict.
Do not commit IBM data, model weights or files over 50 MB. Then tell Luis.
```

## 15. Research sources

- Jacobs et al., "Enhancing Vulnerability Prioritization: Data-Driven Exploit Predictions with Community-Driven Insights" (EPSS): [arXiv 2302.14172](https://arxiv.org/pdf/2302.14172). EPSS is an XGBoost model on ~1,400 features: exploit code, description keywords, CWE, vendor, references.
- Evaluation of EPSS on severe KEV CVEs: [arXiv 2411.02618](https://arxiv.org/html/2411.02618v1). CVEs later added to KEV often didn't have a high EPSS beforehand, consistent with our finding 3.
- EPSS feature overviews: [INCIBE](https://www.incibe.es/en/incibe-cert/blog/epss-advancing-vulnerability-prediction-and-management), [Phoenix Security](https://phoenix.security/what-is-epss).
