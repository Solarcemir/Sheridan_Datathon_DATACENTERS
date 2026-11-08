# Project context · Chen · IBM "Before the Breach" (Sheridan Datathon 2026)

Handoff for teammates and their AI assistants. Last updated **Nov 7, 2026, ~8:00 pm**. Written so someone who never saw the session can continue it. Every number points to a file in `chen/` (§13). Hardware: 4× H100 Linux box (Granite 4.2 30B / 8B via vLLM, LoRA training). Companion: `luis/context_Luis.md` (§11 compares the two).

---

## 1. TL;DR

- **Best model (5-fold purged CV, judge-aligned labels).** An equal-weight **rank average of 3 models**:
  1. a strongly regularized logistic regression, 100 features (§6);
  2. Granite 4.2 8B + LoRA;
  3. Luis's LightGBM trained on Luis's features.

  It gets **32 hits** at scoreboard density (top 517 of 75,148; 75 positives) vs **20** for exploit-code-then-EPSS. Paired bootstrap difference **+5 to +19**. **33 with a cap of 3 per assigner per month.** With KEV-by-Dec-31 labels (119 positives): **47 vs 27**. (`reports/eval_all.txt`)
- **Simpler 2-model version** (LR + LoRA): **30 vs 20** (+4 to +17), the same at cap 3. Forward-only check (each block scored by a model trained only on earlier blocks): the LR alone scores 23 vs 18 for the baseline, almost the same as in the 5-fold CV (24 on the same blocks). **So the 5-fold numbers are not inflated by training on future blocks.** (§5.3)
- **Final 2026 scores for Luis's `final_build.py` (§8.4):** `submissions/lora8b_scoreboard_2026.parquet` (`cve_id, lora_score`) and `submissions/lr_best_scoreboard_2026.parquet` (`cve_id, lr_score`). Both were trained on all 75,148 CVEs of 2024-01..2025-09 (119 positives) and score all 72,670 scoreboard rows; higher = more likely. Luis builds the team's `top500_final.csv`.
- **IBM tech, measured:**
  - **Granite 4.2 30B world-knowledge extraction:** 55 features; +50% average precision on the simple LR. It rated 37% of CVEs "low attacker appeal", and only 5 of 27,716 of those reached KEV.
  - **Granite 4.2 8B LoRA:** ties the LR alone, but adds +4 to +6 hits when averaged with it.
  - **Granite Embedding r2:** drops out of the final LR (TF-IDF + extraction cover it); still used for nearest-KEV explanations.
  - **Granite 30B as a direct zero-shot scorer:** significantly *worse* than the baseline.
  - **Memorization probe:** 0 of 122 exploited CVEs identified from the ID alone.
- **Granite 30B LoRA does not beat the 8B** (§8.3): alone 19 vs 25 hits; in the 3-model average 29 vs 32. Writing Granite's 30B answers into the 8B prompt doesn't help either (29). So no `lora30b_scoreboard_2026.parquet`; the 8B is the LLM voter.
- **Tuning XGBoost doesn't help** (§8.3): a nested random search (60 settings, inner purged CV) raised inner AP from ~0.05 to ~0.08, but on the held-out folds the tuned model ties Luis's fixed settings and LightGBM (24 hits each). Same conclusion as Luis's nested grid search.

## 2. Data views (`scripts/build_historical.py`)

| View | Rows | Rule |
|---|---|---|
| `processed/historical_2024_2025.parquet` (not committed: contains IBM descriptions; rebuilds in ~1 min) | 87,907 | CVEs published 2024-01..2025-12. `snapshot_month` = publish month. EPSS + 4 flags from that month's row of `exploit_signals_*.csv` (no row → blank EPSS, flags 0). **Dropped (196):** CVEs on KEV by the end of their snapshot month, i.e. KEV `date_added` ≤ month end **or** signals `in_kev = 1` |
| `processed/scoreboard_2026_view.parquet` (not committed) | 72,670 | The scoreboard plus the same publish-time columns from `vulnerabilities.csv` |

- **Columns kept from `vulnerabilities.csv`:** description, assigner, CVSS 3 and 4 (score, severity, vector and parts), CWE.
- **CV window = 2024-01..2025-09:** 75,148 CVEs. Oct–Dec 2025 rows are dropped because they have no time to reach KEV before the KEV file ends.
- **1-row difference with Luis's `luis_features_hist_2024_2025.parquet` (87,908):** **CVE-2025-4664** (Chrome, published 2025-05-14). The May 2025 signals row says `in_kev = 1`, but it isn't in `known_exploited.csv`. We drop it (either source says KEV); Luis keeps it (KEV file only). It's a data inconsistency, not a method difference.

## 3. Labels

| Name | Definition | Positives (CV window) | Used for |
|---|---|---|---|
| `label` | added to KEV **after** the end of the snapshot month, on or before 2025-12-31 | 119 (2024: 67, 2025: 52 Jan–Sep) | training (more positives) |
| `nov4` = Luis's `y_nov4` | added after the snapshot month and on or before **Nov 4 of the publish year** (the judges' window) | **75** | **headline numbers** |

## 4. Leakage rules

- **Used:**
  - EPSS percentile within the snapshot month and the 4 exploit flags (snapshot-time);
  - description, assigner, CWE, CVSS 3/4 (score + vector parts; marked "careful": NIST may fill some in later);
  - KEV history **only up to the end of the previous year** for every row (2024 rows ≤ 2023-12-31; 2025 ≤ 2024-12-31; 2026 ≤ 2025-12-31), which mirrors the 2026 staleness exactly.
- **Excluded:**
  - `nvd_status`, `last_modified`, `n_affected_cpes`, `n_references`, `cvss3_source`, CVSS 2, all of `affected_products.csv` and `references.csv` *for the CVE itself*. 100% of 2024/25 KEV CVEs are NVD-analyzed, so these leak.
  - Description text "in the wild / actively exploited / aware of reports": measured **109×** lift on 43 CVEs, but descriptions are today's versions and may have been edited after the attacks. Agreed with Luis to drop it (`txt_in_wild`).
  - CVSS 4 `E` (exploit maturity), for the same reason.
  - Monthly signals of *other* CVEs (there are no 2026 monthly files).
- **Used with a declared exception:** `prior_signals.py` maps **older** CVEs (published before the cutoff) to vendors via their CPE rows. The leak concern is only about a new CVE's own CPE rows. For 2024 rows the reference month is Jan 2024 (the earliest signals month; a one-month look-ahead on *old* CVEs only). This feature ended up neutral and is **not** in the best model.
- **Hidden 2026 KEV CVEs (the "scoreboard gap"):** the 124 Jan–Sep 2026 CVEs in `vulnerabilities.csv` that are missing from the scoreboard (on KEV at their snapshot). Built as features in `fresh_kev.py`, with the same-year mirror from the KEV file for 2024–25. **Neutral, so not in the best model.** A question for the judges is drafted in §10.

## 5. Evaluation protocol (`scripts/cv.py`)

### 5.1 Splits used over the session

| Setup | Train | Test | Positives | Used for | Scored by a model that saw later data? |
|---|---|---|---|---|---|
| Forward holdout | all 2024 | 2025-01..09 | 52 | first experiments (§8.1) | no |
| 2024 validation | 2024-01..06 | 2024-07..12 | 39 | early many-shot choice | no |
| Leave-one-group-out | 2024 minus the CVE's group | each 2024 CVE | 67 | many-shot only | yes (accepted) |
| **5-fold purged time-blocked CV (main)** | 4 blocks minus purged groups | 5th block | **119 / 75** | **all model comparisons** | yes for blocks 0–3 |
| **Forward-only** | only earlier blocks, purged | blocks 1–4 | 103 / 66 | check of the above | no |

### 5.2 How the CV works

- **Folds (`processed/cv_folds.parquet`, `cv_fold{0-4}_{train,test}.txt`):** 5 consecutive publish-date blocks of ~15k CVEs.

| Block | Dates | Positives (Dec-31 label) |
|---|---|---|
| 0 | 2024-01-01..05-15 | 16 |
| 1 | 2024-05-15..10-11 | 32 |
| 2 | 2024-10-11..2025-02-11 | 30 |
| 3 | 2025-02-11..06-04 | 21 |
| 4 | 2025-06-04..09-30 | 20 |

- **Purging with hand-made KEV groups (`processed/kev_groups.csv`, 60 CVEs in 26 groups).** I read every 2024–25 positive with its CISA vendor/product, and grouped same-product CVEs disclosed or added to KEV together and exploit chains. Examples: Palo Alto Expedition ×3, the Linux USB Android chain ×4, Ivanti EPM ×4 (2024+2025), DELMIA Apriso ×3, Craft CMS ×3, plus cross-year same-product groups (FortiOS, SonicOS, Zimbra, Adobe Commerce, TP-Link WR841N). When block f is tested, any CVE whose group has a member in block f is removed from training (3–9 CVEs per fold). Each row has a reason.
- **K = 500 × test rows / 72,670** (517 for all blocks, 414 for blocks 1–4).
- **Metrics:** hits@K; average precision; 1,000-rep (500 for paired) bootstrap 95% CIs; **paired difference vs exploit-then-EPSS**; per-fold hits.
- **Seeds and tuning:**
  - LR: C chosen by an inner grouped 3-fold CV on the training blocks only (average precision; grid 1e-4..1e-2; 15k negatives subsampled for the inner search). It almost always picks C = 1e-4..3e-4.
  - LightGBM: 10 seeds, deterministic, rank-averaged.
  - LoRA: one seed, fixed recipe (§7), final epoch only (no epoch picking).
- **Cap:** at most C CVEs per (assigner, snapshot month) in the ranking, applied after scoring.

### 5.3 Is the 5-fold CV optimistic? (`reports/eval_all.txt`)

Same rows (blocks 1–4, 66 Nov-4 positives, K = 414):

| Model | Forward-only | 5-fold |
|---|---|---|
| Exploit code, then EPSS | 18 | 18 |
| LR, base features | 21 | – |
| LR + meta + embedding + TF-IDF | 23 (+0..+11) | – |
| **Best LR (100 features)** | **23 (+0..+11)** | **24** |
| LR, ours + all of Luis's features | 23 | – |
| LightGBM on Luis's features | 22 | – |
| LightGBM on our features | 15 | 14 |
| avg(Luis-feature LR, Luis LightGBM) | 24 | – |
| avg(best LR, LoRA) | not run (needs 4 forward LoRA trainings) | 27 |

**The difference is ≤ 1 hit, so training on future blocks barely inflates the CV.**

## 6. Features

The best LR uses groups 1, 2, 5 and 6 (**100 features**). All are standardized, then L2 logistic regression with balanced class weights.

| # | Group (file) | Count | Features | Why it's known at the snapshot |
|---|---|---|---|---|
| 1 | Base (`stack.py: features`) | 15 | 4 exploit flags + any; EPSS percentile within the snapshot month + missing; CVSS (3, else 4) + missing; AV network, AC low, PR none, UI none, scope changed; WordPress CNA (Wordfence/Patchstack/WPScan) | signals at the snapshot; CVSS from the CNA at publish ("careful") |
| 2 | Metadata (`meta_features.py`) | 27 | assigner KEV rate (smoothed, 3 yrs), KEV count, ransomware count, days since last KEV; CWE KEV rate; release size, out-of-band, near-identical twins, sibling has exploit code / max EPSS; old reserved ID; Patch Tuesday; text: EOL, unauthenticated, pre-auth, auth bypass, RCE, number of versions; CVSS 4 urgency (Red/Amber), automatable, AT:N, subsequent-system high; vendor and product name found in KEV; unknown CWE; broad CWE | publish-day facts; KEV history ≤ previous year end |
| 3 | Embedding nearest-KEV (`emb_features.py`) | 4 | cosine to the most similar KEV entries (max, top 5), to typical CVEs, and the difference | KEV ≤ previous year end; Granite Embedding r2 |
| 4 | Embedding PCA (`cv.py: emb_pca`) | 64 | description embedding reduced to 64 dims (PCA fit on 2021–23 CVEs only) | description |
| 5 | Luis's TF-IDF (`team_features.py`) | 3 | `text_kev_lr` (as a logit), `text_kev_sim`, adjacent IDs | KEV before Jan 1 of the year; verified: 925 / 1,105 / 1,343 KEV positives, same as Luis |
| 6 | Granite 30B world knowledge (`extract.py` → `llm_features.py`) | 55 | one-hots: product type (16), used by (5), precondition (7), impact (8), fix status (4), typical attacker (5), exploitation style (3); 0–3 scores: internet exposure, install base, attacker appeal; end-of-life; KEV and ransomware history of Granite's normalized vendor / product | Granite reads only description + assigner + CWE names (no EPSS, flags, dates, ID or KEV); history ≤ previous year end |
| – | Tested, not in the best model | – | `age_feats` (days public), `prior_feats` (borrowed EPSS from older CVEs), `fresh_feats` (hidden KEV), Luis's whole feature block | see §8 |

**Strongest single signals** (2024-01..2025-09, base KEV rate 0.16%; `reports/meta.txt`, extraction lifts in §9):

| Signal | Lift |
|---|---|
| VulnCheck as assigner | 14× |
| EOL text | 13× |
| auth-bypass text | 9.5× |
| CVSS 4 urgency Red/Amber | 8× |
| out-of-band release | 5.5× |
| unauthenticated text | 5.6× |
| assigner ransomware history | 4.3× |
| product named in KEV | 3.8× |
| WordPress CNAs | **0 / 17,291** |
| unknown CWE | **0 / 3,551** |
| CVSS 4 E:P | **0 / 2,669** |
| patch batch ≥ 20 that day | 0.3× |
| old reserved ID | 0.3–0.4× |

## 7. Models

| Model | Details |
|---|---|
| LR | `LogisticRegression(C=best, class_weight="balanced")` on standardized features; C from the inner CV. At C = 1 it overfits badly (the metadata variant *lost* 5 hits) |
| LoRA 8B | Granite 4.2 8B, PEFT LoRA r16 / alpha 32 / dropout 0.05 on q, k, v, o, gate, up, down. Prompt = instruction + CVE card (publish month, assigner, CWE names, CVSS 3/4 + vectors without the X metrics, EPSS percentile, 4 flags, description ≤ 600 chars) + "Will this vulnerability be added to the CISA KEV catalog? Answer Yes or No." Non-thinking chat template. Loss = BCE on logit(Yes) − logit(No) at the answer slot, computed by running only the last hidden state through the 2 lm_head rows. Each epoch: all positives + 20 random negatives per positive, resampled; 3 epochs; lr 1e-4 cosine, 5% warmup; batch 16; final epoch used. ~11 min per fold on one H100. Scored with vLLM 0.32 (`--lora`) |
| LightGBM (Luis's) | `n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5, scale_pos_weight=25, deterministic=True, force_row_wise=True`; 10 seeds, rank-averaged |
| Average | mean of per-model percentile ranks over the CVEs scored; then the optional cap |

## 8. Every experiment, with numbers

### 8.1 Granite as a direct scorer (2025 forward holdout, 52 positives, scoreboard-equivalent top 243)

| Run | Top 243 | Top 500 | Top 1,000 | Note |
|---|---|---|---|---|
| CVSS 3 first | 8 | 8 | 14 | |
| EPSS first | 8 | 8 | 13 | |
| Exploit code, then EPSS | 14 | 18 | 22 | |
| Granite 4.2 30B zero-shot (logP Yes − No) | 10 | 16 | 17 | 23/52 positives below rank 5,000 |
| Granite 30B many-shot, 201 examples from 2024 (67 positives + 67 near-misses + 67 k-means representatives), 45k-token prompt | 10 | 15 | 22 | |
| LoRA 8B control, epochs 1 / 2 / 3 | 13 / 13 / 16 | 16 / 15 / 20 | 18 / 20 / 23 | |
| LoRA 8B, 670 **fixed** hard negatives (out-of-fold LR false positives ranks 51–720) + random | **1** / 15 / 11 | 1 / 20 / 16 | 4 / 24 / 23 | collapsed at epoch 1: the hard negatives had exploit code 51% of the time vs 31% for positives |
| LoRA 8B, 670/epoch sampled from hard ranks 51–2,000 + random | 14 / 15 / 11 | 24 / 20 / 16 | 26 / 24 / 23 | overfits by epoch 3 |
| LoRA 8B, metadata lines in the prompt | 6 / 6 / 15 | 6 / 14 / 19 | 15 / 19 / 25 | numbers in prompts don't help |

**Lessons:**
- **The first many-shot example set was wrong:** its "hard negatives" (top exploit code + EPSS) had exploit code 100% of the time vs 31% for positives, which teaches the inverse of the truth.
- **Fixed with semantic near-misses** (most similar non-KEV description, same attack vector, CVSS ±1.5) **plus k-means representatives,** checked with a standardized-mean-difference balance table.
- **The top ~50 out-of-fold false positives look like real exploitation missing from KEV** (Ivanti Connect Secure XXE, GoAnywhere MFT, Splunk path traversal, Flowmon), so they were skipped as negatives.

### 8.2 Main 5-fold CV table (`results/cv_oof_lr.parquet`, `cv_oof_lora.parquet`; `reports/eval_all.txt`)

75,148 CVEs; 119 positives (Dec-31) / 75 (Nov-4); K = 517.

| Model | Hits Dec-31 | **Hits Nov-4** | Nov-4 cap 3 | Nov-4 vs baseline (95% CI) | AP Dec-31 | AP Nov-4 |
|---|---|---|---|---|---|---|
| Exploit code, then EPSS | 27 | **20** | 23 | – | 0.046 | 0.032 |
| Granite 30B zero-shot alone | 22 | – | – | Dec-31: −12..−1 | 0.027 | – |
| Embedding nearest-KEV alone | 7 | – | – | – | 0.008 | – |
| TF-IDF `text_kev_lr` alone | 14 | 10 | 11 | – | 0.020 | 0.010 |
| LR: base | 30 | 20 | 26 | – | 0.042 | 0.031 |
| LR: base + Granite extraction | 32 | 23 | 21 | – | 0.063 | 0.050 |
| LR: base + TF-IDF | 37 | 24 | 25 | – | 0.068 | 0.043 |
| LR: + meta | 34 | 24 | 24 | – | 0.055 | – |
| LR: + meta + embedding (nearest-KEV + PCA-64) | 36 | 26 | 27 | – | 0.065 | 0.065 |
| LR: + meta + embedding + TF-IDF | 40 | 27 | 27 | – | 0.069 | 0.063 |
| LR: + meta + embedding + TF-IDF + Granite | 37 | 24 | 24 | – | 0.074 | 0.067 |
| **LR best: base + meta + TF-IDF + Granite (no embedding), 100 features** | **40** | **26** | **28** | −1..+12 | **0.083** | **0.070** |
| LR best + Luis's sibling features | 40 | 26 | 29 | −1..+14 | 0.083 | 0.071 |
| LR best + all of Luis's safe features | 42 | 28 | 30 | **+1..+16** | 0.073 | 0.052 |
| LR, Luis's features only | 35 | 24 | 25 | −3..+12 | 0.052 | 0.037 |
| LightGBM, Luis's features only | 39 | 24 | 25 | −3..+11 | 0.055 | 0.039 |
| LightGBM, our best features | 31 | 17 | 21 | −10..+3 | 0.043 | 0.024 |
| sklearn HistGBM (depth 3, leaf ≥ 200), earlier holdout | 10–15 / 52 | – | – | – | – | – |
| LoRA 8B (5 folds) | 36 | 25 | 26 | −2..+13 | 0.045 | 0.029 |
| **avg(LR best, LoRA)** | **44** | **30** | 29 | **+4..+17** | **0.093** | 0.055 |
| avg(LR best, LightGBM ours) | 41 | 26 | 29 | −1..+14 | 0.067 | 0.045 |
| avg(LR best, LoRA, LightGBM ours) | 41 | 27 | 28 | +2..+14 | 0.092 | 0.053 |
| **avg(LR best, LoRA, LightGBM on Luis's features)** | **47** | **32** | **33** | **+5..+19** | 0.083 | 0.056 |

**Ablations with paired bootstrap** (Dec-31 labels unless noted):
- **Remove the 64 PCA dims:** 37 → 37, AP +0.002 (CI −0.012..+0.010). They do nothing.
- **Remove all embedding features:** 37 → 40, AP 0.076 → 0.083. Inside the LR+LoRA average: 43 → 44, AP 0.084 → **0.093 (CI on the difference −0.019..−0.003, significant)**. Once TF-IDF and the Granite extraction are in, the embeddings are redundant.
- **Granite extraction, all 55 vs 14 hand-picked vs none** (in the LR+LoRA average): **44** / 40 / 42; Nov-4: **30** / 28 / 28. All within CI. Kept all 55.
- **Same-year hidden-KEV features** (`fresh_kev.py`; name-level lift 3.3×, assigner 2.7× on Nov-4): **44 → 44 and 30 → 30** in the average. Neutral, same as Luis.
- **Time-left adjustment** (score + w·log F(days to Nov 4), F from training-fold positives: median 94 days after snapshot end, 75% within 176), w = 0 / 0.5 / 1 / 1.5 on the average: **29 / 29 / 28 / 26** (Nov-4). Neutral, not used.
- **Days public at snapshot** (`age_feats`): 40 → 38 hits, AP 0.063 → 0.069 (Nov-4). Tie, not used.
- **Borrowed signals from older CVEs** (`prior_signals.py`; assigner and vendor level: p90 EPSS, Nuclei/Metasploit share): 40 → 36–38. Worse (overlaps with assigner KEV rate). The lifts alone were real: assigner p90 EPSS > 0.05 = 4.2×, Metasploit share > 1% = 3.9×.
- **Cap per assigner-month:** helps the baseline (+3) and most single models (+0..+3); the best averages are flat (±1). Keep it as insurance for the 2026 Oracle batch (1,097 CVEs in one day; not seen in 2024–25).

### 8.3 LoRA variants and XGBoost tuning (`reports/lora_cv_compare.txt`, `reports/xgb_compare.txt`)

Same 5-fold purged CV, Nov-4 labels, K = 517.

| Model | Hits Nov-4 | Nov-4 cap 3 | vs baseline (95% CI) | AP Dec-31 | AP Nov-4 |
|---|---|---|---|---|---|
| LoRA 8B control (`cv_control`) | 25 | 26 | −3..+12 | 0.045 | 0.029 |
| LoRA 8B + Granite 30B answers as words in the prompt (`cv_llmx8b`, `--llmx`) | 23 | 26 | −4..+10 | 0.031 | 0.027 |
| LoRA **30B**, same recipe (`cv_lora30b`) | 19 | 25 | −8..+6 | 0.048 | 0.026 |
| **LR + LoRA 8B control + LightGBM (Luis features)** | **32** | **33** | **+5..+20** | 0.083 | 0.056 |
| LR + LoRA 8B + profile + LightGBM | 29 | 31 | +2..+16 | 0.073 | 0.058 |
| LR + LoRA 30B + LightGBM | 29 | 31 | +2..+16 | 0.085 | 0.059 |
| LR + LoRA 30B (no tree) | 28 | 29 | +2..+15 | **0.101** | 0.059 |

- **30B:** better overall ordering when paired with the LR (AP 0.101, the best of any model), but fewer hits in the top 517. Not used; 8B stays.
- **Profile in the prompt:** the LR already has those 55 features, so the LoRA's errors become more like the LR's. Not used.

**Nested XGBoost tuning** (`scripts/xgb_tune.py`, `scripts/xgb_compare.py`; trials in `results/xgb_tune_trials.csv`, out-of-fold scores in `results/cv_xgb.parquet`). For each outer fold, 60 random settings (Luis's XGB_BIG included) were ranked by pooled inner AP on 3 purged time blocks of that fold's training rows only; the winner was refit with 5 seeds. Two feature sets: Luis's 53 features, and ours + his (153).

| Tree model | Alone, hits Nov-4 | AP Nov-4 | In LR + LoRA + tree: hits / cap 3 | vs current ensemble (95% CI) |
|---|---|---|---|---|
| LightGBM, Luis features (current) | 24 | 0.039 | **32 / 33** | – |
| XGBoost, Luis's fixed settings, his features | 24 | 0.041 | 30 / 31 | −6..+0 |
| XGBoost tuned, his features | 24 | 0.043 | 29 / 29 | −7..+0 |
| XGBoost tuned, all 153 features | 24 | 0.037 | 30 / 29 | −7..+1 |
| XGBoost fixed, all 153 features | 24 | 0.038 | 31 / 30 | −6..+2 |

- The inner winner (8 of 10 cases: depth 8, 3,000 trees, lr 0.005, scale_pos_weight ≈ 1) sat at the edge of the space and roughly doubled inner AP (Luis's settings 0.04–0.06 → 0.06–0.09), **but that gain did not survive on the held-out folds.** With ~100 positives per training fold, the search fits noise.
- A round 2 that extended the space (depth to 14, up to 12k trees, GPU) was stopped after ~20 min: it moved in the direction that already failed to transfer, at ~3 h per feature set.
- XGBoost vs LightGBM is a tie. Luis's XGBoost 40-seed tree voter is fine.

### 8.4 Final 2026 scores (handed to Luis's `luis/analysis/final_build.py`)

| File | Model | Training | Checks |
|---|---|---|---|
| `submissions/lora8b_scoreboard_2026.parquet` (`cve_id, lora_score`) | Granite 4.2 8B LoRA, `cv_control` recipe: r16 / alpha 32, 20 random negatives per positive per epoch, 3 epochs, final epoch (`scripts/final_lora.sh`) | all 75,148 CVEs of 2024-01..2025-09, label = KEV after the snapshot month through 2025-12-31 (119 positives) | 72,670 rows, unique `cve_id`, no NaN; score = logP(Yes) − logP(No) via vLLM; mean −2.5 with exploit code vs −6.5 without |
| `submissions/lr_best_scoreboard_2026.parquet` (`cve_id, lr_score`) | `lr_full_llmx_noemb`: base + meta + TF-IDF + Granite 30B, 100 features, C = 1e-4 from inner grouped CV (`scripts/final.py train`) | same rows | 72,670 rows, unique, no NaN; decision function, higher = more likely |

The two agree only moderately (Spearman 0.58; 203 shared CVEs in their top 500s), which is why averaging them helps.

## 9. IBM technology

| Component | Model / version | Served with | Role | Measured |
|---|---|---|---|---|
| World-knowledge extraction | `ibm-granite/granite-4.2-30b` (bf16, non-thinking, JSON-schema constrained via vLLM structured outputs) | vLLM 0.32, 3× H100, ~20 CVEs/s/GPU, 38 min for 147,818 CVEs (111 unparsable) | 55 features (§6) | +50% AP on base LR; in the best model |
| Fine-tuned scorer | `ibm-granite/granite-4.2-8b` + LoRA | PEFT training, vLLM scoring with LoRA | model 2 of the average | ties the LR, +4..+6 hits in the average |
| Embedding | `ibm-granite/granite-embedding-english-r2` (CLS pooling, normalized) | transformers | nearest-KEV features, PCA, explanations | not needed in the final LR; nearest-KEV gives explanations |
| Zero-/many-shot scorer | Granite 4.2 30B | vLLM, prefix caching, pre-tokenized examples | ranking | worse than the baseline |
| Memorization probe | Granite 4.2 30B | vLLM | data-use check | 0/122 KEV CVEs identified from the ID (`results/memorization_probe.parquet`) |

**What Granite's extraction says** (2024-01..2025-09, base 0.16%; `processed/llm_extract.parquet`):

| Field = value | CVEs | KEV | Lift |
|---|---|---|---|
| attacker appeal = low | 27,716 | 5 | 0.1× |
| attacker appeal = high | 12,827 | 59 | 2.9× |
| typical attacker = unlikely target | 15,294 | 1 | 0× |
| typical attacker = ransomware | 26,628 | 89 | 2.1× |
| impact = denial of service | 12,047 | 0 | 0× |
| impact = RCE / file write | 17,682 / 2,725 | 63 / 10 | 2.2× / 2.3× |
| product = WordPress plugin | 20,306 | 0 | 0× |
| product = industrial / OT | 2,196 | 0 | 0× |
| product = remote access / IT management | 340 | 3 | 5.6× |
| product = email / collaboration | 217 | 2 | 5.8× |
| product = enterprise server app | 9,290 | 40 | 2.7× |
| product = network edge device | 15,309 | 30 | 1.2× (label too broad) |
| install base = small / niche | 24,027 | 12 | 0.3× (matches Luis's "niche 0/67") |
| internet exposure | – | – | flat (not useful) |

**Known extraction errors:** Chrome labelled "web app" not browser; `exploitation_style` is almost always "targeted".

## 10. Findings, open questions, next steps

**Findings:**
1. **About 40% of exploited CVEs are invisible at publish time.** Of the 22/52 2025 positives no method ranks in its top 1,000 (`reports/missed.txt`):
   - 0% had public exploit code (vs 73% of found ones); median EPSS percentile 61 vs 96; median CVSS 7.6 vs 9.0;
   - they reached KEV *faster* (median 37 vs 56 days).

   Three groups:
   - targeted zero-days on consumer platforms: Chrome V8, WhatsApp, Samsung Quram ×2, Android Linux kernel, 7-Zip MotW, Windows NTLM, Zimbra XSS;
   - chained or modest-looking edge/enterprise bugs: Ivanti EPM ×3, Fortinet "needs serial numbers", SonicWall SSLVPN, ScreenConnect, DELMIA ×2, VeraCore ×2, Craft CMS ×2;
   - EOL botnet targets: TP-Link, Digiever.
2. **Granite zero-shot ranks obvious targets badly** (SonicWall SSLVPN auth bypass at 17,701). Its prompt's EPSS / no-code lines anchor it.
3. **Exploited CVEs look ordinary at the snapshot** (median EPSS percentile 57 among 2024 positives).
4. **Snapshot artifact.** 63–65% of KEV CVEs published early in a month go onto KEV within the month (so they're dropped from the data), vs 43% for late-month ones. So the KEV rate rises across the month (0.145% → 0.172%) while exploit code falls (2.49% → 1.93%) (`reports/ideas.txt`).
5. **Checked, no signal:**
   - descriptions citing another CVE (564 CVEs; only 9 cite a KEV CVE, 0 positives);
   - IBM X-Force links (36,643 CVEs, mostly MITRE-era or IBM's own; 378 in 2024 and 26 in 2025, 0 positives) (`reports/ideas.txt`).

**Open question for the judges** (needed only if the hidden-KEV feature is used; it's neutral, so it isn't):

> The scoreboard excludes CVEs already on KEV at the end of their snapshot month, so the 124 Jan–Sep 2026 CVEs in `vulnerabilities.csv` that aren't on the scoreboard are presumably 2026 KEV additions. May we use that as a feature ("CVEs from the same vendor, published earlier the same year, that went onto KEV within their snapshot month"), counting only those with a snapshot month ≤ the ranked CVE's, and build the same feature for 2024–25 from `known_exploited.csv`?

**Next steps, in order:**
1. **Final run:**
   - train the best LR (and the Luis-feature LightGBM) on all of 2024-01..2025-09, and one LoRA 8B on the same rows;
   - score `scoreboard_2026_view` **once**;
   - average the ranks, apply cap 3;
   - check the Oracle July batch (≤ cap), WordPress ≈ 0, and the month mix (June CVEs use July);
   - write `submissions/top500_final.csv` and `top500_final_explained.csv` (columns per Luis's brief: rank, cve_id, snapshot_month, assigner, product, category, epss, 4 flags, reason; the reason comes from the nearest-KEV match + the Granite fields).
2. ~~Evaluate §8.3~~ done: the 8B stays (§8.3). Final scores handed to Luis (§8.4).
3. **Forward-only LoRA runs** (4 trainings) to complete §5.3.

## 11. Agreements and disagreements with `luis/context_Luis.md`

| Topic | Luis | Chen | Status |
|---|---|---|---|
| Headline labels | `y_nov4` | adopted | agree |
| TF-IDF `text_kev_lr` | strongest single feature (AUC 0.91) | weak alone (14 hits), +4..+7 inside the LR | agree it's needed, as a feature |
| LightGBM | core model | bad on our features (17); fine on his (24); helps the average (+2) | partly agree: tree + linear pairing confirmed (his §14.5 too) |
| Cap per assigner-month | +6 on A+B | +0..+3; flat for the best averages | agree, keep cap 3–5 |
| `txt_in_wild` | dropped (no loss) | excluded | agree |
| Hidden-KEV gap (`assigner_kev_ytd`) | neutral | neutral (name-level too) | agree; not used |
| KEV vendor name in text | doesn't help (0.11% vs 0.17%) | product-level match 3.8×; Granite-normalized vendor/product history in the best model | different matching; ours keeps it |
| 1-row difference | keeps CVE-2025-4664 | drops it (signals `in_kev = 1`) | data inconsistency, noted |
| Granite Instruct (2B, top 5%) | +2 without cap, 0 with | 30B on all CVEs: +50% AP on base LR, tie in the best average | consistent |

## 12. Environment and run order

- **Hardware:** 4× H100 80GB; **GPU 0 is broken, use 1–3.** Driver 580 / CUDA 13.
- **Python environments:**
  - `~/venv-vllm-cu13`: vLLM 0.32.0, torch 2.13.0+cu130, transformers 5.19.0, pandas, pyarrow, scikit-learn 1.9.1, lightgbm 4.7.0. Used for scoring, embedding and all CPU scripts.
  - `~/venv-sft-cu13`: torch 2.14.1+cu130, transformers 5.19.0, PEFT 0.20.0. Used for LoRA training.
  - `pyproject.toml` / `uv.lock`: pandas + pyarrow for the data build.
- **Before GPU runs:** `unset LD_LIBRARY_PATH; export CUDA_HOME=/usr/local/cuda`.
- **Paths:** scripts resolve paths from their own location: `chen/IBM/data` if present, else `../IBM-20261107T174321Z-1-001/IBM/data/data`. The `scripts/analysis/*` reports expect `chen/IBM/data`, so create it once: `mkdir -p IBM && ln -s ../../IBM-20261107T174321Z-1-001/IBM/data/data IBM/data`. Luis's features are read from `../luis/derived/for_chen`.
- **Gotchas:**
  - `pgrep -f` wait loops match their own command line.
  - Never edit a bash script while it's running (bash reads it as it goes).
  - Re-tokenizing a 45k-token shared prompt per request is the many-shot bottleneck; pre-tokenize (`llm_score_logo.py`).
  - The 30B LoRA peaks at ~77 GB at batch 16.

**Run order** (from `chen/`; `PY=~/venv-vllm-cu13/bin/python`):

```
1  uv run python scripts/build_historical.py                         # data views (~1 min)
2  CUDA_VISIBLE_DEVICES=1 $PY scripts/embed.py                        # Granite Embedding r2 (~5 min)
3  $PY scripts/meta_features.py                                       # metadata (~80 s)
4  CUDA_VISIBLE_DEVICES=1 $PY scripts/emb_features.py embed && $PY scripts/emb_features.py features
5  $PY scripts/team_features.py                                       # Luis's TF-IDF + adjacent IDs (~40 s)
6  for s in 0 1 2; do CUDA_VISIBLE_DEVICES=$((s+1)) $PY scripts/extract.py --shard $s --nshards 3 & done; wait
   $PY scripts/extract.py --merge && $PY scripts/llm_features.py      # Granite 30B extraction (~40 min)
7  $PY scripts/llm_score.py --mode zero --target all2024 ...          # Granite zero-shot (only for old variants)
8  $PY scripts/cv.py folds && OMP_NUM_THREADS=1 $PY scripts/cv.py lr  # all LR / LightGBM variants (CV_ONLY=..., CV_FORWARD=1)
9  scripts/cv_lora.sh control <gpu> <fold> --neg-ratio 20            # LoRA per fold (MODEL=..., --llmx)
10 $PY scripts/analysis/eval_all.py                                   # tables in reports/
```

Optional: `prior_signals.py`, `fresh_kev.py`, `horizon.py` (neutral experiments); `select_shots.py`, `llm_score_logo.py`, `memorization_probe.py` (LLM-scorer experiments).

## 13. File index

| Claim / content | File |
|---|---|
| Main CV table, forward-only, averages (§5.3, §8.2) | `reports/eval_all.txt`, `results/cv_oof_lr.parquet`, `results/cv_fwd_lr.parquet`, `results/cv_oof_lora.parquet`, `results/cv_control_f*.parquet` |
| Best vs baseline with CIs (§1) | `reports/progress.txt` |
| Cap and Nov-4 relabel check | `reports/teamcheck.txt` |
| Missed-positive analysis (§10.1) | `reports/missed.txt`, `reports/assigner_hist.txt` |
| Metadata lifts (§6) | `reports/meta.txt` |
| Scoreboard gap, day-of-month, cross-refs, X-Force (§10) | `reports/ideas.txt`, `reports/fresh.txt` |
| KEV groups / CV folds | `processed/kev_groups.csv`, `processed/cv_folds.parquet`, `processed/cv_fold*_{train,test}.txt` |
| Features | `processed/{meta,team_feat,llm_feat,emb_feat,fresh,prior_sig}_{historical,scoreboard}.parquet` |
| Granite 30B extraction (raw answers) | `processed/llm_extract.parquet`; quality check `results/smoke_extract2.parquet` |
| LLM-scorer runs (§8.1) | `results/holdout_*.parquet` (+ `.json` meta), `results/val2024_*`, `results/all2024_{zero,logo}.parquet` |
| Memorization probe (§9) | `results/memorization_probe.parquet`, `.json` |
| Out-of-fold LR scores for hard negatives | `processed/oof_2024.parquet` |
| Code | `scripts/*.py`, `scripts/*.sh`, `scripts/analysis/*.py` |

**Not committed:** IBM data; `processed/historical_2024_2025.parquet` and `scoreboard_2026_view.parquet` (contain IBM descriptions; step 1 rebuilds them); embeddings `processed/emb_*.npy` (61–129 MB; step 2 rebuilds them); LoRA adapters `runs/` (6.4 GB; on the GPU box at `~/datathon/runs/`).
