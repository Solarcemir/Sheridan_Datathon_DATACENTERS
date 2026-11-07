# Project context · IBM "Before the Breach", GPU-box workstream

Handoff for Luis. Last updated Nov 7, 2026, ~4:40 pm. This is the parallel workstream that ran on a 4× H100 Linux box (Granite 4.2 30B / 8B with vLLM, LoRA fine-tuning, Granite Embedding r2). It's written against your `context_Luis.md`, so §9 maps what each side has and what to merge. Every number below comes from a script in `datathon/scripts/`, and §12 lists the files.

---

## 0. TL;DR

1. **Best result: an equal-weight rank average of a regularized logistic regression and a LoRA-fine-tuned Granite 4.2 8B.** In 5-fold time-blocked, group-purged CV over Jan 2024 to Sep 2025 (75,148 CVEs, 119 positives, top 517 = scoreboard top 500 equivalent), it gets **42 hits (95% bootstrap CI 30–54) vs 27 for exploit code then EPSS.** The paired difference is **+7 to +24**. Average precision is 0.081 vs 0.046, and the average beats or ties the baseline in every time block.
2. **With your judge-aligned labels** (KEV by Nov 4 of the publish year, 75 positives) it's **28 vs 20**, and **30 with a cap of 3 per assigner per month**. Your cap idea holds on our CV too (+1 to +3 hits for most models).
3. **Granite as a direct scorer doesn't win.** 30B zero-shot is significantly worse than the baseline (22 vs 27; paired CI −12 to −1). Many-shot (201 in-prompt examples) ≈ zero-shot. LoRA 8B ties the linear model (36 vs 36), but the two make different mistakes, so averaging them gives +6.
4. **Granite Embedding alone is weak** (nearest-KEV similarity, average precision 0.008). Inside the linear model, the nearest-KEV features plus 64 PCA dims of the description embedding give the best linear variant (36 hits, AP 0.065). The **Granite LLM score adds nothing on top of the embedding** (33–34 hits), which matches your "embeddings ≈ semantics" finding.
5. **Memorization probe:** given only the CVE ID, Granite 4.2 30B identified **0 of 122** exploited 2024–25 CVEs and 0.2% of 2026 CVEs (chance level). Good for the write-up.
6. **Running now:** Granite 4.2 30B is extracting structured world-knowledge features for all 148k CVEs (2024-01..2025-09 + scoreboard): product type, internet exposure, install base, precondition, impact, EOL, fix status, typical attacker, attacker appeal, normalized vendor/product. They go into the linear model through `llm_features.py` and get tested in the same CV. ETA ~5:20 pm.
7. **We haven't produced a 2026 `top500.csv` yet.** Your `top500_v1_cap5.csv` is still the team's submission until this pipeline scores the scoreboard.

---

## 1. Data views (`scripts/build_historical.py`)

- **`processed/historical_2024_2025.parquet`:** 87,907 CVEs published 2024-01..2025-12, built like the scoreboard. `snapshot_month` = publish month; EPSS and flags come from that month's row of the signals files (no row → blank EPSS, 0 flags); CVEs on KEV by the end of the snapshot month are dropped (196 of them). Label `label` = added to KEV after the snapshot month, any time through 2025-12-31. **This is longer than your `y_nov4`.** See §3.
- **`processed/scoreboard_2026_view.parquet`:** the 72,670 scoreboard CVEs with the same columns.
- **Columns dropped everywhere:** `nvd_status`, `last_modified`, `n_affected_cpes`, `n_references`, `cvss3_source`, all of `affected_products.csv` and `references.csv`, CVSS 2. Same leak list as your §3 red row, plus CVSS 2 (NVD-only now) and CVSS source.
- Measured: **100% of 2024/25 KEV CVEs are NVD-analyzed** (same finding as yours).

## 2. Fairness rules we applied (same spirit as yours, a bit stricter)

- **KEV history cutoff = end of the previous year** for every row (2024 rows see KEV ≤ 2023-12-31, 2025 rows ≤ 2024-12-31, 2026 rows ≤ 2025-12-31). This mirrors the 2026 staleness exactly.
- **Excluded as possibly edited after the fact:** description text "exploited in the wild / aware of reports" (measured **109×** lift on 43 CVEs, but description edits can't be dated) and CVSS 4 `E` (exploit maturity). You use `txt_in_wild`, so it's worth deciding together. It's the single strongest text flag and the most leak-prone.
- **Excluded because it's not computable for 2026:** vendor-level monthly signals of *other* CVEs (no 2026 monthly files).
- **The scoreboard gap (124 hidden 2026 KEV CVEs):** we reached the same conclusion as your §6.2 independently. Assigner-level heat measured 2.6× lift on 2024–25 but flags 45% of the scoreboard, so it's too broad. Your test found it neutral. We haven't used it. A product-level version (via Granite vendor/product) is the only variant still worth trying.

## 3. Evaluation (`scripts/cv.py`, `scripts/eval_rank.py`)

| Setup | Train | Test | Positives | Use |
|---|---|---|---|---|
| Forward holdout | all 2024 | 2025-01..09 | 52 | first experiments; noisy |
| 2024 validation | 2024-01..06 | 2024-07..12 | 39 | early method choice |
| Leave-one-group-out | 2024 minus group | each 2024 CVE | 67 | many-shot examples only |
| **5-fold CV (main)** | 4 time blocks minus purged groups | 5th block | **119** | all model comparisons |

- **CV folds:** 5 consecutive publish-date blocks of ~15k CVEs (16 / 32 / 30 / 21 / 20 positives).
- **Purging with hand-made KEV groups:** `processed/kev_groups.csv` has 60 CVEs in 26 groups, built by reading every 2024–25 positive with CISA vendor/product: same product disclosed or added together, exploit chains, plus same-product cross-year groups (FortiOS, SonicOS, Ivanti EPM, Zimbra, Craft CMS, Adobe Commerce, TP-Link WR841N). When scoring block f, any CVE whose group has a member in block f is removed from training.
- **Metrics:** hits@K with K = 500 × n / 72,670 (517 for the CV set), 1,000-rep bootstrap 95% CIs, paired difference vs the baseline, average precision, and per-fold hits.
- **Regularization** for the LR is chosen by an inner grouped 3-fold CV on the training blocks only (average precision; C grid 1e-4..1e-2). It almost always picks C ≈ 1e-4 to 3e-4. With ~95 positives and ~90 features, strong regularization matters a lot: at C = 1 the metadata variant *lost* 5 hits.
- **Label window difference:** our `label` runs to 2025-12-31, so 2024 CVEs get up to 23 months. With your `y_nov4` window, the CV positives drop from 119 to 75. Rankings of methods don't change, but **we should both report judge-aligned numbers.**
- **Noise:** the same LoRA run moves 3–5 hits between epochs on the 52-positive holdout. As you said, only trust gaps you can put a CI on.

## 4. Features

**Base (`stack.py: features`):** 4 exploit flags + any, EPSS percentile within month, EPSS missing, CVSS (3, else 4) + missing, AV network, AC low, PR none, UI none, scope changed, WordPress CNA (Wordfence/Patchstack/WPScan).

**Metadata (`meta_features.py`, history with the previous-year-end cutoff).** Lift is measured on 2024-01..2025-09; base KEV rate is 0.16%.

| Feature | Lift / note |
|---|---|
| Assigner KEV rate (smoothed, last 3 yrs) | 36× median separation; strongest metadata coefficient |
| Assigner ransomware KEV count | 4.3× |
| CWE KEV rate (smoothed) | positive weight |
| Unknown CWE (`NVD-CWE-noinfo` / `Other`) | **0 positives in 3,551**, strongest negative |
| Out-of-band (only CVE from that assigner that day) | 5.5× |
| Patch-batch size ≥ 20 that day | 0.3× |
| Twins (identical description, same assigner+day) | 1.6× |
| A same-release sibling has exploit code | positive |
| Old reserved ID (ID year < publish year) | 0.4×; ≥2 years: 0.3× |
| EOL / unsupported text | 13× |
| "unauthenticated" / pre-auth text | 5.4–5.6× |
| Auth-bypass text | 9.5× |
| RCE text | 4.4× |
| CVSS 4 U:Red/Amber (vendor urgency) | 8× (246 CVEs) |
| CVSS 4 AU:Y | 3× |
| CVSS 4 E:P (mostly VulDB) | **0 positives in 2,669** |
| VulDB "exploit has been disclosed to the public" | 0.2× |
| VulnCheck as assigner | 14× (2.3% KEV rate) |
| Vendor / product named in KEV (text match) | 1.4× / 3.8× |
| Patch Tuesday | 2× |

**Embedding (`emb_features.py`, granite-embedding-english-r2, 768-d, CLS pooling):**
- **Data:** all 2024–26 descriptions, all 1,484 KEV descriptions, and a 40k sample of 2021–23 CVEs.
- **Features:** `kev_sim_max`, `kev_sim_top5` (similarity to KEV entries added by the cutoff, any CVE year, so ~1,100 pre-2024 KEV entries are used), `typ_sim_top5` (similarity to 10k typical CVEs from the 3 years before the cutoff), and `kev_contrast` = the difference.
- **PCA-64:** fit on the 2021–23 sample only, so it's fold-safe.
- **Nearest-KEV examples (good for explanations):**

| CVE | Nearest KEV entry | Similarity |
|---|---|---|
| Chrome V8 CVE-2025-6554 | earlier V8 type confusion | 0.98 |
| Fortinet CVE-2025-24472 | CVE-2022-40684 (FortiOS CWE-288) | 0.94 |
| Zimbra CVE-2025-27915 | CVE-2023-37580 (Zimbra Classic Web Client XSS) | 0.94 |
| Ivanti EPM CVE-2024-13159 | Ivanti CSA path traversal | 0.92 |

**Granite world-knowledge extraction (`extract.py` + `llm_features.py`, running):**
- **Inputs:** Granite 4.2 30B sees only description, assigner and CWE names; no EPSS, flags, dates, ID or KEV.
- **Output:** a JSON-schema-constrained answer (vLLM structured outputs). Fields: normalized vendor and product; `product_type` (16 types); `internet_exposed`; `install_base`; `used_by`; `precondition`; `impact`; `end_of_life`; `fix_status`; `typical_attacker` (ransomware / state espionage / botnets / opportunistic / unlikely); `exploitation_style`; `attacker_appeal`.
- **Spot check (`results/smoke_extract2.parquet`):** SonicWall SSLVPN → edge device, internet-facing, no precondition, auth bypass, ransomware, high. WhatsApp and Samsung → state espionage, high. TP-Link and Digiever → botnets, EOL.
- **Known errors:** Chrome labelled "web app" instead of browser; `exploitation_style` is almost always "targeted".
- **Features:** one-hots, ordinals, and product-level KEV history from the normalized vendor/product matched to KEV vendor/product names, with the same cutoff. **This is your §11 items 1 and 5,** done on all CVEs with the 30B instead of the top 5% with a 2B.

## 5. Results

### 5.1 Main CV table (119 positives, top 517; `results/cv_oof_lr.parquet`, `cv_oof_lora.parquet`)

| Model | Hits@517 (95% CI) | vs baseline (95% CI) | Hits@1000 | Avg precision | Per-fold hits@103 |
|---|---|---|---|---|---|
| Exploit code, then EPSS | 27 (17–37) | – | 35 | 0.046 | 6, 7, 2, 9, 4 |
| Granite 30B zero-shot alone | 22 (13–30) | −12 to −1 | 29 | 0.027 | 3, 6, 3, 7, 2 |
| Embedding `kev_contrast` alone | 7 (2–13) | −32 to −9 | 17 | 0.008 | 1, 3, 1, 1, 2 |
| LR: features | 30 | −5 to +11 | 38 | 0.042 | 5, 8, 4, 9, 4 |
| LR: + meta | 34 | −2 to +15 | 45 | 0.055 | 3, 10, 7, 9, 6 |
| LR: + Granite zero-shot | 33 | −2 to +14 | 40 | 0.057 | 5, 9, 6, 9, 4 |
| LR: + meta + Granite | 34 | −1 to +15 | 46 | 0.057 | 3, 10, 7, 9, 6 |
| LR: + meta + nearest-KEV | 35 | −2 to +18 | 46 | 0.060 | 4, 10, 6, 9, 6 |
| **LR: + meta + nearest-KEV + PCA-64** | **36 (25–47)** | **+0 to +17** | 43 | **0.065** | 5, 9, 6, 9, 5 |
| LR: same + Granite | 33 | −3 to +15 | 44 | 0.065 | 5, 9, 6, 8, 5 |
| LoRA 8B (control, epoch 3) | 36 (26–47) | −1 to +18 | 46 | 0.045 | 6, 10, 5, 12, 4 |
| **Rank average: best LR + LoRA** | **42 (30–54)** | **+7 to +24** | 47 | **0.081** | 6, 11, 8, 10, 7 |

The same rankings under judge-aligned labels (75 positives) and a per-assigner-month cap:

| Model | No cap | Cap 3 | Cap 5 | Cap 8 |
|---|---|---|---|---|
| Exploit code, then EPSS | 20 | 23 | 22 | 22 |
| Best LR | 26 | 27 | 27 | 26 |
| LoRA | 25 | 26 | 27 | 26 |
| **LR + LoRA average** | **28** | **30** | 29 | 28 |

Gradient boosting (scikit-learn histogram GBM, depth 3, leaf ≥ 200) overfit on our setup: 10–15 hits on the 2025 holdout, worse than LR every time. **Your LightGBM settings do better.** Worth running them in our CV.

### 5.2 LLM-as-scorer experiments (2025 holdout, 52 positives, top 243 = scoreboard equivalent)

| Run | Top 243 | Top 500 | Top 1,000 |
|---|---|---|---|
| Exploit code, then EPSS | 14 | 18 | 22 |
| Granite 4.2 30B zero-shot | 10 | 16 | 17 |
| Granite 4.2 30B many-shot (201 examples, 45k-token prompt) | 10 | 15 | 22 |
| LoRA 8B control (1,340 random negatives/epoch), epochs 1/2/3 | 13 / 13 / 16 | 16 / 15 / 20 | 18 / 20 / 23 |
| LoRA 8B, 670 fixed hard negatives + random, epochs 1/2/3 | **1** / 15 / 11 | 1 / 20 / 16 | 4 / 24 / 23 |
| LoRA 8B, hard negatives sampled from a pool of 2,000 + random | 14 / 15 / 11 | **24** / 20 / 16 | 26 / 24 / 23 |
| LoRA 8B, metadata lines in prompt | 6 / 6 / 15 | 6 / 14 / 19 | 15 / 19 / 25 |

**Lessons from these runs:**
- **Hard negatives alone invert the exploit-code signal.** The top out-of-fold false positives had exploit code 51% of the time vs 31% for positives; the fixed hard set collapsed epoch 1 to 1 hit.
- **The top ~50 "hard negatives" look like label noise:** Ivanti Connect Secure XXE, GoAnywhere MFT, Splunk path traversal, Flowmon. Probably exploited but not on KEV, so we skipped them.
- **Metadata in the prompt doesn't help.** The linear model weights it better.
- **Many-shot example choice matters:** the first "hard" example set had exploit code on 100% of negatives and 31% of positives, which teaches the opposite of the truth. The fix was semantic near-misses (most similar non-KEV description, same attack vector, CVSS ±1.5) plus k-means representative negatives. We checked balance with standardized mean differences.

## 6. Findings for the write-up (complements your §10)

1. **The missed positives are invisible at publish time.** Of the 22 of 52 2025 positives that no method ranks in the top 1,000:
   - **0%** had public exploit code (vs 73% of found ones);
   - their median EPSS percentile was 61 (vs 96);
   - they were added to KEV *faster* (median 37 days vs 56);
   - their median CVSS was 7.6 (vs 9.0), and 18% required user interaction (vs 7%).

   Three groups:
   - **Targeted zero-days** on consumer platforms: Chrome V8, WhatsApp, Samsung image codec ×2, the Android Linux kernel, 7-Zip MotW, Windows NTLM, Zimbra XSS.
   - **Chained or modest-looking bugs** in enterprise/edge products: Ivanti EPM ×3, Fortinet "needs serial numbers", SonicWall SSLVPN, ScreenConnect PR:HIGH, DELMIA Apriso ×2, VeraCore ×2, Craft CMS ×2.
   - **End-of-life botnet targets:** TP-Link, Digiever.
2. **Granite zero-shot ranks the obvious ones badly** (SonicWall SSLVPN auth bypass at 17,701; Fortinet at 25,145). The EPSS percentile and "no exploit code" in the prompt anchor it.
3. **Who assigns the CVE matters more than any single CVE field.** Assigner KEV rate, VulnCheck at 14×, WordPress CNAs and unknown CWE both at 0 positives, routine patch batches at 0.3×, out-of-band advisories at 5.5×.
4. **Memorization:** 0 / 122 KEV CVEs identified from the ID alone. Guesses were generic and wrong ("Exchange RCE", "Print Spooler overflow"), including for well-known ones (Jenkins CVE-2024-23897 → "unknown").
5. **Checked, no signal:**
   - Descriptions that mention another CVE: 564 CVEs, only 9 reference a KEV CVE, 0 positives.
   - IBM X-Force links: 36,643 CVEs, almost all MITRE-era or IBM's own; 378 in 2024 and 26 in 2025, 0 positives.
   - Description cross-references to KEV CVEs.
6. **Snapshot artifact.** 63–65% of KEV CVEs published early in a month are dropped as in-month KEV, vs 43% of late-month ones. So survivors skew late-month (KEV rate 0.145% → 0.159% → 0.172% across the month), while late-month CVEs have had less time to get exploit code (2.49% → 1.93%). A "days public at snapshot" feature is planned (your `age_days` covers this).

## 7. Granite / IBM tech used here

| Component | Model | Role | Verdict |
|---|---|---|---|
| Scorer, zero-shot / many-shot | Granite 4.2 30B (vLLM 0.32, non-thinking, logP(Yes) − logP(No)) | rank CVEs | worse than baseline alone |
| Scorer, fine-tuned | Granite 4.2 8B + LoRA r16 (PEFT, BCE on the Yes−No logit gap) | rank CVEs | ties LR, **+6 when averaged with LR** |
| Embedding | granite-embedding-english-r2 | nearest-KEV features, PCA-64 | part of best LR |
| World-knowledge extraction | Granite 4.2 30B + JSON schema | product/attacker features | running |
| Memorization probe | Granite 4.2 30B | data-use check | 0/122 |

All serving uses vLLM, which you listed as an option (Red Hat AI Inference Server is vLLM-based).

## 8. Environment

- 4× H100 80GB; **GPU 0 is broken, use 1–3.** Driver 580 / CUDA 13. Everyone shares the box; check `nvidia-smi` first.
- `~/venv-vllm-cu13`: vLLM 0.32, torch 2.13 cu130, transformers 5.19, plus pandas, pyarrow, scikit-learn. Used for scoring, embedding and CPU scripts.
- `~/venv-sft-cu13`: torch 2.14, transformers 5.19, PEFT 0.20. Used for LoRA training.
- `datathon/pyproject.toml`: uv project (pandas, pyarrow) for the data build.
- Before GPU runs: `unset LD_LIBRARY_PATH; export CUDA_HOME=/usr/local/cuda`.
- Speeds:
  - 30B zero-shot ≈ 120 CVEs/s per GPU (1 output token).
  - Many-shot with a 45k-token shared prefix runs at the same speed once the example tokens are pre-tokenized (`llm_score_logo.py`). Re-tokenizing the prefix per CVE is the bottleneck.
  - LoRA 8B ≈ 8 min for 3 epochs of ~1,400 examples.
  - Extraction ≈ 20 CVEs/s per GPU.

## 9. Merge plan: yours vs ours

**You have, we don't (should be added to our CV):**
1. **`y_nov4` labels.** Adopt for all headline numbers.
2. **TF-IDF KEV text model** (`text_kev_lr`, your strongest feature, AUC 0.91). Exact product names. Our embedding features don't replace it.
3. **The per-assigner-month cap.** Confirmed in our CV (+1 to +3); cap 3–5 plateau. Check the 2026 Oracle July batch in whatever final ranking we submit.
4. **`n_adjacent_ids`** (consecutive CVE numbers).
5. **LightGBM with your settings** (`scale_pos_weight=25`, 5 seeds).
6. **Leak-inflation demo** (+2–4 hits) for the write-up.

**We have, you don't (your §11 list):**
1. **Granite 4.2 30B on all CVEs via vLLM,** plus structured extraction with normalized vendor/product (§11.1, §11.5).
2. **Purged time-blocked CV with 119 positives, hand-made KEV groups and bootstrap CIs** (§11.2).
3. **granite-embedding-english-r2** (§11.4).
4. **LoRA 8B adapters,** with a rank-average partner that's validated in CV.
5. **The memorization probe,** the missed-positive analysis, and the extra metadata features in §4.

**Disagreements to settle:**
- **"KEV vendor named in text doesn't help" (yours) vs 3.8× for product-level matches (ours).** Different matching: your KEV names with ≥3 entries as n-grams, our cleaned product names. Granite's normalized vendor/product should settle it.
- **`txt_in_wild`:** strongest text flag, but possibly an after-the-fact edit. We excluded it; you include it.
- **2024 test positives** (your 27 vs our 67): label window plus your Jan–Nov replicas. Not a real conflict.

**Proposed final pipeline:** one shared feature table (your safe features + TF-IDF + adjacent IDs, plus our metadata + embedding + Granite extraction) → regularized LR and your LightGBM, each tuned only inside training folds → rank average with LoRA 8B → cap of 3–5 per assigner per month → `top500.csv`. Evaluate every step in our CV with `y_nov4`, and score the 2026 scoreboard once at the end.

## 10. Still to do

1. Granite extraction finishes, then `llm_features.py`, then the CV variants `lr_features_meta*_llmx`.
2. Add §9 items 1–5 to the CV.
3. Time-left-to-Nov-4 adjustment. Your `days_to_cutoff` feature or a post-hoc multiplier from the days-to-KEV distribution (median 62 days; 75% within 146).
4. "Days public at snapshot" feature; extra training rows with a snapshot one month later (to match the June→July scoreboard rows).
5. Borrow EPSS / Nuclei / Metasploit coverage from the product's older CVEs (vendor via Granite).
6. Train the final models on 2024-01..2025-09, score the scoreboard, cap, and produce `top500.csv` plus Granite one-line reasons from the nearest-KEV match and features.

## 11. Script run order (from `datathon/`)

```
uv run python scripts/build_historical.py                       # data views
CUDA_VISIBLE_DEVICES=1 ~/venv-vllm-cu13/bin/python scripts/embed.py          # embeddings
~/venv-vllm-cu13/bin/python scripts/meta_features.py            # metadata (80 s)
CUDA_VISIBLE_DEVICES=1 .../python scripts/emb_features.py embed && .../python scripts/emb_features.py features
.../python scripts/llm_score.py --mode zero --target all2024 ... # Granite zero-shot scores (used by LR)
.../python scripts/cv.py folds && .../python scripts/cv.py lr && .../python scripts/cv.py report [extra.parquet]
scripts/cv_lora.sh control <gpu> <fold> --neg-ratio 20          # LoRA per fold
.../python scripts/extract.py --shard i --nshards 3; .../python scripts/extract.py --merge; .../python scripts/llm_features.py
```

## 12. Files

| What | File |
|---|---|
| Data views | `processed/historical_2024_2025.parquet`, `processed/scoreboard_2026_view.parquet` |
| KEV groups (CV purging) | `processed/kev_groups.csv` (2024–25), `processed/kev_groups_2024.csv` |
| CV folds | `processed/cv_folds.parquet`, `processed/cv_fold{0-4}_{train,test}.txt` |
| Metadata features | `processed/meta_{historical,scoreboard}.parquet` |
| Embeddings | `processed/emb_{historical,scoreboard,extra}.npy` (+ `_ids.parquet`), `processed/emb_feat_*.parquet` |
| Granite extraction | `results/extract_*of3.parquet` → `processed/llm_extract.parquet` → `processed/llm_feat_*.parquet` |
| CV out-of-fold scores | `results/cv_oof_lr.parquet`, `results/cv_oof_lora.parquet` |
| LLM scores | `results/{all2024,holdout}_{zero,logo,many}.parquet`, `results/holdout_lora_*` |
| Memorization probe | `results/memorization_probe.parquet` |
| LoRA adapters | `runs/control_r20/`, `runs/cv_control_f{0-4}/`, `runs/hard*/`, `runs/meta_r20/` |
| Code | `scripts/*.py`, `scripts/*.sh` |
