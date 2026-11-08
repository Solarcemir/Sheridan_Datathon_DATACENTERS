# Breach Radar · frontend

The team's story page: how 72,670 CVEs become the 500 we submit, what worked, what didn't, and what the data and the final list look like. One HTML page plus one data file.

| File | What it is |
|---|---|
| `breach-story.html` | The page. Vanilla HTML/CSS/JS, no libraries. Charts are HTML bars; the pipeline is SVG. |
| `story_data.json` | Everything the page reads (≈240 KB). Built by the script below. |
| `build_story_data.py` | Builds `story_data.json` from team files plus aggregates of the local IBM data. |

## Rebuild and publish

```
python -E -P luis/frontend/build_story_data.py      # after any change to the final ranking or the LR weights
```

Publish `breach-story.html` with `story_data.json` next to it (same folder, same name). The page fetches it with a relative URL.

To preview locally, serve the folder over http (opening the file directly blocks the fetch):

```
python -m http.server 8765 --directory luis/frontend
```

Then open `http://127.0.0.1:8765/breach-story.html`.

**Inputs of the build script:**
- `luis/derived/scored_2026_final.csv` and `luis/reports/top500_final_explained.csv` (from `final_build.py`);
- `luis/reports/lr_columns_luis_chen.csv` (both LRs' weights);
- `luis/reports/univariate_lift.csv` and `luis/reports/granite_clusters_2026.csv`;
- `chen/processed/llm_extract.parquet` (Granite 30B answers);
- `scoreboard_2026.csv`, `vulnerabilities.csv` and `known_exploited.csv` from the IBM data at the repo root.

The JSON holds only aggregates plus our own 500 picks. No raw IBM rows go into it.

## Story order

The chapters run in this order on purpose: problem → how we built it → what drives it → what we tried → what the data says → what the result looks like → the list.

## Components and the data each one needs

### Shell header
- Product name, team, and anchor links to the 7 chapters.
- The active chapter is underlined; it follows scroll through an IntersectionObserver.
- No data.

### 1 · Problem (`#problem`)

| Element | Data | Source |
|---|---|---|
| Headline + lede | static | – |
| KPI: CVEs in scope | `pre.n_scoreboard` | scoreboard_2026.csv |
| KPI: share with public exploit code | `pre.share_any_flag` | scoreboard_2026.csv |
| KPI: past KEV additions with no exploit signal | 58% (static) | context_Luis.md §6.1 |
| KPI: expected KEV hits | ≈12 (80%: 7–22) vs baseline 8 (static) | stats_review projection, context §14.12 |

Behavior: numbers count up once on load. The HTML already holds the final values, so the page reads correctly without JS.

### 2 · Pipeline network (`#pipeline`)

A layered network:
1. **Input:** 1 node.
2. **Features:** 7 group nodes.
3. **Models:** 4 nodes, plus 4 "dropped" nodes behind a toggle.
4. **Ensemble:** 1 node.
5. **Post-processing:** 2 nodes.
6. **Output:** 1 node.

| Element | Data |
|---|---|
| Edge thickness into LR A / LR B | Share of that LR's total \|coefficient\| carried by the feature group, from `weights[]` |
| Ring around each feature node | The same share (the larger of the two LRs) |
| Side panel, feature node | Column count, share of each LR's weight, and the 6 heaviest columns (diverging mini-bars) from `weights[]` |
| Side panel, Granite Embedding node | Top clusters from `pre.clusters` |
| Side panel, other nodes | Facts listed in the `L` array inside `buildPipeline()`. They come from `pre` and `post` where possible; model results are static numbers from context_Luis.md §14 and chen/context_Chen.md §8 |
| Validation strip | Static: A/B forward test, 5-fold purged CV, paired bootstrap + Holm |

Columns map to feature groups through the regexes in `GROUPS` (top of the script). A column that matches none falls into "CVSS, CWE & dates".

Behavior:
- **Hover a node:** highlights its upstream and downstream paths and dims the rest.
- **Click a node or press Enter:** opens it in the side panel.
- **"Include what we dropped":** adds the ghost nodes (LightGBM, CatBoost, Granite 30B LoRA, Granite 3.3 2B) with dashed edges.
- Edges animate a flow in the source node's color; the animation is off under reduced motion.
- Below 980px the diagram scrolls horizontally inside its own box.

### 3 · Weights (`#weights`)

| Element | Data |
|---|---|
| Diverging bars | `weights[]` filtered by `lr` (`Luis` = LR A, `Chen` = LR B) and by group. `c` = coefficient per SD; blue is positive, magenta negative |
| Tooltip | `f`, `c`, `d` (definition, LR A only) and `eq` (closest column in the other LR) |
| Signal strength chart | `pre.univariate`: univariate AUC, axis from 0.5 to 1.0 |

Controls: model toggle, Top 20 / All, group chips.

### 4 · What worked (`#results`)
- **Results ladder:** static numbers (hits of 75 on the pooled A/B exam, top 441).
  - Sources: `reports/density_check.txt` (baselines), `stats_review_log.txt`, `chen_round3.csv` (v4f 30, final 32 est.).
  - The vertical line marks the 35/75 ceiling.
- **Experiment ledger:** static list (array `X` in `buildResults()`) from the context's status ledger, filterable by Helped / Didn't.

### 5 · Before the model (`#before`)

| Tile | Data |
|---|---|
| Exploit telemetry | `pre.flags`, `pre.share_any_flag`, `pre.n_scoreboard` |
| Where the signal lives | `pre.group_best_auc` |
| The leak trap | static, from `reports/leak_vs_clean.csv` |
| Where exploitation concentrates | `pre.clusters` (top 8 by KEV rate) |
| Who has been exploited | `pre.kev_vendors`, `pre.kev_total`, `pre.kev_ransomware_share` |
| When CVEs arrive | `pre.months` |

### 6 · After the model (`#after`)

| Tile | Data |
|---|---|
| KPI row | `post.exploit_code.top`, `pre.granite_demoted_share`, `post.cap_pushed_out`, `post.voters.none` |
| Our picks vs all 2026 CVEs | `post.product_type` / `post.impact` / `post.typical_attacker`: `top` % vs `all` %, plus `lift` |
| Who voted for each pick | `post.voters` (each model's own top 500 over the full scoreboard) |
| Vendors in the list | `post.vendors`, `post.cap_biggest_batches` |
| Words the list over-represents | `post.bigrams` (lift vs the whole scoreboard; at least 12 occurrences in our 500) |
| EPSS of our picks | `post.epss_buckets` |

### 7 · The list (`#output`)
- Rows come from `top500[]`.
- Controls:
  - search over CVE, product, reason and type;
  - product-type filter;
  - exploit-code filter;
  - sort by rank, EPSS or KEV-like text;
  - 20 rows per page.
- Selecting a row expands it: reason, typical attacker, nearest past KEV entry.
- The three mini columns are the CVE's percentile in each voter (tree, linear, LLM).

## `story_data.json` contract

```
built                      "YYYY-MM-DD HH:MM"
pre.n_scoreboard           int
pre.share_any_flag         % (1 decimal)
pre.share_no_flag_low_epss %
pre.median_epss            float
pre.flags                  {exploitdb, metasploit, nuclei, poc_github: int}
pre.months                 {"2026-01": int, ...}
pre.granite_demoted_share  %
pre.univariate             [{feature, group, auc, lift_top5}]   top 14 leak-free
pre.group_best_auc         {group: auc}
pre.clusters               [{id, kev_rate %, kev_refs, cves_2026, terms[], products[]}]   top 10
pre.kev_total, pre.kev_vendors {vendor: n}, pre.kev_ransomware_share %, pre.kev_by_year {year: n}
post.product_type|impact|typical_attacker   [{key, top %, all %, lift}]
post.vendors               {vendor: n}  (Granite 30B vendor names, top 12)
post.months                {top: {...}, all: {...}}
post.exploit_code          {top: int, all_share: %}
post.epss_buckets          {labels[], top[] counts, all[] %}
post.voters                {"tree+linear+llm": n, ..., "none": n}
post.voter_corr            Spearman matrix
post.filters_pushed_out, post.cap_pushed_out   int
post.cap_biggest_batches   [{assigner, month, would_have}]
post.bigrams               [{term, top, all, lift}]
post.nearest_kev_ge_085    int
weights                    [{lr: "Luis"|"Chen", f, g, c, d, eq}]   156 rows
top500                     [{r, id, m, p, cat, imp, att, e, fl "0101", tx, nk, ks, v [tree, linear, llm], why}]
```

## Design system

- **Look:** enterprise product shell, in the spirit of open-source Carbon. Dark header bar, square tiles separated by 1px seams, IBM Plex Sans and IBM Plex Mono (Google Fonts, open license). No IBM logo or IBM product branding: this is the team's product, built with IBM Granite.
- **Themes:** light (gray 10 style) and dark (gray 100 style), via tokens on `:root`. The page follows the viewer's theme.
- **Data colors:** 4 categorical slots, validated with the dataviz validator (lightness band, chroma, CVD separation, normal-vision floor, contrast) in both modes.

| Slot | Role in the page | Light | Dark |
|---|---|---|---|
| `--s1` blue | features, "our top 500", positive weights | `#0f62fe` | `#4589ff` |
| `--s2` teal | classic ML | `#009d9a` | `#009d9a` |
| `--s3` purple | IBM Granite, final result | `#8a3ffc` | `#a56eff` |
| `--s4` magenta | post-processing, negative weights, leak | `#d02670` | `#ee5396` |
| `--s0` gray | data nodes, "all 2026 CVEs", baselines | `#8d8d8d` | `#8d8d8d` |

- **Status colors** (`--good`, `--bad`) only appear with a ✓ or ✕ and a label.
- **Bars:** at most 14px thick, 4px rounded data end, square at the baseline. One scale per chart, and no dual axes. Comparisons of different-sized groups use shares (%), never two scales.
- **Motion:** bars grow once (0.45s), KPIs count up, pipeline edges flow. All of it is off under `prefers-reduced-motion`.
- **Copy:** short and direct. Numbers carry the claims; every estimate says so.

## Updating static numbers

When a model result changes, edit these spots in `breach-story.html`:
- the `L` array in `buildPipeline()` (side-panel facts);
- `buildResults()` (ladder and ledger);
- the leak tile in `buildBefore()`;
- the hero KPIs (58%, ≈12).

Everything else comes from `story_data.json`.
