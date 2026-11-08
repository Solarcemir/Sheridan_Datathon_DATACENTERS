# Analyst drill-down dashboard (Honiya)

A Streamlit companion to the team's Breach Radar story page (`luis/frontend/breach-story.html`). Breach Radar covers the method, the benchmark and the list of 500. This app covers two things it doesn't: where our models were blind, and a full file for each of the 500 CVEs we submitted.

All data is precomputed into `data/` (about 1 MB). The app needs no IBM raw data and no API key.

## Run

```bash
cd honiya/dashboard
pip install -r requirements.txt
streamlit run app.py
```

Run it from inside `honiya/dashboard` so Streamlit picks up `.streamlit/config.toml` (light theme).

Open http://localhost:8501.

Or open here: https://oasis-measured-properties-fountain.trycloudflare.com/

## Tabs

- **Where the model was blind.** The 22 of 52 exploited 2025 CVEs that every method missed: no public exploit code, low EPSS, yet exploited faster than the ones we caught. Grouped into consumer-platform zero-days, chained edge/enterprise bugs, and end-of-life botnet targets. Source: `chen/reports/missed.txt`.
- **Open a CVE file.** Any of the 500 CVEs in the team final `top500.csv` (repo root), filterable by vendor, assigner and product type. Each file shows:
  - the advisory text;
  - the CVSS 3 vector at publish and whether public exploit code existed;
  - Granite 4.2 30B's 10-field analyst profile;
  - where each of the three models (trees, linear, Granite 8B LoRA) ranks it;
  - a four-bullet analyst note written by IBM watsonx.ai.

## Folder

| Path | What it is |
|---|---|
| `app.py` | The Streamlit app |
| `explain.py` | watsonx.ai analyst-note client and cache reader |
| `data/dossier.parquet` | One row per submitted CVE: ranks, EPSS, exploit flags, CVSS, Granite fields, NVD text |
| `data/explanations.json` | watsonx notes for 425 of the 500 (the free-plan quota ran out before the rest) |
| `data/missed_2025.csv`, `data/missed_summary.csv` | The 22 missed 2025 KEV CVEs, and missed vs found stats |
| `scripts/build_data.py` | Rebuilds `data/` from the repo |
| `scripts/fetch_nvd.py` | Fetches NVD descriptions and CVSS for 2026 (display only) |
| `scripts/explain_all.py` | Generates missing watsonx notes |

## Honesty notes

- NVD descriptions and current NVD vectors were fetched on Nov 8, 2026 for reading only; no model ever used them. The CVSS 3 fields shown "at publish" come from the IBM data via `luis/derived/for_chen`.
- Leaky fields (NVD status, reference counts, affected-product counts) are not loaded.
- The watsonx notes are written by `meta-llama/llama-3-3-70b-instruct` on watsonx.ai (Toronto, the only region where our project lives; it has no Granite chat model). The model only gets the facts shown on the page. Notes are display-only and never feed the ranking.

## Rebuild data (optional)

```bash
python honiya/dashboard/scripts/fetch_nvd.py          # about 5 min; writes /agent/work/nvd_2026.parquet (edit OUT at the top)
python honiya/dashboard/scripts/build_data.py --repo . --nvd /agent/work/nvd_2026.parquet
```

To regenerate notes, create `honiya/dashboard/.streamlit/secrets.toml` (git-ignored) with `WATSONX_APIKEY`, `WATSONX_PROJECT_ID` and `WATSONX_URL = "https://ca-tor.ml.cloud.ibm.com"`, then run `python honiya/dashboard/scripts/explain_all.py` with the same values exported as environment variables.
