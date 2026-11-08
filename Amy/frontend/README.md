# Blind Spot Investigator (dashboard)

A single-page dashboard for IBM's **Before the Breach** challenge (Sheridan Datathon 2026). It explains the 43 confirmed exploits that the team's final model ranked **below the review line**.

The model ranked 75,148 CVEs published in 2024–2025, and analysts review the top 517. Of the 75 CVEs later added to CISA's Known Exploited Vulnerabilities (KEV) catalog, 32 were inside the top 517 (caught) and 43 were outside it (missed). The dashboard opens a case file on each miss.

It looks back at historical results only. It does not predict 2026 exploitation, and it makes no live AI calls.

## Run it

The page loads its data with `fetch`, so it must be served over HTTP. Opening the file directly (`file://`) will not work.

From the repository root:

```powershell
python -m http.server 8770 --bind 127.0.0.1 --directory Amy
```

Then open <http://127.0.0.1:8770/frontend/blind-spot-investigator.html>.

If the browser shows `ERR_EMPTY_RESPONSE`, an old server is probably still holding port 8770. Stop it, or use another port such as `8771` in both the command and the URL.

## Files

| File | Purpose |
| --- | --- |
| `blind-spot-investigator.html` | The whole dashboard: HTML, CSS and JavaScript in one file. |
| `../feature/output/dashboard_investigations.json` | The only data file the page reads. It is built by `Amy/feature/build_dashboard_data.py`. |

The page has no build step and no npm dependencies. Fonts (Archivo, IBM Plex Sans, IBM Plex Mono) load from Google Fonts. Without internet access the page still works, using fallback fonts.

## What's on the page

1. **The line.** A dot chart of all 75 exploited CVEs by rank, with the review line at 517. Hover to read a rank; click a missed dot to open its case. The four headline numbers underneath also act as filters.
2. **Why they were missed.** How caught and missed CVEs differed, what public exploit databases showed for the misses, missing or weak information, and a test of whether flagging model disagreement would have helped. Clicking a bar or row filters the case files.
3. **Case files.** A searchable, filterable list of the 43 misses. Each case has three tabs:
   - **What happened.** Verified facts: the rank from each model, public exploit code at the time, what lowered or raised the score, missing information, and how similar CVEs turned out.
   - **Possible reasons.** AI-written explanations, each tied to specific facts and checked against them automatically.
   - **What to learn.** AI-written advice on what to check when a new vulnerability looks similar.

The **Tour** button in the bottom dock steps through five highlighted cases automatically.

### Keyboard shortcuts

| Key | Action |
| --- | --- |
| `/` | Search the cases |
| `←` `→` | Previous or next case |
| `1` `2` `3` | Switch tabs |
| `Esc` | Stop the tour |

A link such as `#CVE-2024-27348` opens that case directly.

### Theme

The page follows the operating system's light or dark setting. The ◐ button in the header overrides it, and the choice is saved in the browser (`localStorage` key `bsi-theme`).

## How the content is labelled

- **Verified fact:** calculated directly from historical data.
- **AI explanation** and **AI advice:** informed guesses written from those facts. They are not proven causes.
- **Unknown:** the CVE was never tracked by exploit databases. That means its exploit-code status is unknown, not that no exploit existed.

Every case shows its source record in `watson_miss_explanations.json` and whether its AI text came from IBM watsonx.ai or was written or corrected offline.

## Updating the data

The dashboard never edits data. To refresh it after the investigation records change, rebuild the JSON and reload the page:

```powershell
python Amy/feature/build_dashboard_data.py
```

See `Amy/feature/README.md` for the full pipeline.
