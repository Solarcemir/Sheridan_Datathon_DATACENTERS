"""Download NVD's CVE Change History (public API, https://nvd.nist.gov/developers/vulnerabilities#cve-history)
for every change dated 2024-01-01 .. 2026-10-01, in 30-day windows, 5,000 events per page.

Point-in-time use only: features built from it must use events dated on or before each CVE's snapshot month-end,
and must ignore "CVE CISA KEV Update" events (that is the label).

Resumable: finished windows are skipped. Optional NVD_API_KEY env var (50 req / 30 s instead of 5).
  python scripts/fetch_nvd_history.py   -> external/nvd_history/<start>.jsonl.gz (one change event per line)
"""

import gzip
import http.client
import json
import os
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "external" / "nvd_history"
URL = "https://services.nvd.nist.gov/rest/json/cvehistory/2.0"
KEY = os.environ.get("NVD_API_KEY", "")
PAUSE = 0.8 if KEY else 6.5


def get(params: str) -> dict:
    for attempt in range(8):
        req = urllib.request.Request(f"{URL}?{params}", headers={"apiKey": KEY} if KEY else {})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, http.client.HTTPException) as e:
            wait = 30 * (attempt + 1)
            print(f"  retry {attempt + 1} after {e!r}, sleeping {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"giving up on {params}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    start, stop = date(2024, 1, 1), date(2026, 10, 1)
    while start < stop:
        end = min(start + timedelta(days=30), stop)
        f = OUT / f"{start}.jsonl.gz"
        if f.exists():
            start = end
            continue
        rng = f"changeStartDate={start}T00:00:00.000&changeEndDate={end}T00:00:00.000"
        events, idx, total = [], 0, None
        while total is None or idx < total:
            d = get(f"{rng}&resultsPerPage=5000&startIndex={idx}")
            total = d["totalResults"]
            events += [c["change"] for c in d.get("cveChanges", [])]
            idx += d.get("resultsPerPage", 5000) or 5000
            print(f"{start}..{end}: {len(events):,}/{total:,}", flush=True)
            time.sleep(PAUSE)
        tmp = f.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as g:
            for e in events:
                g.write(json.dumps(e) + "\n")
        tmp.rename(f)
        start = end
    print("done", flush=True)


if __name__ == "__main__":
    main()
