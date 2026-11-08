"""Page NVD 2.0 by publish-date window for Jan-Sep 2026; keep only description + CVSS (display only)."""
import json, time, sys
from datetime import date, timedelta
from pathlib import Path
import requests, pandas as pd

OUT = Path("/agent/work/nvd_raw"); OUT.mkdir(parents=True, exist_ok=True)
URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

def windows():
    d = date(2026, 1, 1)
    while d <= date(2026, 9, 30):
        e = min(d + timedelta(days=29), date(2026, 9, 30))
        yield d, e
        d = e + timedelta(days=1)

def get(params):
    for attempt in range(8):
        try:
            r = requests.get(URL, params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
            print("http", r.status_code, file=sys.stderr)
        except Exception as ex:
            print("err", ex, file=sys.stderr)
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(params)

def cvss(m):
    for key in ("cvssMetricV31", "cvssMetricV30"):
        if key in m and m[key]:
            rows = sorted(m[key], key=lambda x: x.get("type") != "Secondary")  # prefer CNA (Secondary) at publish
            d = rows[0]["cvssData"]
            return d.get("baseScore"), d.get("baseSeverity"), d.get("vectorString"), rows[0].get("source")
    if m.get("cvssMetricV40"):
        d = m["cvssMetricV40"][0]["cvssData"]
        return d.get("baseScore"), d.get("baseSeverity"), d.get("vectorString"), m["cvssMetricV40"][0].get("source")
    return None, None, None, None

rows = []
for s, e in windows():
    start = 0
    while True:
        f = OUT / f"{s}_{start}.json"
        if f.exists():
            j = json.loads(f.read_text())
        else:
            j = get({"pubStartDate": f"{s}T00:00:00.000", "pubEndDate": f"{e}T23:59:59.999",
                     "startIndex": start, "resultsPerPage": 2000})
            f.write_text(json.dumps(j))
            time.sleep(6.5)
        for v in j["vulnerabilities"]:
            c = v["cve"]
            desc = next((d["value"] for d in c.get("descriptions", []) if d["lang"] == "en"), "")
            sc, sev, vec, src = cvss(c.get("metrics", {}))
            cwe = ",".join(sorted({d["value"] for w in c.get("weaknesses", []) for d in w["description"]}))
            rows.append((c["id"], c.get("published", "")[:10], desc, sc, sev, vec, src, cwe))
        print(s, start, j["totalResults"], flush=True)
        start += j["resultsPerPage"]
        if start >= j["totalResults"]:
            break

df = pd.DataFrame(rows, columns=["cve_id", "nvd_published", "description", "cvss_score", "cvss_severity",
                                 "cvss_vector", "cvss_source", "cwe"]).drop_duplicates("cve_id")
df.to_parquet("/agent/work/nvd_2026.parquet", index=False)
print("done", len(df))
