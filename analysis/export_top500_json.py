"""Turn reports/top500_v2_explained.csv into site/top500.json for the hub's "Top 500" table.

Usage:  python -E -P analysis/export_top500_json.py [version label]
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC = os.path.join(ROOT, "reports", "top500_v2_explained.csv")
RES = os.path.join(ROOT, "reports", "granite_instruct_results.csv")
OUT = os.path.join(ROOT, "site", "top500.json")
label = sys.argv[1] if len(sys.argv) > 1 else "v2 draft"

t = pd.read_csv(SRC)
res = pd.read_csv(RES) if os.path.exists(RES) else None


def num(x, nd=3):
    return None if pd.isna(x) else round(float(x), nd)


CATS = ["edge or network device", "network device", "enterprise server software", "operating system", "browser", "mobile",
        "desktop app", "web app or plugin", "library or developer tool", "iot or consumer device", "industrial or medical"]


def clean_product(p):
    """Granite sometimes echoes the template ("<Vendor's name, e.g. ...>"); drop those."""
    p = "" if pd.isna(p) else str(p).strip()
    return "" if (p.startswith("<") or "e.g." in p or "vendor and product" in p.lower()) else p.strip(" .")


def clean_category(c):
    c = "" if pd.isna(c) else str(c).strip().strip("<>").strip().lower()
    if c == "edge or network device":
        c = "network device"
    return next((k for k in CATS if k == c), next((k for k in CATS if k in c), "other"))


rows = [{
    "rank": int(r["rank"]), "cve": r.cve_id, "month": r.snapshot_month, "assigner": str(r.assigner),
    "product": clean_product(r["product"]),
    "category": clean_category(r.category),
    "p_internet": num(r.get("p_internet")), "p_enterprise": num(r.get("p_enterprise")), "p_niche": num(r.get("p_niche")),
    "epss": num(r.epss, 5), "edb": int(r.exploitdb), "msf": int(r.metasploit), "nuc": int(r.nuclei), "poc": int(r.poc_github),
    "text": num(r.text_kev_lr), "reason": "" if pd.isna(r.reason) else str(r.reason),
} for _, r in t.iterrows()]

note = ("The 500 CVEs from the 2026 scoreboard our model would patch first, in order. Each row shows public exploit "
        "code, EPSS, what IBM Granite read in the description, and a one-line reason written by Granite.")
choice_file = os.path.join(ROOT, "reports", "final_ranking_choice.txt")
if res is not None and os.path.exists(choice_file):
    best = open(choice_file).read().strip()
    r = res.set_index("split")[best]
    note += (f" Ranking method: {best}. On our practice exams it found {int(r.iloc[0])} exploited CVEs in the top 500 "
             f"for 2025 (trained on 2024) and {int(r.iloc[1])} for 2024 (trained on 2025).")
meta = {"version": label, "note": note}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"meta": meta, "rows": rows}, f, ensure_ascii=False, separators=(",", ":"))
print(f"wrote {OUT}: {len(rows)} rows, {os.path.getsize(OUT) / 1024:.0f} KB")
