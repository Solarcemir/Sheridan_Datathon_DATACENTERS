"""Pre-generate watsonx analyst notes for every CVE in data/dossier.parquet (skips cached ones).

Usage: WATSONX_APIKEY=... python scripts/explain_all.py [--limit N]
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from explain import generate, load_cache  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--workers", type=int, default=4)
args = ap.parse_args()

d = pd.read_parquet(Path(__file__).resolve().parent.parent / "data" / "dossier.parquet")
todo = [r for r in d.to_dict("records") if r["cve_id"] not in load_cache()][: args.limit]
print(f"{len(todo)} to generate")


def one(r):
    for attempt in range(4):
        try:
            return generate(r)
        except Exception as ex:
            print(r["cve_id"], "retry", attempt, str(ex)[:200], flush=True)
            time.sleep(5 * (attempt + 1))


with ThreadPoolExecutor(args.workers) as pool:
    for i, _ in enumerate(pool.map(one, todo), 1):
        if i % 25 == 0:
            print(i, "done", flush=True)
print("finished; cached:", len(load_cache()))
