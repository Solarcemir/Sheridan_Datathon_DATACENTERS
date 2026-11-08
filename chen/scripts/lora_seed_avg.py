"""Seed-averaged LLM voter: rank-average the final 8B LoRA runs (seed 0 = final_lora8b, seeds 1-3 =
final_lora8b_s{k}), report seed-to-seed stability, write the team file.

  python scripts/lora_seed_avg.py   -> chen-style submissions/lora8b_scoreboard_2026.parquet (cve_id, lora_score)
                                       reports/lora_seed_stability.txt
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "submissions" / "lora8b_scoreboard_2026.parquet"


def main() -> None:
    files = [RES / "final_lora8b_scoreboard.parquet"] + sorted(RES.glob("final_lora8b_s[0-9]*_scoreboard.parquet"))
    runs = {}
    for i, f in enumerate(files):
        d = pd.read_parquet(f).set_index("cve_id").score
        assert len(d) == 72670 and d.index.is_unique and d.notna().all(), f
        runs[f"seed{0 if i == 0 else f.name.split('_s')[1].split('_')[0]}"] = d
    df = pd.DataFrame(runs)
    pct = df.rank(pct=True)
    avg = pct.mean(axis=1)
    top = lambda s: set(s.nlargest(500).index)
    lines = [f"{len(runs)} LoRA runs: {', '.join(runs)} (same recipe, only the seed differs)", "",
             "pairwise Spearman / top-500 overlap:"]
    for a, b in combinations(runs, 2):
        lines.append(f"  {a} vs {b}: {df[a].rank().corr(df[b].rank()):.3f} / {len(top(df[a]) & top(df[b]))}")
    lines.append("each run vs the seed average (top-500 overlap): "
                 + ", ".join(f"{k} {len(top(df[k]) & top(avg))}" for k in runs))
    print("\n".join(lines))
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "lora_seed_stability.txt").write_text("\n".join(lines) + "\n")
    out = pd.DataFrame({"cve_id": avg.index.astype(str), "lora_score": avg.to_numpy(float)})
    assert len(out) == 72670 and out.cve_id.is_unique and out.lora_score.notna().all()
    OUT.parent.mkdir(exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT} (lora_score = mean percentile rank over seeds, higher = more likely)")


if __name__ == "__main__":
    main()
