"""Many-shot scoring with leave-one-group-out over all of 2024.

Every 2024 CVE is scored with the full-2024 example set minus:
  - the CVE itself
  - the other KEV CVEs in its hand-made group (processed/kev_groups_2024.csv: same product
    disclosed or added to KEV together, or one exploit chain)
  - the near-miss partner of any removed positive (it was chosen for resembling it)

CVEs aren't independent (exploit chains, batch disclosures), so plain leave-one-out
would leak siblings' labels.

Prompts are assembled from pre-tokenised pieces so each example is tokenised once, and
requests are sorted by exclusion set so vLLM's prefix cache is reused.

  python scripts/llm_score_logo.py --dry-run          # just count prompts/tokens
  CUDA_VISIBLE_DEVICES=2,3 python scripts/llm_score_logo.py --tp 2 --out results/all2024_logo.parquet
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

import prompts
from llm_score import DATA, FLOOR, PROC, load_target

INTRO = ("Below are past vulnerabilities from 2024 with their outcomes. They are not a random "
         "sample: exploited ones are over-represented so you can see what they look like.")


def exclusion_sets(target: pd.DataFrame, shots: pd.DataFrame) -> list[tuple[int, ...]]:
    groups = pd.read_csv(PROC / "kev_groups_2024.csv")
    group_of = dict(zip(groups.cve_id, groups.group))
    members: dict[str, set[str]] = {}
    for c, g in group_of.items():
        members.setdefault(g, set()).add(c)

    idx = {c: i for i, c in enumerate(shots.cve_id)}
    partner = {p: idx[c] for c, p in zip(shots.cve_id, shots.pair) if isinstance(p, str)}

    out = []
    for c in target.cve_id:
        removed = members.get(group_of.get(c), {c}) | {c}           # itself + its KEV group
        js = {idx[r] for r in removed if r in idx}
        js |= {partner[r] for r in removed if r in partner}         # near-miss chosen for a removed positive
        out.append(tuple(sorted(js)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibm-granite/granite-4.2-30b")
    ap.add_argument("--shots-file", default=str(PROC / "shots_2024.parquet"))
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=49152)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="results/all2024_logo.parquet")
    ap.add_argument("--target", default="all2024",
                    help="all2024 = leave-one-group-out; holdout/scoreboard = plain many-shot with the "
                         "same fast pre-tokenised prompts (nothing to remove)")
    args = ap.parse_args()

    cwe = prompts.load_cwe_names(DATA)
    target = load_target(args.target).reset_index(drop=True)
    shots = pd.read_parquet(args.shots_file).reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    shell = tok.apply_chat_template(
        [{"role": "system", "content": "@@SYS@@"}, {"role": "user", "content": "@@USR@@"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pre, rest = shell.split("@@SYS@@")
    mid, post = rest.split("@@USR@@")

    head = enc(pre + prompts.INSTRUCTION + "\n\n" + INTRO)
    seg = [enc(f"\n\n### Example\n{prompts.format_cve(r, cwe)}\nAdded to KEV: "
               f"{'Yes' if r.label == 1 else 'No'}") for r in shots.itertuples()]
    queries = [enc(mid + prompts.format_cve(r, cwe) + "\n\n" + prompts.QUESTION + post)
               for r in target.itertuples()]

    excl = exclusion_sets(target, shots)
    order = sorted(range(len(target)), key=lambda i: excl[i])
    n_ex = np.array([len(e) for e in excl])
    is_pos = target.label.to_numpy() == 1
    distinct = len(set(excl))
    # uncached prefill estimate: each distinct set recomputes from its first removed example
    seg_start = np.cumsum([len(head)] + [len(s) for s in seg])
    full = int(seg_start[-1])
    uncached = sum(full - int(seg_start[min(e)]) for e in set(excl) if e)
    print(f"target {len(target):,} CVEs ({is_pos.sum()} positives), {len(shots)} examples, "
          f"full prefix {full:,} tokens")
    print(f"examples removed per CVE: mean {n_ex.mean():.2f}; positives mean {n_ex[is_pos].mean():.2f}, "
          f"max {n_ex.max()}; CVEs with none removed {np.mean(n_ex == 0) * 100:.0f}%")
    print(f"distinct prompts {distinct:,}; est. uncached prefix tokens {uncached / 1e6:.1f}M", flush=True)
    if args.dry_run:
        return

    from vllm import LLM, SamplingParams
    reqs = []
    for i in order:
        keep = set(range(len(seg))) - set(excl[i])
        ids = list(head)
        for j in range(len(seg)):
            if j in keep:
                ids += seg[j]
        reqs.append({"prompt_token_ids": ids + queries[i]})

    yes_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ["Yes", " Yes", "yes", " yes", "YES"]}
    no_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ["No", " No", "no", " no", "NO"]}
    llm = LLM(model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=0.92,
              tensor_parallel_size=args.tp, enable_prefix_caching=True, seed=0)
    t0 = time.time()
    outs = llm.generate(reqs, SamplingParams(max_tokens=1, temperature=0.0, logprobs=20))
    elapsed = time.time() - t0

    rows, first = [], {}
    for i, o in zip(order, outs):
        lp = o.outputs[0].logprobs[0]
        best = lambda ids: max([lp[t].logprob for t in ids if t in lp], default=FLOOR)
        ly, ln = best(yes_ids), best(no_ids)
        first[o.outputs[0].text] = first.get(o.outputs[0].text, 0) + 1
        rows.append({"cve_id": target.cve_id[i], "score": ly - ln, "p_yes": math.exp(ly),
                     "p_no": math.exp(ln), "n_removed": len(excl[i])})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    meta = {**vars(args), "rows": len(rows), "distinct_prompts": distinct, "seconds": round(elapsed, 1),
            "first_tokens": dict(sorted(first.items(), key=lambda kv: -kv[1])[:10])}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
