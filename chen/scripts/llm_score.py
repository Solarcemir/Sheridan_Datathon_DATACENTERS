"""Score CVEs with Granite: score = log P(Yes) - log P(No) for the first answer token.

  zero:  instructions only.
  many:  instructions + labelled 2024 examples (all positives + hard/random negatives),
         as one shared prefix so vLLM's prefix cache computes it once.

Usage (run on one GPU per shard):
  python scripts/llm_score.py --mode many --target holdout --shard 0 --nshards 2 --out results/...
"""

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import prompts

ROOT = Path(__file__).resolve().parent.parent
DATA = next((p for p in (ROOT / "IBM" / "data", ROOT.parent / "IBM-20261107T174321Z-1-001" / "IBM" / "data" / "data") if p.exists()), ROOT / "IBM" / "data")
PROC = ROOT / "processed"
FLOOR = -30.0  # logprob used when a token isn't in the top-k


def load_target(name: str) -> pd.DataFrame:
    hist = prompts.add_epss_percentile(pd.read_parquet(PROC / "historical_2024_2025.parquet"))
    if name == "holdout":  # 2025-01..2025-09: every CVE has >= 3 months to reach KEV
        return hist[hist.snapshot_month.between("2025-01", "2025-09")]
    if name == "val2024":  # method selection: shots from 2024-01..06, test on 2024-07..12
        return hist[hist.snapshot_month.between("2024-07", "2024-12")]
    if name == "all2024":  # leave-one-group-out (llm_score_logo.py)
        return hist[hist.snapshot_month.between("2024-01", "2024-12")]
    if name == "scoreboard":
        return prompts.add_epss_percentile(pd.read_parquet(PROC / "scoreboard_2026_view.parquet"))
    if name == "allcves":   # repo-scanner lookup table: every other CVE (scripts/all_cves.py)
        return prompts.add_epss_percentile(pd.read_parquet(PROC / "allcves_view.parquet"))
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["zero", "many"], required=True)
    ap.add_argument("--target", choices=["holdout", "val2024", "all2024", "scoreboard", "allcves", "ids"], default="holdout")
    ap.add_argument("--model", default="ibm-granite/granite-4.2-30b")
    ap.add_argument("--shots-file", help="parquet from select_shots.py (required for --mode many)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="debug: score only the first N rows")
    ap.add_argument("--max-model-len", type=int, default=49152)
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel GPUs")
    ap.add_argument("--lora", help="path to a PEFT LoRA adapter (from train_lora.py)")
    ap.add_argument("--meta", action="store_true", help="add metadata lines to the prompt")
    ap.add_argument("--llmx", action="store_true", help="add Granite 30B profile lines to the prompt")
    ap.add_argument("--ids", help="with --target ids: text file of cve_ids to score")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cwe = prompts.load_cwe_names(DATA)
    if args.target == "ids":
        hist = prompts.add_epss_percentile(pd.read_parquet(PROC / "historical_2024_2025.parquet"))
        target = hist[hist.cve_id.isin(set(Path(args.ids).read_text().split()))]
    else:
        target = load_target(args.target)
    target = target.sort_values("cve_id").reset_index(drop=True)
    if args.meta:
        target = prompts.attach_meta(target, "scoreboard" if args.target == "scoreboard" else "historical")
    if args.llmx:
        target = prompts.attach_llmx(target)
    target = target.iloc[args.shard::args.nshards]
    if args.limit:
        target = target.head(args.limit)

    if args.mode == "zero":
        system = prompts.INSTRUCTION
        n_shots = 0
    else:
        shots = pd.read_parquet(args.shots_file)
        assert not set(shots.cve_id) & set(load_target(args.target).cve_id), "shots overlap target"
        assert shots.published.max() < load_target(args.target).published.min(), "shots not before target"
        system = prompts.many_shot_system(shots, cwe)
        n_shots = len(shots)

    tok = AutoTokenizer.from_pretrained(args.model)
    texts = [
        tok.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": prompts.format_cve(r, cwe) + "\n\n" + prompts.QUESTION}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for r in target.itertuples()
    ]
    prefix_tokens = len(tok(system)["input_ids"])
    print(f"mode={args.mode} rows={len(texts)} shots={n_shots} system_tokens={prefix_tokens}", flush=True)
    print("---- example prompt tail ----\n" + texts[0][-1500:] + "\n-----------------------------", flush=True)

    yes_ids = {i for w in ["Yes", " Yes", "yes", " yes", "YES"] for i in tok.encode(w, add_special_tokens=False)[:1]}
    no_ids = {i for w in ["No", " No", "no", " no", "NO"] for i in tok.encode(w, add_special_tokens=False)[:1]}

    lora_kw = {"enable_lora": True, "max_lora_rank": 64} if args.lora else {}
    llm = LLM(model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=0.92,
              tensor_parallel_size=args.tp, enable_prefix_caching=True, seed=args.seed, **lora_kw)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)

    t0 = time.time()
    if args.lora:
        from vllm.lora.request import LoRARequest
        outs = llm.generate(texts, sp, lora_request=LoRARequest("adapter", 1, str(Path(args.lora).resolve())))
    else:
        outs = llm.generate(texts, sp)
    elapsed = time.time() - t0

    rows, top_tokens = [], {}
    for r, o in zip(target.itertuples(), outs):
        lp = o.outputs[0].logprobs[0]
        def best(ids):
            vals = [lp[i].logprob for i in ids if i in lp]
            return max(vals) if vals else FLOOR
        ly, ln = best(yes_ids), best(no_ids)
        first = o.outputs[0].text
        top_tokens[first] = top_tokens.get(first, 0) + 1
        rows.append({"cve_id": r.cve_id, "score": ly - ln, "p_yes": math.exp(ly), "p_no": math.exp(ln)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    meta = {**vars(args), "rows": len(rows), "shots": n_shots, "system_tokens": prefix_tokens,
            "seconds": round(elapsed, 1), "first_tokens": dict(sorted(top_tokens.items(), key=lambda kv: -kv[1])[:10])}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
