"""LoRA fine-tune Granite to answer Yes/No to "will this CVE be added to KEV?".

Same prompt as zero-shot scoring (prompts.INSTRUCTION + format_cve + QUESTION, thinking off).
Loss: binary cross-entropy on logit(Yes) - logit(No) at the answer position. That gap is
exactly the score llm_score.py ranks by (logP(Yes) - logP(No) share one normaliser).
Only the last position goes through lm_head, so the 100k-vocab logits are never materialised.

Each epoch uses every positive plus a fresh sample of negatives (--neg-ratio per positive;
0 = all negatives). Adapters are saved after every epoch.

  CUDA_VISIBLE_DEVICES=1 python scripts/train_lora.py --name r20 --train-months 2024-01:2024-12 \
      --neg-ratio 20 --epochs 4
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

import prompts

ROOT = Path(__file__).resolve().parent.parent
DATA = next((p for p in (ROOT / "IBM" / "data", ROOT.parent / "IBM-20261107T174321Z-1-001" / "IBM" / "data" / "data") if p.exists()), ROOT / "IBM" / "data")
PROC = ROOT / "processed"


def build_texts(df: pd.DataFrame, tok, cwe) -> list[str]:
    return [
        tok.apply_chat_template(
            [{"role": "system", "content": prompts.INSTRUCTION},
             {"role": "user", "content": prompts.format_cve(r, cwe) + "\n\n" + prompts.QUESTION}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for r in df.itertuples()
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="ibm-granite/granite-4.2-8b")
    ap.add_argument("--train-months", default="2024-01:2024-12")
    ap.add_argument("--neg-ratio", type=int, default=20,
                    help="random negatives per positive per epoch (0 = all); drawn from outside the hard pool")
    ap.add_argument("--hard-skip", type=int, default=50,
                    help="skip the top-N out-of-fold false positives (likely exploited-but-not-in-KEV)")
    ap.add_argument("--hard-pool", type=int, default=0,
                    help="hard pool = negatives at OOF ranks hard_skip+1 .. hard_skip+hard_pool (0 = no hard negatives)")
    ap.add_argument("--hard-per-epoch", type=int, default=0,
                    help="hard negatives per epoch, sampled from the pool (= pool size: same set every epoch)")
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--meta", action="store_true", help="add metadata lines to the prompt")
    ap.add_argument("--llmx", action="store_true", help="add Granite 30B profile lines to the prompt")
    ap.add_argument("--train-ids", help="text file of cve_ids to train on (overrides --train-months)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = ROOT / "runs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    hist = prompts.add_epss_percentile(pd.read_parquet(PROC / "historical_2024_2025.parquet"))
    lo, hi = args.train_months.split(":")
    if args.train_ids:
        ids = set(Path(args.train_ids).read_text().split())
        train = hist[hist.cve_id.isin(ids)].reset_index(drop=True)
        lo, hi = "ids", Path(args.train_ids).name
    else:
        train = hist[hist.snapshot_month.between(lo, hi)].reset_index(drop=True)
    if args.meta:
        train = prompts.attach_meta(train, "historical")
    if args.llmx:
        train = prompts.attach_llmx(train)
    pos = np.flatnonzero(train.label.to_numpy() == 1)
    neg = np.flatnonzero(train.label.to_numpy() == 0)
    hard = np.array([], dtype=int)
    if args.hard_pool:
        oof = pd.read_parquet(PROC / "oof_2024.parquet").set_index("cve_id").oof_score
        neg_by_score = neg[np.argsort(-train.cve_id.iloc[neg].map(oof).to_numpy())]
        hard = neg_by_score[args.hard_skip:args.hard_skip + args.hard_pool]
        skipped = neg_by_score[:args.hard_skip]
        neg = np.setdiff1d(neg, np.concatenate([hard, skipped]))  # random pool excludes hard + skipped
    print(f"train {lo}..{hi}: {len(train):,} CVEs, {len(pos)} positives, hard pool {len(hard)}, "
          f"random pool {len(neg):,}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cwe = prompts.load_cwe_names(DATA)
    texts = build_texts(train, tok, cwe)
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    base = model.base_model.model                       # GraniteForCausalLM
    head_w = base.lm_head.weight[[yes_id, no_id]]       # (2, hidden), frozen
    logits_scaling = getattr(base.config, "logits_scaling", 1.0) or 1.0

    def margin(batch_texts: list[str]) -> torch.Tensor:
        enc = tok(batch_texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        h = base.model(**enc).last_hidden_state[:, -1]  # left padding: last position is the answer slot
        lg = (h @ head_w.T).float() / logits_scaling
        return lg[:, 0] - lg[:, 1]

    n_neg = len(neg) if args.neg_ratio == 0 else min(len(neg), args.neg_ratio * len(pos))
    n_hard = min(args.hard_per_epoch, len(hard))
    steps_per_epoch = math.ceil((len(pos) + n_neg + n_hard) / args.batch)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps_per_epoch * args.epochs),
                                            steps_per_epoch * args.epochs)
    log = []
    t0 = time.time()
    model.train()
    for ep in range(1, args.epochs + 1):
        idx = np.concatenate([pos, np.random.choice(neg, n_neg, replace=False),
                              np.random.choice(hard, n_hard, replace=False) if n_hard else hard[:0]])
        np.random.shuffle(idx)
        y_all = train.label.to_numpy()[idx].astype(np.float32)
        run_loss = 0.0
        for s in range(steps_per_epoch):
            b = idx[s * args.batch:(s + 1) * args.batch]
            y = torch.tensor(y_all[s * args.batch:(s + 1) * args.batch], device="cuda")
            m = margin([texts[i] for i in b])
            loss = F.binary_cross_entropy_with_logits(
                m, y, pos_weight=torch.tensor(args.pos_weight, device="cuda"))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            run_loss += loss.item()
            if (s + 1) % 50 == 0 or s + 1 == steps_per_epoch:
                el = time.time() - t0
                print(f"ep {ep} step {s + 1}/{steps_per_epoch} loss {run_loss / (s + 1):.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {el / 60:.1f} min", flush=True)
        model.save_pretrained(out_dir / f"epoch{ep}")
        log.append({"epoch": ep, "loss": run_loss / steps_per_epoch, "minutes": round((time.time() - t0) / 60, 1)})
        print(f"saved {out_dir / f'epoch{ep}'}", flush=True)
    (out_dir / "train.json").write_text(json.dumps({**vars(args), "n_pos": len(pos), "n_neg_per_epoch": n_neg,
                                                     "n_hard_per_epoch": n_hard,
                                                     "log": log}, indent=2))


if __name__ == "__main__":
    main()
