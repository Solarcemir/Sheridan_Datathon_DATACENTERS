#!/usr/bin/env bash
# Score every saved epoch of a LoRA run on the 2025 holdout, then compare.
# Usage: scripts/eval_lora.sh <run-name> <gpu> [extra llm_score args, e.g. --meta]
cd "$(dirname "$0")/.."
run=$1 gpu=$2; shift 2
PY=$HOME/venv-vllm-cu13/bin/python
unset LD_LIBRARY_PATH VIRTUAL_ENV
export CUDA_HOME=/usr/local/cuda PATH="$HOME/venv-vllm-cu13/bin:/usr/local/cuda/bin:$PATH"
outs=()
for ep in runs/$run/epoch*; do
  e=$(basename "$ep")
  out=results/holdout_lora_${run}_${e}.parquet
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/llm_score.py --mode zero --target holdout \
    --model ibm-granite/granite-4.2-8b --lora "$ep" --max-model-len 4096 "$@" \
    --out "$out" > "logs/eval_${run}_${e}.log" 2>&1
  echo "$(date +%T) $run $e exited $?"
  outs+=("../$out")
done
cd scripts && $PY eval_rank.py holdout ../results/holdout_zero.parquet "${outs[@]}" 2>&1 | grep -vi warn
