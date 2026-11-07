#!/usr/bin/env bash
# Experiments 1 (zero-shot) and 2 (many-shot) in parallel:
#   GPU 1     zero-shot
#   GPU 2+3   many-shot (tensor parallel 2: the example prefix is up to ~45k tokens)
# Usage: setsid nohup scripts/run_zero_many.sh <target> <shots.parquet> [extra args] > logs/x.log 2>&1 < /dev/null &
#   target: val2024 (shots from 2024-01..06) or holdout (shots from all 2024)
# Extra args (e.g. --limit 200) are passed to both runs.
cd "$(dirname "$0")/.."
target=$1 shots=$2; shift 2
PY=$HOME/venv-vllm-cu13/bin/python
unset LD_LIBRARY_PATH VIRTUAL_ENV
export CUDA_HOME=/usr/local/cuda PATH="$HOME/venv-vllm-cu13/bin:/usr/local/cuda/bin:$PATH"
mkdir -p logs results
tag=$target; [[ " $* " == *" --limit "* ]] && tag=${target}_smoke

CUDA_VISIBLE_DEVICES=1 $PY scripts/llm_score.py --mode zero --target "$target" --max-model-len 4096 \
  --out "results/${tag}_zero.parquet" "$@" > "logs/${tag}_zero.log" 2>&1 &
z=$!
CUDA_VISIBLE_DEVICES=2,3 $PY scripts/llm_score.py --mode many --target "$target" --tp 2 \
  --shots-file "$shots" --out "results/${tag}_many.parquet" "$@" > "logs/${tag}_many.log" 2>&1 &
m=$!
wait $z; echo "$(date +%T) zero-shot exited $?"
wait $m; echo "$(date +%T) many-shot exited $?"
cd scripts && $PY eval_rank.py "$target" "../results/${tag}_zero.parquet" "../results/${tag}_many.parquet"
