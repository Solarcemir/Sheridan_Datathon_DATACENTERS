#!/usr/bin/env bash
# Train one CV fold of a LoRA config on its purged training ids, score its test block with the
# final (epoch 3) adapter only (fixed in advance, no epoch picking).
# Usage: scripts/cv_lora.sh <config> <gpu> <fold> [train_lora args, e.g. --neg-ratio 20 --meta]
cd "$(dirname "$0")/.."
cfg=$1 gpu=$2 fold=$3; shift 3
name=cv_${cfg}_f${fold}
unset LD_LIBRARY_PATH VIRTUAL_ENV
export CUDA_HOME=/usr/local/cuda PATH="$HOME/venv-vllm-cu13/bin:/usr/local/cuda/bin:$PATH"
MODEL=${MODEL:-ibm-granite/granite-4.2-8b}
sflags=""; [[ " $* " == *" --meta "* ]] && sflags="$sflags --meta"; [[ " $* " == *" --llmx "* ]] && sflags="$sflags --llmx"
CUDA_VISIBLE_DEVICES=$gpu "$HOME/venv-sft-cu13/bin/python" scripts/train_lora.py --name "$name" \
  --train-ids "processed/cv_fold${fold}_train.txt" --epochs 3 --model "$MODEL" "$@" > "logs/train_${name}.log" 2>&1
echo "$(date +%T) $name train exited $?"
CUDA_VISIBLE_DEVICES=$gpu "$HOME/venv-vllm-cu13/bin/python" scripts/llm_score.py --mode zero --target ids \
  --ids "processed/cv_fold${fold}_test.txt" --model "$MODEL" --lora "runs/$name/epoch3" \
  --max-model-len 4096 $sflags --out "results/${name}.parquet" > "logs/score_${name}.log" 2>&1
echo "$(date +%T) $name score exited $?"
