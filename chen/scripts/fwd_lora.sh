#!/usr/bin/env bash
# Forward-only LoRA 8B (cv_control recipe): fold k trained only on blocks < k (purged), scores block k.
# Usage: setsid nohup scripts/fwd_lora.sh <gpu> <fold> [<fold> ...] > logs/fwd_lora_g<gpu>.log 2>&1 &
cd "$(dirname "$0")/.."
unset LD_LIBRARY_PATH VIRTUAL_ENV; export CUDA_HOME=/usr/local/cuda
gpu=$1; shift
for k in "$@"; do
  CUDA_VISIBLE_DEVICES=$gpu ~/venv-sft-cu13/bin/python scripts/train_lora.py --name cv_fwdlora_f$k \
    --train-ids processed/cv_fwd_fold${k}_train.txt --epochs 3 --neg-ratio 20 --model ibm-granite/granite-4.2-8b
  echo "$(date +%T) fold $k train exited $?"
  CUDA_VISIBLE_DEVICES=$gpu ~/venv-vllm-cu13/bin/python scripts/llm_score.py --mode zero --target ids \
    --ids processed/cv_fold${k}_test.txt --model ibm-granite/granite-4.2-8b --lora runs/cv_fwdlora_f$k/epoch3 \
    --max-model-len 4096 --out results/cv_fwdlora_f$k.parquet
  echo "$(date +%T) fold $k score exited $?"
done
