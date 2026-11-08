#!/usr/bin/env bash
# Extra seeds of the final Granite 4.2 8B LoRA (same cv_control recipe as final_lora.sh; only --seed differs),
# for a seed-averaged LLM voter. Usage: setsid nohup scripts/final_lora_seed.sh <gpu> <seed> > logs/final_lora_s<seed>.log 2>&1 &
cd "$(dirname "$0")/.."
unset LD_LIBRARY_PATH VIRTUAL_ENV; export CUDA_HOME=/usr/local/cuda
gpu=$1 seed=$2
CUDA_VISIBLE_DEVICES=$gpu ~/venv-sft-cu13/bin/python scripts/train_lora.py --name final_lora8b_s$seed --seed $seed \
  --train-months 2024-01:2025-09 --epochs 3 --neg-ratio 20 --model ibm-granite/granite-4.2-8b
echo "$(date +%T) train exited $?"
CUDA_VISIBLE_DEVICES=$gpu ~/venv-vllm-cu13/bin/python scripts/llm_score.py --mode zero --target scoreboard \
  --model ibm-granite/granite-4.2-8b --lora runs/final_lora8b_s$seed/epoch3 --max-model-len 4096 \
  --out results/final_lora8b_s${seed}_scoreboard.parquet
echo "$(date +%T) score exited $?"
