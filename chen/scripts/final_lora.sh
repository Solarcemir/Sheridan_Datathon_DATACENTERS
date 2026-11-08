#!/usr/bin/env bash
# Final Granite 4.2 8B LoRA (the "control" CV config: --neg-ratio 20, 3 epochs) on all training months,
# then score the 72,670 scoreboard CVEs. Usage: setsid nohup scripts/final_lora.sh <gpu> > logs/final_lora.log 2>&1 &
cd "$(dirname "$0")/.."
unset LD_LIBRARY_PATH VIRTUAL_ENV; export CUDA_HOME=/usr/local/cuda
gpu=$1
CUDA_VISIBLE_DEVICES=$gpu ~/venv-sft-cu13/bin/python scripts/train_lora.py --name final_lora8b \
  --train-months 2024-01:2025-09 --epochs 3 --neg-ratio 20 --model ibm-granite/granite-4.2-8b
echo "$(date +%T) train exited $?"
CUDA_VISIBLE_DEVICES=$gpu ~/venv-vllm-cu13/bin/python scripts/llm_score.py --mode zero --target scoreboard \
  --model ibm-granite/granite-4.2-8b --lora runs/final_lora8b/epoch3 --max-model-len 4096 \
  --out results/final_lora8b_scoreboard.parquet
echo "$(date +%T) score exited $?"
