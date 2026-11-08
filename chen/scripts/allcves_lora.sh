#!/usr/bin/env bash
# Score the repo-scanner table (processed/allcves_view.parquet) with the 4 final Granite 4.2 8B LoRA seeds.
# One GPU per shard; each GPU walks through the 4 seeds. Usage: setsid nohup scripts/allcves_lora.sh <gpu> <shard> <nshards> &
cd "$(dirname "$0")/.."
unset LD_LIBRARY_PATH VIRTUAL_ENV; export CUDA_HOME=/usr/local/cuda
gpu=$1; shard=$2; n=$3
mkdir -p results/allcves_lora_parts
for seed in 0 1 2 3; do
  run=runs/final_lora8b; [ "$seed" != 0 ] && run=runs/final_lora8b_s$seed
  CUDA_VISIBLE_DEVICES=$gpu ~/venv-vllm-cu13/bin/python scripts/llm_score.py --mode zero --target allcves \
    --model ibm-granite/granite-4.2-8b --lora $run/epoch3 --max-model-len 4096 --shard $shard --nshards $n \
    --out results/allcves_lora_parts/s${seed}_${shard}of${n}.parquet
  echo "$(date +%T) seed $seed shard $shard exited $?"
done
