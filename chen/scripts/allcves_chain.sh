#!/usr/bin/env bash
# After Granite 30B (extract.py --all) finishes: Granite features, then LoRA scoring on GPUs 1-3 and the unified LR on CPU.
cd "$(dirname "$0")"
while pgrep -f "extract.py --all" > /dev/null; do sleep 20; done
echo "$(date +%T) granite done"; ls -la ../results/xall_*
CUDA_VISIBLE_DEVICES="" ~/venv-vllm-cu13/bin/python llm_features.py allcves
echo "$(date +%T) llm features exit $?"
for s in 0 1 2; do g=$((s+1)); setsid nohup bash allcves_lora.sh $g $s 3 > ../logs/allcves_lora_$s.log 2>&1 < /dev/null & done
CUDA_VISIBLE_DEVICES="" ~/venv-vllm-cu13/bin/python all_cves.py lr
echo "$(date +%T) lr exit $?"
