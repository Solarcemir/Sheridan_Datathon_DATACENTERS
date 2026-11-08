#!/usr/bin/env bash
# Detached finisher for the LoRA CV experiments (survives closing the Claude session).
# Waits for in-flight folds (PIDs given as args), then runs every fold whose result file is still
# missing, one queue per GPU, then writes reports/lora_cv_compare.txt.
# Usage: setsid nohup scripts/finish_lora_cv.sh <pid_gpu1> <pid_gpu2> <pid_gpu3> > logs/finish_lora_cv.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
p1=$1 p2=$2 p3=$3

waitpid() { while [ -n "$1" ] && kill -0 "$1" 2>/dev/null; do sleep 30; done; }
run_missing() {  # cfg gpu model folds... [-- extra args]
  local cfg=$1 gpu=$2 model=$3; shift 3
  local folds=() extra=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do folds+=("$1"); shift; done
  [ "$1" = "--" ] && shift && extra=("$@")
  for f in "${folds[@]}"; do
    if [ ! -f "results/cv_${cfg}_f${f}.parquet" ]; then
      echo "$(date +%T) running $cfg fold $f on GPU $gpu"
      MODEL=$model scripts/cv_lora.sh "$cfg" "$gpu" "$f" "${extra[@]}"
    fi
  done
}

( waitpid "$p1"; run_missing llmx8b 1 ibm-granite/granite-4.2-8b 3 4 -- --neg-ratio 20 --llmx ) &
q1=$!
( waitpid "$p2"; run_missing lora30b 2 ibm-granite/granite-4.2-30b 2 4 -- --neg-ratio 20 ) &
q2=$!
( waitpid "$p3"; run_missing lora30b 3 ibm-granite/granite-4.2-30b 3 -- --neg-ratio 20 ) &
q3=$!
wait $q1 $q2 $q3
# a fold could still be missing if one queue finished early; sweep once more on GPU 3
run_missing lora30b 3 ibm-granite/granite-4.2-30b 0 1 2 3 4 -- --neg-ratio 20
run_missing llmx8b 3 ibm-granite/granite-4.2-8b 0 1 2 3 4 -- --neg-ratio 20 --llmx
echo "$(date +%T) all folds present: $(ls results/cv_lora30b_f*.parquet results/cv_llmx8b_f*.parquet 2>/dev/null | wc -l)/10"
"$HOME/venv-vllm-cu13/bin/python" scripts/lora_cv_compare.py > reports/lora_cv_compare.txt 2>&1
echo "$(date +%T) wrote reports/lora_cv_compare.txt"
