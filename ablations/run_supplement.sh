#!/bin/bash
# Supplement ablations requested by review: K=1+EGV+FCR and K=3+samepage+FCR
unset https_proxy http_proxy ALL_PROXY
MODES=("single_verify_reflect" "samepage_reflect")
MODELS=("qwen3-vl-8b-instruct" "qwen3-vl-32b-instruct" "MiniMax-M2.5" "kimi-k2.5" "qwen3.7-plus")
for model in "${MODELS[@]}"; do
  (
    for mode in "${MODES[@]}"; do
      echo "=== $(date '+%F %T') START $mode $model ==="
      python3 ablations/run_ablation.py "$mode" --model "$model" --concurrency 3
      echo "=== $(date '+%F %T') END $mode $model ==="
    done
  ) > "experiments/supplement_${model}.log" 2>&1 &
done
wait
echo "ALL SUPPLEMENT ABLATIONS DONE $(date '+%F %T')"
