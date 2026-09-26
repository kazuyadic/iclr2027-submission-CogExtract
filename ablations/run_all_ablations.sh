#!/bin/bash
# Run all Cog ablation variants for a given model
# Usage: bash ablations/run_all_ablations.sh [model] [concurrency]

MODEL="${1:-qwen3.7-plus}"
CONCURRENCY="${2:-5}"

echo "=== Cog Ablation Experiments ==="
echo "Model: $MODEL"
echo "Concurrency: $CONCURRENCY"
echo "Start: $(date)"
echo

unset https_proxy http_proxy ALL_PROXY

MODES=("single" "multi_only" "verify_no_reflect" "full")

for mode in "${MODES[@]}"; do
    echo "--- Running: $mode ---"
    python3 ablations/run_ablation.py "$mode" --model "$MODEL" --concurrency "$CONCURRENCY"
    echo
done

echo "=== All ablations complete ==="
echo "End: $(date)"
echo

# Print comparison
echo "=== Ablation Comparison ($MODEL) ==="
printf "%-25s %8s %8s %8s\n" "Mode" "F1" "P" "R"
printf "%-25s %8s %8s %8s\n" "----" "---" "---" "---"

for mode in "${MODES[@]}"; do
    metrics="experiments/ablation_${mode}_${MODEL}/metrics.json"
    if [ -f "$metrics" ]; then
        python3 -c "
import json
m = json.load(open('$metrics'))
printf = lambda fmt, *a: print(fmt % a)
printf('%-25s %8.1f %8.1f %8.1f', m['mode'], m['f1'], m['precision'], m['recall'])
"
    else
        printf "%-25s %8s %8s %8s\n" "$mode" "—" "—" "—"
    fi
done
