#!/bin/bash
# Launch VGS badcase rerun for all models in parallel
# Each model runs as a separate process with 8 internal concurrency

set -e

MODELS=(
    "qwen3.5-27b"
    "qwen3.5-35b-a3b"
    "qwen3.5-122b-a10b"
    "qwen3.7-plus"
    "qwen3.7-max-2026-06-08"
    "qwen3-vl-8b-thinking"
    "qwen3-vl-32b-instruct"
    "qwen3-vl-32b-thinking"
    "qwen3-vl-30b-a3b-instruct"
    "qwen3-vl-30b-a3b-thinking"
    "kimi-k2.5"
    "MiniMax-M2.5"
)

# Skip: qwen3-vl-8b-instruct (already rerun)
# Skip: *_run1, *_run2 (duplicate runs)
# Skip: vgs_baseline (not a specific model)

echo "=== VGS Badcase Rerun — ${#MODELS[@]} models ==="
echo "Start: $(date)"
echo

# Unset proxy for LLM API calls
unset https_proxy http_proxy ALL_PROXY

LOGDIR="output/rerun_logs"
mkdir -p "$LOGDIR"

PIDS=()
for model in "${MODELS[@]}"; do
    logfile="$LOGDIR/rerun_${model}.log"
    echo "Starting: $model → $logfile"
    python3 scripts/rerun_vgs_badcases_cached.py "$model" > "$logfile" 2>&1 &
    PIDS+=($!)
done

echo
echo "All ${#PIDS[@]} processes launched. Waiting..."
echo

# Wait for all and collect exit codes
FAILED=0
for i in "${!PIDS[@]}"; do
    model="${MODELS[$i]}"
    if wait "${PIDS[$i]}"; then
        echo "✓ $model done"
    else
        echo "✗ $model FAILED (check $LOGDIR/rerun_${model}.log)"
        FAILED=$((FAILED + 1))
    fi
done

echo
echo "=== Done: $(date) ==="
echo "Failed: $FAILED / ${#MODELS[@]}"

# Summary
echo
echo "=== F1 Comparison ==="
printf "%-35s %8s %8s\n" "Model" "Old F1" "New F1"
printf "%-35s %8s %8s\n" "-----" "------" "------"

# Old F1 values (before rerun)
declare -A OLD_F1
OLD_F1["qwen3.5-27b"]=33.4
OLD_F1["qwen3.5-35b-a3b"]=30.2
OLD_F1["qwen3.5-122b-a10b"]=10.6
OLD_F1["qwen3.7-plus"]=25.3
OLD_F1["qwen3.7-max-2026-06-08"]=31.2
OLD_F1["qwen3-vl-8b-thinking"]=27.8
OLD_F1["qwen3-vl-32b-instruct"]=14.9
OLD_F1["qwen3-vl-32b-thinking"]=23.6
OLD_F1["qwen3-vl-30b-a3b-instruct"]=21.7
OLD_F1["qwen3-vl-30b-a3b-thinking"]=26.3
OLD_F1["kimi-k2.5"]=23.1
OLD_F1["MiniMax-M2.5"]=18.7

for model in "${MODELS[@]}"; do
    results=$(ls experiments/vgs_${model}/*results.json 2>/dev/null | head -1)
    if [ -n "$results" ]; then
        new_f1=$(python3 -c "import json; r=json.load(open('$results')); print(f'{sum(s.get(\"f1\",0) for s in [x.get(\"eval_score\",{}) for x in r])/len(r)*100:.1f}')")
        old="${OLD_F1[$model]:-?}"
        printf "%-35s %8s %8s\n" "$model" "$old" "$new_f1"
    fi
done
