#!/usr/bin/env bash
# CogExtract (slim) — reproduction driver
#
# Usage:
#   ./run_all.sh quick   # 24-sample subset, offline page cache (needs API key)
#   ./run_all.sh full    # full reproduction (needs API key, ~hours per model)
#   ./run_all.sh smoke   # offline sanity check: dataset load + cache coverage
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-quick}"
MODEL="${MODEL:-qwen3-vl-8b-instruct}"   # override: MODEL=qwen3.7-plus ./run_all.sh quick

case "$MODE" in
  quick)
    # 3 groups x 2 URLs per task type = 24 samples. Pages served from the
    # bundled cache/pages (8 pages only — no live browsing).
    #
    # IMPORTANT: main.py eval writes to the SAME dir (experiments/cog_v8) and
    # resumes from checkpoints NOT keyed by model/method/type:
    #   - results_checkpoint.json  (cog; keyed by (group,url))
    #   - phase1_checkpoint.json   (VGS; keyed by group INDEX 0/1/2 -> collides
    #                               across task types!)
    # A stale checkpoint would be silently reused and re-scored. We wipe the
    # whole output dir before every run and stash metrics under quick_out/.
    OUT="experiments/cog_v8"
    QOUT="quick_out/${MODEL}"
    mkdir -p "$QOUT"
    for METHOD in cog vgs; do
      for T in type_1 type_2 type_3 type_4; do
        echo "== [quick] ${METHOD} ${MODEL} ${T} =="
        rm -rf "$OUT"
        python3 main.py eval --method "$METHOD" --model "$MODEL" \
          --types "$T" --limit 6 --urls-per-entry 2
        cp "${OUT}/metrics.json" "${QOUT}/${METHOD}_${T}.json" 2>/dev/null || true
      done
    done
    python3 scripts/collect_quick.py "$MODEL" 2>/dev/null || \
      echo "Per-type metrics in ${QOUT}/"
    echo "Quick run done."
    ;;

  full)
    # Full reproduction. Requires an API key (configs/config.py or environment).
    # The full page cache is NOT bundled — pages are fetched live.
    echo "== [full] main experiment (Cog+) =="
    python3 scripts/run_multi_model.py --concurrency 3
    echo "== [full] VGS baseline =="
    python3 scripts/run_vgs_multi_model.py --concurrency 3
    echo "== [full] CoT / Reflexion baselines =="
    python3 scripts/run_baselines.py --methods cot reflexion --concurrency 3
    echo "== [full] ablations =="
    bash ablations/run_all_ablations.sh
    echo "Full run done. Results under experiments/*/"
    ;;

  smoke)
    # Offline integrity check — no API key, no network.
    python3 - <<'PY'
import sys, hashlib
from pathlib import Path
root = Path.cwd()
sys.path.insert(0, str(root))
from utils.data_loader import DataLoader
from configs.config import VGSConfig
cfg = VGSConfig()
loader = DataLoader(cfg.data_dir)
urls, keys = set(), set()
for t in ["type_1", "type_2", "type_3", "type_4"]:
    for g in loader.build_grouped_samples(limit=6, urls_per_entry=2, task_types=[t]):
        urls.update(g["urls"])
        keys.update((g.get("sample_id", ""), u) for u in g["urls"])
assert len(keys) == 24, f"quick subset should be 24 (sample,url) pairs, got {len(keys)}"
cache = root / "cache" / "pages"
miss = [u for u in urls if not (cache / hashlib.sha256(u.encode()).hexdigest()[:16] / "page.html").exists()]
print(f"quick_subset_urls={len(urls)} cache_missing={len(miss)}")
assert not miss, miss
miss_marked = []
for u in urls:
    d = cache / hashlib.sha256(u.encode()).hexdigest()[:16]
    for m in ("text", "link", "image"):
        if not (d / f"marked_{m}.png").exists():
            miss_marked.append((u, m))
print(f"marked_screenshots_missing={len(miss_marked)} (regenerate: python3 scripts/precache_stage3b.py)")
assert not miss_marked, miss_marked
print("SMOKE OK")
PY
    ;;

  *)
    echo "usage: $0 {quick|full|smoke}"; exit 1;;
esac
