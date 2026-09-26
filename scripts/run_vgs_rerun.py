#!/usr/bin/env python3
"""Wait for qwen3.7-max to finish, then re-run VGS for 27b and plus (2 rounds each)."""
import asyncio
import json
import logging
import sys
import time
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import VGSConfig
from utils.data_loader import DataLoader
from utils.evaluator import Evaluator
from vgs.pipeline import VGSPipeline
from scripts.experiment_registry import register_experiment, update_result_files

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS = ["qwen3.5-27b", "qwen3.7-plus"]
ROUNDS = 2


async def run_vgs_for_model(model: str, run_id: int, concurrency: int = 3):
    config = VGSConfig(model_name=model)

    suffix = f"_run{run_id}"
    output_dir = config.experiments_dir / f"vgs_{model}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir = output_dir

    exp_id = register_experiment(
        experiment_type="full_eval",
        model=model,
        dataset="LiveWeb-IE",
        method="VGS",
        page_source="cached",
        cache_policy="exclude_no_cache",
        config=config,
    )

    logger.info("=" * 60)
    logger.info("VGS rerun | Model: %s | Run: %d | Output: %s", model, run_id, output_dir)
    logger.info("Exp ID: %s", exp_id)
    logger.info("=" * 60)

    loader = DataLoader(config.data_dir)
    groups = loader.build_grouped_samples()
    total = sum(len(g["urls"]) for g in groups)
    logger.info("Loaded %d groups (%d URLs)", len(groups), total)

    pipeline = VGSPipeline(config, enable_monitor=False)
    t0 = time.time()
    results = await pipeline.run_grouped_batch(groups, concurrency=concurrency)
    elapsed = time.time() - t0
    logger.info("Done: %d results in %.1fs (%.1f min)", len(results), elapsed, elapsed / 60)

    # Evaluate
    data_root = config.data_dir / "LiveWeb_IE"
    lookup = {}
    for g in groups:
        for url in g["urls"]:
            lookup[(g.get("sample_id", ""), url)] = g

    types = []
    for r in results:
        sid_parts = r.get("sample_id", "").split("_")
        sid = "_".join(sid_parts[:6]) if len(sid_parts) >= 6 else r.get("sample_id", "")
        meta = lookup.get((sid, r.get("url", "")), {})
        gt = Evaluator.load_ground_truth(meta.get("label_paths", []), data_root, url=r.get("url", ""))
        pred = {} if "error" in r else r.get("values", {})
        r["eval_score"] = Evaluator.evaluate_sample(pred, gt)
        types.append(meta.get("task_type", ""))

    scores = [r["eval_score"] for r in results]
    by_type = {}
    for s, t in zip(scores, types):
        by_type.setdefault(t, []).append(s)

    def agg(sl):
        n = len(sl)
        if not n:
            return {"precision": 0, "recall": 0, "f1": 0, "count": 0}
        return {k: sum(s[k] for s in sl) / n * 100 for k in ("precision", "recall", "f1")} | {"count": n}

    metrics = {"overall": agg(scores)}
    for t, sl in sorted(by_type.items()):
        metrics[t] = agg(sl)
    metrics["timing"] = {
        "total_seconds": round(elapsed, 1),
        "per_sample_seconds": round(elapsed / max(len(results), 1), 2),
    }

    prefix = f"full_eval_VGS_{model}{suffix}_LiveWeb-IE"
    rp = output_dir / f"{prefix}_results.json"
    mp = output_dir / f"{prefix}_metrics.json"
    with open(rp, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(mp, "w") as f:
        json.dump(metrics, f, indent=2)
    update_result_files(exp_id, {"metrics": str(mp), "results": str(rp)})

    o = metrics["overall"]
    print(f"\n{'=' * 60}")
    print(f"VGS | {model} | Run {run_id}")
    print(f"  F1={o['f1']:.2f}  P={o['precision']:.2f}  R={o['recall']:.2f}  (n={o['count']})")
    print(f"{'=' * 60}\n")

    return metrics


async def wait_for_max():
    """Wait for PID 12715 (qwen3.7-max) to finish."""
    import subprocess
    pid = 12715
    logger.info("Waiting for PID %d (qwen3.7-max) to finish...", pid)
    while True:
        result = subprocess.run(["ps", "-p", str(pid)], capture_output=True)
        if result.returncode != 0:
            logger.info("PID %d finished.", pid)
            return
        await asyncio.sleep(30)


async def main():
    # Wait for max to finish
    await wait_for_max()

    logger.info("Starting VGS rerun: %s x %d rounds", MODELS, ROUNDS)

    all_results = {}
    for model in MODELS:
        for run_id in range(1, ROUNDS + 1):
            try:
                metrics = await run_vgs_for_model(model, run_id, concurrency=3)
                all_results[f"{model}_run{run_id}"] = metrics["overall"]["f1"]
            except Exception as e:
                logger.error("Failed: %s run%d: %s", model, run_id, e, exc_info=True)

    # Summary
    print("\n" + "=" * 60)
    print("VGS RERUN SUMMARY")
    print("=" * 60)
    for k, f1 in all_results.items():
        print(f"  {k}: F1={f1:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
