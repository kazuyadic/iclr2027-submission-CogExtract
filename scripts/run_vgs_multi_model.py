#!/usr/bin/env python3
"""Run VGS baseline across multiple models (no-thinking)."""
import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import VGSConfig
from utils.data_loader import DataLoader
from utils.evaluator import Evaluator
from vgs.pipeline import VGSPipeline
from scripts.experiment_registry import register_experiment, update_result_files

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODELS = [
    "qwen3.5-27b",
    "qwen3.5-122b-a10b",
    "qwen3.5-35b-a3b",
    "qwen3.7-plus",
    "qwen3.7-max-2026-06-08",
]


async def run_vgs_for_model(model: str, concurrency: int = 3):
    config = VGSConfig(model_name=model)

    exp_id = register_experiment(
        experiment_type="full_eval",
        model=model,
        dataset="LiveWeb-IE",
        method="VGS",
        page_source="cached",
        cache_policy="exclude_no_cache",
        config=config,
    )

    output_dir = config.experiments_dir / f"vgs_{model}"
    output_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir = output_dir

    logger.info("=" * 60)
    logger.info("VGS baseline | Model: %s | Output: %s", model, output_dir)
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

    prefix = f"full_eval_VGS_{model}_LiveWeb-IE"
    rp = output_dir / f"{prefix}_results.json"
    mp = output_dir / f"{prefix}_metrics.json"
    with open(rp, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(mp, "w") as f:
        json.dump(metrics, f, indent=2)
    update_result_files(exp_id, {"metrics": str(mp), "results": str(rp)})

    print(f"\n{'=' * 60}")
    print(f"VGS | {model}")
    for k, v in metrics.items():
        if k == "timing":
            continue
        print(f"  {k:12s}: P={v['precision']:.2f}  R={v['recall']:.2f}  F1={v['f1']:.2f}  (n={v['count']})")
    print()

    return metrics


async def main():
    parser = argparse.ArgumentParser(description="Run VGS baseline across models")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    print(f"Models: {args.models}")
    print(f"Method: VGS (baseline, no thinking)")
    print(f"Dataset: LiveWeb-IE (cached)")
    print("=" * 60)

    all_metrics = {}
    for model in args.models:
        print(f"\n{'#' * 60}")
        print(f"# Starting: {model}")
        print(f"{'#' * 60}")
        try:
            metrics = await run_vgs_for_model(model, concurrency=args.concurrency)
            all_metrics[model] = metrics
        except Exception as e:
            logger.error("Failed for %s: %s", model, e, exc_info=True)

    # Summary
    print("\n" + "=" * 60)
    print("VGS MULTI-MODEL SUMMARY")
    print("=" * 60)
    print(f"{'Model':<30} {'P':>8} {'R':>8} {'F1':>8} {'n':>8}")
    print("-" * 62)
    for model, m in all_metrics.items():
        o = m["overall"]
        print(f"{model:<30} {o['precision']:8.2f} {o['recall']:8.2f} {o['f1']:8.2f} {o['count']:8d}")


if __name__ == "__main__":
    asyncio.run(main())
