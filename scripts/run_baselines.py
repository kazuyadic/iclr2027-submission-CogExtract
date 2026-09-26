#!/usr/bin/env python3
"""Run CoT and Reflexion baselines across multiple models.

Usage:
    # Run both methods on all thinking-ablation models
    python3 scripts/run_baselines.py

    # Run specific method
    python3 scripts/run_baselines.py --methods cot

    # Run specific models
    python3 scripts/run_baselines.py --models qwen3-vl-8b-instruct qwen3-vl-32b-instruct
"""
import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import VGSConfig
from utils.data_loader import DataLoader
from utils.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

MODELS = [
    "qwen3-vl-8b-instruct",
    "qwen3-vl-8b-thinking",
    "qwen3-vl-30b-a3b-instruct",
    "qwen3-vl-30b-a3b-thinking",
    "qwen3-vl-32b-instruct",
    "qwen3-vl-32b-thinking",
]

METHODS = ["cot", "reflexion"]


async def run_experiment(model: str, method: str, concurrency: int = 3):
    """Run a single model + method combination."""
    config = VGSConfig(model_name=model)

    output_dir = config.experiments_dir / f"{method}_{model}"
    output_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir = output_dir

    logger.info("=" * 60)
    logger.info("%s | Model: %s | Output: %s", method.upper(), model, output_dir)
    logger.info("=" * 60)

    # Select pipeline
    if method == "cot":
        from baselines.cot_pipeline import CoTPipeline
        pipeline = CoTPipeline(config, enable_monitor=False)
    else:
        from baselines.reflexion_pipeline import ReflexionPipeline
        pipeline = ReflexionPipeline(config, max_rounds=3, enable_monitor=False)

    loader = DataLoader(config.data_dir)
    groups = loader.build_grouped_samples()
    total = sum(len(g["urls"]) for g in groups)
    logger.info("Loaded %d groups (%d URLs)", len(groups), total)

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

    prefix = f"full_eval_{method}_{model}_LiveWeb-IE"
    rp = output_dir / f"{prefix}_results.json"
    mp = output_dir / f"{prefix}_metrics.json"
    with open(rp, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(mp, "w") as f:
        json.dump(metrics, f, indent=2)

    o = metrics["overall"]
    logger.info("%s | %s → F1=%.2f  P=%.2f  R=%.2f  (n=%d)",
                method.upper(), model, o["f1"], o["precision"], o["recall"], o["count"])
    return metrics


async def main():
    parser = argparse.ArgumentParser(description="Run CoT/Reflexion baselines")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        choices=["cot", "reflexion"])
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    print(f"Models: {args.models}")
    print(f"Methods: {args.methods}")
    print(f"Total experiments: {len(args.models) * len(args.methods)}")
    print("=" * 60)

    all_metrics = {}
    for model in args.models:
        for method in args.methods:
            key = f"{method}|{model}"
            print(f"\n{'#' * 60}")
            print(f"# {method.upper()} | {model}")
            print(f"{'#' * 60}")
            try:
                metrics = await run_experiment(model, method, concurrency=args.concurrency)
                all_metrics[key] = metrics
            except Exception as e:
                logger.error("Failed for %s: %s", key, e, exc_info=True)

    # Summary
    print("\n" + "=" * 70)
    print("BASELINES SUMMARY")
    print("=" * 70)
    print(f"{'Method':<12} {'Model':<30} {'P':>8} {'R':>8} {'F1':>8}")
    print("-" * 70)
    for key, m in all_metrics.items():
        method, model = key.split("|")
        o = m["overall"]
        print(f"{method:<12} {model:<30} {o['precision']:8.2f} {o['recall']:8.2f} {o['f1']:8.2f}")


if __name__ == "__main__":
    asyncio.run(main())
