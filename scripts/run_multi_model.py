"""Batch run main experiments for multiple models.

Usage:
    # Dry-run: just register and print config
    python3 scripts/run_multi_model.py --dry-run

    # Run all models
    python3 scripts/run_multi_model.py

    # Run specific models
    python3 scripts/run_multi_model.py --models qwen3.5-122b-a10b qwen3.7-plus
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
from scripts.experiment_registry import register_experiment, update_result_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODELS = [
    "qwen3.5-122b-a10b",
    "qwen3.5-35b-a3b",
    "qwen3.7-plus",
    "qwen3.7-max-2026-06-08",
]


async def run_model(model_name: str, dry_run: bool = False, concurrency: int = 3):
    """Run the full CogExtract (cog) pipeline for a single model."""

    # ── Register experiment ──
    config = VGSConfig(model_name=model_name)
    exp_id = register_experiment(
        experiment_type="full_eval",
        model=model_name,
        dataset="LiveWeb-IE",
        method="CogExtract_v8",
        config=config,
        dry_run=dry_run,
    )

    if dry_run:
        return

    # ── Setup output directory ──
    model_output = config.experiments_dir / f"main_{model_name}"
    model_output.mkdir(parents=True, exist_ok=True)
    config.output_dir = model_output

    logger.info("=" * 60)
    logger.info("Experiment: %s", exp_id)
    logger.info("Model:      %s", model_name)
    logger.info("Output:     %s", model_output)
    logger.info("=" * 60)

    # ── Load data ──
    loader = DataLoader(config.data_dir)
    groups = loader.build_grouped_samples()
    total_urls = sum(len(g["urls"]) for g in groups)
    logger.info("Loaded %d groups (%d total URLs)", len(groups), total_urls)

    # ── Create pipeline ──
    from cog.pipeline_cog import CogPipeline
    pipeline = CogPipeline(config, max_rounds=1, enable_monitor=False)

    # ── Run ──
    t_start = time.time()
    results = await pipeline.run_grouped_batch(groups, concurrency=concurrency)
    elapsed = time.time() - t_start
    logger.info("Extraction complete: %d results in %.1fs", len(results), elapsed)

    # ── Evaluate ──
    data_root = config.data_dir / "LiveWeb_IE"
    sample_url_to_meta = {}
    for g in groups:
        sample_id = g.get("sample_id", "")
        for url in g["urls"]:
            sample_url_to_meta[(sample_id, url)] = g

    task_types_list = []
    for res in results:
        url = res.get("url", "")
        sid_parts = res.get("sample_id", "").split("_")
        sample_id = "_".join(sid_parts[:6]) if len(sid_parts) >= 6 else res.get("sample_id", "")
        meta = sample_url_to_meta.get((sample_id, url), {})
        label_paths = meta.get("label_paths", [])
        gt = Evaluator.load_ground_truth(label_paths, data_root, url=url)
        pred = {} if "error" in res else res.get("values", {})
        score = Evaluator.evaluate_sample(pred, gt)
        res["eval_score"] = score
        res["ground_truth"] = gt
        task_types_list.append(meta.get("task_type", ""))

    # ── Aggregate ──
    all_scores = [r.get("eval_score", {"precision": 0, "recall": 0, "f1": 0}) for r in results]
    type_scores = {}
    for score, task_type in zip(all_scores, task_types_list):
        type_scores.setdefault(task_type, []).append(score)

    def aggregate(scores_list):
        count = len(scores_list)
        if count == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "count": 0}
        return {
            "precision": sum(s["precision"] for s in scores_list) / count * 100,
            "recall": sum(s["recall"] for s in scores_list) / count * 100,
            "f1": sum(s["f1"] for s in scores_list) / count * 100,
            "count": count,
        }

    metrics = {"overall": aggregate(all_scores)}
    for task_type, scores in sorted(type_scores.items()):
        metrics[task_type] = aggregate(scores)

    metrics["timing"] = {
        "total_seconds": round(elapsed, 1),
        "per_sample_seconds": round(elapsed / max(len(results), 1), 2),
    }

    # ── Save with descriptive names ──
    prefix = f"full_eval_CogExtract_v8_{model_name}_LiveWeb-IE"
    results_path = model_output / f"{prefix}_results.json"
    metrics_path = model_output / f"{prefix}_metrics.json"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Register result files ──
    update_result_files(exp_id, {
        "metrics": str(metrics_path),
        "results": str(results_path),
    })

    # ── Print ──
    print(f"\n{'=' * 60}")
    print(f"Experiment {exp_id}")
    print(f"Model: {model_name} | Dataset: LiveWeb-IE")
    print(f"Time: {elapsed:.1f}s ({len(results)} samples)")
    print(f"{'=' * 60}")
    for key, val in metrics.items():
        if key == "timing":
            continue
        print(
            f"  {key:12s}: P={val['precision']:.2f}  "
            f"R={val['recall']:.2f}  F1={val['f1']:.2f}  (n={val['count']})"
        )

    return metrics


async def main():
    parser = argparse.ArgumentParser(description="Batch run main experiments")
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help=f"Models to evaluate (default: {DEFAULT_MODELS})",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config without running")
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    print(f"Models to evaluate: {args.models}")
    print(f"Method: CogExtract v8 (cog)")
    print(f"Dataset: LiveWeb-IE (cached)")
    print(f"{'=' * 60}\n")

    all_results = {}
    for model in args.models:
        print(f"\n{'#'*60}")
        print(f"# Starting: {model}")
        print(f"{'#'*60}\n")

        try:
            metrics = await run_model(model, dry_run=args.dry_run, concurrency=args.concurrency)
            if metrics:
                all_results[model] = metrics
        except Exception as e:
            logger.error("Failed to run %s: %s", model, e, exc_info=True)
            all_results[model] = {"error": str(e)}

    # ── Summary ──
    if all_results and not args.dry_run:
        print(f"\n{'=' * 70}")
        print("MULTI-MODEL SUMMARY")
        print(f"{'=' * 70}")
        print(f"{'Model':<30s} {'P':>8s} {'R':>8s} {'F1':>8s} {'n':>8s}")
        print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        for model, m in all_results.items():
            if "error" in m:
                print(f"{model:<30s} {'ERROR':>8s}")
            else:
                o = m["overall"]
                print(f"{model:<30s} {o['precision']:>8.2f} {o['recall']:>8.2f} {o['f1']:>8.2f} {o['count']:>8d}")


if __name__ == "__main__":
    asyncio.run(main())
