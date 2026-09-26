"""Ablation Study Runner for CogExtract.

Usage:
    python scripts/ablation/run_ablation.py --ablation A1
    python scripts/ablation/run_ablation.py --ablation A2 --model qwen2.5-7b
    python scripts/ablation/run_ablation.py --ablation A3

Each ablation modifies the v8 pipeline minimally:
  A1 (w/o MHG):  truncate initial candidates to K=1
  A2 (w/o CPSV): skip cross-page verification, use first candidate
  A3 (w/o UIR):  set MAX_REFLECTION_ROUNDS = 0

Results are saved to experiments/ablation_{ablation}/
"""
import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# scripts/ablation/ → project root (3 levels up)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
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


# ═══════════════════════════════════════════════════════
#  Ablation-aware Pipeline Factory
# ═══════════════════════════════════════════════════════

def create_pipeline(ablation: str, config: VGSConfig):
    """Create an CogPipeline with ablation-specific modifications."""
    from cog.pipeline_cog import CogPipeline

    if ablation == "A0":
        return CogPipeline(config, max_rounds=1, enable_monitor=False)

    elif ablation == "A1":
        pipeline = CogPipeline(config, max_rounds=1, enable_monitor=False)
        pipeline._ablation_mode = "no_mhg"
        return pipeline

    elif ablation == "A2":
        pipeline = CogPipeline(config, max_rounds=1, enable_monitor=False)
        pipeline._ablation_mode = "no_cpsv"
        return pipeline

    elif ablation == "A3":
        pipeline = CogPipeline(config, max_rounds=1, enable_monitor=False)
        pipeline.MAX_REFLECTION_ROUNDS = 0
        pipeline._ablation_mode = "no_uir"
        return pipeline

    else:
        raise ValueError(f"Unknown ablation: {ablation}")


def patch_pipeline(pipeline, ablation: str):
    """Monkey-patch the pipeline's _verify_and_refine for A1 and A2."""
    original_verify = pipeline._verify_and_refine

    if ablation == "A1":
        async def patched_verify(attribute, query, candidates, seed_html, val_html):
            candidates = candidates[:1]
            return original_verify(attribute, query, candidates, seed_html, val_html)
        pipeline._verify_and_refine = patched_verify

    elif ablation == "A2":
        def patched_verify(attribute, query, candidates, seed_html, val_html):
            from cog.reflector import ReflectionTrace
            trace = ReflectionTrace(
                attribute=attribute,
                initial_xpath=candidates[0] if candidates else "",
                final_xpath=candidates[0] if candidates else "",
                success=bool(candidates),
                total_rounds=0,
            )
            xpath = candidates[0] if candidates else ""
            return xpath, trace
        pipeline._verify_and_refine = patched_verify


# ═══════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument(
        "--ablation", required=True,
        choices=["A0", "A1", "A2", "A3"],
        help="Ablation ID: A0=full, A1=no-MHG, A2=no-CPSV, A3=no-UIR",
    )
    parser.add_argument("--model", default=None,
                        help="Override model name (e.g. qwen2.5-7b, qwen2.5-72b)")
    parser.add_argument("--dataset", default="LiveWeb-IE",
                        help="Dataset name (default: LiveWeb-IE)")
    parser.add_argument("--page-source", default="cached",
                        choices=["cached", "live"],
                        help="How pages are served: cached (local) or live (URL)")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiment config without running")
    args = parser.parse_args()

    ablation = args.ablation
    config = VGSConfig()

    # Override model if specified
    if args.model:
        config.model_name = args.model

    # ── Experiment Registration ──
    exp_id = register_experiment(
        experiment_type="ablation",
        ablation=ablation,
        model=config.model_name,
        dataset=args.dataset,
        page_source=args.page_source,
        config=config,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return

    # Output directory: experiments/ablation_{ablation}/{exp_id}/
    ablation_output = config.experiments_dir / f"ablation_{ablation}" / exp_id
    ablation_output.mkdir(parents=True, exist_ok=True)
    config.output_dir = ablation_output

    logger.info("=" * 60)
    logger.info("Experiment: %s", exp_id)
    logger.info("Ablation:   %s", ablation)
    logger.info("Model:      %s", config.model_name)
    logger.info("Dataset:    %s", args.dataset)
    logger.info("Output:     %s", ablation_output)
    logger.info("=" * 60)

    # Load data
    loader = DataLoader(config.data_dir)
    groups = loader.build_grouped_samples()
    logger.info("Total groups: %d", len(groups))

    # Create pipeline
    pipeline = create_pipeline(ablation, config)

    # Apply monkey patches for A1/A2
    if ablation in ("A1", "A2"):
        patch_pipeline(pipeline, ablation)
        logger.info("Applied patch for %s", ablation)

    # Run
    t_start = time.time()
    results = await pipeline.run_grouped_batch(groups, concurrency=args.concurrency)
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
    all_scores = [r.get("eval_score", {"precision": 0, "recall": 0, "f1": 0})
                  for r in results]
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

    # Add timing info
    metrics["timing"] = {
        "total_seconds": round(elapsed, 1),
        "per_sample_seconds": round(elapsed / max(len(results), 1), 2),
    }

    # ── Save ──
    metrics_path = ablation_output / f"{exp_id}_metrics.json"
    results_path = ablation_output / f"{exp_id}_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Register result files ──
    result_files = {
        "metrics": str(metrics_path),
        "results": str(results_path),
    }
    update_result_files(exp_id, result_files)

    # ── Print ──
    print(f"\n{'=' * 60}")
    print(f"Experiment {exp_id} — Ablation {ablation}")
    print(f"Model: {config.model_name} | Dataset: {args.dataset}")
    print(f"Time: {elapsed:.1f}s ({len(results)} samples)")
    print(f"{'=' * 60}")
    for key, val in metrics.items():
        if key == "timing":
            continue
        print(
            f"  {key:12s}: P={val['precision']:.2f}  "
            f"R={val['recall']:.2f}  F1={val['f1']:.2f}  (n={val['count']})"
        )


if __name__ == "__main__":
    asyncio.run(main())
