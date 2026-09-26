#!/usr/bin/env python3
"""Run Cog ablation experiments.

Ablation variants:
  single               — K=1 (like VGS), no verification/reflection
  multi_only           — K=3, pick first candidate, no verification/reflection
  verify_no_reflect    — K=3 + structural verification, no reflection
  single_verify_reflect— K=1 + cross-page verification + reflection (no diversity)
  samepage_reflect     — K=3 + reflection, EGV degraded to same-page check
  full                 — K=3 + verification + reflection (full Cog+)

Usage:
    python3 ablations/run_ablation.py <mode> [--model qwen3.7-plus] [--concurrency 5]
    python3 ablations/run_ablation.py single --model qwen3.7-plus
    python3 ablations/run_ablation.py verify_no_reflect --model qwen3.5-27b
"""
import argparse, asyncio, json, logging, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import VGSConfig
from utils.data_loader import DataLoader
from utils.evaluator import Evaluator

VALID_MODES = ["single", "multi_only", "verify_no_reflect",
               "single_verify_reflect", "samepage_reflect", "full"]


async def run_ablation(mode: str, model: str, concurrency: int = 5):
    logging.basicConfig(level=logging.INFO,
                        format=f"%(asctime)s [ablation·{mode}] %(message)s")
    logger = logging.getLogger(__name__)

    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode: {mode}. Must be one of {VALID_MODES}")

    config = VGSConfig(model_name=model)

    # Output directory
    exp_dir = config.experiments_dir / f"ablation_{mode}_{model}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir = exp_dir

    logger.info("=" * 60)
    logger.info("Ablation: %s", mode)
    logger.info("Model:    %s", model)
    logger.info("Output:   %s", exp_dir)
    logger.info("=" * 60)

    # Load data
    loader = DataLoader(config.data_dir)
    groups = loader.build_grouped_samples()
    total_urls = sum(len(g["urls"]) for g in groups)
    logger.info("Loaded %d groups (%d total URLs)", len(groups), total_urls)

    # Create pipeline with ablation mode
    from cog.pipeline_cog import CogPipeline
    pipeline = CogPipeline(
        config, max_rounds=1, enable_monitor=False,
        ablation_mode=mode,
    )

    # Run
    t_start = time.time()
    results = await pipeline.run_grouped_batch(groups, concurrency=concurrency)
    elapsed = time.time() - t_start
    logger.info("Extraction complete: %d results in %.1fs", len(results), elapsed)

    # Evaluate
    data_root = config.data_dir / "LiveWeb_IE"
    sample_url_to_meta = {}
    for g in groups:
        sample_id = g.get("sample_id", "")
        for url in g["urls"]:
            sample_url_to_meta[(sample_id, url)] = g

    total_f1 = 0
    total_p = 0
    total_r = 0
    for res in results:
        url = res.get("url", "")
        sid_parts = res.get("sample_id", "").split("_")
        sample_id = "_".join(sid_parts[:6]) if len(sid_parts) >= 6 else res.get("sample_id", "")
        meta = sample_url_to_meta.get((sample_id, url), {})

        gt_paths = meta.get("label_paths", [])
        gt = Evaluator.load_ground_truth(gt_paths, data_root, url) if gt_paths else {}

        if gt and res.get("values"):
            score = Evaluator.evaluate_sample(res["values"], gt)
        else:
            score = {"f1": 0, "precision": 0, "recall": 0}

        res["eval_score"] = score
        total_f1 += score.get("f1", 0)
        total_p += score.get("precision", 0)
        total_r += score.get("recall", 0)

    n = max(len(results), 1)
    avg_f1 = total_f1 / n * 100
    avg_p = total_p / n * 100
    avg_r = total_r / n * 100

    logger.info("Results: F1=%.1f  P=%.1f  R=%.1f", avg_f1, avg_p, avg_r)

    # Save results
    result_file = exp_dir / f"ablation_{mode}_{model}_LiveWeb-IE_results.json"
    result_file.write_text(json.dumps(results, ensure_ascii=False))
    logger.info("Saved: %s", result_file)

    # Save summary
    summary = {
        "mode": mode,
        "model": model,
        "total_samples": len(results),
        "f1": round(avg_f1, 1),
        "precision": round(avg_p, 1),
        "recall": round(avg_r, 1),
        "elapsed_seconds": round(elapsed, 1),
    }
    summary_file = exp_dir / "metrics.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    logger.info("Summary: %s", summary_file)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Cog Ablation Experiments")
    parser.add_argument("mode", choices=VALID_MODES,
                        help="Ablation variant to run")
    parser.add_argument("--model", default="qwen3.7-plus",
                        help="Model name (default: qwen3.7-plus)")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Concurrent workers (default: 5)")
    args = parser.parse_args()

    summary = asyncio.run(run_ablation(args.mode, args.model, args.concurrency))
    print(f"\n{'='*40}")
    print(f"Ablation: {summary['mode']}")
    print(f"Model:    {summary['model']}")
    print(f"F1: {summary['f1']:.1f}  P: {summary['precision']:.1f}  R: {summary['recall']:.1f}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
