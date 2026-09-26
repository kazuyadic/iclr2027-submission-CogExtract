"""Main entry point for VGS reproduction."""
import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs.config import VGSConfig
from vgs.pipeline import VGSPipeline
from utils.data_loader import DataLoader
from utils.evaluator import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="VGS — Visual Grounding Scraper")
    sub = parser.add_subparsers(dest="command")

    # ── single query ──
    single = sub.add_parser("single", help="Run VGS on a single URL + query")
    single.add_argument("--url", required=True)
    single.add_argument("--query", required=True)

    # ── batch evaluation ──
    batch = sub.add_parser("eval", help="Evaluate on LiveWeb-IE dataset")
    batch.add_argument("--limit", type=int, default=None, help="Max samples to evaluate")
    batch.add_argument("--urls-per-entry", type=int, default=None,
                       help="Max URLs per entry (default: all URLs in group)")
    batch.add_argument("--skip", type=int, default=0,
                       help="Skip first N samples (resume from breakpoint)")
    batch.add_argument("--method", choices=["vgs", "cog"], default="vgs",
                       help="Extraction method: vgs | cog (DOM-enhanced)")
    batch.add_argument("--types", nargs="+", default=None,
                       help="Filter by task types, e.g. --types type_3 type_4")

    # ── review results ──
    review = sub.add_parser("review", help="Launch web UI to review results")
    review.add_argument("--port", type=int, default=5555, help="Port for review server")
    review.add_argument("--output-dir", default=None, help="Output directory to review")

    # ── common ──
    for sub_parser in [single, batch]:
        sub_parser.add_argument("--model", default=None)
        sub_parser.add_argument("--api-key", default=None)
        sub_parser.add_argument("--api-base", default=None)
        sub_parser.add_argument("--headless", action="store_true", default=True)
        sub_parser.add_argument("--monitor", action="store_true", default=False,
                                help="Save per-sample trace data for review")

    return parser.parse_args()


async def run_single(args):
    kwargs = dict(headless=args.headless)
    if args.model:
        kwargs["model_name"] = args.model
    if args.api_key:
        kwargs["api_key"] = args.api_key
    if args.api_base:
        kwargs["api_base"] = args.api_base
    config = VGSConfig(**kwargs)

    pipeline = VGSPipeline(config, enable_monitor=args.monitor)
    result = await pipeline.run(args.url, args.query)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.monitor:
        from monitor.report import generate_report
        report_path = generate_report(config.output_dir / "monitor_report.html")
        print(f"\n📊 Monitor report: {report_path}")


async def run_eval(args):
    kwargs = dict(headless=args.headless)
    if args.model:
        kwargs["model_name"] = args.model
    if args.api_key:
        kwargs["api_key"] = args.api_key
    if args.api_base:
        kwargs["api_base"] = args.api_base
    config = VGSConfig(**kwargs)

    loader = DataLoader(config.data_dir)

    # Paper §5.1: "XPaths generated from the first web page in a group
    # and then applied to all other pages within that group."
    groups = loader.build_grouped_samples(
        limit=args.limit,
        urls_per_entry=args.urls_per_entry,
        task_types=getattr(args, 'types', None),
    )
    total_urls = sum(len(g["urls"]) for g in groups)
    logger.info("Loaded %d groups (%d total URLs)", len(groups), total_urls)

    if args.method == "cog":
        from cog.pipeline_cog import CogPipeline
        logger.info("Using Cog (DOM-enhanced reflection) pipeline")
        pipeline = CogPipeline(config, max_rounds=1, enable_monitor=args.monitor)
    else:
        logger.info("Using VGS (baseline) pipeline")
        pipeline = VGSPipeline(config, enable_monitor=args.monitor)

    results = await pipeline.run_grouped_batch(groups, concurrency=3)

    # Save raw results
    output_path = config.output_dir / "results.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s (%d results)", output_path, len(results))

    # Build a (sample_id, url) → group_meta lookup for evaluation
    # Use (sample_id, url) as key to handle cases where same URL is used by multiple queries
    sample_url_to_meta: dict[tuple[str, str], dict] = {}
    for group in groups:
        sample_id = group.get("sample_id", "")
        for url in group["urls"]:
            sample_url_to_meta[(sample_id, url)] = group

    # Evaluate per-result
    data_root = config.data_dir / "LiveWeb_IE"
    task_types_list = []

    for res in results:
        url = res.get("url", "")
        # Extract sample_id from result (format: w_XXX_g_YYY_q_ZZZ_N)
        # Need to extract w_XXX_g_YYY_q_ZZZ (first 6 parts: w, XXX, g, YYY, q, ZZZ)
        sample_id_parts = res.get("sample_id", "").split("_")
        # Format: w_013_g_000_q_002_1 -> w_013_g_000_q_002
        if len(sample_id_parts) >= 6:
            sample_id = "_".join(sample_id_parts[:6])
        else:
            sample_id = res.get("sample_id", "")
        meta = sample_url_to_meta.get((sample_id, url), {})
        label_paths = meta.get("label_paths", [])
        gt = Evaluator.load_ground_truth(label_paths, data_root, url=url)
        pred = {} if "error" in res else res.get("values", {})
        score = Evaluator.evaluate_sample(pred, gt)
        res["eval_score"] = score
        res["ground_truth"] = gt
        task_types_list.append(meta.get("task_type", ""))

    # Re-save with eval scores
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    # Aggregate metrics
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

    metrics_path = config.output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    print("\n=== Evaluation Results (Grouped Protocol) ===")
    print(f"  Groups: {len(groups)}, URLs: {len(results)}")
    for key, val in metrics.items():
        print(f"  {key}: P={val['precision']:.2f}  R={val['recall']:.2f}  F1={val['f1']:.2f}  (n={val['count']})")

    if args.monitor:
        from monitor.report import generate_report
        report_path = generate_report(config.output_dir / "monitor_report.html")
        print(f"\n📊 Monitor report: {report_path}")


def run_review(args):
    from monitor.reviewer import ReviewServer
    config = VGSConfig()
    output_dir = Path(args.output_dir) if args.output_dir else config.output_dir
    server = ReviewServer(output_dir=output_dir, port=args.port)
    server.start()


def main():
    args = parse_args()
    if args.command == "single":
        asyncio.run(run_single(args))
    elif args.command == "eval":
        asyncio.run(run_eval(args))
    elif args.command == "review":
        run_review(args)
    else:
        print("Usage: python main.py {single|eval|review} --help")


if __name__ == "__main__":
    main()
