"""Run ablation experiments for MiniMax-M2.5: multi_only + verify_no_reflect."""
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import VGSConfig
from cog.pipeline_cog import CogPipeline
from utils.data_loader import DataLoader
from utils.evaluator import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "MiniMax-M2.5"
DATASET = "LiveWeb-IE"


async def run_ablation(ablation_mode: str):
    """Run one ablation variant for MiniMax-M2.5."""
    output_dir = Path("experiments") / f"ablation_{ablation_mode}_{MODEL}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = VGSConfig(model_name=MODEL)
    config.output_dir = output_dir

    loader = DataLoader(config.data_dir)
    groups = loader.build_grouped_samples()
    logger.info("Loaded %d groups", len(groups))

    pipeline = CogPipeline(
        config,
        max_rounds=1,
        enable_monitor=False,
        ablation_mode=ablation_mode,
    )

    results = await pipeline.run_grouped_batch(groups, concurrency=3)
    logger.info("Got %d results", len(results))

    # Save results
    results_file = output_dir / f"ablation_{ablation_mode}_{MODEL}_{DATASET}_results.json"
    results_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info("Saved results to %s", results_file)

    # Evaluate
    scores, tscores = [], {}
    for r in results:
        url = r.get("url", "")
        gt = r.get("ground_truth", {})
        pred = {} if r.get("error") else r.get("values", {})
        score = Evaluator.evaluate_sample(pred, gt)
        scores.append(score)
        sid = r.get("sample_id", "")
        for t in ["type_1", "type_2", "type_3", "type_4"]:
            if f"_{t}_" in sid or sid.startswith(f"{t}_"):
                tscores.setdefault(t, []).append(score)
                break

    n = len(scores)
    metrics = {
        "overall": {
            "f1": sum(s["f1"] for s in scores) / n * 100,
            "precision": sum(s["precision"] for s in scores) / n * 100,
            "recall": sum(s["recall"] for s in scores) / n * 100,
            "count": n,
        }
    }
    for t, ss in tscores.items():
        tn = len(ss)
        metrics[t] = {
            "f1": sum(s["f1"] for s in ss) / tn * 100,
            "precision": sum(s["precision"] for s in ss) / tn * 100,
            "recall": sum(s["recall"] for s in ss) / tn * 100,
            "count": tn,
        }

    metrics_file = output_dir / "metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2))
    logger.info("Saved metrics to %s", metrics_file)
    logger.info("Overall F1=%.1f P=%.1f R=%.1f (n=%d)",
                metrics["overall"]["f1"], metrics["overall"]["precision"],
                metrics["overall"]["recall"], n)


async def main():
    for mode in ["multi_only", "verify_no_reflect"]:
        logger.info("=" * 60)
        logger.info("Running ablation: %s for %s", mode, MODEL)
        logger.info("=" * 60)
        await run_ablation(mode)


if __name__ == "__main__":
    asyncio.run(main())
