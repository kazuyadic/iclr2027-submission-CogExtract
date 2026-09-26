"""Quick evaluation: re-run only conference websites (w_001, w_002, w_003)
and merge with v7.1 results for all other websites.

This is much faster than full eval (~546 samples instead of 35758).
"""
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from configs.config import VGSConfig
from utils.data_loader import DataLoader
from utils.evaluator import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Conference website group IDs
CONF_WEBSITES = {"001", "002", "003"}  # ICLR, NeurIPS, ICML


async def main():
    config = VGSConfig()
    loader = DataLoader(config.data_dir)

    # Load all groups
    groups = loader.build_grouped_samples()
    logger.info("Total groups: %d", len(groups))

    # Split into conference vs non-conference
    conf_groups = []
    non_conf_groups = []
    for g in groups:
        sid = g.get("sample_id", "")
        w_num = sid.split("_")[1] if "_" in sid else ""
        if w_num in CONF_WEBSITES:
            conf_groups.append(g)
        else:
            non_conf_groups.append(g)

    logger.info("Conference groups: %d, Non-conference: %d",
                len(conf_groups), len(non_conf_groups))

    # Run pipeline on conference groups
    from cog.pipeline_cog import CogPipeline
    pipeline = CogPipeline(config, max_rounds=1, enable_monitor=False)

    logger.info("Running pipeline on %d conference groups...", len(conf_groups))
    conf_results = await pipeline.run_grouped_batch(conf_groups, concurrency=3)
    logger.info("Conference results: %d", len(conf_results))

    # Load v7.1 results for non-conference
    v71_path = config.project_root / "output" / "cog_plus_v7_1" / "results.json"
    with open(v71_path) as f:
        v71_results = json.load(f)

    # Filter v7.1 results to non-conference
    non_conf_sids = set()
    for g in non_conf_groups:
        non_conf_sids.add(g.get("sample_id", ""))

    v71_non_conf = [r for r in v71_results
                    if "_".join(r.get("sample_id", "").split("_")[:6]) in non_conf_sids
                    or r.get("sample_id", "").split("_")[1] not in CONF_WEBSITES]

    logger.info("v7.1 non-conference results: %d", len(v71_non_conf))

    # Merge results
    all_results = conf_results + v71_non_conf

    # Evaluate
    data_root = config.data_dir / "LiveWeb_IE"
    sample_url_to_meta = {}
    for g in groups:
        sample_id = g.get("sample_id", "")
        for url in g["urls"]:
            sample_url_to_meta[(sample_id, url)] = g

    task_types_list = []
    for res in all_results:
        url = res.get("url", "")
        sid_parts = res.get("sample_id", "").split("_")
        if len(sid_parts) >= 6:
            sample_id = "_".join(sid_parts[:6])
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

    # Aggregate
    all_scores = [r.get("eval_score", {"precision": 0, "recall": 0, "f1": 0})
                  for r in all_results]
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

    # Save
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== v7.3 Quick Eval (conference re-run + v7.1 baseline) ===")
    print(f"  Total: {len(all_results)}, Conference: {len(conf_results)}, v7.1: {len(v71_non_conf)}")
    for key, val in metrics.items():
        print(f"  {key}: P={val['precision']:.2f}  R={val['recall']:.2f}  F1={val['f1']:.2f}  (n={val['count']})")

    # Compare with v7.1
    v71_metrics_path = config.project_root / "output" / "cog_plus_v7_1" / "metrics.json"
    with open(v71_metrics_path) as f:
        v71_metrics = json.load(f)

    print("\n=== Comparison with v7.1 ===")
    for key in ["overall", "type_1", "type_2", "type_3", "type_4"]:
        v71_f1 = v71_metrics.get(key, {}).get("f1", 0)
        v73_f1 = metrics.get(key, {}).get("f1", 0)
        delta = v73_f1 - v71_f1
        print(f"  {key}: v71={v71_f1:.2f}  v73={v73_f1:.2f}  delta={delta:+.2f}")


if __name__ == "__main__":
    asyncio.run(main())
