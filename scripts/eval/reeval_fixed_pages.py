"""Re-evaluate specific non-conference URLs that had fixed screenshots.
These are fueleconomy (w_006) and books.toscrape (w_011) pages that
previously had no screenshots.
"""
import asyncio
import hashlib
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


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


async def main():
    config = VGSConfig()
    loader = DataLoader(config.data_dir)
    groups = loader.build_grouped_samples()

    # Build mappings
    sid_to_type = {g['sample_id']: g.get('task_type', '') for g in groups}
    sid_to_group = {g['sample_id']: g for g in groups}

    # Load current results
    results_path = config.output_dir / "results.json"
    with open(results_path) as f:
        results = json.load(f)

    # Find non-conference results that need re-eval
    # These are w_006 (fueleconomy) and w_011 (books) URLs with recently fixed screenshots
    target_websites = {'006', '011'}
    target_results = []
    other_results = []

    for r in results:
        sid = r.get('sample_id', '')
        parts = sid.split('_')
        if len(parts) < 6:
            other_results.append(r)
            continue
        w = parts[1]
        group_sid = '_'.join(parts[:6])
        url = r.get('url', '')

        if w in target_websites:
            h = url_hash(url)
            cache = Path("cache/pages") / h
            ss = cache / "screenshot.png"
            # Check if screenshot was recently fixed (within last 3 hours)
            import time
            if ss.exists() and (time.time() - ss.stat().st_mtime) < 10800:
                target_results.append(r)
            else:
                other_results.append(r)
        else:
            other_results.append(r)

    logger.info(f"Found {len(target_results)} results to re-evaluate")
    if not target_results:
        logger.info("Nothing to do!")
        return

    # Re-run pipeline on target URLs
    from cog.pipeline_cog import CogPipeline
    pipeline = CogPipeline(config, max_rounds=1, enable_monitor=False)

    # Build groups for target URLs
    target_groups = []
    target_group_sids = set()
    for r in target_results:
        parts = r.get('sample_id', '').split('_')
        if len(parts) >= 6:
            group_sid = '_'.join(parts[:6])
            if group_sid not in target_group_sids:
                target_group_sids.add(group_sid)
                # Find the actual group
                for g in groups:
                    if g['sample_id'] == group_sid:
                        target_groups.append(g)
                        break

    logger.info(f"Re-running pipeline on {len(target_groups)} groups ({len(target_results)} URL-results)")
    new_results = await pipeline.run_grouped_batch(target_groups, concurrency=3)
    logger.info(f"Pipeline produced {len(new_results)} new results")

    # Replace old results with new ones
    # Build lookup: (sample_id, url) -> new result
    new_lookup = {}
    for r in new_results:
        key = (r.get('sample_id', ''), r.get('url', ''))
        new_lookup[key] = r

    # Evaluate new results
    data_root = config.data_dir / "LiveWeb_IE"
    replaced = 0
    final_results = []
    for r in other_results:
        final_results.append(r)

    for r in target_results:
        key = (r.get('sample_id', ''), r.get('url', ''))
        if key in new_lookup:
            new_r = new_lookup[key]
            # Evaluate
            sid = new_r.get('sample_id', '')
            parts = sid.split('_')
            group_sid = '_'.join(parts[:6]) if len(parts) >= 6 else sid
            meta = sid_to_group.get(group_sid, {})
            label_paths = meta.get('label_paths', [])
            url = new_r.get('url', '')
            gt = Evaluator.load_ground_truth(label_paths, data_root, url=url)
            pred = {} if "error" in new_r else new_r.get("values", {})
            score = Evaluator.evaluate_sample(pred, gt)
            new_r["eval_score"] = score
            new_r["ground_truth"] = gt
            final_results.append(new_r)
            replaced += 1
        else:
            final_results.append(r)

    logger.info(f"Replaced {replaced}/{len(target_results)} results")

    # Save
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    # Recalculate metrics
    all_scores = [r.get("eval_score", {"precision": 0, "recall": 0, "f1": 0})
                  for r in final_results]
    task_types = []
    for r in final_results:
        sid = r.get('sample_id', '')
        parts = sid.split('_')
        group_sid = '_'.join(parts[:6]) if len(parts) >= 6 else sid
        task_types.append(sid_to_type.get(group_sid, ''))

    type_scores = {}
    for score, tt in zip(all_scores, task_types):
        type_scores.setdefault(tt, []).append(score)

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
    for tt, scores in sorted(type_scores.items()):
        metrics[tt] = aggregate(scores)

    with open(config.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Compare with v7.1
    v71_metrics_path = config.project_root / "output" / "cog_plus_v7_1" / "metrics.json"
    with open(v71_metrics_path) as f:
        v71_metrics = json.load(f)

    print("\n=== v7.3 Final (with all crawl fixes) ===")
    print(f"  Total: {len(final_results)}, Re-evaluated: {replaced}")
    for key in ["overall", "type_1", "type_2", "type_3", "type_4"]:
        v71_f1 = v71_metrics.get(key, {}).get("f1", 0)
        v73_f1 = metrics.get(key, {}).get("f1", 0)
        delta = v73_f1 - v71_f1
        print(f"  {key}: v71={v71_f1:.2f}  v73={v73_f1:.2f}  delta={delta:+.2f}")


if __name__ == "__main__":
    asyncio.run(main())
