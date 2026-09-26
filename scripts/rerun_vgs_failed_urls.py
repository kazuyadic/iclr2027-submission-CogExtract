#!/usr/bin/env python3
"""Re-run fully-failed URLs: one VGS run per URL, apply result to all queries."""
import asyncio, json, sys, time, logging
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from configs.config import VGSConfig
from vgs.pipeline import VGSPipeline
from utils.evaluator import Evaluator
from utils.data_loader import DataLoader

EXP_NAME = "vgs_qwen3-vl-8b-instruct"
RESULTS_PATH = Path(f"experiments/{EXP_NAME}/full_eval_VGS_qwen3-vl-8b-instruct_LiveWeb-IE_results.json")
NUM_BROWSERS = 4

async def worker(wid, pipeline, url_queue, all_results, url_to_gt, url_to_indices, stats):
    """Worker: pull URL from queue, run VGS on first query, apply xpaths to all queries."""
    while True:
        try:
            url, first_query = url_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        try:
            # Run full VGS pipeline on first query for this URL
            result = await pipeline.run(url, first_query)
            new_xpaths = result.get("xpaths", {})
            new_values = result.get("values", {})
            new_attrs = result.get("attributes", [])

            # Apply to ALL queries for this URL
            indices = url_to_indices.get(url, [])
            for idx in indices:
                old = all_results[idx]
                gt = old.get("ground_truth", {}) or url_to_gt.get(url, {})

                # Use new xpaths/values
                updated = dict(old)
                updated["xpaths"] = new_xpaths
                updated["values"] = new_values
                if new_attrs:
                    updated["attributes"] = new_attrs

                # Re-evaluate
                if new_values and gt:
                    score = Evaluator.evaluate_sample(new_values, gt)
                    updated["eval_score"] = score
                    f1 = score.get("f1", 0)
                else:
                    f1 = 0

                old_f1 = old.get("eval_score", {}).get("f1", 0)
                if f1 > old_f1:
                    stats["up"] += 1
                else:
                    stats["same"] += 1

                all_results[idx] = updated

            stats["urls"] += 1
            stats["queries"] += len(indices)

            if stats["urls"] % 5 == 0:
                logger.info(f"  W{wid} url={stats['urls']} ↑{stats['up']} ={stats['same']} !{stats['err']}")

        except Exception as e:
            stats["err"] += 1
            stats["urls"] += 1
            logger.warning(f"  W{wid} [!] {url[:50]}: {e}")

async def main():
    results = json.load(open(RESULTS_PATH))
    logger.info(f"Loaded {len(results)} results")

    # Group by URL, find fully-failed
    url_groups = defaultdict(list)
    for i, r in enumerate(results):
        url_groups[r["url"]].append(i)

    fully_failed = {}
    for url, indices in url_groups.items():
        if all(results[i].get("eval_score", {}).get("f1", 0) == 0 for i in indices):
            fully_failed[url] = indices

    logger.info(f"Fully-failed URLs: {len(fully_failed)}")
    total_queries = sum(len(v) for v in fully_failed.values())
    logger.info(f"Total queries affected: {total_queries}")

    # Load GT
    config = VGSConfig(model_name="qwen3-vl-8b-instruct")
    loader = DataLoader(config.data_dir)
    gt_root = config.data_dir / "LiveWeb_IE"
    groups = loader.build_grouped_samples(exclude_websites=[])
    url_to_gt = {}
    for g in groups:
        for url in g["urls"]:
            gt = Evaluator.load_ground_truth(g["label_paths"], gt_root, url)
            if gt:
                url_to_gt[url] = gt

    # Build queue & index map
    queue = asyncio.Queue()
    url_to_indices = {}
    for url, indices in fully_failed.items():
        first_query = results[indices[0]]["query"]
        queue.put_nowait((url, first_query))
        url_to_indices[url] = indices

    # Create pipelines
    pipelines = []
    for i in range(NUM_BROWSERS):
        p = VGSPipeline(config, enable_monitor=False)
        await p.browser.start()
        pipelines.append(p)
    logger.info(f"{NUM_BROWSERS} browsers started")

    stats = {"up": 0, "same": 0, "err": 0, "urls": 0, "queries": 0}
    t0 = time.time()

    workers = [asyncio.create_task(worker(i, pipelines[i], queue, results, url_to_gt, url_to_indices, stats)) for i in range(NUM_BROWSERS)]

    # Periodic checkpoint
    async def saver():
        while stats["urls"] < len(fully_failed):
            await asyncio.sleep(60)
            RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False))
            elapsed = time.time() - t0
            rate = stats["urls"] / elapsed * 60 if elapsed > 0 else 0
            remaining = len(fully_failed) - stats["urls"]
            eta = remaining / rate / 60 if rate > 0 else 0
            logger.info(f"[checkpoint] {stats['urls']}/{len(fully_failed)} urls rate={rate:.1f}/min ETA={eta:.1f}h ↑{stats['up']} ={stats['same']} !{stats['err']}")

    saver_task = asyncio.create_task(saver())
    await asyncio.gather(*workers)
    saver_task.cancel()

    elapsed = time.time() - t0
    for p in pipelines:
        await p.browser.stop()

    # Final save
    backup = RESULTS_PATH.with_suffix(".json.bak")
    if not backup.exists():
        backup.write_text(open(RESULTS_PATH).read())
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False))

    rate = stats["urls"] / elapsed * 60
    logger.info(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info(f"URLs: {stats['urls']}, Queries updated: {stats['queries']}")
    logger.info(f"Rate: {rate:.1f} URL/min")
    logger.info(f"Improved: {stats['up']}, Same: {stats['same']}, Errors: {stats['err']}")
    logger.info(f"Backup: {backup}")

asyncio.run(main())
