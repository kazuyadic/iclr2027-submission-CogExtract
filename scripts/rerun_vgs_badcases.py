#!/usr/bin/env python3
"""Re-run VGS badcases with multiple Playwright instances (multi-browser)."""
import asyncio, json, sys, time, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from configs.config import VGSConfig
from vgs.pipeline import VGSPipeline
from utils.evaluator import Evaluator
from utils.data_loader import DataLoader

EXP_NAME = "vgs_qwen3-vl-8b-instruct"
RESULTS_PATH = Path(f"experiments/{EXP_NAME}/full_eval_VGS_qwen3-vl-8b-instruct_LiveWeb-IE_results.json")
NUM_BROWSERS = 4          # 4 Playwright instances
CONCURRENCY_PER_BROWSER = 2  # 2 workers per browser = 8 total
MAX_SAMPLES = 999999

async def worker(worker_id, pipeline, task_queue, results, url_to_gt, stats):
    """Worker that pulls tasks from queue and processes them."""
    while True:
        try:
            idx = task_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        old = results[idx]
        url = old["url"]
        query = old["query"]

        for attempt in range(3):
            try:
                new_result = await pipeline._run_inner(url, query)
                pred = new_result.get("values", {})
                gt = old.get("ground_truth", {}) or url_to_gt.get(url, {})
                if pred and gt:
                    new_score = Evaluator.evaluate_sample(pred, gt)
                    new_f1 = new_score.get("f1", 0)
                else:
                    new_f1 = 0
                    new_score = {}

                old_f1 = old.get("eval_score", {}).get("f1", 0)

                if new_f1 > old_f1:
                    stats["up"] += 1
                    tag = "↑"
                elif new_f1 < old_f1:
                    stats["down"] += 1
                    tag = "↓"
                else:
                    stats["same"] += 1
                    tag = "="

                results[idx] = {
                    "url": url, "query": query,
                    "attributes": new_result.get("attributes", old.get("attributes", [])),
                    "xpaths": new_result.get("xpaths", {}),
                    "values": new_result.get("values", {}),
                    "sample_id": old.get("sample_id", ""),
                    "eval_score": new_score if new_score else old.get("eval_score", {}),
                }
                stats["done"] += 1
                if stats["done"] % 20 == 0:
                    logger.info(f"  W{worker_id} [{tag}] idx={idx} {old_f1:.2f}→{new_f1:.2f} | total={stats['done']} {stats}")
                break

            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(3)
                else:
                    stats["err"] += 1
                    stats["done"] += 1
                    logger.warning(f"  W{worker_id} [!] idx={idx}: {e}")

async def main():
    results = json.load(open(RESULTS_PATH))
    logger.info(f"Loaded {len(results)} results")

    bad_indices = [i for i, r in enumerate(results) if r.get("eval_score", {}).get("f1", 0) == 0]
    logger.info(f"Bad cases (F1=0): {len(bad_indices)}")

    rerun_indices = bad_indices[:MAX_SAMPLES]
    logger.info(f"Re-running {len(rerun_indices)} samples")
    logger.info(f"Browsers: {NUM_BROWSERS} × {CONCURRENCY_PER_BROWSER} = {NUM_BROWSERS * CONCURRENCY_PER_BROWSER} total concurrency")

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
    logger.info(f"GT loaded: {len(url_to_gt)} URLs")

    # Create multiple pipelines (each with own browser)
    pipelines = []
    for i in range(NUM_BROWSERS):
        p = VGSPipeline(config, enable_monitor=False)
        await p.browser.start()
        pipelines.append(p)
        logger.info(f"Browser {i} started")

    # Build task queue
    task_queue = asyncio.Queue()
    for idx in rerun_indices:
        task_queue.put_nowait(idx)

    stats = {"up": 0, "same": 0, "down": 0, "err": 0, "done": 0}

    t0 = time.time()

    # Launch workers (CONCURRENCY_PER_BROWSER workers per pipeline)
    workers = []
    for wid, pipeline in enumerate(pipelines):
        for j in range(CONCURRENCY_PER_BROWSER):
            w = asyncio.create_task(worker(wid * CONCURRENCY_PER_BROWSER + j, pipeline, task_queue, results, url_to_gt, stats))
            workers.append(w)

    # Periodic checkpoint
    async def checkpoint_saver():
        while stats["done"] < len(rerun_indices):
            await asyncio.sleep(120)
            if stats["done"] > 0:
                RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False))
                elapsed = time.time() - t0
                rate = stats["done"] / elapsed * 60
                remaining = len(rerun_indices) - stats["done"]
                eta_h = remaining / rate / 60 if rate > 0 else 0
                logger.info(f"[checkpoint] done={stats['done']}/{len(rerun_indices)} rate={rate:.1f}/min ETA={eta_h:.1f}h {stats}")

    saver = asyncio.create_task(checkpoint_saver())
    await asyncio.gather(*workers)
    saver.cancel()

    elapsed = time.time() - t0

    # Stop all browsers
    for p in pipelines:
        await p.browser.stop()

    # Save final
    backup = RESULTS_PATH.with_suffix(".json.bak")
    if not backup.exists():
        backup.write_text(open(RESULTS_PATH).read())
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False))

    rate = stats["done"] / elapsed * 60
    logger.info(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info(f"Stats: ↑{stats['up']} ={stats['same']} ↓{stats['down']} !{stats['err']}")
    logger.info(f"Rate: {rate:.1f}/min")
    logger.info(f"Backup: {backup}")

asyncio.run(main())
