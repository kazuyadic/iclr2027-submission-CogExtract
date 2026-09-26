#!/usr/bin/env python3
"""Run VGS on 100 random badcases and save traces with full stage data."""
import asyncio, json, sys, time, logging, random, hashlib
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
TRACE_DIR = Path(f"experiments/{EXP_NAME}/traces")
NUM_SAMPLES = 100
CONCURRENCY = 5

async def main():
    results = json.load(open(RESULTS_PATH))
    logger.info(f"Loaded {len(results)} results")

    # Find all badcases
    bad_indices = [i for i, r in enumerate(results) if r.get("eval_score", {}).get("f1", 0) == 0]
    logger.info(f"Total badcases (F1=0): {len(bad_indices)}")

    # Random sample
    random.seed(42)
    sampled = random.sample(bad_indices, min(NUM_SAMPLES, len(bad_indices)))
    logger.info(f"Random sample: {len(sampled)} badcases")

    # Setup
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    config = VGSConfig(model_name="qwen3-vl-8b-instruct")
    pipeline = VGSPipeline(config, enable_monitor=False)
    await pipeline.browser.start()
    logger.info("Browser started")

    # Load GT
    loader = DataLoader(config.data_dir)
    gt_root = config.data_dir / "LiveWeb_IE"
    groups = loader.build_grouped_samples(exclude_websites=[])
    url_to_gt = {}
    for g in groups:
        for url in g["urls"]:
            gt = Evaluator.load_ground_truth(g["label_paths"], gt_root, url)
            if gt:
                url_to_gt[url] = gt

    stats = {"up": 0, "same": 0, "err": 0}
    t0 = time.time()

    sem = asyncio.Semaphore(CONCURRENCY)

    async def run_one(idx, seq):
        async with sem:
            old = results[idx]
            url = old["url"]
            query = old["query"]

            try:
                result = await pipeline._run_inner(url, query)

                # Evaluate
                pred = result.get("values", {})
                gt = old.get("ground_truth", {}) or url_to_gt.get(url, {})
                if pred and gt:
                    score = Evaluator.evaluate_sample(pred, gt)
                    f1 = score.get("f1", 0)
                else:
                    score = {}
                    f1 = 0

                old_f1 = old.get("eval_score", {}).get("f1", 0)
                tag = "↑" if f1 > old_f1 else "="
                if f1 > old_f1:
                    stats["up"] += 1
                else:
                    stats["same"] += 1

                # Save trace with stages
                url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
                trace_file = TRACE_DIR / f"badcase_{seq:03d}_{url_hash}.json"
                trace_data = {
                    "idx": idx,
                    "url": url,
                    "query": query,
                    "attributes": result.get("attributes", []),
                    "xpaths": result.get("xpaths", {}),
                    "values": result.get("values", {}),
                    "ground_truth": gt,
                    "old_f1": old_f1,
                    "new_f1": f1,
                    "old_xpaths": old.get("xpaths", {}),
                    "old_values": old.get("values", {}),
                    "stages": result.get("stages", []),
                }
                trace_file.write_text(json.dumps(trace_data, ensure_ascii=False, indent=2, default=str))

                # Update results
                results[idx] = {
                    "url": url, "query": query,
                    "attributes": result.get("attributes", old.get("attributes", [])),
                    "xpaths": result.get("xpaths", {}),
                    "values": result.get("values", {}),
                    "sample_id": old.get("sample_id", ""),
                    "eval_score": score if score else old.get("eval_score", {}),
                }

                elapsed = time.time() - t0
                done = stats["up"] + stats["same"] + stats["err"]
                logger.info(f"  [{tag}] #{seq} idx={idx} {old_f1:.2f}→{f1:.2f} url={url[:50]} | {done}/{NUM_SAMPLES} ↑{stats['up']} ={stats['same']} !{stats['err']} ({elapsed:.0f}s)")

            except Exception as e:
                stats["err"] += 1
                logger.warning(f"  [!] #{seq} idx={idx}: {e}")

    tasks = [run_one(idx, seq) for seq, idx in enumerate(sampled)]
    await asyncio.gather(*tasks)

    elapsed = time.time() - t0
    await pipeline.browser.stop()

    # Save updated results
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False))

    logger.info(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info(f"Improved: {stats['up']}, Same: {stats['same']}, Errors: {stats['err']}")
    logger.info(f"Traces saved to: {TRACE_DIR}")
    logger.info(f"Results updated: {RESULTS_PATH}")

asyncio.run(main())
