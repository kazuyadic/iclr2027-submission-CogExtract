#!/usr/bin/env python3
"""Re-run VGS badcases for a specific model with Phase 1 + Phase 2 reuse (cached).

Usage:
    python3 scripts/rerun_vgs_badcases_cached.py <model_name>
    python3 scripts/rerun_vgs_badcases_cached.py qwen3.7-plus
    python3 scripts/rerun_vgs_badcases_cached.py qwen3-vl-32b-instruct
"""
import asyncio, json, sys, time, logging, argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import VGSConfig
from vgs.pipeline import VGSPipeline
from utils.evaluator import Evaluator
from utils.data_loader import DataLoader

CONCURRENCY = 8


def find_results_file(model: str) -> Path:
    """Find the results JSON file for a given model."""
    exp_dir = Path(f"experiments/vgs_{model}")
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment dir not found: {exp_dir}")
    candidates = list(exp_dir.glob("*results.json"))
    if not candidates:
        raise FileNotFoundError(f"No results.json in {exp_dir}")
    return candidates[0]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model name (e.g. qwen3.7-plus)")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    model = args.model
    logging.basicConfig(level=logging.INFO,
                        format=f"%(asctime)s [{model}] %(message)s")
    logger = logging.getLogger(__name__)

    results_path = find_results_file(model)
    checkpoint_path = results_path.parent / "rerun_checkpoint.json"

    logger.info(f"Results: {results_path}")
    results = json.load(open(results_path))
    logger.info(f"Loaded {len(results)} results")

    # Current F1
    cur_f1 = sum(r.get("eval_score", {}).get("f1", 0) for r in results) / len(results) * 100
    logger.info(f"Current F1: {cur_f1:.1f}")

    # Find badcases
    bad_indices = [i for i, r in enumerate(results)
                   if r.get("eval_score", {}).get("f1", 0) == 0
                   and not r.get("error")]
    logger.info(f"Badcases: {len(bad_indices)}")

    # Group by query
    query_to_indices = defaultdict(list)
    for idx in bad_indices:
        query_to_indices[results[idx]["query"]].append(idx)
    groups = list(query_to_indices.items())
    logger.info(f"Groups (queries): {len(groups)}")

    # Resume
    done_queries = set()
    stats = {"improved": 0, "same": 0, "err": 0}
    if checkpoint_path.exists():
        ckpt = json.load(open(checkpoint_path))
        done_queries = set(ckpt.get("done_queries", []))
        stats = ckpt.get("stats", stats)
        for entry in ckpt.get("updated_results", []):
            results[entry["idx"]] = entry["result"]
        logger.info(f"Resumed: {len(done_queries)} done, ↑{stats['improved']} ={stats['same']} !{stats['err']}")

    remaining = [(q, idxs) for q, idxs in groups if q not in done_queries]
    logger.info(f"Remaining: {len(remaining)}")

    if not remaining:
        logger.info("Nothing to do!")
        return

    # Setup pipeline
    config = VGSConfig(model_name=model)
    pipeline = VGSPipeline(config, enable_monitor=False)

    # Load GT
    loader = DataLoader(config.data_dir)
    gt_root = config.data_dir / "LiveWeb_IE"
    all_groups = loader.build_grouped_samples(exclude_websites=[])
    url_to_gt = {}
    for g in all_groups:
        for url in g["urls"]:
            gt = Evaluator.load_ground_truth(g["label_paths"], gt_root, url)
            if gt:
                url_to_gt[url] = gt

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    t0 = time.time()
    updated_results_buf = []
    done_counter = [0]

    from tqdm import tqdm
    pbar = tqdm(total=len(remaining), desc=f"VGS {model}",
                bar_format=f"{{l_bar}}{{bar}}| {{n_fmt}}/{{total_fmt}} [{model}] [{{elapsed}}<{{remaining}}]")

    async def run_group(query, indices):
        async with sem:
            url_indices = defaultdict(list)
            for idx in indices:
                url_indices[results[idx]["url"]].append(idx)
            urls = list(url_indices.keys())

            first_url = urls[0]
            try:
                phase1 = await pipeline._run_inner(first_url, query)
                xpaths = phase1.get("xpaths", {})
            except Exception as e:
                async with lock:
                    stats["err"] += len(indices)
                    done_queries.add(query)
                    done_counter[0] += 1
                    pbar.update(1)
                logger.warning(f"  [!] P1 {first_url[:40]}: {e}")
                return

            async with lock:
                for url, idxs_for_url in url_indices.items():
                    for idx in idxs_for_url:
                        old = results[idx]
                        gt = old.get("ground_truth", {}) or url_to_gt.get(url, {})

                        if url != first_url:
                            values = await _exec_xpaths(url, xpaths)
                        else:
                            values = phase1.get("values", {})

                        score = Evaluator.evaluate_sample(values, gt) if values and gt else old.get("eval_score", {"f1": 0})
                        old_f1 = old.get("eval_score", {}).get("f1", 0)
                        new_f1 = score.get("f1", 0)

                        results[idx] = {
                            "url": url, "query": query,
                            "attributes": phase1.get("attributes", old.get("attributes", [])),
                            "xpaths": xpaths, "values": values,
                            "sample_id": old.get("sample_id", ""),
                            "eval_score": score,
                        }
                        updated_results_buf.append({"idx": idx, "result": results[idx]})

                        if new_f1 > old_f1:
                            stats["improved"] += 1
                        else:
                            stats["same"] += 1

                done_queries.add(query)
                done_counter[0] += 1

            pbar.update(1)
            elapsed = time.time() - t0
            rate = done_counter[0] / max(elapsed, 1)
            pbar.set_postfix_str(f"↑{stats['improved']} ={stats['same']} !{stats['err']} {rate:.1f}/s")

            if done_counter[0] % 10 == 0:
                async with lock:
                    save_ckpt()

    async def _exec_xpaths(url, xpaths):
        page = await pipeline.browser.load_page(url)
        values = {}
        for attr, xp in xpaths.items():
            values[attr] = await pipeline.browser.execute_xpath(page, xp)
        try:
            await page.context.close()
        except Exception:
            pass
        return values

    def save_ckpt():
        ckpt = {"done_queries": list(done_queries), "stats": stats,
                "updated_results": updated_results_buf[-5000:]}
        checkpoint_path.write_text(json.dumps(ckpt, ensure_ascii=False))

    tasks = [run_group(q, idxs) for q, idxs in remaining]
    await asyncio.gather(*tasks)

    pbar.close()
    results_path.write_text(json.dumps(results, ensure_ascii=False))
    save_ckpt()

    elapsed = time.time() - t0
    new_f1 = sum(r.get("eval_score", {}).get("f1", 0) for r in results) / len(results) * 100
    logger.info(f"Done in {elapsed:.0f}s | F1: {cur_f1:.1f} → {new_f1:.1f} | "
                f"↑{stats['improved']} ={stats['same']} !{stats['err']}")


asyncio.run(main())
