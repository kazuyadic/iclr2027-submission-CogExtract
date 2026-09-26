"""SWDE Cog Runner — runs Cog pipeline on SWDE offline HTML.

Uses full Cog pipeline (Playwright for screenshots + VLM) on first page,
then applies generated XPaths to remaining pages via lxml.

Usage:
    python3 scripts/run_swde_cog.py --model qwen3-vl-8b-instruct --concurrency 20
    python3 scripts/run_swde_cog.py --model qwen3-vl-32b-instruct --verticals auto movie
"""

import asyncio
import argparse
import json
import logging
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lxml import html as lxml_html, etree

from scripts.swde_loader import load_swde_dataset, save_swde_manifest
from utils.evaluator import Evaluator
from configs.config import VGSConfig
from cog.pipeline_cog import CogPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SWDE_DATA_ROOT = str(
    Path(__file__).resolve().parent.parent
    / "data" / "swde" / "swde" / "sourceCode" / "sourceCode"
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "swde"


def html_path_to_file_url(html_path: str) -> str:
    """Convert absolute HTML path to file:// URL."""
    abs_path = str(Path(html_path).resolve())
    return f"file://{abs_path}"


def execute_xpath_on_html(html_content: str, xpath: str) -> list[str]:
    """Execute XPath on local HTML content using lxml."""
    try:
        tree = lxml_html.fromstring(html_content, parser=lxml_html.HTMLParser(encoding="utf-8"))
        results = tree.xpath(xpath)
        if results is None:
            return []
        values = []
        for r in results:
            if isinstance(r, str):
                v = r.strip()
                if v:
                    values.append(v)
            elif isinstance(r, etree._Element):
                text = r.text_content().strip()
                if text:
                    values.append(text)
            elif r is not None:
                values.append(str(r).strip())
        return values
    except Exception as e:
        logger.debug(f"XPath error '{xpath}': {e}")
        return []


async def run_swde_cog(
    groups: list[dict],
    model: str,
    checkpoint_dir: Path,
    concurrency: int = 2,
    max_rounds: int = 1,
) -> list[dict]:
    """Run Cog on SWDE groups.
    
    Phase 1: Run full Cog pipeline on first page (Playwright + VLM)
    Phase 2: Execute generated XPaths on remaining pages via lxml
    """
    config = VGSConfig()
    config.model_name = model
    config.output_dir = checkpoint_dir
    config.screenshot_dir = checkpoint_dir / "screenshots"
    config.screenshot_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / "results_checkpoint.json"
    phase1_checkpoint_path = checkpoint_dir / "phase1_checkpoint.json"
    traces_dir = checkpoint_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    
    # Resume
    all_results = []
    done_urls = set()
    if checkpoint_path.exists():
        all_results = json.loads(checkpoint_path.read_text())
        done_urls = {r.get("url", "") for r in all_results}
        logger.info(f"Resuming: {len(all_results)} page results done")
    
    # Phase 1 checkpoint
    phase1_results = {}
    phase1_done = set()
    if phase1_checkpoint_path.exists():
        for entry in json.loads(phase1_checkpoint_path.read_text()):
            idx = entry.get("_group_idx")
            if idx is not None:
                phase1_results[idx] = entry
                phase1_done.add(idx)
        logger.info(f"Phase 1 checkpoint: {len(phase1_done)} groups done")
    
    start_time = time.time()
    
    # Build groups with file:// URLs
    swde_groups = []
    for g in groups:
        swde_g = dict(g)
        swde_g["urls"] = [html_path_to_file_url(p) for p in g["html_paths"]]
        swde_groups.append(swde_g)
    
    # Create Cog pipeline
    pipeline = CogPipeline(config, max_rounds=max_rounds, enable_monitor=False)
    await pipeline.browser.start()
    
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)
    p1_ok = 0
    p1_err = 0
    p2_ok = 0
    p2_err = 0
    
    async def process_group(gi: int, group: dict) -> None:
        nonlocal p1_ok, p1_err, p2_ok, p2_err
        
        sid = group["sample_id"]
        
        # Check if Phase 1 already done
        if gi in phase1_done:
            p1_result = phase1_results[gi]
        else:
            # Phase 1: Run Cog on first page
            async with sem:
                first_url = group["urls"][0]
                first_html = group["html_paths"][0]
                
                try:
                    logger.info(f"P1 [{gi+1}/{len(groups)}] {sid}: {group['query']}")
                    
                    # Cog needs: seed_url, validation_urls, remaining_urls, query, predefined_attributes
                    seed_url = group["urls"][0]
                    remaining = group["urls"][1:] if len(group["urls"]) > 1 else []
                    validation = remaining[:1]  # use second page for validation
                    
                    cog_result = await pipeline._run_inner(
                        seed_url, validation, remaining, group["query"], None
                    )
                    
                    # Convert CogResult to dict
                    p1_result = {
                        "xpaths": cog_result.xpaths if hasattr(cog_result, 'xpaths') else {},
                        "values": cog_result.values if hasattr(cog_result, 'values') else {},
                        "attributes": cog_result.attributes if hasattr(cog_result, 'attributes') else [],
                        "stages": [],
                    }
                    # Serialize reflection traces per attribute
                    if hasattr(cog_result, 'traces') and cog_result.traces:
                        for attr, rt in cog_result.traces.items():
                            p1_result["stages"].append({
                                "attribute": attr,
                                "initial_xpath": rt.initial_xpath,
                                "final_xpath": rt.final_xpath,
                                "total_rounds": rt.total_rounds,
                                "success": rt.success,
                                "rounds": [
                                    {"round": i+1, "revised_xpath": getattr(r, 'xpath', ''), "reasoning": getattr(r, 'reasoning', '')}
                                    for i, r in enumerate(rt.rounds)
                                ] if rt.rounds else [],
                            })
                    p1_result["_group_idx"] = gi
                    p1_result["sample_id"] = f"{sid}_{Path(first_html).stem}"
                    p1_ok += 1
                    
                    # Save per-group trace
                    trace_file = traces_dir / f"{sid}.json"
                    trace_file.write_text(json.dumps(p1_result, ensure_ascii=False, indent=2))
                    
                    # Save Phase 1 checkpoint
                    async with lock:
                        phase1_results[gi] = p1_result
                        if len(phase1_results) % 10 == 0:
                            phase1_checkpoint_path.write_text(
                                json.dumps(list(phase1_results.values()), ensure_ascii=False, indent=2)
                            )
                
                except Exception as e:
                    logger.error(f"  P1 error on {sid}: {e}")
                    p1_result = {"error": str(e), "_group_idx": gi}
                    p1_err += 1
        
        # Phase 2: Apply XPaths to remaining pages via lxml
        xpaths = p1_result.get("xpaths", {})
        if not xpaths or "error" in p1_result:
            async with lock:
                all_results.append({
                    "sample_id": f"{sid}_phase1_error",
                    "url": "",
                    "query": group["query"],
                    "error": p1_result.get("error", "No XPaths from Phase 1"),
                })
            return
        
        html_paths = group["html_paths"]
        
        # First page result (from Phase 1)
        first_page_id = Path(html_paths[0]).stem
        first_url = group["urls"][0]
        
        async with lock:
            all_results.append({
                "sample_id": f"{sid}_{first_page_id}",
                "url": first_url,
                "query": group["query"],
                "attributes": list(xpaths.keys()),
                "xpaths": xpaths,
                "values": p1_result.get("values", {}),
            })
            p2_ok += 1
        
        # Remaining pages: execute XPaths via lxml
        for html_path in html_paths[1:]:
            page_id = Path(html_path).stem
            url = html_path_to_file_url(html_path)
            
            if url in done_urls:
                continue
            
            try:
                with open(html_path, "r", encoding="utf-8", errors="replace") as f:
                    page_html = f.read()
                
                values = {}
                for attr, xpath in xpaths.items():
                    if xpath:
                        values[attr] = execute_xpath_on_html(page_html, xpath)
                    else:
                        values[attr] = []
                
                async with lock:
                    all_results.append({
                        "sample_id": f"{sid}_{page_id}",
                        "url": url,
                        "query": group["query"],
                        "attributes": list(xpaths.keys()),
                        "xpaths": xpaths,
                        "values": values,
                    })
                    p2_ok += 1
            
            except Exception as e:
                async with lock:
                    all_results.append({
                        "sample_id": f"{sid}_{page_id}",
                        "url": url,
                        "query": group["query"],
                        "error": str(e),
                    })
                    p2_err += 1
        
        # Periodic checkpoint
        async with lock:
            done_count = p1_ok + p1_err
            if done_count % 5 == 0 and done_count > 0:
                checkpoint_path.write_text(
                    json.dumps(all_results, ensure_ascii=False, indent=2)
                )
                elapsed = time.time() - start_time
                rate = done_count / elapsed if elapsed > 0 else 1
                remaining = (len(groups) - done_count) / rate
                logger.info(
                    f"  Progress: {done_count}/{len(groups)} groups, "
                    f"P1(✓{p1_ok}✗{p1_err}) P2(✓{p2_ok}✗{p2_err}), "
                    f"ETA: {remaining/60:.0f} min"
                )
    
    try:
        tasks = [process_group(i, g) for i, g in enumerate(swde_groups)]
        await asyncio.gather(*tasks)
    finally:
        await pipeline.browser.stop()
    
    # Final save
    checkpoint_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2)
    )
    phase1_checkpoint_path.write_text(
        json.dumps(list(phase1_results.values()), ensure_ascii=False, indent=2)
    )
    
    logger.info(f"Done: {len(all_results)} page results, P1(✓{p1_ok}✗{p1_err}) P2(✓{p2_ok}✗{p2_err})")
    return all_results


def evaluate_results(groups: list[dict], results: list[dict]) -> dict:
    """Evaluate Cog results against SWDE GT."""
    gt_lookup = {}
    for g in groups:
        sid = g["sample_id"]
        for page_id, page_gt in g["gt"].items():
            key = f"{sid}_{page_id}"
            gt_lookup[key] = page_gt
    
    vertical_scores = {}
    overall_scores = []
    
    for r in results:
        rid = r.get("sample_id", "")
        gt = gt_lookup.get(rid)
        if gt is None:
            continue
        
        pred = {} if "error" in r else r.get("values", {})
        score = Evaluator.evaluate_sample(pred, gt)
        overall_scores.append(score)
        
        parts = rid.split("_")
        vertical = parts[1] if len(parts) >= 2 else "unknown"
        vertical_scores.setdefault(vertical, []).append(score)
    
    def aggregate(scores):
        if not scores:
            return {"precision": 0, "recall": 0, "f1": 0, "count": 0}
        n = len(scores)
        return {
            "precision": sum(s["precision"] for s in scores) / n * 100,
            "recall": sum(s["recall"] for s in scores) / n * 100,
            "f1": sum(s["f1"] for s in scores) / n * 100,
            "count": n,
        }
    
    metrics = {"overall": aggregate(overall_scores)}
    for v in sorted(vertical_scores.keys()):
        metrics[v] = aggregate(vertical_scores[v])
    
    return metrics


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3-vl-8b-instruct")
    parser.add_argument("--verticals", nargs="*", default=None)
    parser.add_argument("--sample-per-site", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=1)
    args = parser.parse_args()
    
    run_name = f"swde_cog_{args.model}"
    checkpoint_dir = OUTPUT_DIR / "experiments" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"=== SWDE Cog Evaluation ===")
    logger.info(f"Model: {args.model}, Max rounds: {args.max_rounds}")
    logger.info(f"Output: {checkpoint_dir}")
    
    # Load dataset
    groups = load_swde_dataset(
        SWDE_DATA_ROOT,
        sample_per_site=args.sample_per_site,
        verticals=args.verticals,
    )
    save_swde_manifest(groups, str(checkpoint_dir / "manifest.json"))
    
    # Run Cog+
    results = await run_swde_cog(
        groups, args.model, checkpoint_dir, args.concurrency, args.max_rounds
    )
    
    # Evaluate
    metrics = evaluate_results(groups, results)
    
    # Save metrics
    metrics_path = checkpoint_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    
    # Print results
    logger.info(f"\n{'='*60}")
    logger.info(f"=== SWDE Cog Results ({args.model}) ===")
    logger.info(f"{'='*60}")
    for key, val in metrics.items():
        logger.info(
            f"  {key:<15} P={val['precision']:.2f}  R={val['recall']:.2f}  "
            f"F1={val['f1']:.2f}  (n={val['count']})"
        )
    logger.info(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    asyncio.run(main())
