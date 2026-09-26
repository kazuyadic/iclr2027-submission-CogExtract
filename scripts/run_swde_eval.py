"""SWDE Evaluation Runner for CoT and VGS methods.

Adapts the LiveWeb-IE evaluation pipeline to run on SWDE offline HTML data.
Matches the paper's protocol (Appendix B.2):
- 100 pages per website
- Excluded sites: CollegeToolkit, FanHouse  
- Final: ~312 cases

Usage:
    python3 scripts/run_swde_eval.py --method cot --model qwen3.7-plus
    python3 scripts/run_swde_eval.py --method vgs --model qwen3-vl-32b-instruct
    python3 scripts/run_swde_eval.py --method cot --model qwen3.7-plus --verticals auto movie
"""

import asyncio
import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.swde_loader import load_swde_dataset, save_swde_manifest
from utils.evaluator import Evaluator
from utils.html_processor import HTMLProcessor

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


async def run_cot_on_swde(
    groups: list,
    model: str,
    output_dir: Path,
    checkpoint_dir: Path,
    concurrency: int = 5,
) -> list:
    """Run CoT baseline on SWDE groups.
    
    For each group (website × attribute):
    1. Phase 1: CoT on first URL → generate XPath
    2. Phase 2: Apply XPath to remaining URLs
    3. Evaluate against GT
    """
    from baselines.cot_pipeline import CoTPipeline
    from utils.browser import BrowserManager
    from utils.llm_client import LLMClient
    
    llm = LLMClient(model=model)
    browser = BrowserManager(headless=True)
    html_processor = HTMLProcessor()
    
    pipeline = CoTPipeline(llm, browser, html_processor)
    
    all_results = []
    checkpoint_path = checkpoint_dir / "results_checkpoint.json"
    
    # Load checkpoint if exists
    processed_ids = set()
    if checkpoint_path.exists():
        existing = json.loads(checkpoint_path.read_text())
        for r in existing:
            processed_ids.add(r.get("sample_id", ""))
        all_results = existing
        logger.info(f"Resuming from checkpoint: {len(all_results)} groups done")
    
    start_time = time.time()
    
    for gi, group in enumerate(groups):
        sid = group["sample_id"]
        if sid in processed_ids:
            continue
        
        logger.info(f"[{gi+1}/{len(groups)}] {sid}: {group['query']}")
        
        html_paths = group["html_paths"]
        gt = group["gt"]
        
        try:
            # Phase 1: CoT on first HTML
            first_html = html_paths[0]
            first_page_id = Path(first_html).stem
            
            # Load and simplify HTML
            with open(first_html, "r", encoding="utf-8", errors="replace") as f:
                raw_html = f.read()
            
            simplified = html_processor.simplify_html(raw_html)
            simplified_truncated = simplified[:60000]
            
            # Call LLM to generate XPath
            response = await llm.generate(
                prompt=pipeline._build_prompt(simplified_truncated, group["query"]),
                temperature=0.0,
            )
            xpaths = pipeline._parse_response(response)
            
            # Phase 2: Apply XPaths to all pages
            page_results = []
            for html_path in html_paths:
                page_id = Path(html_path).stem
                url = group["urls"][html_paths.index(html_path)]
                
                try:
                    with open(html_path, "r", encoding="utf-8", errors="replace") as f:
                        page_html = f.read()
                    
                    page_obj = await browser.load_page_from_html(page_html, url=url)
                    values = {}
                    for attr, xpath in xpaths.items():
                        extracted = await browser.execute_xpath(page_obj, xpath)
                        values[attr] = extracted
                    
                    page_results.append({
                        "sample_id": sid,
                        "page_id": page_id,
                        "url": url,
                        "query": group["query"],
                        "xpaths": str(xpaths),
                        "values": values,
                    })
                except Exception as e:
                    page_results.append({
                        "sample_id": sid,
                        "page_id": page_id,
                        "url": url,
                        "error": str(e),
                    })
            
            all_results.append({
                "sample_id": sid,
                "query": group["query"],
                "xpaths": str(xpaths),
                "pages": page_results,
                "num_pages": len(page_results),
            })
            
        except Exception as e:
            logger.error(f"  Error on {sid}: {e}")
            all_results.append({
                "sample_id": sid,
                "query": group["query"],
                "error": str(e),
            })
        
        # Save checkpoint every 10 groups
        if (gi + 1) % 10 == 0:
            checkpoint_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
            elapsed = time.time() - start_time
            remaining = elapsed / (gi + 1) * (len(groups) - gi - 1)
            logger.info(f"  Checkpoint saved. ETA: {remaining/60:.0f} min")
    
    # Final save
    checkpoint_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    return all_results


async def run_vgs_on_swde(
    groups: list,
    model: str,
    output_dir: Path,
    checkpoint_dir: Path,
    screenshot_dir: Path,
    concurrency: int = 3,
) -> list:
    """Run VGS on SWDE groups.
    
    Requires rendering HTML to screenshots first.
    """
    from vgs.pipeline import VGSipeline
    from utils.browser import BrowserManager
    from utils.llm_client import LLMClient
    
    llm = LLMClient(model=model)
    browser = BrowserManager(headless=True)
    
    pipeline = VGSipeline(llm, browser)
    
    all_results = []
    checkpoint_path = checkpoint_dir / "results_checkpoint.json"
    
    # Load checkpoint
    processed_ids = set()
    if checkpoint_path.exists():
        existing = json.loads(checkpoint_path.read_text())
        for r in existing:
            processed_ids.add(r.get("sample_id", ""))
        all_results = existing
        logger.info(f"Resuming from checkpoint: {len(all_results)} groups done")
    
    start_time = time.time()
    
    for gi, group in enumerate(groups):
        sid = group["sample_id"]
        if sid in processed_ids:
            continue
        
        logger.info(f"[{gi+1}/{len(groups)}] {sid}: {group['query']}")
        
        html_paths = group["html_paths"]
        
        # Render screenshots
        screenshot_paths = []
        for html_path in html_paths:
            page_id = Path(html_path).stem
            ss_path = screenshot_dir / sid / f"{page_id}.png"
            
            if not ss_path.exists():
                try:
                    ss_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(html_path, "r", encoding="utf-8", errors="replace") as f:
                        html_content = f.read()
                    
                    page = await browser.load_page_from_html(
                        html_content, 
                        url=group["urls"][html_paths.index(html_path)]
                    )
                    await page.screenshot(path=str(ss_path), full_page=True)
                    screenshot_paths.append(str(ss_path))
                except Exception as e:
                    logger.warning(f"  Screenshot failed for {page_id}: {e}")
            else:
                screenshot_paths.append(str(ss_path))
        
        if not screenshot_paths:
            all_results.append({
                "sample_id": sid,
                "query": group["query"],
                "error": "No screenshots rendered",
            })
            continue
        
        # Run VGS pipeline
        try:
            result = await pipeline.run_group(
                screenshot_paths=screenshot_paths,
                query=group["query"],
                max_rounds=1,
            )
            all_results.append({
                "sample_id": sid,
                "query": group["query"],
                "result": result,
            })
        except Exception as e:
            logger.error(f"  VGS error on {sid}: {e}")
            all_results.append({
                "sample_id": sid,
                "query": group["query"],
                "error": str(e),
            })
        
        # Checkpoint
        if (gi + 1) % 10 == 0:
            checkpoint_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
            elapsed = time.time() - start_time
            remaining = elapsed / (gi + 1) * (len(groups) - gi - 1)
            logger.info(f"  Checkpoint saved. ETA: {remaining/60:.0f} min")
    
    checkpoint_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    return all_results


def evaluate_swde_results(
    groups: list,
    results: list,
    method: str,
) -> dict:
    """Evaluate SWDE results against GT.
    
    Returns per-vertical and overall metrics.
    """
    gt_lookup = {}
    for g in groups:
        for page_id, page_gt in g["gt"].items():
            gt_lookup[(g["sample_id"], page_id)] = page_gt
    
    vertical_scores = {}
    overall_scores = []
    
    for r in results:
        sid = r.get("sample_id", "")
        if "error" in r and not r.get("pages"):
            continue
        
        vertical = sid.split("_")[1] if "_" in sid else "unknown"
        
        if method == "cot":
            for page in r.get("pages", []):
                page_id = page.get("page_id", "")
                gt = gt_lookup.get((sid, page_id))
                if gt is None:
                    continue
                
                pred = {} if "error" in page else page.get("values", {})
                score = Evaluator.evaluate_sample(pred, gt)
                overall_scores.append(score)
                vertical_scores.setdefault(vertical, []).append(score)
        
        elif method == "vgs":
            result = r.get("result", {})
            pages = result.get("pages", [])
            for page in pages:
                page_id = page.get("page_id", "")
                gt = gt_lookup.get((sid, page_id))
                if gt is None:
                    continue
                pred = page.get("values", {})
                score = Evaluator.evaluate_sample(pred, gt)
                overall_scores.append(score)
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
    for v, scores in sorted(vertical_scores.items()):
        metrics[v] = aggregate(scores)
    
    return metrics


async def main():
    parser = argparse.ArgumentParser(description="Run SWDE evaluation")
    parser.add_argument("--method", choices=["cot", "vgs"], required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--verticals", nargs="*", default=None)
    parser.add_argument("--sample-per-site", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()
    
    # Setup output
    run_name = f"swde_{args.method}_{args.model}"
    checkpoint_dir = OUTPUT_DIR / "experiments" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    screenshot_dir = OUTPUT_DIR / "screenshots" / run_name
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Method: {args.method}, Model: {args.model}")
    logger.info(f"Output: {checkpoint_dir}")
    
    # Load dataset
    groups = load_swde_dataset(
        SWDE_DATA_ROOT,
        sample_per_site=args.sample_per_site,
        verticals=args.verticals,
    )
    
    # Save manifest
    save_swde_manifest(groups, str(checkpoint_dir / "manifest.json"))
    
    # Run evaluation
    if args.method == "cot":
        results = await run_cot_on_swde(
            groups, args.model, OUTPUT_DIR, checkpoint_dir, args.concurrency
        )
    elif args.method == "vgs":
        results = await run_vgs_on_swde(
            groups, args.model, OUTPUT_DIR, checkpoint_dir, screenshot_dir, args.concurrency
        )
    
    # Evaluate
    metrics = evaluate_swde_results(groups, results, args.method)
    
    # Save metrics
    metrics_path = checkpoint_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    
    logger.info(f"\n=== Results ({args.method}, {args.model}) ===")
    for key, val in metrics.items():
        logger.info(f"  {key}: P={val['precision']:.2f}  R={val['recall']:.2f}  F1={val['f1']:.2f}  (n={val['count']})")
    
    logger.info(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    asyncio.run(main())
