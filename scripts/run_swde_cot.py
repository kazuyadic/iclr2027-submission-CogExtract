"""SWDE CoT Runner — runs CoT baseline on SWDE offline HTML.

Since SWDE HTML files are local snapshots, we can:
1. Read HTML directly from disk
2. Simplify and feed to LLM for XPath generation
3. Execute XPath using lxml (no browser needed!)

This is much simpler and faster than loading via Playwright.

Usage:
    python3 scripts/run_swde_cot.py --model qwen3.7-plus
    python3 scripts/run_swde_cot.py --model qwen3.7-plus --verticals auto movie
"""

import asyncio
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

from lxml import etree, html as lxml_html

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.swde_loader import load_swde_dataset, save_swde_manifest, VERTICAL_ATTRS
from utils.html_processor import HTMLProcessor
from utils.llm_client import LLMClient
from utils.evaluator import Evaluator
from baselines.prompts import TOP_DOWN_PROMPT
from configs.config import VGSConfig

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

MAX_HTML_CHARS = 60000


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


async def run_swde_cot(
    groups: list[dict],
    model: str,
    checkpoint_dir: Path,
    concurrency: int = 3,
) -> list[dict]:
    """Run CoT on SWDE groups.
    
    For each group:
    1. Read first HTML → simplify → LLM → XPath
    2. Execute XPath on all HTML files in group
    3. Evaluate against GT
    """
    config = VGSConfig()
    config.model_name = model
    
    llm = LLMClient(
        api_key=config.api_key,
        api_base=config.api_base,
        model=model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        enable_monitor=False,
    )
    
    checkpoint_path = checkpoint_dir / "results_checkpoint.json"
    traces_dir = checkpoint_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    
    # Resume
    all_results = []
    done_sids = set()
    if checkpoint_path.exists():
        all_results = json.loads(checkpoint_path.read_text())
        done_sids = {r.get("sample_id", "") for r in all_results}
        logger.info(f"Resuming: {len(all_results)} groups done")
    
    start_time = time.time()
    
    # Process with semaphore for concurrency
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    
    async def process_one_group(gi: int, group: dict) -> None:
        sid = group["sample_id"]
        if sid in done_sids:
            return
        
        async with sem:
            html_paths = group["html_paths"]
            gt = group["gt"]
            query = group["query"]
            
            try:
                # Phase 1: Read first HTML, simplify, call LLM
                first_html_path = html_paths[0]
                with open(first_html_path, "r", encoding="utf-8", errors="replace") as f:
                    raw_html = f.read()
                
                simplified = HTMLProcessor.simplify(raw_html)
                if len(simplified) > MAX_HTML_CHARS:
                    simplified = simplified[:MAX_HTML_CHARS]
                
                prompt = TOP_DOWN_PROMPT.format(instruction=query, html=simplified)
                result = await asyncio.to_thread(
                    llm.text_query, prompt, "CoT·SWDE"
                )
                
                xpaths_raw = result.get("xpath", {})
                
                # Normalize xpaths
                xpaths = {}
                for attr, xpath_val in xpaths_raw.items():
                    if isinstance(xpath_val, list):
                        xpaths[attr] = xpath_val[0] if xpath_val else ""
                    else:
                        xpaths[attr] = xpath_val or ""
                
                # Save trace
                trace = {
                    "sample_id": sid,
                    "query": query,
                    "html_path": first_html_path,
                    "simplified_html_len": len(simplified),
                    "prompt_len": len(prompt),
                    "llm_response": result,
                    "xpaths": xpaths,
                }
                trace_file = traces_dir / f"{sid}.json"
                trace_file.write_text(json.dumps(trace, ensure_ascii=False, indent=2))
                
                # Phase 2: Execute XPaths on all HTML files
                page_results = []
                for html_path in html_paths:
                    page_id = Path(html_path).stem
                    url = group["urls"][html_paths.index(html_path)]
                    
                    try:
                        with open(html_path, "r", encoding="utf-8", errors="replace") as f:
                            page_html = f.read()
                        
                        values = {}
                        for attr, xpath in xpaths.items():
                            if xpath:
                                values[attr] = execute_xpath_on_html(page_html, xpath)
                            else:
                                values[attr] = []
                        
                        page_results.append({
                            "sample_id": f"{sid}_{page_id}",
                            "url": url,
                            "query": query,
                            "attributes": list(xpaths.keys()),
                            "xpaths": xpaths,
                            "values": values,
                        })
                    except Exception as e:
                        page_results.append({
                            "sample_id": f"{sid}_{page_id}",
                            "url": url,
                            "query": query,
                            "error": str(e),
                        })
                
                async with lock:
                    all_results.extend(page_results)
                    done_sids.add(sid)
            
            except Exception as e:
                logger.error(f"  Error on {sid}: {e}")
                async with lock:
                    all_results.append({
                        "sample_id": sid,
                        "url": "",
                        "query": query,
                        "error": str(e),
                    })
                    done_sids.add(sid)
            
            # Checkpoint every 20 groups
            done_count = len(done_sids)
            if done_count % 20 == 0 and done_count > 0:
                checkpoint_path.write_text(
                    json.dumps(all_results, ensure_ascii=False, indent=2)
                )
                elapsed = time.time() - start_time
                rate = done_count / elapsed if elapsed > 0 else 1
                remaining = (len(groups) - done_count) / rate
                logger.info(
                    f"  Progress: {done_count}/{len(groups)} groups, "
                    f"ETA: {remaining/60:.0f} min"
                )
    
    # Run all groups
    tasks = [process_one_group(i, g) for i, g in enumerate(groups)]
    await asyncio.gather(*tasks)
    
    # Final save
    checkpoint_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2)
    )
    
    logger.info(f"Done: {len(all_results)} page results from {len(done_sids)} groups")
    return all_results


def evaluate_results(groups: list[dict], results: list[dict]) -> dict:
    """Evaluate CoT results against SWDE GT."""
    # Build GT lookup: (sample_id_prefix, page_id) → gt
    gt_lookup = {}
    for g in groups:
        sid = g["sample_id"]
        for page_id, page_gt in g["gt"].items():
            key = f"{sid}_{page_id}"
            gt_lookup[key] = page_gt
    
    # Evaluate
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
        
        # Extract vertical from sample_id: swde_{vertical}_{site}_{attr}_{page}
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
    parser.add_argument("--model", type=str, default="qwen3.7-plus")
    parser.add_argument("--verticals", nargs="*", default=None)
    parser.add_argument("--sample-per-site", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()
    
    run_name = f"swde_cot_{args.model}"
    checkpoint_dir = OUTPUT_DIR / "experiments" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"=== SWDE CoT Evaluation ===")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {checkpoint_dir}")
    
    # Load dataset
    groups = load_swde_dataset(
        SWDE_DATA_ROOT,
        sample_per_site=args.sample_per_site,
        verticals=args.verticals,
    )
    save_swde_manifest(groups, str(checkpoint_dir / "manifest.json"))
    
    # Run CoT
    results = await run_swde_cot(
        groups, args.model, checkpoint_dir, args.concurrency
    )
    
    # Evaluate
    metrics = evaluate_results(groups, results)
    
    # Save metrics
    metrics_path = checkpoint_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    
    # Print results
    logger.info(f"\n{'='*60}")
    logger.info(f"=== SWDE CoT Results ({args.model}) ===")
    logger.info(f"{'='*60}")
    for key, val in metrics.items():
        logger.info(
            f"  {key:<15} P={val['precision']:.2f}  R={val['recall']:.2f}  "
            f"F1={val['f1']:.2f}  (n={val['count']})"
        )
    logger.info(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    asyncio.run(main())
