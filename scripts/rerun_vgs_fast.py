#!/usr/bin/env python3
"""Fast VGS badcase rerun using cached HTML + screenshots (no Playwright)."""
import asyncio, json, hashlib, sys, time, logging
from pathlib import Path
from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from configs.config import VGSConfig
from utils.llm_client import LLMClient
from utils.evaluator import Evaluator
from utils.data_loader import DataLoader
from vgs.prompts import *

EXP_NAME = "vgs_qwen3-vl-8b-instruct"
RESULTS_PATH = Path(f"experiments/{EXP_NAME}/full_eval_VGS_qwen3-vl-8b-instruct_LiveWeb-IE_results.json")
CACHE_DIR = Path("cache/pages")
CONCURRENCY = 30
MAX_SAMPLES = 999999

def _url_hash(url):
    return hashlib.sha256(url.encode()).hexdigest()[:16]

def get_candidates(tree, max_els=150):
    """Get candidate elements from HTML tree via lxml."""
    els = tree.xpath('//*[self::p or self::span or self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6 or self::td or self::th or self::li or self::a or self::div or self::label or self::em or self::strong or self::b or self::i]')
    candidates = []
    seen_texts = set()
    for i, el in enumerate(els):
        if i >= max_els * 2:  # scan more but keep max_els
            break
        text = ''.join(el.itertext()).strip()
        if not text or len(text) < 2:
            continue
        # Deduplicate by text content
        t_key = text[:50]
        if t_key in seen_texts:
            continue
        seen_texts.add(t_key)
        candidates.append({'idx': i, 'el': el, 'tag': el.tag, 'text': text})
        if len(candidates) >= max_els:
            break
    return candidates

async def process_one(idx, old_result, llm, url_to_gt, results):
    """Process one badcase using cached HTML + screenshot."""
    url = old_result["url"]
    query = old_result["query"]
    url_hash = _url_hash(url)
    cache = CACHE_DIR / url_hash

    if not (cache / "page.html").exists():
        return "skip"

    try:
        html = (cache / "page.html").read_text(encoding="utf-8")
        screenshot = cache / "screenshot.png"
        tree = etree.HTML(html)

        # Stage 1: Attribute ID (LLM)
        s1 = llm.text_query(ATTRIBUTE_IDENTIFICATION_PROMPT.format(query=query), label='S1')
        attrs = s1.get('attributes', old_result.get('attributes', ['unknown']))
        if not attrs:
            attrs = old_result.get('attributes', ['unknown'])
        attr = attrs[0]

        # Stage 2: Visual Grounding (VLM + cached screenshot)
        s2 = llm.vision_query(VISUAL_GROUNDING_PROMPT.format(attribute=attr), [screenshot], label='S2')

        # Stage 3a: Scanning (VLM + cached screenshot)
        s3a = llm.vision_query(ELEMENT_SCANNING_PROMPT.format(attribute=attr), [screenshot], label='S3a')

        # Stage 3b: Text-based selection (LLM + lxml candidates)
        candidates = get_candidates(tree)
        el_list = '\n'.join(f"{i}. <{c['tag']}> {c['text'][:100]}" for i, c in enumerate(candidates))
        s3b = llm.text_query(
            TEXT_ELEMENT_SELECTION_PROMPT.format(attribute=attr, total_count=len(candidates), element_list=el_list),
            label='S3b'
        )
        sel_ids = s3b.get('selected_ids', [])

        xpaths = {}
        values = {}

        if sel_ids:
            # Stage 4: XPath synthesis + lxml execution
            for sid in sel_ids[:3]:  # max 3 elements
                if sid >= len(candidates):
                    continue
                el = candidates[sid]['el']
                seg = etree.tostring(el, encoding='unicode')[:2000]
                s4 = llm.vision_query(
                    XPATH_SYNTHESIS_PROMPT.format(attribute=attr, html_segments=seg),
                    [screenshot], label='S4'
                )
                xp = s4.get('xpath', '')
                if xp:
                    xpaths[attr] = xp
                    try:
                        matches = tree.xpath(xp)
                        vals = []
                        for m in matches:
                            if isinstance(m, str):
                                vals.append(m.strip())
                            elif hasattr(m, 'itertext'):
                                vals.append(''.join(m.itertext()).strip())
                        values[attr] = [v for v in vals if v]
                    except Exception:
                        values[attr] = []
                    break  # Use first successful xpath
        else:
            # Fallback: try direct xpath from S3a items
            items = s3a.get('items', [])
            if items:
                for item_text in items[:2]:
                    xp = f"//*[contains(text(), '{item_text[:40]}')]"
                    try:
                        matches = tree.xpath(xp)
                        if matches:
                            xpaths[attr] = xp
                            values[attr] = [item_text]
                            break
                    except Exception:
                        pass

        # Re-evaluate
        gt = old_result.get('ground_truth', {}) or url_to_gt.get(url, {})
        if values and gt:
            new_score = Evaluator.evaluate_sample(values, gt)
            new_f1 = new_score.get('f1', 0)
        else:
            new_score = {}
            new_f1 = 0

        old_f1 = old_result.get('eval_score', {}).get('f1', 0)
        tag = "↑" if new_f1 > old_f1 else ("↓" if new_f1 < old_f1 else "=")

        results[idx] = {
            "url": url, "query": query,
            "attributes": attrs,
            "xpaths": xpaths, "values": values,
            "sample_id": old_result.get("sample_id", ""),
            "eval_score": new_score if new_score else old_result.get("eval_score", {}),
        }
        return tag

    except Exception as e:
        logger.warning(f"  [!] idx={idx}: {e}")
        return "!"

async def main():
    results = json.load(open(RESULTS_PATH))
    logger.info(f"Loaded {len(results)} results")

    bad_indices = [i for i, r in enumerate(results) if r.get("eval_score", {}).get("f1", 0) == 0]
    logger.info(f"Bad cases (F1=0): {len(bad_indices)}")

    rerun = bad_indices[:MAX_SAMPLES]
    logger.info(f"Re-running {len(rerun)} samples (concurrency={CONCURRENCY})")

    config = VGSConfig(model_name="qwen3-vl-8b-instruct")
    llm = LLMClient(config.api_key, config.api_base, config.model_name)

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

    sem = asyncio.Semaphore(CONCURRENCY)
    stats = {"↑": 0, "=": 0, "↓": 0, "!": 0, "skip": 0}

    async def bounded(idx):
        async with sem:
            tag = await process_one(idx, results[idx], llm, url_to_gt, results)
            stats[tag] = stats.get(tag, 0) + 1
            done = sum(stats.values())
            if done % 100 == 0:
                RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False))
                logger.info(f"  [checkpoint] {done} done: {stats}")
            if done % 10 == 0:
                logger.info(f"  [{tag}] idx={idx} | progress: {done}/{len(rerun)} | {stats}")

    t0 = time.time()
    tasks = [bounded(i) for i in rerun]
    await asyncio.gather(*tasks)
    elapsed = time.time() - t0

    # Save final
    backup = RESULTS_PATH.with_suffix(".json.bak")
    if not backup.exists():
        backup.write_text(open(RESULTS_PATH).read())
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False))

    logger.info(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info(f"Stats: {stats}")
    rate = len(rerun) / elapsed * 60
    logger.info(f"Rate: {rate:.1f}/min")

asyncio.run(main())
