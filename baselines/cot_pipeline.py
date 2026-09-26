"""CoT Baseline — single-pass XPath generation from simplified HTML.

Paper reference: Chain-of-Thought (Wei et al., 2022) adapted for WIE.
Generates XPath in a single LLM call given the simplified HTML + query.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from lxml import etree

from configs.config import VGSConfig
from utils.llm_client import LLMClient
from utils.cached_browser import CachedBrowserManager
from utils.html_processor import HTMLProcessor
from baselines.prompts import TOP_DOWN_PROMPT

logger = logging.getLogger(__name__)

# Max HTML chars to feed LLM (prevent context overflow)
MAX_HTML_CHARS = 60000


class CoTPipeline:
    """CoT baseline: single-pass XPath generation from simplified HTML."""

    def __init__(self, config: VGSConfig, enable_monitor: bool = False):
        self.config = config
        self._enable_monitor = enable_monitor
        self.llm = LLMClient(
            api_key=config.api_key,
            api_base=config.api_base,
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            enable_monitor=enable_monitor,
        )
        self.browser = CachedBrowserManager(
            cache_dir=config.cache_dir,
            viewport_width=config.viewport_width,
            viewport_height=config.viewport_height,
            headless=config.headless,
            timeout_ms=config.page_load_timeout_ms,
        )

    async def _run_inner(self, url: str, query: str,
                         predefined_attributes: list[str] | None = None) -> dict:
        """Run CoT on a single URL + query."""
        # Load page and get simplified HTML
        page = await self.browser.load_page(url)
        full_html = await self.browser.get_full_html(page)
        simplified_html = HTMLProcessor.simplify(full_html)

        # Truncate if too long
        if len(simplified_html) > MAX_HTML_CHARS:
            simplified_html = simplified_html[:MAX_HTML_CHARS]

        # Single-pass LLM call
        prompt = TOP_DOWN_PROMPT.format(instruction=query, html=simplified_html)
        result = self.llm.text_query(prompt, label="CoT·TopDown")

        # Parse response
        xpaths_dict = result.get("xpath", {})
        values_dict = result.get("value", {})

        # Normalize: ensure xpaths are single strings (take first if list)
        normalized_xpaths = {}
        for attr, xpath_val in xpaths_dict.items():
            if isinstance(xpath_val, list):
                # Join multiple XPaths with | (union) or take first
                normalized_xpaths[attr] = xpath_val[0] if xpath_val else ""
            else:
                normalized_xpaths[attr] = xpath_val or ""

        # Execute XPaths to get actual values
        extracted_values = {}
        for attr, xpath in normalized_xpaths.items():
            if not xpath:
                extracted_values[attr] = []
                continue
            try:
                vals = await self.browser.execute_xpath(page, xpath)
                extracted_values[attr] = vals
            except Exception as e:
                logger.warning("[CoT] XPath execution error for %s: %s", attr, e)
                extracted_values[attr] = []

        await page.context.close()

        return {
            "url": url,
            "query": query,
            "attributes": list(normalized_xpaths.keys()),
            "xpaths": normalized_xpaths,
            "values": extracted_values,
        }

    async def run_grouped_batch(self, groups: list[dict], concurrency: int = 3) -> list[dict]:
        """Run CoT with Phase 1 → Phase 2 grouped evaluation protocol.

        Phase 1: Generate XPaths on first URL of each group.
        Phase 2: Reuse XPaths on remaining URLs.
        """
        from tqdm import tqdm

        checkpoint_path = self.config.output_dir / "results_checkpoint.json"
        total_urls = sum(len(g["urls"]) for g in groups)

        # Shared state
        lock = asyncio.Lock()
        all_results: list[dict] = []
        pipeline_ok = 0
        pipeline_err = 0
        reuse_ok = 0
        reuse_err = 0

        # Resume from checkpoint
        done_urls: set[str] = set()
        if checkpoint_path.exists():
            with open(checkpoint_path, encoding="utf-8") as fh:
                prior = json.load(fh)
            done_urls = {r.get("url", "") for r in prior}
            all_results.extend(prior)
            logger.info("Resuming from checkpoint: %d URLs done", len(done_urls))

        # Progress bar
        pbar = tqdm(total=total_urls, desc="CoT·Pipeline",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")
        pbar.update(len(done_urls))

        # Workers
        workers: list[CoTPipeline] = []
        for _ in range(concurrency):
            worker = CoTPipeline(self.config, enable_monitor=self._enable_monitor)
            await worker.browser.start()
            workers.append(worker)

        pipeline_sem = asyncio.Semaphore(concurrency)
        worker_available = list(range(concurrency))
        pool_lock = asyncio.Lock()

        # Phase 2 browser pool
        reuse_concurrency = min(2, concurrency)
        reuse_browsers: list[CachedBrowserManager] = []
        for _ in range(reuse_concurrency):
            browser = CachedBrowserManager(
                cache_dir=self.config.cache_dir,
                headless=self.config.headless,
                viewport_width=self.config.viewport_width,
                viewport_height=self.config.viewport_height,
                timeout_ms=self.config.page_load_timeout_ms,
            )
            await browser.start()
            reuse_browsers.append(browser)
        reuse_sem = asyncio.Semaphore(reuse_concurrency)
        browser_pool = list(range(reuse_concurrency))
        browser_lock = asyncio.Lock()

        async def reuse_single_url(group_idx: int, url: str, query: str,
                                   xpaths: dict, attrs: list):
            nonlocal reuse_ok, reuse_err
            sample_id = groups[group_idx]["sample_id"]
            uid = f"{sample_id}_{url.split('/')[-1][:20]}"

            if url in done_urls:
                return

            if not xpaths:
                async with lock:
                    all_results.append({
                        "sample_id": uid, "url": url, "query": query,
                        "error": "No XPath from Phase 1",
                    })
                    reuse_err += 1
                    pbar.update(1)
                    pbar.set_postfix_str(
                        f"P1✓{pipeline_ok}✗{pipeline_err} P2✓{reuse_ok}✗{reuse_err}")
                return

            await asyncio.sleep(1.0)
            async with reuse_sem:
                async with browser_lock:
                    bidx = browser_pool.pop()
                browser = reuse_browsers[bidx]
                try:
                    page = await browser.load_page(url)
                    values: dict[str, list] = {}
                    for attr, xpath in xpaths.items():
                        try:
                            extracted = await browser.execute_xpath(page, xpath)
                        except Exception:
                            extracted = []
                        values[attr] = extracted
                    await page.context.close()

                    async with lock:
                        all_results.append({
                            "url": url, "query": query,
                            "attributes": attrs, "xpaths": xpaths,
                            "values": values, "sample_id": uid,
                        })
                        reuse_ok += 1
                except Exception as exc:
                    async with lock:
                        all_results.append({
                            "sample_id": uid, "url": url, "query": query,
                            "error": str(exc),
                        })
                        reuse_err += 1
                finally:
                    async with browser_lock:
                        browser_pool.append(bidx)
                    async with lock:
                        pbar.update(1)
                        pbar.set_postfix_str(
                            f"P1✓{pipeline_ok}✗{pipeline_err} P2✓{reuse_ok}✗{reuse_err}")
                        if len(all_results) % 50 == 0:
                            with open(checkpoint_path, "w", encoding="utf-8") as fh:
                                json.dump(all_results, fh, indent=2, ensure_ascii=False)

        async def process_group(group_idx: int, group: dict):
            nonlocal pipeline_ok, pipeline_err
            sample_id = group["sample_id"]
            first_url = group["urls"][0]
            first_uid = f"{sample_id}_{first_url.split('/')[-1][:20]}"

            if first_url in done_urls:
                # Skip already done
                async with lock:
                    pbar.update(1)
                return

            # Phase 1: CoT on first URL
            async with pipeline_sem:
                async with pool_lock:
                    widx = worker_available.pop()
                worker = workers[widx]
                try:
                    result = await worker._run_inner(first_url, group["query"])
                    result["sample_id"] = first_uid
                    async with lock:
                        all_results.append(result)
                        pipeline_ok += 1
                        pbar.update(1)
                        pbar.set_postfix_str(
                            f"P1✓{pipeline_ok}✗{pipeline_err} P2✓{reuse_ok}✗{reuse_err}")
                except Exception as exc:
                    result = {
                        "sample_id": first_uid, "url": first_url,
                        "query": group["query"], "error": str(exc),
                    }
                    async with lock:
                        all_results.append(result)
                        pipeline_err += 1
                        pbar.update(1)
                        pbar.set_postfix_str(
                            f"P1✓{pipeline_ok}✗{pipeline_err} P2✓{reuse_ok}✗{reuse_err}")
                finally:
                    async with pool_lock:
                        worker_available.append(widx)

            # Phase 2: Reuse on remaining URLs
            xpaths = result.get("xpaths", {}) if "error" not in result else {}
            attrs = result.get("attributes", [])
            reuse_urls = [u for u in group["urls"][1:] if u not in done_urls]
            if reuse_urls:
                reuse_tasks = [
                    reuse_single_url(group_idx, url, group["query"], xpaths, attrs)
                    for url in reuse_urls
                ]
                await asyncio.gather(*reuse_tasks)

        try:
            all_tasks = [process_group(i, g) for i, g in enumerate(groups)]
            await asyncio.gather(*all_tasks)
        finally:
            pbar.close()
            for worker in workers:
                await worker.browser.stop()
            for browser in reuse_browsers:
                await browser.stop()

        # Final save
        with open(checkpoint_path, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2, ensure_ascii=False)

        logger.info("Done: %d results, P1(✓%d ✗%d) P2(✓%d ✗%d)",
                    len(all_results), pipeline_ok, pipeline_err, reuse_ok, reuse_err)
        return all_results
