"""Reflexion Baseline — iterative XPath refinement based on execution feedback.

Paper reference: Reflexion (Shinn et al., 2023) adapted for WIE.
Generates XPath, executes it, and if results are empty/wrong, reflects and retries.
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
from baselines.prompts import TOP_DOWN_PROMPT, SELF_REFLECTION_PROMPT

logger = logging.getLogger(__name__)

MAX_HTML_CHARS = 60000
MAX_ROUNDS = 3  # Maximum reflection iterations


class ReflexionPipeline:
    """Reflexion baseline: iterative XPath refinement with execution feedback."""

    def __init__(self, config: VGSConfig, max_rounds: int = MAX_ROUNDS,
                 enable_monitor: bool = False):
        self.config = config
        self.max_rounds = max_rounds
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

    def _execute_xpath_on_html(self, html: str, xpath: str) -> list[str]:
        """Execute XPath on static HTML using lxml."""
        try:
            tree = etree.HTML(html)
            if tree is None:
                return []
            nodes = tree.xpath(xpath)
        except Exception:
            return []

        results = []
        for node in nodes[:50]:
            if isinstance(node, str):
                text = node.strip()
            elif hasattr(node, "text_content"):
                if node.tag == "img":
                    text = node.get("src", node.get("alt", ""))
                elif node.tag == "a":
                    text = node.get("href", node.text_content().strip())
                else:
                    text = node.text_content().strip()
            else:
                text = str(node).strip()
            if text:
                results.append(text[:200])
        return results

    async def _run_inner(self, url: str, query: str,
                         predefined_attributes: list[str] | None = None) -> dict:
        """Run Reflexion on a single URL + query."""
        # Load page and get simplified HTML
        page = await self.browser.load_page(url)
        full_html = await self.browser.get_full_html(page)
        simplified_html = HTMLProcessor.simplify(full_html)

        if len(simplified_html) > MAX_HTML_CHARS:
            simplified_html = simplified_html[:MAX_HTML_CHARS]

        # Round 1: Initial top-down generation
        prompt = TOP_DOWN_PROMPT.format(instruction=query, html=simplified_html)
        result = self.llm.text_query(prompt, label="Reflexion·TopDown")

        xpaths_dict = result.get("xpath", {})
        values_dict = result.get("value", {})

        # Build history for reflection
        history_entries = []
        history_entries.append({
            "round": 1,
            "thought": result.get("thought", ""),
            "xpath": xpaths_dict,
            "extracted_values": {},
        })

        # Execute and check
        current_xpaths = {}
        for attr, xpath_val in xpaths_dict.items():
            if isinstance(xpath_val, list):
                current_xpaths[attr] = xpath_val[0] if xpath_val else ""
            else:
                current_xpaths[attr] = xpath_val or ""

        # Execute on the simplified HTML to check results
        execution_results = {}
        has_failures = False
        for attr, xpath in current_xpaths.items():
            if not xpath:
                execution_results[attr] = []
                has_failures = True
                continue
            vals = self._execute_xpath_on_html(simplified_html, xpath)
            execution_results[attr] = vals
            if not vals:
                has_failures = True

        history_entries[0]["extracted_values"] = execution_results

        # Iterative reflection rounds
        for round_num in range(2, self.max_rounds + 1):
            if not has_failures:
                break

            # Format history
            history_str = ""
            for entry in history_entries:
                history_str += f"\n--- Round {entry['round']} ---\n"
                history_str += f"Thought: {entry['thought']}\n"
                history_str += f"XPaths: {json.dumps(entry['xpath'], ensure_ascii=False)}\n"
                history_str += f"Extracted: {json.dumps(entry['extracted_values'], ensure_ascii=False)}\n"

            # Self-reflection
            reflection_prompt = SELF_REFLECTION_PROMPT.format(
                instruction=query,
                history=history_str,
                html=simplified_html,
            )
            reflection_result = self.llm.text_query(
                reflection_prompt, label=f"Reflexion·Reflect·R{round_num}")

            # Check if consistent
            consistent = reflection_result.get("consistent", "").lower().strip()
            if consistent == "yes":
                break

            # Update xpaths from reflection
            new_xpaths_dict = reflection_result.get("xpath", {})
            if not new_xpaths_dict:
                break

            # Normalize
            current_xpaths = {}
            for attr, xpath_val in new_xpaths_dict.items():
                if isinstance(xpath_val, list):
                    current_xpaths[attr] = xpath_val[0] if xpath_val else ""
                else:
                    current_xpaths[attr] = xpath_val or ""

            # Execute new xpaths
            execution_results = {}
            has_failures = False
            for attr, xpath in current_xpaths.items():
                if not xpath:
                    execution_results[attr] = []
                    has_failures = True
                    continue
                vals = self._execute_xpath_on_html(simplified_html, xpath)
                execution_results[attr] = vals
                if not vals:
                    has_failures = True

            history_entries.append({
                "round": round_num,
                "thought": reflection_result.get("thought", ""),
                "xpath": new_xpaths_dict,
                "extracted_values": execution_results,
            })

        # Final execution via browser for accurate results
        extracted_values = {}
        for attr, xpath in current_xpaths.items():
            if not xpath:
                extracted_values[attr] = []
                continue
            try:
                vals = await self.browser.execute_xpath(page, xpath)
                extracted_values[attr] = vals
            except Exception as e:
                logger.warning("[Reflexion] XPath exec error for %s: %s", attr, e)
                extracted_values[attr] = []

        await page.context.close()

        return {
            "url": url,
            "query": query,
            "attributes": list(current_xpaths.keys()),
            "xpaths": current_xpaths,
            "values": extracted_values,
            "rounds_used": len(history_entries),
        }

    async def run_grouped_batch(self, groups: list[dict], concurrency: int = 3) -> list[dict]:
        """Run Reflexion with Phase 1 → Phase 2 grouped evaluation."""
        from tqdm import tqdm

        checkpoint_path = self.config.output_dir / "results_checkpoint.json"
        total_urls = sum(len(g["urls"]) for g in groups)

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

        pbar = tqdm(total=total_urls, desc="Reflexion·Pipeline",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")
        pbar.update(len(done_urls))

        # Workers
        workers: list[ReflexionPipeline] = []
        for _ in range(concurrency):
            worker = ReflexionPipeline(self.config, max_rounds=self.max_rounds,
                                       enable_monitor=self._enable_monitor)
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
                async with lock:
                    pbar.update(1)
                return

            # Phase 1: Reflexion on first URL
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

        with open(checkpoint_path, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2, ensure_ascii=False)

        logger.info("Done: %d results, P1(✓%d ✗%d) P2(✓%d ✗%d)",
                    len(all_results), pipeline_ok, pipeline_err, reuse_ok, reuse_err)
        return all_results
