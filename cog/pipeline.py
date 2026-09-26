"""Cog Pipeline — Reflection-guided XPath Refinement.

Generate → Execute → Verify → Reflect → Revise (loop up to k rounds)

Stage 1: XPath Generation (VGS Stage 1-4 on seed page)
Stage 2: Seed Execution (verify XPath works on seed page)
Stage 3: Cross-page Validation (test on other pages in the group)
Stage 4: Failure Diagnosis (classify failure type)
Stage 5: Reflection & Revision (reflect + revise + re-validate)
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from configs.config import VGSConfig
from utils.llm_client import LLMClient
from utils.browser import BrowserManager
from utils.cached_browser import CachedBrowserManager
from cog.generator import XPathGenerator
from cog.executor import CrossPageExecutor
from cog.diagnoser import FailureDiagnoser, DiagnosisResult
from cog.reflector import ReflectionRefiner, RevisionResult, ReflectionTrace

logger = logging.getLogger(__name__)


@dataclass
class CogResult:
    """Full result of the Cog pipeline for one group."""
    url: str
    query: str
    attributes: list[str]
    xpaths: dict[str, str]
    values: dict[str, list]
    traces: dict[str, ReflectionTrace] = field(default_factory=dict)
    elapsed: float = 0.0


class CogBasePipeline:
    """Reflection-guided XPath Refinement pipeline.

    Given a group of same-site URLs and a query:
    1. Generate XPaths on the first URL (seed)
    2. Validate on a second URL
    3. If failures: diagnose → reflect → revise → re-validate (up to max_rounds)
    4. Apply final XPaths to all remaining URLs
    """

    def __init__(
        self,
        config: VGSConfig,
        max_rounds: int = 3,
        enable_monitor: bool = False,
    ):
        self.config = config
        self.max_rounds = max_rounds

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
        self.generator = XPathGenerator(
            config, self.llm, self.browser, enable_monitor
        )
        self.executor = CrossPageExecutor(self.browser)
        self.diagnoser = FailureDiagnoser(self.llm)
        self.reflector = ReflectionRefiner(self.llm, max_rounds)

    async def run(
        self,
        urls: list[str],
        query: str,
        predefined_attributes: Optional[list[str]] = None,
    ) -> CogResult:
        """Run the full Cog pipeline on a group of URLs."""
        start_time = time.time()

        if len(urls) < 1:
            raise ValueError("Need at least 1 URL")

        seed_url = urls[0]
        validation_urls = urls[1:2] if len(urls) > 1 else []
        remaining_urls = urls[2:] if len(urls) > 2 else []

        await self.browser.start()

        try:
            result = await self._run_inner(
                seed_url, validation_urls, remaining_urls,
                query, predefined_attributes,
            )
        finally:
            await self.browser.stop()

        result.elapsed = round(time.time() - start_time, 2)
        return result

    async def _run_inner(
        self,
        seed_url: str,
        validation_urls: list[str],
        remaining_urls: list[str],
        query: str,
        predefined_attributes: Optional[list[str]],
    ) -> CogResult:
        """Core pipeline logic."""

        # ── Stage 1: Attribute Identification ──
        if predefined_attributes:
            attributes = predefined_attributes
            logger.info("[Cog] Stage 1: predefined attributes: %s", attributes)
        else:
            attributes = self.generator.identify_attributes(query)
            logger.info("[Cog] Stage 1: identified attributes: %s", attributes)

        # ── Load seed page ──
        seed_page = await self.browser.load_page(seed_url)
        seed_id = seed_url.split("/")[-1][:30].replace("/", "_")
        screenshot_dir = self.config.screenshot_dir / f"cog_{seed_id}_{int(time.time())}"
        seed_html = await self.browser.get_full_html(seed_page)

        # ── Load validation page (if available) ──
        val_page = None
        val_html = ""
        if validation_urls:
            val_page = await self.browser.load_page(validation_urls[0])
            val_html = await self.browser.get_full_html(val_page)

        xpaths: dict[str, str] = {}
        values: dict[str, list] = {}
        traces: dict[str, ReflectionTrace] = {}

        try:
            for attribute in attributes:
                logger.info("[Cog] ═══ Processing attribute: '%s' ═══", attribute)

                # ── Stage 1 (cont): Generate initial XPath ──
                xpath, gen_metadata = await self.generator.generate(
                    seed_page, attribute, screenshot_dir,
                )

                # ── Stage 2: Seed Execution ──
                seed_result = await self.executor.execute_on_page(
                    seed_page, xpath, seed_url,
                )

                if not seed_result.success:
                    logger.warning(
                        "[Cog] Stage 2: seed extraction failed for '%s', triggering reflection",
                        attribute,
                    )
                    # Seed fails → treat as validation failure, trigger reflection
                    if val_page is not None:
                        xpath, trace = await self._validation_loop(
                            attribute, xpath,
                            seed_html, val_html,
                            [],  # no seed values
                            seed_page, val_page, validation_urls[0],
                        )
                        traces[attribute] = trace
                    else:
                        xpaths[attribute] = xpath
                        values[attribute] = []
                        continue
                else:
                    logger.info(
                        "[Cog] Stage 2: seed OK — %d value(s)", seed_result.count
                    )

                    # ── Stage 3: Cross-page Validation ──
                    if val_page is not None:
                        xpath, trace = await self._validation_loop(
                            attribute, xpath,
                            seed_html, val_html,
                            seed_result.values,
                            seed_page, val_page, validation_urls[0],
                        )
                        traces[attribute] = trace
                    else:
                        logger.info("[Cog] No validation URL — skipping Stage 3-5")

                # Final extraction from seed page with (possibly refined) XPath
                final_result = await self.executor.execute_on_page(
                    seed_page, xpath, seed_url,
                )
                xpaths[attribute] = xpath
                values[attribute] = final_result.values
        finally:
            # ── Always close seed and validation pages ──
            try:
                await seed_page.context.close()
            except Exception:
                pass
            if val_page is not None:
                try:
                    await val_page.context.close()
                except Exception:
                    pass

        # ── Apply final XPaths to remaining URLs ──
        if remaining_urls:
            await self._apply_to_remaining(
                xpaths, values, attributes, remaining_urls,
            )

        return CogResult(
            url=seed_url,
            query=query,
            attributes=attributes,
            xpaths=xpaths,
            values=values,
            traces=traces,
        )

    async def _validation_loop(
        self,
        attribute: str,
        xpath: str,
        html_a: str,
        html_b: str,
        values_a: list[str],
        seed_page,
        val_page,
        val_url: str,
    ) -> tuple[str, ReflectionTrace]:
        """Stage 3-4-5 loop: validate → diagnose → reflect → revise."""

        trace = ReflectionTrace(
            attribute=attribute,
            initial_xpath=xpath,
            final_xpath=xpath,
        )

        # Stage 3: validate on Page B using lxml on raw HTML (same as Cog)
        from lxml import etree as _etree

        tree_b = _etree.HTML(html_b)
        validation_passed = False
        count_b = 0

        if tree_b is not None and xpath:
            try:
                hits_b = tree_b.xpath(xpath)
                count_b = len(hits_b)
                validation_passed = count_b > 0
            except Exception:
                validation_passed = False

        if validation_passed:
            logger.info("[Cog] Stage 3: ✓ XPath generalizes (lxml: B=%d)", count_b)
            trace.final_xpath = xpath
            trace.success = True
            return xpath, trace

        logger.info(
            "[Cog] Stage 3: ✗ XPath fails on validation page (lxml: %d matches)",
            count_b,
        )

        # ── Reflection loop ──
        # Extract current page_B values via lxml for diagnosis
        from lxml import etree as _etree_diag
        def _lxml_extract(html: str, xp: str) -> list[str]:
            tree = _etree_diag.HTML(html)
            if tree is None or not xp:
                return []
            try:
                hits = tree.xpath(xp)
                results = []
                for node in hits:
                    if isinstance(node, str):
                        results.append(node.strip()[:200])
                    elif hasattr(node, "text_content"):
                        text = node.text_content().strip()[:200]
                        if text:
                            results.append(text)
                return results
            except Exception:
                return []

        history: list[RevisionResult] = []
        current_values_b = _lxml_extract(html_b, xpath)

        for round_num in range(1, self.max_rounds + 1):
            logger.info("[Cog] ── Reflection round %d/%d ──", round_num, self.max_rounds)

            # Stage 4: Failure Diagnosis
            diagnosis = self.diagnoser.diagnose(
                attribute, xpath, html_a, html_b, values_a, current_values_b,
            )
            logger.info(
                "[Cog] Stage 4: type=%s, diag='%s'",
                diagnosis.failure_type, diagnosis.diagnosis[:80],
            )

            # Stage 5: Reflection & Revision
            if history:
                revision = self.reflector.reflect_with_history(
                    attribute, xpath, diagnosis, history, html_a, html_b,
                )
            else:
                revision = self.reflector.reflect_and_revise(
                    attribute, xpath, diagnosis, html_a, html_b,
                )

            revised_xpath = revision.revised_xpath

            # Validate the revised XPath on both pages (using lxml for speed)
            works, new_vals_a, new_vals_b = self.reflector.validate_xpath(
                revised_xpath, html_a, html_b,
            )

            if works:
                logger.info(
                    "[Cog] Stage 5: ✓ Revision accepted (A=%d, B=%d)",
                    len(new_vals_a), len(new_vals_b),
                )
                revision.accepted = True
                history.append(revision)
                xpath = revised_xpath
                values_a = new_vals_a
                trace.rounds = history
                trace.total_rounds = round_num
                trace.final_xpath = xpath
                trace.success = True
                break
            else:
                logger.info(
                    "[Cog] Stage 5: ✗ Revision still fails (A=%d, B=%d)",
                    len(new_vals_a), len(new_vals_b),
                )
                revision.accepted = False
                history.append(revision)

                # Update for next round
                current_values_b = _lxml_extract(html_b, revised_xpath)
                xpath = revised_xpath

        if not trace.success:
            logger.warning(
                "[Cog] Reflection loop exhausted (%d rounds) for '%s'",
                self.max_rounds, attribute,
            )
            trace.total_rounds = self.max_rounds
            # Fall back to initial xpath if all revisions failed
            trace.final_xpath = trace.initial_xpath
            xpath = trace.initial_xpath

        trace.rounds = history
        return xpath, trace

    async def _apply_to_remaining(
        self,
        xpaths: dict[str, str],
        values: dict[str, list],
        attributes: list[str],
        remaining_urls: list[str],
    ):
        """Apply finalized XPaths to remaining group URLs."""
        for url in remaining_urls:
            page = None
            try:
                page = await self.browser.load_page(url)
                for attr in attributes:
                    xpath = xpaths.get(attr, "")
                    if xpath:
                        result = await self.executor.execute_on_page(page, xpath, url)
                        # Note: values dict only holds seed page values;
                        # remaining URL values go to the evaluation loop
            except Exception as exc:
                logger.warning("[Cog] Failed on remaining URL %s: %s", url[:50], exc)
            finally:
                if page is not None and hasattr(page, 'context'):
                    try:
                        await page.context.close()
                    except Exception:
                        pass

    # ── Grouped batch interface (for evaluation protocol) ──

    async def run_grouped_batch(
        self, groups: list[dict], concurrency: int = 3
    ) -> list[dict]:
        """Run Cog on all groups with pipelined Phase 1 → Phase 2.

        Phase 1: Cog pipeline on seed URL (generate + refine XPaths)
        Phase 2: Reuse XPaths on remaining URLs via browser pool (concurrent)

        Each group dict has: sample_id, query, urls, predefined_attributes, ...
        Output: list of per-URL result dicts
        """
        import json as _json
        from tqdm import tqdm

        total_urls = sum(len(g["urls"]) for g in groups)
        checkpoint_path = self.config.output_dir / "results_checkpoint.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # ── Shared state ──
        lock = asyncio.Lock()
        all_results: list[dict] = []
        phase1_ok = 0
        phase1_err = 0
        phase2_ok = 0
        phase2_err = 0

        # ── Resume from checkpoint ──
        # Track (group_sample_id, url) pairs to allow same URL in different queries
        done_keys: set[tuple[str, str]] = set()
        if checkpoint_path.exists():
            with open(checkpoint_path, encoding="utf-8") as fh:
                prior = _json.load(fh)
            for r in prior:
                # Extract group sample_id (w_XXX_g_YYY_q_ZZZ) from result sample_id
                sid_parts = r.get("sample_id", "").split("_")
                group_sid = "_".join(sid_parts[:6]) if len(sid_parts) >= 6 else r.get("sample_id", "")
                done_keys.add((group_sid, r.get("url", "")))
            all_results.extend(prior)
            logger.info("Checkpoint: %d (group,url) pairs already done", len(done_keys))

        pbar = tqdm(
            total=total_urls,
            desc="Cog·Pipeline",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )
        pbar.update(len(done_keys))

        # ── Phase 2 browser pool (higher concurrency for cached mode) ──
        reuse_concurrency = min(8, concurrency * 3)
        reuse_browsers: list[BrowserManager] = []
        for _ in range(reuse_concurrency):
            browser = CachedBrowserManager(
                cache_dir=self.config.cache_dir,
                viewport_width=self.config.viewport_width,
                viewport_height=self.config.viewport_height,
                headless=self.config.headless,
                timeout_ms=self.config.page_load_timeout_ms,
            )
            await browser.start()
            reuse_browsers.append(browser)
        reuse_sem = asyncio.Semaphore(reuse_concurrency)
        browser_pool = list(range(reuse_concurrency))
        browser_lock = asyncio.Lock()

        # ── Phase 2: reuse XPath on a single URL ──
        async def reuse_single_url(group: dict, url: str, xpaths: dict, attrs: list,
                                   traces: dict | None = None):
            nonlocal phase2_ok, phase2_err
            sample_id = group.get("sample_id", "")
            query = group["query"]
            uid = f"{sample_id}_{url.split('/')[-1][:20]}"

            if not xpaths:
                async with lock:
                    all_results.append({
                        "sample_id": uid, "url": url, "query": query,
                        "error": "No XPath from Phase 1",
                    })
                    phase2_err += 1
                    pbar.update(1)
                return

            await asyncio.sleep(0.01)  # minimal yield for concurrency (cached mode)
            async with reuse_sem:
                async with browser_lock:
                    bidx = browser_pool.pop()
                browser = reuse_browsers[bidx]
                last_exc = None

                for attempt in range(1, 4):
                    page = None
                    try:
                        page = await browser.load_page(url)
                        url_values: dict[str, list] = {}
                        for attr in attrs:
                            xpath = xpaths.get(attr, "")
                            if xpath:
                                exec_r = await self.executor.execute_on_page(
                                    page, xpath, url)
                                url_values[attr] = exec_r.values
                            else:
                                url_values[attr] = []
                        await page.context.close()
                        page = None  # mark as closed

                        entry = {
                            "sample_id": uid, "url": url, "query": query,
                            "attributes": attrs, "xpaths": xpaths,
                            "values": url_values,
                        }
                        if traces:
                            entry["cog_traces"] = {
                                attr: {
                                    "initial": t.initial_xpath,
                                    "final": t.final_xpath,
                                    "rounds": t.total_rounds,
                                    "success": t.success,
                                    "candidates": getattr(t, 'candidates', []),
                                    "scores": getattr(t, 'scores', []),
                                    "ablation_mode": getattr(t, 'ablation_mode', ''),
                                }
                                for attr, t in traces.items()
                            }

                        async with lock:
                            all_results.append(entry)
                            phase2_ok += 1
                        last_exc = None
                        break
                    except Exception as exc:
                        last_exc = exc
                        if page is not None:
                            try:
                                await page.context.close()
                            except Exception:
                                pass
                            page = None
                        if attempt < 3:
                            await asyncio.sleep(attempt * 3)

                if last_exc is not None:
                    async with lock:
                        all_results.append({
                            "sample_id": uid, "url": url, "query": query,
                            "error": str(last_exc),
                        })
                        phase2_err += 1

                async with browser_lock:
                    browser_pool.append(bidx)
                async with lock:
                    pbar.update(1)
                    pbar.set_postfix_str(
                        f"P1✓{phase1_ok}✗{phase1_err} P2✓{phase2_ok}✗{phase2_err}")
                    # Checkpoint every 50 results
                    if len(all_results) % 50 == 0:
                        with open(checkpoint_path, "w", encoding="utf-8") as fh:
                            _json.dump(all_results, fh, indent=2, ensure_ascii=False)

        # ── Process one group ──
        async def process_group(group: dict):
            nonlocal phase1_ok, phase1_err
            sample_id = group.get("sample_id", "")
            query = group["query"]
            urls = group["urls"]
            # Use predefined attributes from dataset (avoid LLM Stage 1 errors)
            predef = group.get("predefined_attributes") or group.get("attributes")

            # Skip fully-done groups (check by (group_sample_id, url) pairs)
            remaining = [u for u in urls if (sample_id, u) not in done_keys]
            if not remaining:
                return

            seed_url = urls[0]

            # ── Phase 1: Cog on seed URL ──
            if (sample_id, seed_url) not in done_keys:
                try:
                    await asyncio.sleep(1.0)
                    result = await self.run(urls[:2], query, predef)
                    phase1_ok += 1

                    # Record seed URL result
                    seed_values = {}
                    for attr in result.attributes:
                        seed_values[attr] = result.values.get(attr, [])

                    entry = {
                        "sample_id": f"{sample_id}_{seed_url.split('/')[-1][:20]}",
                        "url": seed_url, "query": query,
                        "attributes": result.attributes,
                        "xpaths": result.xpaths,
                        "values": seed_values,
                    }
                    if result.traces:
                        entry["cog_traces"] = {
                            attr: {
                                "initial": t.initial_xpath,
                                "final": t.final_xpath,
                                "rounds": t.total_rounds,
                                "success": t.success,
                            }
                            for attr, t in result.traces.items()
                        }
                    async with lock:
                        all_results.append(entry)
                        done_keys.add((sample_id, seed_url))
                        pbar.update(1)

                    # Phase 2: dispatch remaining URLs concurrently
                    phase2_urls = [u for u in urls[1:] if (sample_id, u) not in done_keys]
                    tasks = [
                        reuse_single_url(group, u, result.xpaths, result.attributes)
                        for u in phase2_urls
                    ]
                    await asyncio.gather(*tasks)

                except Exception as exc:
                    logger.error("[Cog] Phase 1 failed for group %s: %s", sample_id, exc)
                    phase1_err += 1
                    # Mark all URLs as errored
                    for url in remaining:
                        async with lock:
                            all_results.append({
                                "sample_id": f"{sample_id}_{url.split('/')[-1][:20]}",
                                "url": url, "query": query,
                                "error": str(exc),
                            })
                            done_keys.add((sample_id, url))
                            pbar.update(1)
            else:
                # Seed already done — find its xpaths from prior results
                seed_entry = next(
                    (r for r in all_results
                     if r.get("url") == seed_url and sample_id in r.get("sample_id", "")),
                    None)
                if seed_entry and seed_entry.get("xpaths"):
                    phase2_urls = [u for u in urls[1:] if (sample_id, u) not in done_keys]
                    tasks = [
                        reuse_single_url(
                            group, u, seed_entry["xpaths"],
                            seed_entry.get("attributes", list(seed_entry["xpaths"].keys())))
                        for u in phase2_urls
                    ]
                    await asyncio.gather(*tasks)
                else:
                    for url in remaining:
                        async with lock:
                            all_results.append({
                                "sample_id": f"{sample_id}_{url.split('/')[-1][:20]}",
                                "url": url, "query": query,
                                "error": "Seed XPath unavailable",
                            })
                            pbar.update(1)

            # Save checkpoint after each group
            async with lock:
                with open(checkpoint_path, "w", encoding="utf-8") as fh:
                    _json.dump(all_results, fh, indent=2, ensure_ascii=False)

        # ── Process groups sequentially (Phase 1 is serial; Phase 2 is concurrent) ──
        for group in groups:
            await process_group(group)

        # ── Cleanup ──
        for browser in reuse_browsers:
            await browser.stop()
        pbar.close()

        # Final save
        with open(checkpoint_path, "w", encoding="utf-8") as fh:
            _json.dump(all_results, fh, indent=2, ensure_ascii=False)

        return all_results
