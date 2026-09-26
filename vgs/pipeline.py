"""VGS Pipeline — orchestrates the four stages end-to-end."""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from configs.config import VGSConfig
from utils.llm_client import LLMClient
from utils.browser import BrowserManager
from utils.cached_browser import CachedBrowserManager, CachedPage
from utils.html_processor import HTMLProcessor
from vgs.attribute_identification import AttributeIdentifier
from vgs.visual_grounding import VisualGrounder
from vgs.element_pinpointing import ElementPinpointer
from vgs.xpath_synthesis import XPathSynthesizer

logger = logging.getLogger(__name__)


class VGSPipeline:
    """Full VGS extraction pipeline."""

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
        self.identifier = AttributeIdentifier(self.llm)
        self.grounder = VisualGrounder(self.llm)
        self.pinpointer = ElementPinpointer(self.llm, self.browser)
        self.synthesizer = XPathSynthesizer(self.llm, config.neighbor_distance)

    async def run(self, url: str, query: str) -> dict:
        """Execute VGS on a single (url, query) pair.

        Returns
        -------
        dict with keys:
            attributes : list[str]
            xpaths     : dict[str, str]   attribute → xpath
            values     : dict[str, list]  attribute → extracted values
        """
        await self.browser.start()
        try:
            return await self._run_inner(url, query)
        finally:
            await self.browser.stop()

    async def _run_inner(self, url: str, query: str,
                         predefined_attributes: list[str] | None = None) -> dict:
        """Core pipeline with full trace capture (prompt + response per stage)."""
        from vgs.prompts import (
            ATTRIBUTE_IDENTIFICATION_PROMPT,
            VISUAL_GROUNDING_PROMPT,
            ELEMENT_SCANNING_PROMPT,
            ELEMENT_SELECTION_PROMPT,
            XPATH_SYNTHESIS_PROMPT,
            XPATH_SYNTHESIS_LINK_PROMPT,
            XPATH_SYNTHESIS_IMAGE_PROMPT,
        )
        import re as _re

        stages_trace: list[dict] = []

        # ── Stage 1: Attribute Identification ──
        if predefined_attributes:
            attributes = predefined_attributes
            logger.info("Stage 1: Using predefined attributes: %s", attributes)
            stages_trace.append({
                "stage": 1, "name": "Attribute Identification",
                "prompt": "(predefined)",
                "response": {"attributes": attributes},
            })
        else:
            logger.info("Stage 1: Identifying attributes from query …")
            s1_prompt = ATTRIBUTE_IDENTIFICATION_PROMPT.format(query=query)
            s1_result = self.identifier.identify(query)
            attributes = s1_result if isinstance(s1_result, list) else s1_result
            logger.info("  → attributes: %s", attributes)
            stages_trace.append({
                "stage": 1, "name": "Attribute Identification",
                "prompt": s1_prompt,
                "response": {"attributes": attributes},
            })

        # ── Load page via cache or real browser ──
        page = await self.browser.load_page(url)
        sample_id = url.split("/")[-1][:30].replace("/", "_")
        screenshot_dir = self.config.screenshot_dir / sample_id
        region_paths = await self.browser.capture_regions(page, screenshot_dir)
        logger.info("  → captured %d regions", len(region_paths))

        full_html = await self.browser.get_full_html(page)
        simplified_html = HTMLProcessor.simplify(full_html)

        xpaths: dict[str, str] = {}
        values: dict[str, list] = {}

        try:
            for attribute in attributes:
                logger.info("Processing attribute: %s", attribute)

                # ── Stage 2: Visual Grounding ──
                logger.info("  Stage 2: Visual grounding …")
                s2_prompt = VISUAL_GROUNDING_PROMPT.format(attribute=attribute)
                labels = [f"Region {i}" for i in range(len(region_paths))]
                s2_result = self.llm.vision_query(
                    s2_prompt, region_paths,
                    image_labels=labels, label="Stage2·VisualGround"
                )
                matching = s2_result.get("matching_region", "")
                match = _re.search(r"(\d+)", str(matching))
                region_index = int(match.group(1)) if match and int(match.group(1)) < len(region_paths) else 0
                logger.info("  → matched region %d", region_index)
                stages_trace.append({
                    "stage": 2, "name": "Visual Grounding",
                    "attribute": attribute,
                    "prompt": s2_prompt,
                    "response": s2_result,
                    "result": {"region_index": region_index},
                    "num_regions": len(region_paths),
                    "screenshot": str(region_paths[region_index]) if region_index < len(region_paths) else None,
                })

                scroll_y = region_index * self.config.viewport_height
                await page.evaluate(f"window.scrollTo(0, {scroll_y})")
                await asyncio.sleep(0.3)

                # ── Stage 3: Element Pinpointing ──
                logger.info("  Stage 3: Element pinpointing …")
                modality = self.pinpointer.classify_modality(attribute)
                css_selector = self.pinpointer._modality_to_selector(modality)

                region_screenshot = screenshot_dir / f"region_{region_index}.png"
                marked_path = screenshot_dir / f"region_{region_index}_marked.png"
                s3a_prompt = ELEMENT_SCANNING_PROMPT.format(attribute=attribute)
                s3b_prompt = ELEMENT_SELECTION_PROMPT.format(attribute=attribute)

                # S3a: Scanning — VLM analyzes region screenshot
                s3a_result = self.llm.vision_query(
                    s3a_prompt, [region_screenshot], label="Stage3·Scanning"
                )
                # S3b: Selection — inject Set-of-Mark bounding boxes + screenshot + VLM
                # Use pre-cached marked screenshots if available (offline mode)
                cached_marked = None
                if isinstance(page, CachedPage):
                    cached_marked = page._cache_dir / f"marked_{modality}.png"
                    if not cached_marked.exists():
                        cached_marked = None

                if cached_marked:
                    # Offline: use pre-cached marked screenshot
                    marked_path = cached_marked
                else:
                    # Online: inject bbox + screenshot
                    marked_path = screenshot_dir / f"region_{region_index}_marked.png"
                    await self.browser.inject_bounding_boxes(page, css_selector)
                    await page.screenshot(path=str(marked_path))
                    await self.browser.remove_bounding_boxes(page)
                s3b_result = self.llm.vision_query(
                    s3b_prompt, [marked_path], label="Stage3·Selection"
                )
                selected_ids = s3b_result.get("selected_ids", [])
                selected_ids = [int(x) for x in selected_ids]

                logger.info("  → selected IDs: %s", selected_ids)

                stages_trace.append({
                    "stage": 3, "name": "Element Pinpointing",
                    "attribute": attribute,
                    "modality": modality,
                    "css_selector": css_selector,
                    "scan_prompt": s3a_prompt,
                    "scan_response": s3a_result,
                    "select_prompt": s3b_prompt,
                    "select_response": s3b_result,
                    "selected_ids": selected_ids,
                    "screenshot_region": str(region_screenshot),
                    "screenshot_marked": str(marked_path),
                })

                # ── Stage 4: XPath Synthesis ──
                logger.info("  Stage 4: XPath synthesis …")
                html_segments = await self._build_html_segments(
                    page, css_selector, selected_ids
                )
                prompt_template = self.synthesizer._select_prompt(attribute)
                s4_prompt = prompt_template.format(
                    attribute=attribute, html_segments=html_segments
                )
                # Use marked screenshot if exists, else region screenshot
                s4_img = marked_path if marked_path.exists() else region_screenshot
                s4_result = self.llm.vision_query(
                    s4_prompt, [s4_img], label="Stage4·XPathSynth"
                )
                xpath = s4_result.get("xpath", "")
                logger.info("  → XPath: %s", xpath)

                # Execute XPath to get values
                extracted = await self.browser.execute_xpath(page, xpath)
                xpaths[attribute] = xpath
                values[attribute] = extracted
                logger.info("  → extracted %d values", len(extracted))

                stages_trace.append({
                    "stage": 4, "name": "XPath Synthesis",
                    "attribute": attribute,
                    "html_segments": html_segments[:2000],
                    "prompt": s4_prompt,
                    "response": s4_result,
                    "result": {"xpath": xpath, "extracted_values": extracted},
                })
        finally:
            try:
                await page.context.close()
            except Exception:
                pass

        return {
            "url": url,
            "query": query,
            "attributes": attributes,
            "xpaths": xpaths,
            "values": values,
            "stages": stages_trace,
        }

    async def _build_html_segments(self, page, css_selector: str,
                                    selected_ids: list[int]) -> str:
        """Build HTML segments around selected elements."""
        full_html = await page.content()
        tags = self.synthesizer._css_to_tags(css_selector)
        elements = HTMLProcessor.get_elements_by_multi_tags(full_html, tags)
        segments: list[str] = []
        for box_id in selected_ids:
            if box_id >= len(elements):
                continue
            elem_info = elements[box_id]
            xpath_guess = elem_info["xpath"]
            segment = HTMLProcessor.extract_local_segment(
                full_html, xpath_guess, self.synthesizer.neighbor_distance
            )
            if segment:
                segments.append(f"<!-- Element {box_id} -->\n{segment}")
        return "\n\n".join(segments) if segments else "(no segments extracted)"

    @staticmethod
    def _make_xpaths_robust(xpaths: dict[str, str], html: str) -> dict[str, str]:
        """Generalize XPaths for cross-page reuse.

        Strategies:
        1. Simplify compound class predicates → contains(@class, first-token)
        2. Remove positional predicates
        3. Convert deeply nested paths to descendant axis
        """
        import re
        from lxml import etree

        tree = etree.HTML(html)
        if tree is None:
            return xpaths

        robust = {}
        for attr, xpath in xpaths.items():
            if not xpath:
                robust[attr] = xpath
                continue

            candidates = [xpath]

            # Strategy 1: Simplify compound class predicates
            if "@class=" in xpath:
                simplified = re.sub(
                    r"@class=['\"]([^'\"]+)['\"]",
                    lambda m: f"contains(@class,'{m.group(1).split()[0]}')",
                    xpath,
                )
                if simplified != xpath:
                    candidates.append(simplified)

            # Strategy 2: Remove positional predicates
            if re.search(r'/\w+\[\d+\]/', xpath):
                no_pos = re.sub(r'\[\d+\]', '', xpath)
                candidates.append(no_pos)

            # Strategy 3: Convert deeply nested paths to descendant axis
            parts = xpath.split('/')
            non_empty = [p for p in parts if p]
            if len(non_empty) >= 5:
                tail = []
                for part in reversed(non_empty):
                    tail.insert(0, part)
                    if '[' in part or len(tail) >= 3:
                        break
                descendant_xpath = '//' + '/'.join(tail)
                candidates.append(descendant_xpath)

            # Strategy 4: tag[@class='x'] → tag[contains(@class,'x')]
            if "[@class=" in xpath and "contains" not in xpath:
                contains_ver = re.sub(
                    r"\[@class=['\"]([^'\"]+)['\"]\]",
                    lambda m: f"[contains(@class,'{m.group(1).split()[0]}')]",
                    xpath,
                )
                if contains_ver != xpath:
                    candidates.append(contains_ver)

            # Pick best: still matches on current page but shorter/more general
            best = xpath
            for candidate in candidates[1:]:
                try:
                    matches = tree.xpath(candidate)
                    orig_matches = tree.xpath(xpath)
                    if matches and len(matches) >= len(orig_matches):
                        if len(candidate) <= len(best):
                            best = candidate
                except (etree.XPathError, Exception):
                    continue

            robust[attr] = best
            if best != xpath:
                logger.info("  [Robust] %s: %s → %s", attr, xpath, best)

        return robust

    @staticmethod
    def _adaptive_xpath_repair(xpath: str, page) -> str | None:
        """Lightweight XPath repair when reuse returns empty (no LLM).

        Strategies:
        1. Simplify class predicates → contains()
        2. Strip positional indices
        3. Use last 2-3 path steps as descendant
        4. Keep contains(@class) but simplify path
        """
        import re

        candidates = []

        # Strategy 1: Simplify class predicates
        if "@class=" in xpath:
            simplified = re.sub(
                r"@class=['\"]([^'\"]+)['\"]",
                lambda m: f"contains(@class,'{m.group(1).split()[0]}')",
                xpath,
            )
            if simplified != xpath:
                candidates.append(simplified)

        # Strategy 2: Remove positional indices
        no_pos = re.sub(r'\[\d+\]', '', xpath)
        if no_pos != xpath:
            candidates.append(no_pos)

        # Strategy 3: Last 2-3 path steps as descendant
        parts = [p for p in xpath.split('/') if p]
        if len(parts) >= 3:
            tail = parts[-2:]
            tail_clean = [re.sub(r'\[\d+\]', '', p) for p in tail]
            candidates.append('//' + '/'.join(tail_clean))
            if len(parts) >= 4:
                tail3 = parts[-3:]
                tail3_clean = [re.sub(r'\[\d+\]', '', p) for p in tail3]
                candidates.append('//' + '/'.join(tail3_clean))

        # Strategy 4: Keep contains(@class) but simplify path
        class_match = re.search(
            r"(\w+)\[contains\(@class,['\"]([^'\"]+)['\"]\)\]", xpath
        )
        if class_match:
            tag = class_match.group(1)
            cls = class_match.group(2)
            candidates.append(f"//{tag}[contains(@class,'{cls}')]")

        for candidate in candidates:
            try:
                from lxml import etree
                etree.XPath(candidate)
                return candidate
            except Exception:
                continue

        return None

    async def run_batch(self, samples: list[dict], index_offset: int = 0,
                        concurrency: int = 1) -> list[dict]:
        """Run VGS on a batch of samples with optional concurrency.

        Each sample should have keys: url, query
        Args:
            index_offset: Global index offset for unique_id generation (used for resume).
            concurrency: Number of parallel workers (each gets its own browser).
        """
        import json
        from tqdm import tqdm

        # Set up trace directory if monitor enabled
        if self._enable_monitor:
            from monitor.collector import StageCollector
            collector = StageCollector.get()
            traces_dir = self.config.output_dir / "traces"
            collector.set_traces_dir(traces_dir)

        checkpoint_path = self.config.output_dir / "results_checkpoint.json"

        # Create independent pipeline instances for each worker
        workers: list[VGSPipeline] = []
        for _ in range(concurrency):
            worker = VGSPipeline(self.config, enable_monitor=self._enable_monitor)
            await worker.browser.start()
            workers.append(worker)

        semaphore = asyncio.Semaphore(concurrency)
        worker_available = list(range(concurrency))
        pool_lock = asyncio.Lock()

        results = [None] * len(samples)
        success_count = 0
        error_count = 0
        lock = asyncio.Lock()

        pbar = tqdm(total=len(samples), desc="VGS",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")

        async def process_sample(i: int, sample: dict):
            nonlocal success_count, error_count
            global_idx = i + index_offset
            unique_id = f"{sample.get('sample_id', global_idx)}_{global_idx}"

            async with semaphore:
                async with pool_lock:
                    worker_idx = worker_available.pop()
                worker = workers[worker_idx]

                if self._enable_monitor:
                    from monitor.collector import StageCollector
                    StageCollector.get().begin_sample(unique_id)

                try:
                    result = await worker._run_inner(sample["url"], sample["query"])
                    result["sample_id"] = unique_id
                    async with lock:
                        results[i] = result
                        success_count += 1
                except Exception as exc:
                    logger.error("Failed on sample %s: %s", unique_id, exc)
                    async with lock:
                        results[i] = {
                            "sample_id": unique_id,
                            "url": sample["url"],
                            "query": sample["query"],
                            "error": str(exc),
                        }
                        error_count += 1
                finally:
                    async with pool_lock:
                        worker_available.append(worker_idx)

                    if self._enable_monitor:
                        from monitor.collector import StageCollector
                        StageCollector.get().end_sample(unique_id)

                async with lock:
                    pbar.set_postfix_str(f"✓{success_count} ✗{error_count}")
                    pbar.update(1)
                    completed = success_count + error_count
                    if completed % 10 == 0 or completed == len(samples):
                        partial = [r for r in results if r is not None]
                        with open(checkpoint_path, "w", encoding="utf-8") as fh:
                            json.dump(partial, fh, indent=2, ensure_ascii=False)

        try:
            tasks = [process_sample(i, sample) for i, sample in enumerate(samples)]
            await asyncio.gather(*tasks)
        finally:
            pbar.close()
            for worker in workers:
                await worker.browser.stop()

        final_results = [r for r in results if r is not None]
        return final_results

    async def run_grouped_batch(self, groups: list[dict], concurrency: int = 3) -> list[dict]:
        """Run VGS with pipelined Phase 1 → Phase 2 (mirrors CogExtract optimizations).

        Each group's Phase 2 starts immediately after its Phase 1 completes.
        Includes: predefined_attributes, 3× retry, 1s rate-limit, adaptive
        XPath repair, and Phase 1 checkpoint/resume.
        """
        import json as _json
        from tqdm import tqdm

        checkpoint_path = self.config.output_dir / "results_checkpoint.json"
        phase1_checkpoint_path = self.config.output_dir / "phase1_checkpoint.json"
        total_urls = sum(len(g["urls"]) for g in groups)

        # ── Shared state ──
        lock = asyncio.Lock()
        all_results: list[dict] = []
        pipeline_ok = 0
        pipeline_err = 0
        reuse_ok = 0
        reuse_err = 0

        # ── Resume Phase 1 checkpoint ──
        pipeline_results: dict[int, dict] = {}
        phase1_done_indices: set[int] = set()
        if phase1_checkpoint_path.exists():
            with open(phase1_checkpoint_path, encoding="utf-8") as _fh:
                saved = _json.load(_fh)
            for entry in saved:
                idx = entry.get("_group_idx")
                if idx is not None and 0 <= idx < len(groups):
                    pipeline_results[idx] = entry
                    phase1_done_indices.add(idx)
            pipeline_ok = sum(1 for e in saved if e.get("xpaths"))
            pipeline_err = sum(1 for e in saved if e.get("error"))
            logger.info("Resuming — %d/%d groups from Phase 1 checkpoint",
                        len(phase1_done_indices), len(groups))

        # ── Resume Phase 2 checkpoint ──
        done_urls: set[str] = set()
        if checkpoint_path.exists():
            with open(checkpoint_path, encoding="utf-8") as _fh:
                prior = _json.load(_fh)
            done_urls = {r.get("url", "") for r in prior}
            all_results.extend(prior)
            logger.info("Phase 2 checkpoint: %d URLs already done", len(done_urls))

        remaining_groups = [(i, g) for i, g in enumerate(groups)
                           if i not in phase1_done_indices]

        # ── Progress bar ──
        pbar = tqdm(total=total_urls, desc="VGS·Pipeline",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")
        pbar.update(len(done_urls) + len(phase1_done_indices))

        # ── Phase 1 workers ──
        pipeline_workers: list[VGSPipeline] = []
        for _ in range(concurrency):
            worker = VGSPipeline(self.config, enable_monitor=self._enable_monitor)
            await worker.browser.start()
            pipeline_workers.append(worker)

        pipeline_sem = asyncio.Semaphore(concurrency)
        worker_available = list(range(concurrency))
        pool_lock = asyncio.Lock()

        # ── Phase 2 browser pool (shared, 2 concurrency) ──
        reuse_concurrency = min(2, concurrency)
        reuse_browsers: list[BrowserManager] = []
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

        # ── Phase 2: reuse XPath on a single URL ──
        async def reuse_single_url(group_idx: int, url: str, query: str,
                                   xpaths: dict, attrs: list):
            nonlocal reuse_ok, reuse_err
            sample_id = groups[group_idx]["sample_id"]
            uid = f"{sample_id}_{url.split('/')[-1][:20]}"

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

            await asyncio.sleep(1.0)  # rate limit
            async with reuse_sem:
                async with browser_lock:
                    bidx = browser_pool.pop()
                browser = reuse_browsers[bidx]
                last_exc = None
                for attempt in range(1, 4):
                    page = None
                    try:
                        page = await browser.load_page(url)
                        values: dict[str, list] = {}
                        used_xpaths: dict[str, str] = {}
                        for attr, xpath in xpaths.items():
                            extracted = await browser.execute_xpath(page, xpath)
                            if not extracted:
                                repaired = self._adaptive_xpath_repair(xpath, page)
                                if repaired:
                                    extracted = await browser.execute_xpath(
                                        page, repaired)
                                    used_xpaths[attr] = repaired if extracted else xpath
                                else:
                                    used_xpaths[attr] = xpath
                            else:
                                used_xpaths[attr] = xpath
                            values[attr] = extracted
                        await page.context.close()
                        page = None

                        async with lock:
                            all_results.append({
                                "url": url, "query": query,
                                "attributes": attrs, "xpaths": used_xpaths,
                                "values": values, "sample_id": uid,
                            })
                            reuse_ok += 1
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
                        reuse_err += 1

                async with browser_lock:
                    browser_pool.append(bidx)
                async with lock:
                    pbar.update(1)
                    pbar.set_postfix_str(
                        f"P1✓{pipeline_ok}✗{pipeline_err} P2✓{reuse_ok}✗{reuse_err}")
                    if len(all_results) % 50 == 0:
                        with open(checkpoint_path, "w", encoding="utf-8") as fh:
                            _json.dump(all_results, fh, indent=2, ensure_ascii=False)

        # ── Process one group: Phase 1 → immediate Phase 2 ──
        async def process_group(group_idx: int, group: dict):
            nonlocal pipeline_ok, pipeline_err
            sample_id = group["sample_id"]
            first_url = group["urls"][0]
            first_uid = f"{sample_id}_{first_url.split('/')[-1][:20]}"

            # Phase 1: pipeline on first URL (with 3× retry)
            result = pipeline_results.get(group_idx)
            if result is None:
                async with pipeline_sem:
                    async with pool_lock:
                        widx = worker_available.pop()
                    worker = pipeline_workers[widx]
                    last_error = None
                    for attempt in range(1, 4):
                        try:
                            predefined_attrs = group.get("attributes") or None
                            result = await worker._run_inner(
                                first_url, group["query"],
                                predefined_attributes=predefined_attrs,
                            )
                            result["sample_id"] = first_uid
                            result["_group_idx"] = group_idx
                            async with lock:
                                pipeline_ok += 1
                                pbar.update(1)
                                pbar.set_postfix_str(
                                    f"P1✓{pipeline_ok}✗{pipeline_err} "
                                    f"P2✓{reuse_ok}✗{reuse_err}")
                            last_error = None
                            break
                        except Exception as exc:
                            last_error = exc
                            if attempt < 3:
                                wait = attempt * 5
                                logger.warning(
                                    "[%s] Attempt %d/3 failed: %s — retry in %ds",
                                    sample_id, attempt, exc, wait)
                                await asyncio.sleep(wait)

                    if last_error is not None:
                        result = {
                            "sample_id": first_uid, "url": first_url,
                            "query": group["query"], "error": str(last_error),
                            "_group_idx": group_idx,
                        }
                        async with lock:
                            pipeline_err += 1
                            pbar.update(1)
                            pbar.set_postfix_str(
                                f"P1✓{pipeline_ok}✗{pipeline_err} "
                                f"P2✓{reuse_ok}✗{reuse_err}")

                    async with pool_lock:
                        worker_available.append(widx)

                    # Save Phase 1 checkpoint
                    pipeline_results[group_idx] = result
                    async with lock:
                        done = list(pipeline_results.values())
                        with open(phase1_checkpoint_path, "w", encoding="utf-8") as _fh:
                            _json.dump(done, _fh, indent=2, ensure_ascii=False)

            # Add Phase 1 result
            async with lock:
                all_results.append(result)

            # Phase 2: immediately reuse XPaths on remaining URLs
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
            all_tasks = []
            for idx in sorted(phase1_done_indices):
                all_tasks.append(process_group(idx, groups[idx]))
            for idx, group in remaining_groups:
                all_tasks.append(process_group(idx, group))
            await asyncio.gather(*all_tasks)
        finally:
            pbar.close()
            for worker in pipeline_workers:
                await worker.browser.stop()
            for browser in reuse_browsers:
                await browser.stop()

        # Final save
        with open(checkpoint_path, "w", encoding="utf-8") as fh:
            _json.dump(all_results, fh, indent=2, ensure_ascii=False)

        total_ok = pipeline_ok + reuse_ok
        total_err = pipeline_err + reuse_err
        logger.info("Done: %d results, P1(✓%d ✗%d) P2(✓%d ✗%d)",
                    len(all_results), pipeline_ok, pipeline_err, reuse_ok, reuse_err)
        return all_results
