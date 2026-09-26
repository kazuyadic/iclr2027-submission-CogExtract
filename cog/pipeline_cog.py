"""Cog v8 Pipeline — Unified Hypothesis Generation with Iterative Refinement.

Paper contribution (v8):
1. Multi-Hypothesis Generation: Stage 4 produces K=3 diverse XPath candidates
2. Cross-Page Structural Verification: Deterministic scoring selects best candidate
3. Unified Iterative Refinement: ALL failure modes handled by a single feedback loop
   — no branching into separate rescue mechanisms

Key change from v7.3:
- Removed: List Pattern Rescue, DOM Rescue, Nav filtering (all rule-based)
- Added: Unified reflection loop that handles all failure types via VLM feedback
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

from lxml import etree

from configs.config import VGSConfig
from utils.llm_client import LLMClient
from utils.cached_browser import CachedBrowserManager
from cog.pipeline import CogBasePipeline, CogResult
from cog.generator_cog import EnhancedXPathGenerator
from cog.executor import CrossPageExecutor
from cog.diagnoser import FailureDiagnoser
from cog.reflector import ReflectionRefiner, ReflectionTrace
from cog.structural_verifier import StructuralVerifier, CandidateScore
from cog.prompts import UNIFIED_REFLECTION_PROMPT

logger = logging.getLogger(__name__)


class CogPipeline(CogBasePipeline):
    """Cog v8: Unified iterative refinement pipeline.

    Ablation modes:
        "full"             — K=3 + verification + reflection (default)
        "verify_no_reflect"— K=3 + verification, NO reflection
        "multi_only"       — K=3, pick first candidate, NO verification/reflection
        "single"           — K=1 (like VGS), NO verification/reflection

    Core loop:
    1. Generate K=3 candidates (VLM)
    2. Cross-page verification (deterministic)
    3. If pass → done
    4. If fail → unified reflection (VLM gets structured feedback) → goto 2
    """

    MAX_REFLECTION_ROUNDS = 3

    def __init__(
        self,
        config: VGSConfig,
        max_rounds: int = 1,
        enable_monitor: bool = False,
        ablation_mode: str = "full",
    ):
        self.config = config
        self.max_rounds = max_rounds
        self.ablation_mode = ablation_mode

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
        self.generator = EnhancedXPathGenerator(
            config, self.llm, self.browser, enable_monitor
        )
        self.executor = CrossPageExecutor(self.browser)
        self.diagnoser = FailureDiagnoser(self.llm)
        self.reflector = ReflectionRefiner(self.llm, max_rounds)
        self.verifier = StructuralVerifier()

    async def _run_inner(
        self,
        seed_url: str,
        validation_urls: list[str],
        remaining_urls: list[str],
        query: str,
        predefined_attributes: Optional[list[str]],
    ) -> CogResult:
        """Core pipeline: generate → verify → unified reflection loop."""

        # ── Stage 1: Attribute Identification ──
        if predefined_attributes:
            attributes = predefined_attributes
        else:
            attributes = self.generator.identify_attributes(query)
        logger.info("[Cog+] Stage 1: attributes = %s", attributes)

        # ── Load pages ──
        seed_page = await self.browser.load_page(seed_url)
        seed_id = seed_url.split("/")[-1][:30].replace("/", "_")
        screenshot_dir = self.config.screenshot_dir / f"cog_{seed_id}_{int(time.time())}"
        seed_html = await self.browser.get_full_html(seed_page)

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
                logger.info("[Cog+] ═══ Processing attribute: '%s' ═══", attribute)

                # ── Stage 2-3-4: Generate K candidates ──
                candidates, gen_metadata = await self.generator.generate_candidates(
                    seed_page, attribute, screenshot_dir,
                )
                logger.info("[Cog+] Initial candidates: %s", candidates)

                # ── Ablation: apply mode-specific logic ──
                if self.ablation_mode == "single":
                    # K=1: use only first candidate, skip verification
                    xpath = candidates[0] if candidates else ""
                    xpaths[attribute] = xpath
                    traces[attribute] = ReflectionTrace(
                        attribute=attribute, initial_xpath=xpath,
                        final_xpath=xpath, success=bool(xpath), total_rounds=0,
                    )
                    logger.info("[Cog+·single] Using first candidate: %s", xpath)

                elif self.ablation_mode == "multi_only":
                    # K=3 but pick first, no verification
                    xpath = candidates[0] if candidates else ""
                    xpaths[attribute] = xpath
                    traces[attribute] = ReflectionTrace(
                        attribute=attribute, initial_xpath=xpath,
                        final_xpath=xpath, success=bool(xpath), total_rounds=0,
                    )
                    logger.info("[Cog+·multi_only] Picked first of %d: %s",
                                len(candidates), xpath)

                elif self.ablation_mode == "verify_no_reflect":
                    # K=3 + verification (score + select best), NO reflection
                    if val_html and candidates:
                        scored = self.verifier.score_candidates(
                            candidates, seed_html, val_html, attribute
                        )
                        best = self.verifier.select_best(
                            candidates, seed_html, val_html, attribute
                        )
                        xpath = best.xpath if best else (candidates[0] if candidates else "")
                        xpaths[attribute] = xpath
                        traces[attribute] = ReflectionTrace(
                            attribute=attribute, initial_xpath=candidates[0] if candidates else "",
                            final_xpath=xpath, success=best.passes_basic if best else False,
                            total_rounds=0,
                        )
                        logger.info("[Cog+·verify] Best: %s (score=%.3f)",
                                    xpath, best.total_score if best else 0)
                    else:
                        xpath = candidates[0] if candidates else ""
                        xpaths[attribute] = xpath
                        traces[attribute] = ReflectionTrace(
                            attribute=attribute, initial_xpath=xpath,
                            final_xpath=xpath, success=bool(xpath), total_rounds=0,
                        )

                elif self.ablation_mode == "single_verify_reflect":
                    # K=1 + cross-page verification + reflection:
                    # identical to "full" except diversity (only first candidate).
                    if val_html and candidates:
                        xpath, trace = self._verify_and_refine(
                            attribute, query, candidates[:1], seed_html, val_html,
                        )
                        xpaths[attribute] = xpath
                        traces[attribute] = trace
                    else:
                        xpath = candidates[0] if candidates else ""
                        xpaths[attribute] = xpath
                        traces[attribute] = ReflectionTrace(
                            attribute=attribute, initial_xpath=xpath,
                            final_xpath=xpath, success=bool(xpath), total_rounds=0,
                        )
                    logger.info("[Cog+·single_verify_reflect] Final: %s", xpath)

                elif self.ablation_mode == "samepage_reflect":
                    # K=3 + reflection, but EGV degraded to a same-page execution
                    # check (val page = seed page, i.e. no cross-page evidence).
                    if candidates:
                        xpath, trace = self._verify_and_refine(
                            attribute, query, candidates, seed_html, seed_html,
                        )
                        xpaths[attribute] = xpath
                        traces[attribute] = trace
                    else:
                        xpath = ""
                        xpaths[attribute] = ""
                        traces[attribute] = ReflectionTrace(
                            attribute=attribute, initial_xpath="",
                            final_xpath="", success=False,
                        )
                    logger.info("[Cog+·samepage_reflect] Final: %s", xpath)

                else:  # "full" — original behavior
                    # ── Cross-page verification ──
                    if val_html and candidates:
                        xpath, trace = self._verify_and_refine(
                            attribute, query, candidates, seed_html, val_html,
                        )
                        xpaths[attribute] = xpath
                        traces[attribute] = trace
                    elif candidates:
                        xpath = candidates[0]
                        xpaths[attribute] = xpath
                        traces[attribute] = ReflectionTrace(
                            attribute=attribute, initial_xpath=xpath,
                            final_xpath=xpath, success=True,
                        )
                        logger.info("[Cog+] No validation page, using first: %s", xpath)
                    else:
                        xpaths[attribute] = ""
                        traces[attribute] = ReflectionTrace(
                            attribute=attribute, initial_xpath="",
                            final_xpath="", success=False,
                        )

                # Final extraction from seed page
                final_result = await self.executor.execute_on_page(
                    seed_page, xpaths[attribute], seed_url,
                )
                values[attribute] = final_result.values if final_result.success else []
        finally:
            # Always close pages to prevent Chromium process leak
            try:
                await seed_page.context.close()
            except Exception:
                pass
            if val_page is not None:
                try:
                    await val_page.context.close()
                except Exception:
                    pass

        # Apply to remaining URLs
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

    def _verify_and_refine(
        self,
        attribute: str,
        query: str,
        candidates: list[str],
        seed_html: str,
        val_html: str,
    ) -> tuple[str, ReflectionTrace]:
        """Unified verify-then-refine loop.

        1. Score candidates on both pages
        2. If any passes → return best
        3. If none passes → unified reflection → verify new candidates → repeat
        """
        trace = ReflectionTrace(
            attribute=attribute,
            initial_xpath=candidates[0] if candidates else "",
            final_xpath="",
        )

        # Track all failed attempts across rounds
        all_failed: list[CandidateScore] = []

        for round_num in range(self.MAX_REFLECTION_ROUNDS + 1):
            # Score current candidates
            scores = self.verifier.score_candidates(
                candidates, seed_html, val_html, attribute
            )

            # Check for a winner
            for s in scores:
                if s.passes_basic:
                    logger.info(
                        "[Cog+] ✓ Round %d: %s (score=%.3f, A=%d, B=%d)",
                        round_num, s.xpath, s.total_score, s.count_a, s.count_b,
                    )
                    trace.final_xpath = s.xpath
                    trace.success = True
                    trace.total_rounds = round_num
                    return s.xpath, trace

            # No winner — collect failures and reflect
            all_failed.extend(scores)
            failed_summary = "\n".join(
                f"  - {s.xpath} (A={s.count_a}, B={s.count_b}, score={s.total_score:.2f})"
                for s in all_failed
            )
            logger.info(
                "[Cog+] ✗ Round %d: no candidate passes. %d total failures.",
                round_num, len(all_failed),
            )

            if round_num >= self.MAX_REFLECTION_ROUNDS:
                break

            # ── Unified Reflection ──
            failed_xpaths = [s.xpath for s in all_failed]
            new_candidates = self._unified_reflection(
                attribute, query, failed_summary, failed_xpaths,
                seed_html, val_html, round_num,
            )
            if new_candidates:
                candidates = new_candidates
                logger.info(
                    "[Cog+] Reflection produced %d new candidates: %s",
                    len(candidates), candidates,
                )
            else:
                logger.warning("[Cog+] Reflection returned no candidates")
                break

        # Exhausted all rounds — fallback to best scoring candidate
        if all_failed:
            best = max(all_failed, key=lambda s: s.total_score)
            trace.final_xpath = best.xpath
            logger.warning(
                "[Cog+] Exhausted rounds. Fallback: %s (score=%.3f)",
                best.xpath, best.total_score,
            )
        else:
            trace.final_xpath = trace.initial_xpath

        trace.total_rounds = self.MAX_REFLECTION_ROUNDS
        return trace.final_xpath, trace

    def _unified_reflection(
        self,
        attribute: str,
        query: str,
        failed_summary: str,
        failed_xpaths: list[str],
        seed_html: str,
        val_html: str,
        round_num: int,
    ) -> list[str]:
        """Generate new candidates via VLM reflection on structured failure info."""

        # Structural analysis (deterministic, no LLM)
        tree_a = etree.HTML(seed_html)
        tree_b = etree.HTML(val_html)
        depth_a = self._avg_depth(tree_a, failed_xpaths)
        depth_b = self._avg_depth(tree_b, failed_xpaths)
        common_classes = self._find_common_classes(seed_html, val_html)

        # Extract HTML sections for context
        html_a_section = self._extract_section(seed_html)
        html_b_section = self._extract_section(val_html)

        # Get extraction results from the best failed candidate
        values_a = "(no values extracted)"
        values_b = "(no values extracted)"
        # Parse the summary to find actual values from score objects
        # For simplicity, show count info
        last_line = failed_summary.strip().split("\n")[-1] if failed_summary else ""
        if "A=" in last_line:
            values_a = f"See counts in: {last_line.strip()}"
            values_b = f"See counts in: {last_line.strip()}"

        prompt = UNIFIED_REFLECTION_PROMPT.format(
            attribute=attribute,
            query=query,
            failed_candidates=failed_summary,
            values_a_summary=values_a,
            values_b_summary=values_b,
            common_classes=", ".join(common_classes[:10]) or "(none found)",
            depth_a=f"~{depth_a:.0f}" if depth_a else "unknown",
            depth_b=f"~{depth_b:.0f}" if depth_b else "unknown",
            html_a_section=html_a_section,
            html_b_section=html_b_section,
        )

        parsed = self.llm.text_query(
            prompt, label=f"Cog+·UnifiedReflect·R{round_num}·{attribute}"
        )

        # Extract 3 candidates from response
        new_candidates = []
        for key in ("xpath_1", "xpath_2", "xpath_3"):
            xp = parsed.get(key, "")
            if xp and xp not in new_candidates:
                new_candidates.append(xp)

        diagnosis = parsed.get("diagnosis", "")
        strategy = parsed.get("strategy", "")
        if diagnosis:
            logger.info("[Cog+] Diagnosis: %s", diagnosis[:100])
        if strategy:
            logger.info("[Cog+] Strategy: %s", strategy[:100])

        return new_candidates

    # ── Structural Analysis Helpers (deterministic, no hardcode) ──

    @staticmethod
    def _avg_depth(tree, xpaths: list[str]) -> float:
        """Average DOM depth of matched elements."""
        if tree is None:
            return 0
        all_depths = []
        for xpath in xpaths:
            try:
                nodes = tree.xpath(xpath)
                for node in nodes[:5]:
                    if not hasattr(node, "getparent"):
                        continue
                    depth = 0
                    p = node.getparent()
                    while p is not None:
                        depth += 1
                        p = p.getparent()
                    all_depths.append(depth)
            except Exception:
                continue
        return sum(all_depths) / len(all_depths) if all_depths else 0

    @staticmethod
    def _find_common_classes(html_a: str, html_b: str, top_k: int = 10) -> list[str]:
        """Find class names that appear in both pages."""
        def extract_classes(html: str) -> set[str]:
            classes = set()
            for match in re.finditer(r'class="([^"]*)"', html):
                for cls in match.group(1).split():
                    if len(cls) > 2 and not any(c.isdigit() for c in cls[-4:]):
                        classes.add(cls)
            return classes

        classes_a = extract_classes(html_a)
        classes_b = extract_classes(html_b)
        common = classes_a & classes_b
        return sorted(common, key=len)[:top_k]

    @staticmethod
    def _extract_section(html: str, max_chars: int = 2000) -> str:
        """Extract concise body HTML for prompt context."""
        body_match = re.search(
            r"<body[^>]*>(.*)</body>", html, re.DOTALL | re.IGNORECASE
        )
        body = body_match.group(1) if body_match else html
        body = re.sub(r"<script[^>]*>.*?</script>", "", body, flags=re.DOTALL | re.IGNORECASE)
        body = re.sub(r"<style[^>]*>.*?</style>", "", body, flags=re.DOTALL | re.IGNORECASE)
        body = re.sub(r"\s+", " ", body).strip()
        return body[:max_chars]
