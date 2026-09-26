"""Cog Enhanced XPath Generator — Multi-Hypothesis Stage 4.

Paper contribution: Diverse Hypothesis Generation with Structural Verification.

Key differences from base Cog generator:
- Stage 3: Same as Cog (proven reliable)
- Stage 4: Generates K=3 diverse XPath candidates in ONE LLM call
           Then uses StructuralVerifier to select the best via cross-page scoring
           No LLM cost increase (single call produces multiple hypotheses)
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from configs.config import VGSConfig
from utils.llm_client import LLMClient
from utils.browser import BrowserManager
from utils.html_processor import HTMLProcessor
from vgs.attribute_identification import AttributeIdentifier
from vgs.visual_grounding import VisualGrounder
from vgs.element_pinpointing import ElementPinpointer
from cog.dom_analyzer import DOMPatternAnalyzer, StructuralContextBuilder
from cog.prompts import (
    MULTI_CANDIDATE_SYNTHESIS_PROMPT,
    MULTI_CANDIDATE_LINK_PROMPT,
    MULTI_CANDIDATE_IMAGE_PROMPT,
)

logger = logging.getLogger(__name__)

# Keywords for modality detection
_LINK_KEYWORDS = {"link", "url", "href", "hyperlink", "redirect", "homepage", "website"}
_IMAGE_KEYWORDS = {"image", "photo", "picture", "thumbnail", "logo", "icon", "banner",
                   "poster", "fanart", "artwork", "cover", "avatar", "badge",
                   "flag", "screenshot", "gallery"}


class EnhancedXPathGenerator:
    """Cog Generator: Multi-hypothesis generation with structural verification.

    Stage 1: Attribute Identification (same as Cog)
    Stage 2: Visual Grounding (same as Cog)
    Stage 3: Element Pinpointing (same as Cog — reliable hardcoded selectors)
    Stage 4: Multi-Candidate XPath Synthesis (K=3 diverse hypotheses)
    """

    def __init__(
        self,
        config: VGSConfig,
        llm: LLMClient,
        browser: BrowserManager,
        enable_monitor: bool = False,
    ):
        self.config = config
        self.llm = llm
        self.browser = browser
        self.identifier = AttributeIdentifier(llm)
        self.grounder = VisualGrounder(llm)
        self.pinpointer = ElementPinpointer(llm, browser)
        self.context_builder = StructuralContextBuilder()

    def identify_attributes(self, query: str) -> list[str]:
        """Stage 1: identify target attributes from query."""
        attributes = self.identifier.identify(query)
        logger.info("[Generator+] Stage 1: attributes = %s", attributes)
        return attributes

    async def generate(
        self,
        page,
        attribute: str,
        screenshot_dir: Path,
    ) -> tuple[str, dict]:
        """Run enhanced Stage 2-3-4 → returns (best_xpath, metadata).

        Stage 4 now returns MULTIPLE candidates. The pipeline_cog will
        use StructuralVerifier to select the best one.
        """
        metadata = {"attribute": attribute, "enhanced": True}

        # Get full HTML
        full_html = await page.content()

        # ── Stage 2: Visual Grounding (unchanged) ──
        region_paths = await self.browser.capture_regions(page, screenshot_dir)
        matched_region = self.grounder.ground(attribute, region_paths)
        region_index = int(matched_region.stem.split("_")[-1])
        scroll_y = region_index * self.config.viewport_height
        await page.evaluate(f"window.scrollTo(0, {scroll_y})")
        await asyncio.sleep(0.3)

        metadata["stage2"] = {
            "matched_region": region_index,
            "region_path": str(matched_region),
        }
        logger.info("[Generator+] Stage 2: matched region %d", region_index)

        # ── Stage 3: Element Pinpointing (SAME as base Cog) ──
        selected_ids = await self.pinpointer.pinpoint(
            page, attribute, region_index, screenshot_dir,
        )
        modality = self.pinpointer.classify_modality(attribute)
        css_selector = self.pinpointer._modality_to_selector(modality)
        marked_path = screenshot_dir / f"region_{region_index}_marked.png"

        metadata["stage3"] = {
            "modality": modality,
            "selected_ids": selected_ids,
            "css_selector": css_selector,
        }
        logger.info(
            "[Generator+] Stage 3: modality=%s, selected_ids=%s", modality, selected_ids
        )

        # ── Stage 4: Multi-Candidate XPath Synthesis ──
        # Build structural context for context injection
        tags = self._css_to_tags(css_selector)
        elements = HTMLProcessor.get_elements_by_multi_tags(full_html, tags)
        element_xpaths = [
            elements[sid]["xpath"]
            for sid in selected_ids
            if sid < len(elements)
        ]

        structural_context = self.context_builder.build_from_elements(
            full_html, element_xpaths
        )
        context_text = structural_context.format()

        # Generate K=3 diverse XPath candidates
        candidates = await self._generate_multi_candidates(
            page, attribute, selected_ids, css_selector,
            marked_path, full_html, context_text,
        )

        metadata["stage4"] = {
            "candidates": candidates,
            "structural_context": context_text[:300],
        }
        logger.info(
            "[Generator+] Stage 4: %d candidates generated: %s",
            len(candidates), candidates,
        )

        # Return the first candidate as default (pipeline will verify all)
        best_xpath = candidates[0] if candidates else ""
        return best_xpath, metadata

    async def generate_candidates(
        self,
        page,
        attribute: str,
        screenshot_dir: Path,
    ) -> tuple[list[str], dict]:
        """Generate multiple XPath candidates (called by pipeline_cog).

        Returns (candidates_list, metadata) instead of single (xpath, metadata).
        """
        metadata = {"attribute": attribute, "enhanced": True}
        full_html = await page.content()

        # Stage 2
        region_paths = await self.browser.capture_regions(page, screenshot_dir)
        matched_region = self.grounder.ground(attribute, region_paths)
        region_index = int(matched_region.stem.split("_")[-1])
        scroll_y = region_index * self.config.viewport_height
        await page.evaluate(f"window.scrollTo(0, {scroll_y})")
        await asyncio.sleep(0.3)

        metadata["stage2"] = {"matched_region": region_index}
        logger.info("[Generator+] Stage 2: matched region %d", region_index)

        # Stage 3
        selected_ids = await self.pinpointer.pinpoint(
            page, attribute, region_index, screenshot_dir,
        )
        modality = self.pinpointer.classify_modality(attribute)
        css_selector = self.pinpointer._modality_to_selector(modality)
        marked_path = screenshot_dir / f"region_{region_index}_marked.png"

        metadata["stage3"] = {
            "modality": modality,
            "selected_ids": selected_ids,
        }
        logger.info("[Generator+] Stage 3: modality=%s, ids=%s", modality, selected_ids)

        # Stage 4: Multi-candidate
        tags = self._css_to_tags(css_selector)
        elements = HTMLProcessor.get_elements_by_multi_tags(full_html, tags)
        element_xpaths = [
            elements[sid]["xpath"]
            for sid in selected_ids
            if sid < len(elements)
        ]

        structural_context = self.context_builder.build_from_elements(
            full_html, element_xpaths
        )
        context_text = structural_context.format()

        candidates = await self._generate_multi_candidates(
            page, attribute, selected_ids, css_selector,
            marked_path, full_html, context_text,
        )

        metadata["stage4"] = {"candidates": candidates}
        logger.info("[Generator+] Stage 4: %d candidates: %s", len(candidates), candidates)

        return candidates, metadata

    async def _generate_multi_candidates(
        self,
        page,
        attribute: str,
        selected_ids: list[int],
        css_selector: str,
        marked_screenshot: Path,
        full_html: str,
        structural_context: str,
    ) -> list[str]:
        """Stage 4: Generate K=3 diverse XPath candidates in one LLM call."""
        # Collect HTML segments
        tags = self._css_to_tags(css_selector)
        elements = HTMLProcessor.get_elements_by_multi_tags(full_html, tags)
        segments: list[str] = []

        for box_id in selected_ids:
            if box_id >= len(elements):
                continue
            elem_info = elements[box_id]
            xpath_guess = elem_info["xpath"]
            segment = HTMLProcessor.extract_local_segment(
                full_html, xpath_guess, distance=2
            )
            if segment:
                segments.append(f"<!-- Element {box_id} -->\n{segment}")

        html_segments = "\n\n".join(segments) if segments else "(no segments)"

        # Select prompt
        prompt_template = self._select_multi_prompt(attribute)
        prompt = prompt_template.format(
            attribute=attribute,
            html_segments=html_segments,
            structural_context=structural_context,
        )

        logger.info(
            "[Generator+] Stage 4: sending multi-candidate prompt (%d chars)",
            len(prompt),
        )

        result = self.llm.vision_query(
            prompt, [marked_screenshot],
            label="Stage4+·MultiXPath",
        )

        # Parse multiple candidates from LLM response
        candidates = []
        for key in ["xpath_1", "xpath_2", "xpath_3"]:
            xpath = result.get(key, "")
            if xpath and xpath.strip():
                candidates.append(xpath.strip())

        # Fallback: if LLM returned single "xpath" key
        if not candidates and result.get("xpath"):
            candidates.append(result["xpath"])

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        return unique if unique else [""]

    @staticmethod
    def _select_multi_prompt(attribute: str) -> str:
        """Select modality-aware multi-candidate prompt."""
        attr_lower = attribute.lower().replace("_", " ").replace("-", " ")
        tokens = set(attr_lower.split())
        if tokens & _LINK_KEYWORDS:
            return MULTI_CANDIDATE_LINK_PROMPT
        if tokens & _IMAGE_KEYWORDS:
            return MULTI_CANDIDATE_IMAGE_PROMPT
        return MULTI_CANDIDATE_SYNTHESIS_PROMPT

    @staticmethod
    def _css_to_tags(css_selector: str) -> list[str]:
        """Parse CSS selector into tag names."""
        if css_selector == "img":
            return ["img"]
        if css_selector == "a[href]":
            return ["a"]
        tags = set()
        for part in css_selector.split(","):
            part = part.strip()
            for token in part.split():
                tag = token.split(".")[0].split("[")[0].split("#")[0]
                if tag and tag.isalpha():
                    tags.add(tag)
        return list(tags) if tags else ["p", "span", "h1", "h2", "h3", "div"]
