"""Stage 4: XPath Synthesis — generate reusable XPaths from pinpointed elements."""
from __future__ import annotations

from pathlib import Path

from utils.llm_client import LLMClient
from utils.html_processor import HTMLProcessor
from vgs.prompts import (
    XPATH_SYNTHESIS_PROMPT,
    XPATH_SYNTHESIS_LINK_PROMPT,
    XPATH_SYNTHESIS_IMAGE_PROMPT,
)

# Keywords that indicate link or image modality
_LINK_KEYWORDS = {"link", "url", "href", "hyperlink", "redirect", "homepage", "website"}
_IMAGE_KEYWORDS = {"image", "photo", "picture", "thumbnail", "logo", "icon", "banner",
                    "poster", "fanart", "artwork", "cover", "avatar", "badge",
                    "flag", "screenshot", "gallery"}


class XPathSynthesizer:
    """Synthesize generalizable XPaths from validated bounding boxes."""

    def __init__(self, llm: LLMClient, neighbor_distance: int = 2):
        self.llm = llm
        self.neighbor_distance = neighbor_distance

    @staticmethod
    def _select_prompt(attribute: str) -> str:
        """Select modality-aware prompt based on attribute name."""
        attr_lower = attribute.lower().replace("_", " ").replace("-", " ")
        tokens = set(attr_lower.split())
        if tokens & _LINK_KEYWORDS:
            return XPATH_SYNTHESIS_LINK_PROMPT
        if tokens & _IMAGE_KEYWORDS:
            return XPATH_SYNTHESIS_IMAGE_PROMPT
        return XPATH_SYNTHESIS_PROMPT

    async def synthesize(self, page, attribute: str,
                         selected_ids: list[int], css_selector: str,
                         marked_screenshot: Path) -> str:
        """Generate a single XPath for the given attribute.

        Parameters
        ----------
        page : Playwright Page
        attribute : target attribute name
        selected_ids : bounding-box IDs from Element Pinpointing
        css_selector : CSS selector used during pinpointing
        marked_screenshot : screenshot with Set-of-Mark annotations
        """
        full_html = await page.content()
        html_segments = self._collect_local_segments(
            full_html, css_selector, selected_ids
        )

        prompt_template = self._select_prompt(attribute)
        prompt = prompt_template.format(
            attribute=attribute,
            html_segments=html_segments,
        )
        print(f"\n  [Stage4] HTML segments ({len(html_segments)} chars):")
        print(html_segments[:1000] + ("…(truncated)" if len(html_segments) > 1000 else ""))

        result = self.llm.vision_query(prompt, [marked_screenshot],
                                       label="Stage4·XPathSynth")
        return result.get("xpath", "")

    def _collect_local_segments(self, html: str, css_selector: str,
                                selected_ids: list[int]) -> str:
        """Build local HTML segments around each selected element."""
        tags = self._css_to_tags(css_selector)
        elements = HTMLProcessor.get_elements_by_multi_tags(html, tags)
        segments: list[str] = []

        for box_id in selected_ids:
            if box_id >= len(elements):
                continue
            elem_info = elements[box_id]
            xpath_guess = elem_info["xpath"]
            segment = HTMLProcessor.extract_local_segment(
                html, xpath_guess, self.neighbor_distance
            )
            if segment:
                segments.append(f"<!-- Element {box_id} -->\n{segment}")

        return "\n\n".join(segments) if segments else "(no segments extracted)"

    @staticmethod
    def _css_to_tags(css_selector: str) -> list[str]:
        """Parse a CSS selector into a list of tag names."""
        if css_selector == "img":
            return ["img"]
        if css_selector == "a[href]":
            return ["a"]
        # Multi-tag selector like "p, span, h1, h2, ..."
        return [tag.strip().split("[")[0] for tag in css_selector.split(",")]
