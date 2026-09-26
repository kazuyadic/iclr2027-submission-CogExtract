"""Stage 3: Element Pinpointing — locate exact elements via Set-of-Mark.

For cached pages (no browser), uses text-based element selection from HTML
instead of visual bounding box injection.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from utils.llm_client import LLMClient
from utils.browser import BrowserManager
from utils.html_processor import HTMLProcessor
from vgs.prompts import (
    ELEMENT_SCANNING_PROMPT,
    ELEMENT_SELECTION_PROMPT,
    TEXT_ELEMENT_SELECTION_PROMPT,
)


class ElementPinpointer:
    """Two-step element pinpointing: scanning then selection."""

    IMAGE_KEYWORDS = {
        "image", "photo", "picture", "thumbnail", "logo", "icon",
        "banner", "poster", "fanart", "artwork", "cover", "avatar",
        "badge", "flag", "screenshot", "gallery",
    }
    LINK_KEYWORDS = {
        "link", "url", "href", "hyperlink", "redirect",
        "profile link", "homepage", "website",
    }

    def __init__(self, llm: LLMClient, browser: BrowserManager):
        self.llm = llm
        self.browser = browser

    def classify_modality(self, attribute: str) -> str:
        lower = attribute.lower()
        if any(kw in lower for kw in self.IMAGE_KEYWORDS):
            return "image"
        if any(kw in lower for kw in self.LINK_KEYWORDS):
            return "link"
        return "text"

    async def pinpoint(self, page, attribute: str,
                       region_index: int, screenshot_dir: Path) -> list[int]:
        """Run the two-step pinpointing and return selected bounding-box IDs.

        For cached pages (_is_cached=True), uses text-based element selection
        from HTML — no browser needed.
        """
        is_cached = getattr(page, '_is_cached', False)

        if is_cached:
            return await self._pinpoint_offline(
                page, attribute, region_index, screenshot_dir
            )

        return await self._pinpoint_visual(
            page, attribute, region_index, screenshot_dir
        )

    async def _pinpoint_visual(self, page, attribute: str,
                               region_index: int, screenshot_dir: Path) -> list[int]:
        """Original visual pinpointing with Set-of-Mark bounding boxes."""
        modality = self.classify_modality(attribute)
        css_selector = self._modality_to_selector(modality)

        # Step 1 — Scanning
        region_screenshot = screenshot_dir / f"region_{region_index}.png"
        scan_prompt = ELEMENT_SCANNING_PROMPT.format(attribute=attribute)
        print(f"\n  [Stage3] Modality: {modality}, CSS selector: {css_selector}")
        scan_result = self.llm.vision_query(scan_prompt, [region_screenshot],
                                            label="Stage3·Scanning")
        candidate_items = scan_result.get("items", [])
        print(f"  [Stage3] Scan found {len(candidate_items)} candidate items: {candidate_items}")

        # Step 2 — Inject Set-of-Mark bounding boxes and let VLM select
        await self.browser.inject_bounding_boxes(page, css_selector)
        marked_path = screenshot_dir / f"region_{region_index}_marked.png"
        await page.screenshot(path=str(marked_path))
        print(f"  [Stage3] Marked screenshot saved: {marked_path}")

        select_prompt = ELEMENT_SELECTION_PROMPT.format(attribute=attribute)
        select_result = self.llm.vision_query(select_prompt, [marked_path],
                                              label="Stage3·Selection")
        selected_ids = select_result.get("selected_ids", [])

        await self.browser.remove_bounding_boxes(page)
        return [int(sid) for sid in selected_ids]

    async def _pinpoint_offline(self, page, attribute: str,
                                region_index: int, screenshot_dir: Path) -> list[int]:
        """Text-based element pinpointing for cached pages (no browser).

        Uses lxml to find candidate elements from HTML, then asks the LLM
        to select elements based on text descriptions + the region screenshot.
        """
        modality = self.classify_modality(attribute)
        css_selector = self._modality_to_selector(modality)
        tags = self._css_to_tags(css_selector)

        # Get all elements matching the selector in document order
        html = await page.content()
        elements = HTMLProcessor.get_elements_by_multi_tags(html, tags)

        # Build concise text descriptions for each element
        descriptions = []
        from lxml import etree
        tree = etree.HTML(html)

        for el_info in elements:
            desc = f"[{el_info['index']}] <{el_info['tag']}"
            for attr in ['class', 'id', 'href', 'src', 'alt']:
                if attr in el_info:
                    val = str(el_info[attr])[:80]
                    desc += f' {attr}="{val}"'
            desc += ">"
            # Add text content snippet from xpath
            if tree is not None:
                try:
                    nodes = tree.xpath(el_info.get('xpath', ''))
                    if nodes:
                        text = (nodes[0].text or '').strip()[:60]
                        if text:
                            desc += f" {text}"
                except Exception:
                    pass
            descriptions.append(desc)

        # Limit to reasonable number for prompt
        element_list = "\n".join(descriptions[:80])
        total_count = len(descriptions)

        print(f"\n  [Stage3·Offline] Modality: {modality}, {total_count} candidate elements")

        # Step 1 — Use LLM with region screenshot + text element list
        region_screenshot = screenshot_dir / f"region_{region_index}.png"
        prompt = TEXT_ELEMENT_SELECTION_PROMPT.format(
            attribute=attribute,
            total_count=total_count,
            element_list=element_list,
        )
        result = self.llm.vision_query(
            prompt, [region_screenshot],
            label="Stage3·OfflineSelection",
        )
        selected_ids = result.get("selected_ids", [])

        print(f"  [Stage3·Offline] Selected IDs: {selected_ids}")

        # Copy region screenshot as "marked" version for Stage 4
        marked_path = screenshot_dir / f"region_{region_index}_marked.png"
        region_screenshot = screenshot_dir / f"region_{region_index}.png"
        if region_screenshot.exists():
            shutil.copy2(region_screenshot, marked_path)

        return [int(sid) for sid in selected_ids]

    @staticmethod
    def _modality_to_selector(modality: str) -> str:
        if modality == "image":
            return "img"
        if modality == "link":
            return "a[href]"
        return "p, span, h1, h2, h3, h4, h5, h6, li, td, th, dd, dt, label"

    @staticmethod
    def _css_to_tags(css_selector: str) -> list[str]:
        """Parse a CSS selector into a list of tag names."""
        if css_selector == "img":
            return ["img"]
        if css_selector == "a[href]":
            return ["a"]
        return [tag.strip().split("[")[0] for tag in css_selector.split(",")]