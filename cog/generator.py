"""Stage 1: XPath Generator — reuses VGS Stage 1-4 to generate initial XPaths."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from configs.config import VGSConfig
from utils.llm_client import LLMClient
from utils.browser import BrowserManager
from vgs.attribute_identification import AttributeIdentifier
from vgs.visual_grounding import VisualGrounder
from vgs.element_pinpointing import ElementPinpointer
from vgs.xpath_synthesis import XPathSynthesizer

logger = logging.getLogger(__name__)


class XPathGenerator:
    """Generate initial XPaths using VGS Stage 1-4 on a seed page.

    Stage 1: Attribute Identification — LLM parses query → attribute list
    Stage 2: Visual Grounding — VLM matches attribute to page region
    Stage 3: Element Pinpointing — Set-of-Mark annotation → VLM selects elements
    Stage 4: XPath Synthesis — VLM generates XPath from selected elements + HTML
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
        self.synthesizer = XPathSynthesizer(llm, config.neighbor_distance)

    def identify_attributes(self, query: str) -> list[str]:
        """Stage 1: identify target attributes from query."""
        attributes = self.identifier.identify(query)
        logger.info("[Generator] Stage 1: attributes = %s", attributes)
        return attributes

    async def generate(
        self,
        page,
        attribute: str,
        screenshot_dir: Path,
    ) -> tuple[str, dict]:
        """Run VGS Stage 2-3-4 on page for one attribute → (xpath, metadata).

        Returns:
            xpath: generated XPath string
            metadata: dict with intermediate results for tracing
        """
        metadata = {"attribute": attribute}

        # Stage 2: Visual Grounding
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
        logger.info("[Generator] Stage 2: matched region %d", region_index)

        # Stage 3: Element Pinpointing
        selected_ids = await self.pinpointer.pinpoint(
            page, attribute, region_index, screenshot_dir,
        )
        modality = self.pinpointer.classify_modality(attribute)
        css_selector = self.pinpointer._modality_to_selector(modality)
        marked_path = screenshot_dir / f"region_{region_index}_marked.png"

        metadata["stage3"] = {
            "modality": modality,
            "selected_ids": selected_ids,
            "marked_path": str(marked_path),
        }
        logger.info(
            "[Generator] Stage 3: modality=%s, selected_ids=%s", modality, selected_ids
        )

        # Stage 4: XPath Synthesis
        xpath = await self.synthesizer.synthesize(
            page, attribute, selected_ids, css_selector, marked_path,
        )
        metadata["stage4"] = {"xpath": xpath}
        logger.info("[Generator] Stage 4: xpath = %s", xpath)

        return xpath, metadata
