"""Stage 2: Visual Grounding — locate the relevant region for each attribute."""
from __future__ import annotations

from pathlib import Path

from utils.llm_client import LLMClient
from vgs.prompts import VISUAL_GROUNDING_PROMPT


class VisualGrounder:
    """Use a VLM to find which viewport region contains the target attribute."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def ground(self, attribute: str, region_paths: list[Path]) -> Path:
        """Return the path of the region screenshot that best matches *attribute*."""
        prompt = VISUAL_GROUNDING_PROMPT.format(attribute=attribute)
        labels = [f"Region {i}" for i in range(len(region_paths))]

        result = self.llm.vision_query(prompt, region_paths, image_labels=labels,
                                       label="Stage2·VisualGround")
        matching = result.get("matching_region", "")

        # Parse "Region X" → index
        region_index = self._parse_region_id(matching, len(region_paths))
        return region_paths[region_index]

    @staticmethod
    def _parse_region_id(raw: str, total: int) -> int:
        """Extract integer region index from LLM output like 'Region 2'."""
        import re
        match = re.search(r"(\d+)", str(raw))
        if match:
            idx = int(match.group(1))
            if 0 <= idx < total:
                return idx
        return 0
