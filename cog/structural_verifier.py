"""Structural Consistency Verifier for XPath candidates.

Paper contribution: Cross-page structural verification.
Instead of relying on the LLM to judge XPath quality, we use deterministic
DOM structure signals to score and rank XPath candidates.

Scoring dimensions:
1. Cross-page Coverage: XPath must extract values on BOTH pages
2. Count Stability: Similar extraction counts across pages (not wildly different)
3. Depth Consistency: Matched elements sit at similar DOM depths
4. Semantic Coherence: Extracted values share textual characteristics
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from lxml import etree

logger = logging.getLogger(__name__)


@dataclass
class CandidateScore:
    """Score breakdown for one XPath candidate."""
    xpath: str
    values_a: list[str] = field(default_factory=list)
    values_b: list[str] = field(default_factory=list)
    count_a: int = 0
    count_b: int = 0
    coverage_score: float = 0.0      # Both pages have hits
    stability_score: float = 0.0      # Count ratio similarity
    depth_score: float = 0.0          # DOM depth consistency
    coherence_score: float = 0.0      # Value length consistency
    plausibility_score: float = 0.0   # Values look like the target attribute
    specificity_score: float = 0.0    # Prefer precise (fewer matches) XPaths
    total_score: float = 0.0

    @property
    def passes_basic(self) -> bool:
        """Candidate works on both pages."""
        return self.count_a > 0 and self.count_b > 0


class StructuralVerifier:
    """Score and rank XPath candidates by cross-page structural consistency.

    Design rationale:
    - A correct XPath extracts semantically similar content at similar DOM positions
      across different pages of the same website.
    - We exploit this invariance as a verification signal WITHOUT needing ground truth.
    - We also use a semantic plausibility check: extracted values should look like
      the target attribute (e.g., titles should be multi-word text, not single labels).
    """

    def __init__(self, weight_coverage: float = 0.30,
                 weight_stability: float = 0.20,
                 weight_depth: float = 0.10,
                 weight_coherence: float = 0.10,
                 weight_plausibility: float = 0.15,
                 weight_specificity: float = 0.15):
        self.w_cov = weight_coverage
        self.w_stab = weight_stability
        self.w_dep = weight_depth
        self.w_coh = weight_coherence
        self.w_plaus = weight_plausibility
        self.w_spec = weight_specificity

    def score_candidates(
        self,
        candidates: list[str],
        html_a: str,
        html_b: str,
        attribute: str = "",
    ) -> list[CandidateScore]:
        """Score all candidates and return sorted (best first)."""
        tree_a = etree.HTML(html_a)
        tree_b = etree.HTML(html_b)

        if tree_a is None or tree_b is None:
            return []

        scores = []
        for xpath in candidates:
            if not xpath or not xpath.strip():
                continue
            score = self._score_one(xpath, tree_a, tree_b, html_a, html_b, attribute)
            scores.append(score)

        # Sort by total_score descending
        scores.sort(key=lambda s: s.total_score, reverse=True)
        return scores

    def select_best(
        self,
        candidates: list[str],
        html_a: str,
        html_b: str,
        attribute: str = "",
    ) -> Optional[CandidateScore]:
        """Select the best candidate that passes basic validation."""
        scored = self.score_candidates(candidates, html_a, html_b, attribute)
        for s in scored:
            if s.passes_basic:
                return s
        # If none pass basic, return highest-scoring anyway (for reflection)
        return scored[0] if scored else None

    def _score_one(
        self, xpath: str,
        tree_a, tree_b,
        html_a: str, html_b: str,
        attribute: str = "",
    ) -> CandidateScore:
        """Compute composite score for one XPath candidate."""
        score = CandidateScore(xpath=xpath)

        # Extract values
        score.values_a = self._extract(tree_a, xpath)
        score.values_b = self._extract(tree_b, xpath)
        score.count_a = len(score.values_a)
        score.count_b = len(score.values_b)

        # 1. Coverage score: binary — do both pages yield results?
        if score.count_a > 0 and score.count_b > 0:
            score.coverage_score = 1.0
        elif score.count_a > 0 or score.count_b > 0:
            score.coverage_score = 0.3  # partial credit
        else:
            score.coverage_score = 0.0

        # 2. Count stability: |count_A - count_B| / max(count_A, count_B)
        if score.count_a > 0 and score.count_b > 0:
            max_c = max(score.count_a, score.count_b)
            min_c = min(score.count_a, score.count_b)
            score.stability_score = min_c / max_c
        else:
            score.stability_score = 0.0

        # 3. Depth consistency: similar average DOM depth
        depths_a = self._get_depths(tree_a, xpath)
        depths_b = self._get_depths(tree_b, xpath)
        if depths_a and depths_b:
            avg_a = sum(depths_a) / len(depths_a)
            avg_b = sum(depths_b) / len(depths_b)
            max_depth = max(avg_a, avg_b, 1)
            depth_diff = abs(avg_a - avg_b) / max_depth
            score.depth_score = max(0, 1.0 - depth_diff)
        else:
            score.depth_score = 0.0

        # 4. Semantic coherence: value lengths are consistent
        if score.values_a and score.values_b:
            avg_len_a = sum(len(v) for v in score.values_a) / len(score.values_a)
            avg_len_b = sum(len(v) for v in score.values_b) / len(score.values_b)
            max_len = max(avg_len_a, avg_len_b, 1)
            len_ratio = min(avg_len_a, avg_len_b) / max_len
            score.coherence_score = len_ratio
        else:
            score.coherence_score = 0.0

        # 5. Plausibility: extracted values look like the target attribute
        all_values = score.values_a + score.values_b
        score.plausibility_score = self._plausibility(all_values, attribute)

        # 6. Specificity: prefer XPaths that match fewer elements (more precise)
        # Rationale: A correct XPath targets the minimal DOM subtree containing
        # the desired value. Overly broad XPaths (e.g., //*) match many unrelated
        # elements. We reward precision through inverse count scoring.
        score.specificity_score = self._specificity(score.count_a, score.count_b)

        # Composite
        score.total_score = (
            self.w_cov * score.coverage_score +
            self.w_stab * score.stability_score +
            self.w_dep * score.depth_score +
            self.w_coh * score.coherence_score +
            self.w_plaus * score.plausibility_score +
            self.w_spec * score.specificity_score
        )

        return score

    @staticmethod
    def _plausibility(values: list[str], attribute: str) -> float:
        """Score how plausible the extracted values are for the given attribute.

        Heuristics:
        - Labels/descriptors are short (< 15 chars) and often end with ':'
        - Actual content values tend to be longer
        - For 'title' attributes, expect multi-word text (> 20 chars average)
        - For 'link/url', expect strings starting with http or /
        - Penalize extractions where ALL values are very short labels
        """
        if not values or not attribute:
            return 0.5  # neutral when no info

        attr_lower = attribute.lower().replace("_", " ")

        # Check for label-like values (short, ending with colon)
        label_count = sum(
            1 for v in values
            if len(v.strip()) < 15 and (v.strip().endswith(":") or v.strip().endswith("："))
        )
        label_ratio = label_count / len(values) if values else 0

        # If most values look like labels, penalize heavily
        if label_ratio > 0.5:
            return 0.1

        # Average value length
        avg_len = sum(len(v) for v in values) / len(values)

        # For title-like attributes, prefer longer values
        if any(kw in attr_lower for kw in ["title", "name", "description", "abstract"]):
            if avg_len < 10:
                return 0.2  # too short for a title
            elif avg_len < 30:
                return 0.5
            else:
                return 1.0

        # For link attributes
        if any(kw in attr_lower for kw in ["link", "url", "href"]):
            url_count = sum(1 for v in values if v.startswith(("http", "/", "#")))
            return url_count / len(values) if values else 0.5

        # For image attributes
        if any(kw in attr_lower for kw in ["image", "img", "photo", "thumbnail"]):
            img_count = sum(1 for v in values if v.startswith(("http", "/", "data:")) or v.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")))
            return img_count / len(values) if values else 0.5

        # Generic: prefer non-trivial length
        if avg_len < 5:
            return 0.3
        elif avg_len < 20:
            return 0.6
        else:
            return 0.9

    @staticmethod
    def _specificity(count_a: int, count_b: int) -> float:
        """Adaptive specificity — auto-detect detail vs listing tasks.

        Key insight: if BOTH pages yield similar high counts (>10), the task
        is likely a listing/collection extraction. In that case, high count
        is CORRECT behavior, not over-extraction.

        Detection logic:
        - count_a ≈ count_b AND both > 10 → listing task → reward stability
        - count_a ≈ count_b AND both 4-10 → ambiguous → neutral score
        - count ≤ 3 → detail task → reward precision
        - count_a >> count_b → unstable → penalize (over-extraction signal)
        """
        if count_a == 0 or count_b == 0:
            return 0.0

        avg_count = (count_a + count_b) / 2
        min_c = min(count_a, count_b)
        max_c = max(count_a, count_b)
        ratio = min_c / max_c  # 1.0 = perfectly stable

        # Listing detection: both pages have >10 items with very stable counts
        if avg_count > 10 and ratio >= 0.7:
            # Clearly a listing task → count stability IS the quality signal
            return 0.8 * ratio + 0.2

        # Detail task: prefer minimal count
        if avg_count <= 1:
            return 1.0
        elif avg_count <= 3:
            return 0.7
        elif avg_count <= 10:
            # Ambiguous zone: could be listing or over-extraction
            # Use ratio as tiebreaker: very stable → tolerable, unstable → penalize
            if ratio >= 0.8:
                return 0.5  # stable but uncertain — neutral
            else:
                return 0.3  # unstable → likely over-extraction
        else:
            # High count (>10) but unstable → over-extraction
            return 0.2 * ratio

    @staticmethod
    def _extract(tree, xpath: str, limit: int = 20) -> list[str]:
        """Extract text values using lxml."""
        try:
            nodes = tree.xpath(xpath)
        except Exception:
            return []

        results = []
        for node in nodes[:limit]:
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
                results.append(text[:300])
        return results

    @staticmethod
    def _get_depths(tree, xpath: str, limit: int = 20) -> list[int]:
        """Get DOM depths of matched elements."""
        try:
            nodes = tree.xpath(xpath)
        except Exception:
            return []

        depths = []
        for node in nodes[:limit]:
            if not hasattr(node, "getparent"):
                continue
            depth = 0
            parent = node.getparent()
            while parent is not None:
                depth += 1
                parent = parent.getparent()
            depths.append(depth)
        return depths
