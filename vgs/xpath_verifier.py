"""XPath Self-Verify & Repair — multi-page validation and automatic XPath refinement.

Key insight: VGS generates XPath from a single seed page, but never validates if it
generalizes to other pages in the same group. This module:
1. Verifies XPath on 2-3 additional pages
2. If verification fails, applies repair strategies:
   - Relaxation: remove overly specific position/class constraints
   - Ancestor generalization: go up one level and re-target
   - LLM re-synthesis: regenerate with failure context
3. Votes on the most stable XPath variant
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

XPATH_REPAIR_PROMPT = """You are an expert XPath debugger. The original XPath failed to extract values from some pages.

Original XPath: {original_xpath}
Target attribute: {attribute}

Seed page extraction (worked): {seed_values}
Failed pages: {failure_info}

Candidate relaxed variants that partially work:
{candidate_info}

HTML snippet from a verification page (showing relevant structure):
{html_context}

Task: Generate an improved XPath that will work across ALL pages in this website.
Consider:
- The original XPath may target elements that don't exist — check the HTML snippet above
- Class names or structural patterns may be more stable than exact paths
- If the target is inside a label-value pair, anchor on the label text
- If the attribute info is in meta tags, use //meta[@name='..']/@content patterns

Output JSON:
{{
    "xpath": "the improved xpath",
    "reasoning": "why this will generalize better"
}}"""


@dataclass
class VerifyResult:
    """Result of XPath verification across multiple pages."""
    xpath: str
    success_rate: float  # 0.0 to 1.0
    total_pages: int
    successful_pages: int
    values_per_page: dict = field(default_factory=dict)  # page_id -> values
    is_valid: bool = False


@dataclass
class RepairCandidate:
    """A candidate XPath from repair strategies."""
    xpath: str
    strategy: str  # "relaxed", "ancestor", "llm_regen"
    success_rate: float = 0.0
    values_per_page: dict = field(default_factory=dict)


class XPathVerifier:
    """Verify and repair XPaths using multi-page validation."""

    # Minimum success rate to consider an XPath valid
    VALIDITY_THRESHOLD = 0.6
    # Number of verification pages
    VERIFY_PAGE_COUNT = 3

    def __init__(self, llm: LLMClient, browser_manager=None):
        self.llm = llm
        self.browser = browser_manager

    async def verify_and_repair(
        self,
        xpath: str,
        attribute: str,
        seed_page,
        verification_pages: list,
        seed_values: list[str] = None,
    ) -> tuple[str, VerifyResult]:
        """Verify XPath on multiple pages and repair if needed.

        Parameters
        ----------
        xpath : the candidate XPath to verify
        attribute : target attribute name
        seed_page : Playwright page where XPath was generated (already loaded)
        verification_pages : list of Playwright pages to verify on
        seed_values : values extracted from seed page (for reference)

        Returns
        -------
        (best_xpath, verify_result) : the best XPath and its verification stats
        """
        if not xpath:
            return xpath, VerifyResult(xpath="", success_rate=0.0, total_pages=0,
                                       successful_pages=0, is_valid=False)

        # Step 1: Get seed values if not provided
        if seed_values is None:
            seed_values = await self.browser.execute_xpath(seed_page, xpath)

        # Step 2: Verify on other pages
        verify_result = await self._verify_xpath(xpath, verification_pages)
        logger.info("  [Verify] %s: success_rate=%.0f%% (%d/%d pages)",
                    attribute, verify_result.success_rate * 100,
                    verify_result.successful_pages, verify_result.total_pages)

        # Step 3: If valid, return as-is
        if verify_result.is_valid:
            return xpath, verify_result

        # Step 4: Repair — try relaxation strategies
        logger.info("  [Repair] XPath failed verification, attempting repair …")
        repaired_xpath, repair_result = await self._repair_xpath(
            xpath, attribute, seed_values, verification_pages
        )

        if repair_result and repair_result.is_valid:
            logger.info("  [Repair] ✓ Repaired: %s (success=%.0f%%)",
                        repaired_xpath, repair_result.success_rate * 100)
            return repaired_xpath, repair_result

        # Step 5: If repair failed, return the best candidate (even if imperfect)
        if repair_result and repair_result.success_rate > verify_result.success_rate:
            logger.info("  [Repair] Partial improvement: %.0f%% → %.0f%%",
                        verify_result.success_rate * 100, repair_result.success_rate * 100)
            return repaired_xpath, repair_result

        # No improvement — return original
        logger.info("  [Repair] No improvement found, keeping original")
        return xpath, verify_result

    async def _verify_xpath(self, xpath: str, pages: list) -> VerifyResult:
        """Verify XPath across multiple pages."""
        total = len(pages)
        if total == 0:
            return VerifyResult(xpath=xpath, success_rate=1.0, total_pages=0,
                                successful_pages=0, is_valid=True)

        successful = 0
        values_map = {}
        for i, page in enumerate(pages):
            try:
                values = await self.browser.execute_xpath(page, xpath)
                values_map[str(i)] = values
                if values:  # Non-empty result = success
                    successful += 1
            except Exception:
                values_map[str(i)] = []

        success_rate = successful / total if total > 0 else 0.0
        return VerifyResult(
            xpath=xpath,
            success_rate=success_rate,
            total_pages=total,
            successful_pages=successful,
            values_per_page=values_map,
            is_valid=success_rate >= self.VALIDITY_THRESHOLD,
        )

    async def _repair_xpath(
        self,
        original_xpath: str,
        attribute: str,
        seed_values: list[str],
        verification_pages: list,
    ) -> tuple[Optional[str], Optional[VerifyResult]]:
        """Apply repair strategies and return the best candidate."""
        candidates: list[RepairCandidate] = []

        # Strategy 1: Relaxation — remove positional indices
        relaxed_variants = self._generate_relaxed_variants(original_xpath)
        for variant in relaxed_variants:
            result = await self._verify_xpath(variant, verification_pages)
            candidates.append(RepairCandidate(
                xpath=variant, strategy="relaxed",
                success_rate=result.success_rate, values_per_page=result.values_per_page
            ))

        # Strategy 2: Ancestor generalization
        ancestor_variants = self._generate_ancestor_variants(original_xpath)
        for variant in ancestor_variants:
            result = await self._verify_xpath(variant, verification_pages)
            candidates.append(RepairCandidate(
                xpath=variant, strategy="ancestor",
                success_rate=result.success_rate, values_per_page=result.values_per_page
            ))

        # Strategy 3: LLM re-synthesis (only if relaxation didn't work)
        best_relaxed = max(candidates, key=lambda c: c.success_rate) if candidates else None
        if not best_relaxed or best_relaxed.success_rate < self.VALIDITY_THRESHOLD:
            logger.info("  [Repair] Trying LLM re-synthesis …")
            llm_xpath = await self._llm_repair(
                original_xpath, attribute, seed_values,
                verification_pages, candidates
            )
            logger.info("  [Repair] LLM returned: %s", llm_xpath)
            if llm_xpath:
                result = await self._verify_xpath(llm_xpath, verification_pages)
                logger.info("  [Repair] LLM xpath verify: rate=%.0f%%", result.success_rate * 100)
                candidates.append(RepairCandidate(
                    xpath=llm_xpath, strategy="llm_regen",
                    success_rate=result.success_rate, values_per_page=result.values_per_page
                ))

        # Select best candidate
        if not candidates:
            return None, None

        best = max(candidates, key=lambda c: c.success_rate)
        verify_result = VerifyResult(
            xpath=best.xpath,
            success_rate=best.success_rate,
            total_pages=len(verification_pages),
            successful_pages=int(best.success_rate * len(verification_pages)),
            values_per_page=best.values_per_page,
            is_valid=best.success_rate >= self.VALIDITY_THRESHOLD,
        )
        return best.xpath, verify_result

    def _generate_relaxed_variants(self, xpath: str) -> list[str]:
        """Generate XPath variants by removing overly specific constraints."""
        variants = []

        # Remove positional indices: [1], [2], etc.
        no_position = re.sub(r'\[\d+\]', '', xpath)
        if no_position != xpath:
            variants.append(no_position)

        # Remove last() constraints
        no_last = re.sub(r'\[last\(\)[^\]]*\]', '', xpath)
        if no_last != xpath and no_last not in variants:
            variants.append(no_last)

        # Remove specific text() content matches but keep structure
        # e.g., text()[2] → text() or just get parent element
        no_text_idx = re.sub(r'/text\(\)\[\d+\]', '', xpath)
        if no_text_idx != xpath and no_text_idx not in variants:
            variants.append(no_text_idx)

        # Replace contains(@class, 'specific') with just the structural path
        # Only if there are other structural anchors
        simplified_class = re.sub(
            r"\[contains\(@class,\s*'[^']+'\)\s*and\s*", "[", xpath
        )
        if simplified_class != xpath and simplified_class not in variants:
            variants.append(simplified_class)

        # Remove following-sibling::text() → use following-sibling::*
        text_to_elem = xpath.replace('/following-sibling::text()[1]',
                                     '/following-sibling::*[1]')
        if text_to_elem != xpath and text_to_elem not in variants:
            variants.append(text_to_elem)

        # Add // prefix variant if starts with specific path
        if xpath.startswith('/html/'):
            double_slash = '//' + xpath.split('/')[-1]
            if double_slash not in variants:
                variants.append(double_slash)

        # Tag generalization: strong↔b, span↔div are common HTML variations
        tag_swaps = [('strong', 'b'), ('b', 'strong'), ('span', 'div'), ('div', 'span')]
        for old_tag, new_tag in tag_swaps:
            if old_tag in xpath:
                swapped = re.sub(rf'\b{old_tag}\b', new_tag, xpath)
                if swapped != xpath and swapped not in variants:
                    variants.append(swapped)
                # Combined: tag swap + remove positional index
                swapped_no_pos = re.sub(r'\[\d+\]', '', swapped)
                if swapped_no_pos != swapped and swapped_no_pos not in variants:
                    variants.append(swapped_no_pos)
                # Also try wildcard: strong → *
                wildcard = re.sub(rf'\b{old_tag}\b', '*', xpath)
                if wildcard != xpath and wildcard not in variants:
                    variants.append(wildcard)
                # Combined: wildcard + remove positional index
                wildcard_no_pos = re.sub(r'\[\d+\]', '', wildcard)
                if wildcard_no_pos != wildcard and wildcard_no_pos not in variants:
                    variants.append(wildcard_no_pos)
                break  # Only one swap per call

        return variants[:8]  # Limit to 8 variants

    def _generate_ancestor_variants(self, xpath: str) -> list[str]:
        """Generate XPath variants by going up to ancestor level."""
        variants = []

        # If xpath targets a child, try the parent element instead
        # e.g., //td[...]/following-sibling::td[1] → //td[...]/following-sibling::td
        no_sibling_idx = re.sub(r'(/following-sibling::\w+)\[\d+\]', r'\1', xpath)
        if no_sibling_idx != xpath:
            variants.append(no_sibling_idx)

        # Try ancestor axis: if we're deep, try matching at a higher level
        # e.g., //div[@class='price']//span → //div[@class='price']
        parts = xpath.split('//')
        if len(parts) > 2:
            # Try with just first two parts
            shorter = '//' + '//'.join(parts[1:-1])
            variants.append(shorter)

        # If there's a contains(text(), 'X') pattern, make it the primary anchor
        text_match = re.search(r"contains\((?:text\(\)|\.),\s*'([^']+)'\)", xpath)
        if text_match:
            anchor_text = text_match.group(1)
            # Try: get the parent of the element containing this text
            parent_variant = f"//*[contains(text(), '{anchor_text}')]/.."
            variants.append(parent_variant)
            # Try: get next sibling
            sibling_variant = f"//*[contains(text(), '{anchor_text}')]/following-sibling::*[1]"
            if sibling_variant != xpath:
                variants.append(sibling_variant)

        return variants[:4]  # Limit

    async def _llm_repair(
        self,
        original_xpath: str,
        attribute: str,
        seed_values: list[str],
        verification_pages: list,
        candidates: list[RepairCandidate],
    ) -> Optional[str]:
        """Use LLM to generate a repaired XPath given failure context."""
        # Build failure info
        failure_info = f"XPath '{original_xpath}' returned empty on {len(verification_pages)} verification pages."

        # Build candidate info
        candidate_lines = []
        for c in candidates[:3]:
            if c.success_rate > 0:
                candidate_lines.append(
                    f"  - [{c.strategy}] {c.xpath} → success={c.success_rate:.0%}"
                )
        candidate_info = "\n".join(candidate_lines) if candidate_lines else "None worked."

        # Get HTML context from first verification page
        html_context = await self._get_page_html_snippet(
            verification_pages[0] if verification_pages else None, attribute
        )

        prompt = XPATH_REPAIR_PROMPT.format(
            original_xpath=original_xpath,
            attribute=attribute,
            seed_values=str(seed_values[:3]),
            failure_info=failure_info,
            candidate_info=candidate_info,
            html_context=html_context,
        )

        try:
            result = self.llm.text_query(prompt, label="XPathRepair")
            xpath_result = result.get("xpath", "")
            logger.info("  [Repair] LLM repair result: %s", xpath_result)
            return xpath_result if xpath_result else None
        except Exception as e:
            logger.warning("  [Repair] LLM repair exception: %s", e)
            return None

    async def _get_page_html_snippet(self, page, attribute: str) -> str:
        """Extract a relevant HTML snippet from the page for LLM context."""
        if page is None:
            return "(no page available)"
        try:
            full_html = await page.content()
            # Search for attribute-related text in the HTML
            # Take a representative snippet (head meta + body structure)
            lines = full_html.split('\n')

            # Collect meta tags (often contain attribute info)
            meta_lines = [l.strip() for l in lines if '<meta' in l.lower()][:5]

            # Collect title
            title_lines = [l.strip() for l in lines if '<title' in l.lower()][:1]

            # Search for attribute keyword in visible elements
            attr_lower = attribute.lower()
            relevant_lines = []
            for i, line in enumerate(lines):
                if attr_lower in line.lower() and '<script' not in line.lower():
                    context_start = max(0, i - 1)
                    context_end = min(len(lines), i + 2)
                    relevant_lines.extend(lines[context_start:context_end])
                    if len(relevant_lines) > 20:
                        break

            snippet_parts = []
            if title_lines:
                snippet_parts.append("<!-- Title -->")
                snippet_parts.extend(title_lines)
            if meta_lines:
                snippet_parts.append("<!-- Meta tags -->")
                snippet_parts.extend(meta_lines)
            if relevant_lines:
                snippet_parts.append(f"<!-- Elements mentioning '{attribute}' -->")
                snippet_parts.extend(relevant_lines[:20])

            snippet = '\n'.join(snippet_parts)
            # Limit to ~2000 chars
            return snippet[:2000] if snippet else "(attribute not found in page HTML)"
        except Exception:
            return "(failed to extract HTML)"
