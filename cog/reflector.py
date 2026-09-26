"""Stage 5: Reflection Refiner — reflect on failure and revise XPath."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from lxml import etree

from utils.llm_client import LLMClient
from cog.diagnoser import DiagnosisResult
from cog.prompts import REFLECTION_PROMPT, REFLECTION_WITH_HISTORY_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class RevisionResult:
    """Structured output from one reflection-revision round."""
    round_num: int
    original_xpath: str
    revised_xpath: str
    reasoning: str
    revision_strategy: str
    diagnosis: DiagnosisResult
    accepted: bool = False  # whether the revision passed validation


@dataclass
class ReflectionTrace:
    """Full trace of the reflection loop for one attribute."""
    attribute: str
    initial_xpath: str
    final_xpath: str
    rounds: list[RevisionResult] = field(default_factory=list)
    total_rounds: int = 0
    success: bool = False


class ReflectionRefiner:
    """Reflect on XPath failures and iteratively revise.

    Core loop (up to max_rounds):
      1. Receive diagnosis from Diagnoser
      2. LLM reflects: why did it fail? what should change?
      3. LLM produces revised XPath
      4. Validate revised XPath on both pages
      5. If still fails → accumulate history, try again
    """

    def __init__(self, llm: LLMClient, max_rounds: int = 3):
        self.llm = llm
        self.max_rounds = max_rounds

    def reflect_and_revise(
        self,
        attribute: str,
        xpath: str,
        diagnosis: DiagnosisResult,
        html_a: str,
        html_b: str,
    ) -> RevisionResult:
        """Single round of reflection → revision (no history)."""
        html_a_section = self._extract_section(html_a)
        html_b_section = self._extract_section(html_b)

        prompt = REFLECTION_PROMPT.format(
            attribute=attribute,
            xpath=xpath,
            failure_type=diagnosis.failure_type,
            diagnosis=diagnosis.diagnosis,
            failing_predicate=diagnosis.failing_predicate,
            count_a=diagnosis.count_a,
            values_a="\n".join(f"  - {v}" for v in diagnosis.values_a) or "  (none)",
            count_b=diagnosis.count_b,
            values_b="\n".join(f"  - {v}" for v in diagnosis.values_b) or "  (none)",
            html_a_section=html_a_section,
            html_b_section=html_b_section,
        )

        parsed = self.llm.text_query(prompt, label=f"Cog·Reflect·{attribute}")
        revised_xpath = parsed.get("revised_xpath", "")
        reasoning = parsed.get("reasoning", "")
        strategy = parsed.get("revision_strategy", "")

        if not revised_xpath:
            logger.warning("[Reflector] No revised XPath for '%s'", attribute)
            revised_xpath = xpath

        result = RevisionResult(
            round_num=1,
            original_xpath=xpath,
            revised_xpath=revised_xpath,
            reasoning=reasoning,
            revision_strategy=strategy,
            diagnosis=diagnosis,
        )
        logger.info(
            "[Reflector] %s: '%s' → '%s' (%s)",
            attribute, xpath[:50], revised_xpath[:50], strategy[:60],
        )
        return result

    def reflect_with_history(
        self,
        attribute: str,
        xpath: str,
        diagnosis: DiagnosisResult,
        history: list[RevisionResult],
        html_a: str,
        html_b: str,
    ) -> RevisionResult:
        """Reflection with accumulated history of past failed attempts."""
        html_a_section = self._extract_section(html_a)
        html_b_section = self._extract_section(html_b)

        history_text = self._format_history(history)

        prompt = REFLECTION_WITH_HISTORY_PROMPT.format(
            attribute=attribute,
            xpath=xpath,
            failure_type=diagnosis.failure_type,
            diagnosis=diagnosis.diagnosis,
            repair_history=history_text,
            html_a_section=html_a_section,
            html_b_section=html_b_section,
        )

        parsed = self.llm.text_query(
            prompt, label=f"Cog·Reflect·R{len(history)+1}·{attribute}"
        )
        revised_xpath = parsed.get("revised_xpath", "")
        reasoning = parsed.get("reasoning", "")
        strategy = parsed.get("revision_strategy", "")

        if not revised_xpath:
            revised_xpath = xpath

        result = RevisionResult(
            round_num=len(history) + 1,
            original_xpath=xpath,
            revised_xpath=revised_xpath,
            reasoning=reasoning,
            revision_strategy=strategy,
            diagnosis=diagnosis,
        )
        logger.info(
            "[Reflector] Round %d: '%s' → '%s' (%s)",
            result.round_num, xpath[:40], revised_xpath[:40], strategy[:60],
        )
        return result

    def validate_xpath(
        self, xpath: str, html_a: str, html_b: str
    ) -> tuple[bool, list[str], list[str]]:
        """Validate XPath works on both pages using lxml."""
        tree_a = etree.HTML(html_a)
        tree_b = etree.HTML(html_b)

        values_a = self._extract_values(tree_a, xpath)
        values_b = self._extract_values(tree_b, xpath)

        works = len(values_a) > 0 and len(values_b) > 0
        return works, values_a, values_b

    @staticmethod
    def _extract_values(tree, xpath: str, limit: int = 10) -> list[str]:
        """Extract text values from XPath matches."""
        if tree is None:
            return []
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
                results.append(text[:200])
        return results

    @staticmethod
    def _format_history(history: list[RevisionResult]) -> str:
        """Format past revision attempts for the history-aware prompt."""
        lines = []
        for rev in history:
            lines.append(
                f"Round {rev.round_num}:\n"
                f"  XPath: {rev.revised_xpath}\n"
                f"  Strategy: {rev.revision_strategy}\n"
                f"  Result: FAILED ({rev.diagnosis.failure_type})\n"
                f"  Diagnosis: {rev.diagnosis.diagnosis}"
            )
        return "\n\n".join(lines) if lines else "(no history)"

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
