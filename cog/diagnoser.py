"""Stage 4: Failure Diagnoser — classify why an XPath fails on a new page."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from lxml import etree

from utils.llm_client import LLMClient
from cog.prompts import FAILURE_DIAGNOSIS_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class DiagnosisResult:
    """Structured output from failure diagnosis."""
    failure_type: str  # empty | wrong | over_extraction | format_mismatch
    diagnosis: str
    failing_predicate: str
    count_a: int
    count_b: int
    values_a: list[str] = field(default_factory=list)
    values_b: list[str] = field(default_factory=list)


class FailureDiagnoser:
    """Diagnose why an XPath fails to generalize to a new page.

    Two-tier diagnosis:
      1. Rule-based pre-classification (fast, no LLM)
      2. LLM-based deep diagnosis (when rule-based is uncertain)
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def diagnose(
        self,
        attribute: str,
        xpath: str,
        html_a: str,
        html_b: str,
        values_a: list[str],
        values_b: list[str],
    ) -> DiagnosisResult:
        """Diagnose failure: classify type and identify failing predicate."""
        count_a = len(values_a)
        count_b = len(values_b)

        # ── Tier 1: Rule-based pre-classification ──
        rule_result = self._rule_based_classify(
            xpath, count_a, count_b, values_a, values_b
        )
        if rule_result and rule_result.failure_type == "empty":
            # For empty failures, also try to find the failing predicate
            failing_pred = self._find_content_predicate(xpath)
            if failing_pred:
                rule_result.failing_predicate = failing_pred
                rule_result.diagnosis = (
                    f"XPath contains content-specific predicate '{failing_pred}' "
                    f"that does not exist on Page B"
                )
                logger.info(
                    "[Diagnoser] Rule-based: %s — %s",
                    rule_result.failure_type, rule_result.diagnosis,
                )
                return rule_result

        # ── Tier 2: LLM-based deep diagnosis ──
        return self._llm_diagnose(
            attribute, xpath, html_a, html_b,
            values_a, values_b, count_a, count_b,
        )

    def _rule_based_classify(
        self,
        xpath: str,
        count_a: int,
        count_b: int,
        values_a: list[str],
        values_b: list[str],
    ) -> Optional[DiagnosisResult]:
        """Fast rule-based classification without LLM."""
        if count_b == 0:
            return DiagnosisResult(
                failure_type="empty",
                diagnosis="XPath matches 0 nodes on Page B",
                failing_predicate="",
                count_a=count_a,
                count_b=0,
                values_a=values_a[:5],
                values_b=[],
            )

        if count_a > 0 and count_b > count_a * 5:
            return DiagnosisResult(
                failure_type="over_extraction",
                diagnosis=f"XPath matches {count_b} nodes on Page B vs {count_a} on Page A — too broad",
                failing_predicate="",
                count_a=count_a,
                count_b=count_b,
                values_a=values_a[:5],
                values_b=values_b[:5],
            )

        return None  # uncertain — need LLM

    @staticmethod
    def _find_content_predicate(xpath: str) -> str:
        """Find content-specific predicates that likely cause empty failures."""
        # Match text() predicates with literal values
        patterns = [
            r"text\(\)\s*=\s*['\"]([^'\"]+)['\"]",
            r"contains\(\s*text\(\)\s*,\s*['\"]([^'\"]+)['\"]\s*\)",
            r"contains\(\s*@alt\s*,\s*['\"]([^'\"]+)['\"]\s*\)",
            r"contains\(\s*@src\s*,\s*['\"]([^'\"]+)['\"]\s*\)",
            r"contains\(\s*@title\s*,\s*['\"]([^'\"]+)['\"]\s*\)",
            r"contains\(\s*@href\s*,\s*['\"]([^'\"]+)['\"]\s*\)",
            r"@alt\s*=\s*['\"]([^'\"]+)['\"]",
        ]
        for pattern in patterns:
            match = re.search(pattern, xpath)
            if match:
                return match.group(0)
        return ""

    def _llm_diagnose(
        self,
        attribute: str,
        xpath: str,
        html_a: str,
        html_b: str,
        values_a: list[str],
        values_b: list[str],
        count_a: int,
        count_b: int,
    ) -> DiagnosisResult:
        """Use LLM for deep failure diagnosis."""
        html_b_section = self._extract_html_section(html_b, max_chars=2000)

        prompt = FAILURE_DIAGNOSIS_PROMPT.format(
            attribute=attribute,
            xpath=xpath,
            count_a=count_a,
            values_a="\n".join(f"  - {v[:100]}" for v in values_a[:5]) or "  (none)",
            count_b=count_b,
            values_b="\n".join(f"  - {v[:100]}" for v in values_b[:5]) or "  (none)",
            html_b_section=html_b_section,
        )

        parsed = self.llm.text_query(prompt, label=f"Cog·Diagnose·{attribute}")

        failure_type = parsed.get("failure_type", "empty")
        if failure_type not in ("empty", "wrong", "over_extraction", "format_mismatch"):
            failure_type = "empty"

        result = DiagnosisResult(
            failure_type=failure_type,
            diagnosis=parsed.get("diagnosis", ""),
            failing_predicate=parsed.get("failing_predicate", ""),
            count_a=count_a,
            count_b=count_b,
            values_a=values_a[:5],
            values_b=values_b[:5],
        )
        logger.info(
            "[Diagnoser] LLM: type=%s, predicate='%s', diag='%s'",
            result.failure_type, result.failing_predicate, result.diagnosis[:80],
        )
        return result

    @staticmethod
    def _extract_html_section(html: str, max_chars: int = 2000) -> str:
        """Extract concise body HTML for prompt context."""
        body_match = re.search(
            r"<body[^>]*>(.*)</body>", html, re.DOTALL | re.IGNORECASE
        )
        body = body_match.group(1) if body_match else html
        body = re.sub(r"<script[^>]*>.*?</script>", "", body, flags=re.DOTALL | re.IGNORECASE)
        body = re.sub(r"<style[^>]*>.*?</style>", "", body, flags=re.DOTALL | re.IGNORECASE)
        body = re.sub(r"\s+", " ", body).strip()
        return body[:max_chars]
