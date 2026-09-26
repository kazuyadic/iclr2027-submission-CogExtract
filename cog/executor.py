"""Stage 2-3: Executor — execute XPath and validate across pages."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from lxml import etree

from utils.browser import BrowserManager

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of executing an XPath on one page."""
    url: str
    xpath: str
    values: list[str] = field(default_factory=list)
    count: int = 0
    success: bool = False


@dataclass
class ValidationResult:
    """Result of cross-page validation."""
    seed_result: ExecutionResult
    validation_results: list[ExecutionResult] = field(default_factory=list)
    all_passed: bool = False
    failed_pages: list[ExecutionResult] = field(default_factory=list)


class CrossPageExecutor:
    """Execute XPath on seed page, then validate on other pages.

    Stage 2: Execute on seed page → confirm extraction works
    Stage 3: Execute on validation pages → identify failures
    """

    def __init__(self, browser: BrowserManager):
        self.browser = browser

    async def execute_on_page(
        self, page, xpath: str, url: str = ""
    ) -> ExecutionResult:
        """Execute XPath on a single page via browser."""
        try:
            values = await self.browser.execute_xpath(page, xpath)
        except Exception as exc:
            logger.warning("[Executor] XPath execution error: %s", exc)
            values = []

        result = ExecutionResult(
            url=url,
            xpath=xpath,
            values=values,
            count=len(values),
            success=len(values) > 0,
        )
        logger.info(
            "[Executor] %s → %d value(s) on %s",
            xpath[:50], result.count, url[:50],
        )
        return result

    def execute_on_html(
        self, html: str, xpath: str, url: str = ""
    ) -> ExecutionResult:
        """Execute XPath on static HTML using lxml (no browser needed)."""
        tree = etree.HTML(html)
        values = self._extract_values(tree, xpath)
        result = ExecutionResult(
            url=url,
            xpath=xpath,
            values=values,
            count=len(values),
            success=len(values) > 0,
        )
        return result

    async def validate_cross_page(
        self,
        xpath: str,
        seed_page,
        seed_url: str,
        validation_pages: list[tuple],  # list of (page, url)
    ) -> ValidationResult:
        """Execute XPath on seed + validation pages, identify failures."""
        seed_result = await self.execute_on_page(seed_page, xpath, seed_url)

        validation_results = []
        failed_pages = []
        for val_page, val_url in validation_pages:
            vr = await self.execute_on_page(val_page, xpath, val_url)
            validation_results.append(vr)
            if not vr.success:
                failed_pages.append(vr)

        all_passed = seed_result.success and len(failed_pages) == 0

        result = ValidationResult(
            seed_result=seed_result,
            validation_results=validation_results,
            all_passed=all_passed,
            failed_pages=failed_pages,
        )
        logger.info(
            "[Executor] Validation: %d/%d passed (seed=%s)",
            len(validation_results) - len(failed_pages),
            len(validation_results),
            "✓" if seed_result.success else "✗",
        )
        return result

    @staticmethod
    def _extract_values(tree, xpath: str, limit: int = 20) -> list[str]:
        """Extract text values from XPath matches on an lxml tree."""
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
