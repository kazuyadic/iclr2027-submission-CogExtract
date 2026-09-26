"""DOM Pattern Analyzer — structural analysis for Cog enhanced pipeline.

Two core components:
1. DOMPatternAnalyzer: Detects repeated DOM patterns → dynamic CSS selectors
2. StructuralContextBuilder: Builds ancestor path + sibling pattern + attribute stability context

These replace hardcoded CSS selectors in Stage 3 and inject structural context into Stage 4.
"""
from __future__ import annotations

import re
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from lxml import etree

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# DOMPatternAnalyzer — Stage 3 Enhancement
# ═══════════════════════════════════════════════════════════════════


@dataclass
class RepeatingPattern:
    """A detected repeating DOM pattern."""
    tag: str
    class_name: str  # first class token (or empty)
    count: int
    css_selector: str  # e.g., "div.product-card"
    content_selectors: list[str] = field(default_factory=list)


class DOMPatternAnalyzer:
    """Detect repeating DOM sub-structures and generate dynamic CSS selectors.

    Replaces the hardcoded "p, span, h1, h2, ..." selector in ElementPinpointer.

    Algorithm:
    1. Count (tag, first_class_token) combinations across the DOM
    2. Identify patterns that repeat >= min_repeat times
    3. For each repeating container, enumerate content elements inside
    4. Build targeted CSS selectors for content elements within containers
    """

    # Tags to skip (non-content structural/script elements)
    SKIP_TAGS = frozenset({
        "script", "style", "link", "meta", "head", "html", "noscript",
        "br", "hr", "svg", "path", "circle", "rect", "polygon",
        "defs", "clippath", "use", "symbol", "g", "line",
    })

    # Content-bearing tags that might contain extractable values
    CONTENT_TAGS = frozenset({
        "p", "span", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "td", "th", "dd", "dt", "label", "strong", "em",
        "b", "i", "a", "img", "time", "small", "cite",
        "div", "section", "article",
    })

    def __init__(self, min_repeat: int = 3, max_patterns: int = 5):
        self.min_repeat = min_repeat
        self.max_patterns = max_patterns

    def analyze(self, html: str, modality: str = "text") -> tuple[str, list[RepeatingPattern]]:
        """Analyze HTML and return (dynamic_css_selector, detected_patterns).

        Parameters
        ----------
        html : Full page HTML
        modality : "text", "image", or "link"

        Returns
        -------
        css_selector : Dynamic CSS selector string for element pinpointing
        patterns : List of detected repeating patterns (for context injection)
        """
        tree = etree.HTML(html)
        if tree is None:
            return self._fallback_selector(modality), []

        # Step 1: Count (tag, class) combinations
        pattern_counter = Counter()
        pattern_elements = defaultdict(list)

        for elem in tree.iter():
            if elem.tag in self.SKIP_TAGS or not isinstance(elem.tag, str):
                continue
            class_attr = elem.get("class", "")
            first_class = class_attr.split()[0] if class_attr else ""
            # Only track elements with at least one class (gives specificity)
            if first_class and not self._is_dynamic_class(first_class):
                key = (elem.tag, first_class)
                pattern_counter[key] += 1
                pattern_elements[key].append(elem)

        # Step 2: Filter to repeating patterns (>= min_repeat)
        repeating = [
            (key, count) for key, count in pattern_counter.most_common(50)
            if count >= self.min_repeat
        ]

        if not repeating:
            return self._fallback_selector(modality), []

        # Step 3: Build RepeatingPattern objects with content selectors
        patterns = []
        for (tag, cls), count in repeating[:self.max_patterns]:
            container_sel = f"{tag}.{cls}"
            content_sels = self._find_content_selectors(
                pattern_elements[(tag, cls)], modality
            )
            pattern = RepeatingPattern(
                tag=tag,
                class_name=cls,
                count=count,
                css_selector=container_sel,
                content_selectors=content_sels,
            )
            patterns.append(pattern)

        # Step 4: Compose final CSS selector
        css_selector = self._compose_selector(patterns, modality)
        logger.info(
            "[DOMAnalyzer] Found %d patterns, selector: %s",
            len(patterns), css_selector[:100],
        )
        return css_selector, patterns

    def _find_content_selectors(
        self, container_elems: list, modality: str
    ) -> list[str]:
        """Find common content-bearing sub-elements within containers."""
        child_tag_counter = Counter()

        for container in container_elems[:10]:  # sample first 10
            for child in container.iter():
                if child is container:
                    continue
                if not isinstance(child.tag, str):
                    continue
                if child.tag in self.SKIP_TAGS:
                    continue

                # Filter by modality
                if modality == "image" and child.tag != "img":
                    continue
                if modality == "link" and child.tag != "a":
                    continue
                if modality == "text" and child.tag not in self.CONTENT_TAGS:
                    continue

                child_cls = (child.get("class", "").split() or [""])[0]
                if child_cls and not self._is_dynamic_class(child_cls):
                    child_tag_counter[(child.tag, child_cls)] += 1
                else:
                    child_tag_counter[(child.tag, "")] += 1

        # Return top content selectors that appear in most containers
        results = []
        sample_size = min(10, len(container_elems))
        for (tag, cls), count in child_tag_counter.most_common(8):
            # Must appear in at least half the sampled containers
            if count >= sample_size * 0.4:
                if cls:
                    results.append(f"{tag}.{cls}")
                else:
                    results.append(tag)
        return results

    def _compose_selector(
        self, patterns: list[RepeatingPattern], modality: str
    ) -> str:
        """Compose a CSS selector from detected patterns."""
        selectors = set()

        for pattern in patterns:
            if pattern.content_selectors:
                for content_sel in pattern.content_selectors[:3]:
                    selectors.add(f"{pattern.css_selector} {content_sel}")
            else:
                # Fallback: select content tags inside container
                if modality == "image":
                    selectors.add(f"{pattern.css_selector} img")
                elif modality == "link":
                    selectors.add(f"{pattern.css_selector} a")
                else:
                    selectors.add(f"{pattern.css_selector} span")
                    selectors.add(f"{pattern.css_selector} p")
                    selectors.add(f"{pattern.css_selector} h1")
                    selectors.add(f"{pattern.css_selector} h2")
                    selectors.add(f"{pattern.css_selector} h3")

        if not selectors:
            return self._fallback_selector(modality)

        # Add fallback generic selectors to ensure coverage
        if modality == "text":
            selectors.update({"h1", "h2", "h3", "h4"})
        elif modality == "image":
            selectors.add("img")
        elif modality == "link":
            selectors.add("a[href]")

        return ", ".join(sorted(selectors)[:15])

    @staticmethod
    def _fallback_selector(modality: str) -> str:
        """Fallback to hardcoded selectors when no patterns detected."""
        if modality == "image":
            return "img"
        if modality == "link":
            return "a[href]"
        return "p, span, h1, h2, h3, h4, h5, h6, li, td, th, dd, dt, label"

    @staticmethod
    def _is_dynamic_class(class_name: str) -> bool:
        """Check if a class name is likely dynamic/generated."""
        # Contains hash-like sequences (css-xxxxx, sc-xxxxx, etc.)
        if re.search(r'[a-f0-9]{5,}', class_name):
            return True
        # Contains underscore + long random chars (e.g., emotion_xxx)
        if re.search(r'_[a-zA-Z0-9]{6,}$', class_name):
            return True
        # Very short meaningless (1-2 chars)
        if len(class_name) <= 2 and class_name.isalpha():
            return True
        return False


# ═══════════════════════════════════════════════════════════════════
# StructuralContextBuilder — Stage 4 Enhancement
# ═══════════════════════════════════════════════════════════════════


@dataclass
class StructuralContext:
    """Assembled structural context for XPath synthesis prompt injection."""
    ancestor_path: str
    sibling_pattern: str
    attribute_stability: str

    def format(self) -> str:
        """Format as a prompt-injectable text block."""
        parts = []
        if self.ancestor_path:
            parts.append(f"=== Ancestor Path ===\n{self.ancestor_path}")
        if self.sibling_pattern:
            parts.append(f"=== Sibling Repetition Pattern ===\n{self.sibling_pattern}")
        if self.attribute_stability:
            parts.append(f"=== Attribute Stability ===\n{self.attribute_stability}")
        return "\n\n".join(parts) if parts else "(no structural context available)"


class StructuralContextBuilder:
    """Build structural context for a target element in the DOM.

    Provides three types of context:
    1. Ancestor path: full path from element to <body> with annotations
    2. Sibling pattern: how many siblings share the same structure
    3. Attribute stability: which class/id values are stable vs dynamic
    """

    def build(self, html: str, target_xpath: str) -> StructuralContext:
        """Build structural context for a target element identified by XPath.

        Parameters
        ----------
        html : Full page HTML
        target_xpath : XPath to the target element (from Stage 3 pinpointing)

        Returns
        -------
        StructuralContext with ancestor path, sibling pattern, and attribute stability
        """
        tree = etree.HTML(html)
        if tree is None:
            return StructuralContext("", "", "")

        # Try to locate the target element
        target = self._find_target(tree, target_xpath)
        if target is None:
            return StructuralContext("", "", "")

        ancestor_path = self._build_ancestor_path(target)
        sibling_pattern = self._analyze_sibling_pattern(target)
        attribute_stability = self._classify_attributes(target)

        return StructuralContext(
            ancestor_path=ancestor_path,
            sibling_pattern=sibling_pattern,
            attribute_stability=attribute_stability,
        )

    def build_from_elements(
        self, html: str, element_xpaths: list[str]
    ) -> StructuralContext:
        """Build context from multiple pinpointed elements (takes first valid)."""
        tree = etree.HTML(html)
        if tree is None:
            return StructuralContext("", "", "")

        for xpath in element_xpaths:
            target = self._find_target(tree, xpath)
            if target is not None:
                ancestor_path = self._build_ancestor_path(target)
                sibling_pattern = self._analyze_sibling_pattern(target)
                attribute_stability = self._classify_attributes(target)
                return StructuralContext(
                    ancestor_path=ancestor_path,
                    sibling_pattern=sibling_pattern,
                    attribute_stability=attribute_stability,
                )

        return StructuralContext("", "", "")

    @staticmethod
    def _find_target(tree, xpath: str):
        """Locate target element by XPath."""
        try:
            results = tree.xpath(xpath)
            if results and hasattr(results[0], 'tag'):
                return results[0]
        except Exception:
            pass
        return None

    def _build_ancestor_path(self, elem) -> str:
        """Build annotated path from element to body.

        Format: body > div.main-content > div.paper-list > div.paper-item[x20] > span.title
        """
        path_parts = []
        current = elem

        while current is not None:
            if not isinstance(current.tag, str):
                current = current.getparent()
                continue

            tag = current.tag
            if tag == "html":
                break

            cls = (current.get("class", "").split() or [""])[0]
            elem_id = current.get("id", "")

            # Count siblings with same tag+class
            parent = current.getparent()
            sibling_count = 0
            if parent is not None:
                for sib in parent:
                    if not isinstance(sib.tag, str):
                        continue
                    if sib.tag == tag:
                        sib_cls = (sib.get("class", "").split() or [""])[0]
                        if sib_cls == cls:
                            sibling_count += 1

            # Build part description
            part = tag
            if cls and not self._is_dynamic(cls):
                part += f".{cls}"
            elif elem_id and not self._is_dynamic(elem_id):
                part += f"#{elem_id}"

            if sibling_count > 1:
                part += f"[x{sibling_count}]"

            path_parts.append(part)
            current = current.getparent()

        path_parts.reverse()
        return " > ".join(path_parts) if path_parts else ""

    def _analyze_sibling_pattern(self, elem) -> str:
        """Analyze how many siblings share the same structure."""
        parent = elem.getparent()
        if parent is None:
            return ""

        tag = elem.tag
        cls = (elem.get("class", "").split() or [""])[0]

        # Count structurally identical siblings
        matching_siblings = []
        for sib in parent:
            if not isinstance(sib.tag, str):
                continue
            if sib.tag == tag:
                sib_cls = (sib.get("class", "").split() or [""])[0]
                if sib_cls == cls:
                    matching_siblings.append(sib)

        count = len(matching_siblings)
        if count <= 1:
            return ""

        # Find shared vs varying attributes
        shared_attrs = {}
        varying_attrs = set()

        for attr_name in ("class", "id", "data-type", "data-testid", "role"):
            values = set()
            for sib in matching_siblings[:10]:
                values.add(sib.get(attr_name, ""))
            if len(values) == 1 and "" not in values:
                shared_attrs[attr_name] = values.pop()
            elif len(values) > 1:
                varying_attrs.add(attr_name)

        lines = [f"{count} siblings share: tag={tag}"]
        if cls:
            lines[0] += f", class={cls}"
        if shared_attrs:
            shared_str = ", ".join(f"{k}={v}" for k, v in shared_attrs.items()
                                   if k != "class")
            if shared_str:
                lines.append(f"  Shared (stable anchors): {shared_str}")
        if varying_attrs:
            lines.append(f"  Varying (do NOT use as selector): {', '.join(sorted(varying_attrs))}")

        return "\n".join(lines)

    def _classify_attributes(self, elem) -> str:
        """Classify class/id values as stable vs dynamic."""
        lines = []

        class_attr = elem.get("class", "")
        if class_attr:
            tokens = class_attr.split()
            stable = [t for t in tokens if not self._is_dynamic(t)]
            dynamic = [t for t in tokens if self._is_dynamic(t)]
            if stable:
                lines.append(f"Stable classes (safe for XPath): {', '.join(stable[:5])}")
            if dynamic:
                lines.append(f"Dynamic classes (AVOID in XPath): {', '.join(dynamic[:5])}")

        elem_id = elem.get("id", "")
        if elem_id:
            if self._is_dynamic(elem_id):
                lines.append(f"Dynamic ID (AVOID): {elem_id}")
            else:
                lines.append(f"Stable ID (safe): {elem_id}")

        # Check parent stability too
        parent = elem.getparent()
        if parent is not None and isinstance(parent.tag, str):
            parent_cls = parent.get("class", "")
            if parent_cls:
                parent_tokens = parent_cls.split()
                parent_stable = [t for t in parent_tokens if not self._is_dynamic(t)]
                if parent_stable:
                    lines.append(
                        f"Parent ({parent.tag}) stable classes: {', '.join(parent_stable[:3])}"
                    )

        return "\n".join(lines) if lines else ""

    @staticmethod
    def _is_dynamic(value: str) -> bool:
        """Check if a class/id value is likely dynamic."""
        if not value:
            return False
        # Hash-like
        if re.search(r'[a-f0-9]{5,}', value):
            return True
        # Underscore + random suffix
        if re.search(r'_[a-zA-Z0-9]{6,}$', value):
            return True
        # Purely numeric
        if value.isdigit():
            return True
        # Very short meaningless
        if len(value) <= 2 and not value.isalpha():
            return True
        return False
