"""HTML simplification utilities following the paper's preprocessing rules."""
from __future__ import annotations

from lxml import etree


class HTMLProcessor:
    """Simplify raw HTML for LLM consumption.

    Rules (from Appendix A):
    - Remove <script> and <style> elements
    - Keep only @class, @href, @src, @alt attributes
    """

    KEEP_ATTRS = {"class", "href", "src", "alt", "id"}
    REMOVE_TAGS = {"script", "style", "noscript", "svg"}

    @classmethod
    def simplify(cls, html: str) -> str:
        tree = etree.HTML(html)
        if tree is None:
            return html

        for tag_name in cls.REMOVE_TAGS:
            for elem in tree.xpath(f"//{tag_name}"):
                elem.getparent().remove(elem)

        for elem in tree.iter():
            removable = [a for a in elem.attrib if a not in cls.KEEP_ATTRS]
            for attr in removable:
                del elem.attrib[attr]

        return etree.tostring(tree, encoding="unicode", method="html")

    @classmethod
    def extract_local_segment(cls, html: str, xpath: str, distance: int = 2) -> str:
        """Extract a local HTML segment around the element matched by *xpath*.

        Returns the element itself plus *distance* siblings on each side,
        mirroring the paper's §4.4 context-window strategy.
        """
        tree = etree.HTML(html)
        if tree is None:
            return ""

        matches = tree.xpath(xpath)
        if not matches:
            return ""

        target = matches[0]
        parent = target.getparent()
        if parent is None:
            return etree.tostring(target, encoding="unicode", method="html")

        siblings = list(parent)
        target_index = siblings.index(target)
        start = max(0, target_index - distance)
        end = min(len(siblings), target_index + distance + 1)

        fragment_parts = [
            etree.tostring(siblings[i], encoding="unicode", method="html")
            for i in range(start, end)
        ]
        return "\n".join(fragment_parts)

    @classmethod
    def get_elements_by_tag(cls, html: str, tag: str) -> list[dict]:
        """Return a list of dicts with index, tag, and key attributes."""
        tree = etree.HTML(html)
        if tree is None:
            return []

        results = []
        for index, elem in enumerate(tree.xpath(f"//{tag}")):
            info = {"index": index, "tag": tag}
            for attr in cls.KEEP_ATTRS:
                val = elem.get(attr)
                if val:
                    info[attr] = val
            results.append(info)
        return results

    @classmethod
    def get_elements_by_multi_tags(cls, html: str, tags: list[str]) -> list[dict]:
        """Return elements matching multiple tags in document order.

        This mirrors how querySelectorAll('p, span, h1, ...') works in the browser,
        ensuring element IDs align with the Set-of-Mark bounding box labels.
        """
        tree = etree.HTML(html)
        if tree is None:
            return []

        # Use an ElementTree for getpath() support
        doc_tree = etree.ElementTree(tree)
        tag_set = set(t.strip().lower() for t in tags)
        results = []
        for elem in tree.iter():
            try:
                local_tag = etree.QName(elem.tag).localname if isinstance(elem.tag, str) else None
            except ValueError:
                continue
            if local_tag and local_tag.lower() in tag_set:
                xpath = doc_tree.getpath(elem)
                info = {"index": len(results), "tag": local_tag, "xpath": xpath}
                for attr in cls.KEEP_ATTRS:
                    val = elem.get(attr)
                    if val:
                        info[attr] = val
                results.append(info)
        return results
