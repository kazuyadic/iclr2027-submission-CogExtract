"""Transparent caching layer over BrowserManager.

First access for a URL goes through Playwright and saves HTML + screenshots
to a local cache directory. Subsequent accesses read from cache and use
lxml for XPath execution — no browser needed.

Cache layout:
    cache_dir/
      <url_hash>/
        page.html          # Full rendered HTML
        meta.json           # {url, viewport, timestamp, total_height, ...}
        region_0.png        # Viewport screenshots
        region_1.png
        ...
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from lxml import etree
from playwright.async_api import Page

from utils.browser import BrowserManager

logger = logging.getLogger(__name__)


def _url_hash(url: str) -> str:
    """Deterministic, filesystem-safe hash for a URL."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


class CachedPage:
    """Lightweight proxy that stands in for a Playwright Page when serving
    from cache.  Only the subset of Page API used by the pipeline is exposed.
    """

    def __init__(self, url: str, html: str, cache_dir: Path, meta: dict):
        self.url = url
        self._html = html
        self._cache_dir = cache_dir
        self._meta = meta
        self._tree: etree._Element | None = None
        # Mimic a real Page context so `page.context.close()` doesn't crash
        self.context = _FakeContext()
        self._is_cached = True
        # Real browser page reference (set after inject_bounding_boxes)
        self._real_page = None
        # Tracked scroll position for region identification
        self._scroll_y: int = 0

    @property
    def tree(self) -> etree._Element | None:
        if self._tree is None:
            self._tree = etree.HTML(self._html)
        return self._tree

    async def content(self) -> str:
        return self._html

    async def evaluate(self, expression: str, *args, **kwargs) -> Any:
        """Handle common evaluate calls offline.

        Covers all patterns used by the VGS/Cog pipelines:
        - scrollHeight queries → return cached total_height
        - scrollTo calls → no-op
        - DOM queries (querySelectorAll, etc.) → return empty
        - Bounding box / SPA detection JS → return safe defaults

        If a real page is attached, delegate to it for JS execution.
        """
        # If real page is active, delegate JS execution to it
        if self._real_page is not None:
            return await self._real_page.evaluate(expression, *args, **kwargs)

        expr = expression.strip()

        # Scroll height queries
        if "scrollHeight" in expr:
            return self._meta.get("total_height", 5000)

        # Scroll commands — track position for offline region identification
        if "scrollTo" in expr:
            import re as _re
            m = _re.search(r'scrollTo\s*\(\s*\d+\s*,\s*(\d+)', expr)
            if m:
                self._scroll_y = int(m.group(1))
            return None

        # Bounding box injection (inject_script from inject_bounding_boxes)
        if "vgs-som-box" in expr or "querySelectorAll" in expr:
            return None

        # Bounding box removal
        if "forEach" in expr and "remove()" in expr:
            return None

        # SPA detection / hydration checks
        if "reactRootContainer" in expr or "data-reactroot" in expr:
            return False
        if "__vue_app__" in expr or "data-v-app" in expr:
            return False
        if "ng-version" in expr or "app-root" in expr:
            return False

        # MutationObserver / stability check
        if "MutationObserver" in expr:
            return True

        # XPath execution via document.evaluate
        if "document.evaluate" in expr:
            # Fall back to lxml — extract the xpath string from the JS
            import re
            match = re.search(r'`([^`]+)`', expr)
            if match:
                xpath_str = match.group(1)
                from utils.cached_browser import CachedBrowserManager
                return CachedBrowserManager._lxml_execute_xpath(
                    self._html, xpath_str, self.url
                )
            return []

        # Default: log warning and return None instead of crashing
        logger.warning(
            "[CachedPage] Unhandled evaluate expression: %s",
            expr[:120],
        )
        return None

    async def screenshot(self, path: str = "", **kwargs) -> bytes:
        """Return cached screenshot bytes or copy cached file to target path.

        If a real page is attached (after inject_bounding_boxes), delegate to it.
        """
        # If real page is active (e.g., after bounding box injection), use it
        if self._real_page is not None:
            if path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            return await self._real_page.screenshot(path=path, **kwargs)

        # Determine which region this corresponds to
        if path:
            target = Path(path)
            stem = target.stem  # e.g. "region_0"
            cached = self._cache_dir / f"{stem}.png"
            if cached.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cached, target)
                return cached.read_bytes()
        # Fallback: return first region
        fallback = self._cache_dir / "region_0.png"
        if fallback.exists():
            return fallback.read_bytes()
        raise FileNotFoundError(f"No cached screenshot in {self._cache_dir}")

    async def query_selector(self, selector: str):
        return None

    async def wait_for_load_state(self, *args, **kwargs):
        pass


class _FakeContext:
    """Minimal context stub so `page.context.close()` works on cached pages."""

    async def close(self):
        pass


class CachedBrowserManager:
    """Drop-in wrapper around BrowserManager with transparent disk cache.

    Usage:
        browser = CachedBrowserManager(cache_dir=Path("./cache"))
        await browser.start()         # starts real browser (lazy)
        page = await browser.load_page(url)  # cache hit → CachedPage
        html = await browser.get_full_html(page)
        values = await browser.execute_xpath(page, xpath)
        await browser.stop()
    """

    def __init__(
        self,
        cache_dir: Path = Path("./cache/pages"),
        viewport_width: int = 1280,
        viewport_height: int = 1100,
        headless: bool = True,
        timeout_ms: int = 30_000,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.headless = headless
        self.timeout_ms = timeout_ms

        # Real browser — created lazily on first cache miss
        self._real: BrowserManager | None = None
        self._real_started = False
        self._start_lock = asyncio.Lock()

    # ── Lifecycle ──

    async def start(self):
        """No-op on start; real browser starts lazily on first miss."""
        pass

    async def _ensure_real_browser(self) -> BrowserManager:
        """Start the real Playwright browser if not already running."""
        if self._real_started and self._real is not None:
            return self._real
        async with self._start_lock:
            if self._real_started and self._real is not None:
                return self._real
            logger.info("[Cache] Starting real Playwright browser (first cache miss)")
            self._real = BrowserManager(
                viewport_width=self.viewport_width,
                viewport_height=self.viewport_height,
                headless=self.headless,
                timeout_ms=self.timeout_ms,
            )
            await self._real.start()
            self._real_started = True
            return self._real

    async def stop(self):
        if self._real and self._real_started:
            await self._real.stop()
            self._real_started = False

    # ── Cache helpers ──

    def _cache_path(self, url: str) -> Path:
        return self.cache_dir / _url_hash(url)

    def _is_cached(self, url: str) -> bool:
        cache = self._cache_path(url)
        return (cache / "page.html").exists() and (cache / "meta.json").exists()

    def _read_cache(self, url: str) -> CachedPage:
        cache = self._cache_path(url)
        html = (cache / "page.html").read_text(encoding="utf-8")
        # Normalize Unicode confusable characters so XPath contains() matches
        html = self._normalize_unicode(html)
        meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
        return CachedPage(url=url, html=html, cache_dir=cache, meta=meta)

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Normalize visually identical but codepoint-different characters.

        This ensures XPath `contains(text(), '·')` matches regardless of whether
        the HTML uses U+00B7 (MIDDLE DOT), U+22C5 (DOT OPERATOR), etc.
        Only normalizes text content — does NOT touch HTML tags/attributes.
        """
        import unicodedata
        # Map confusable Unicode chars to canonical ASCII equivalents
        # These are characters that look the same but have different codepoints
        _MAP = str.maketrans({
            '\u22C5': '\u00B7',  # DOT OPERATOR → MIDDLE DOT
            '\u2027': '\u00B7',  # HYPHENATION POINT → MIDDLE DOT
            '\u0387': '\u00B7',  # GREEK ANO TELEIA → MIDDLE DOT
            '\u2022': '\u00B7',  # BULLET → MIDDLE DOT
            '\u2024': '.',       # ONE DOT LEADER → period
            '\uFE52': '.',       # SMALL FULL STOP → period
            '\uFF0E': '.',       # FULLWIDTH FULL STOP → period
            '\u00A0': ' ',       # NO-BREAK SPACE → space
            '\u2009': ' ',       # THIN SPACE → space
            '\u200A': ' ',       # HAIR SPACE → space
            '\u202F': ' ',       # NARROW NO-BREAK SPACE → space
            '\u205F': ' ',       # MEDIUM MATHEMATICAL SPACE → space
            '\u3000': ' ',       # IDEOGRAPHIC SPACE → space
            '\u2018': "'",       # LEFT SINGLE QUOTATION MARK → apostrophe
            '\u2019': "'",       # RIGHT SINGLE QUOTATION MARK → apostrophe
            '\u201C': '"',       # LEFT DOUBLE QUOTATION MARK → quote
            '\u201D': '"',       # RIGHT DOUBLE QUOTATION MARK → quote
            '\uFF0D': '-',       # FULLWIDTH HYPHEN-MINUS → hyphen
            '\u2010': '-',       # HYPHEN → hyphen
            '\u2011': '-',       # NON-BREAKING HYPHEN → hyphen
            '\u2012': '-',       # FIGURE DASH → hyphen
            '\u2013': '-',       # EN DASH → hyphen
            '\u2014': '-',       # EM DASH → hyphen
        })
        return text.translate(_MAP)

    # ── Inner-scroll container handling for screenshots ──

    _EXPAND_INNER_SCROLL_JS = """
    () => {
        const expanded = [];
        const allEls = document.querySelectorAll('*');
        for (const el of allEls) {
            if (el === document.documentElement || el === document.body) continue;
            const style = getComputedStyle(el);
            const overflowY = style.overflowY;
            if ((overflowY === 'auto' || overflowY === 'scroll')
                && el.scrollHeight > el.clientHeight + 10) {
                expanded.push({
                    el: el,
                    origOverflow: el.style.overflow,
                    origHeight: el.style.height,
                    origMaxHeight: el.style.maxHeight,
                    origPosition: el.style.position,
                });
                el.style.overflow = 'visible';
                el.style.height = 'auto';
                el.style.maxHeight = 'none';
                el.style.position = 'static';
            }
        }
        // Store on window for restore
        window.__innerScrollExpanded = expanded;
        return expanded.length;
    }
    """

    _RESTORE_INNER_SCROLL_JS = """
    () => {
        const expanded = window.__innerScrollExpanded || [];
        for (const item of expanded) {
            item.el.style.overflow = item.origOverflow;
            item.el.style.height = item.origHeight;
            item.el.style.maxHeight = item.origMaxHeight;
            item.el.style.position = item.origPosition;
        }
        window.__innerScrollExpanded = [];
        return expanded.length;
    }
    """

    async def _expand_inner_scroll(self, page: Page):
        """Temporarily expand inner-scroll containers for full-page screenshot."""
        try:
            count = await page.evaluate(self._EXPAND_INNER_SCROLL_JS)
            if count > 0:
                logger.info("[Cache] Expanded %d inner-scroll container(s) for screenshot", count)
                # Give the browser a moment to reflow
                import asyncio
                await asyncio.sleep(0.5)
        except Exception as e:
            logger.debug("[Cache] Inner-scroll expansion failed: %s", e)

    async def _restore_inner_scroll(self, page: Page):
        """Restore inner-scroll containers to their original state."""
        try:
            await page.evaluate(self._RESTORE_INNER_SCROLL_JS)
        except Exception:
            pass

    async def _write_cache(self, url: str, page: Page):
        """Persist page HTML + region screenshots to cache."""
        cache = self._cache_path(url)
        cache.mkdir(parents=True, exist_ok=True)

        # Save HTML
        html = await page.content()
        (cache / "page.html").write_text(html, encoding="utf-8")

        # Save metadata
        total_height = await page.evaluate(
            "document.documentElement.scrollHeight"
        )
        meta = {
            "url": url,
            "viewport_width": self.viewport_width,
            "viewport_height": self.viewport_height,
            "total_height": total_height,
            "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (cache / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Save full-page screenshot (original, unsegmented)
        # For pages with inner-scroll containers (overflow:auto/scroll on a div),
        # document.documentElement.scrollHeight doesn't reflect actual content height.
        # We temporarily expand those containers so the screenshot captures all content.
        screenshot_path = cache / "screenshot.png"
        try:
            await self._expand_inner_scroll(page)
            await page.screenshot(path=str(screenshot_path), full_page=True, timeout=30000)
        except Exception as screenshot_exc:
            logger.warning(
                "[Cache] Screenshot failed for %s: %s (HTML saved, screenshot skipped)",
                url[:60], screenshot_exc,
            )
        finally:
            await self._restore_inner_scroll(page)

        logger.info(
            "[Cache] Saved %s → %s (full_page screenshot, %d bytes HTML)",
            url[:60], cache.name, len(html),
        )

    # ── Public API (mirrors BrowserManager) ──

    async def load_page(self, url: str) -> CachedPage | Page:
        """Load page from cache or via Playwright (auto-caching)."""
        if self._is_cached(url):
            logger.debug("[Cache] HIT  %s", url[:60])
            return self._read_cache(url)

        # Cache miss → real browser
        logger.info("[Cache] MISS %s — fetching via Playwright", url[:60])
        real = await self._ensure_real_browser()
        page = await real.load_page(url)

        # Write cache before returning
        try:
            await self._write_cache(url, page)
        except Exception as exc:
            logger.warning("[Cache] Failed to write cache for %s: %s", url, exc)

        return page

    async def load_page_live(self, url: str) -> "Page":
        """Always load page via real Playwright browser (bypass cache).

        Used by VGS pipeline which needs a real DOM for bounding-box injection.
        """
        real = await self._ensure_real_browser()
        page = await real.load_page(url)
        return page

    async def get_full_html(self, page: CachedPage | Page) -> str:
        if isinstance(page, CachedPage):
            return page._html
        return await page.content()

    async def capture_regions(
        self, page: CachedPage | Page, output_dir: Path
    ) -> list[Path]:
        """Copy cached screenshots or capture live.

        If only a full-page screenshot.png exists (no region_*.png),
        automatically slice it into viewport-height regions.
        """
        if isinstance(page, CachedPage):
            output_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            idx = 0
            while True:
                src = page._cache_dir / f"region_{idx}.png"
                if not src.exists():
                    break
                dst = output_dir / f"region_{idx}.png"
                shutil.copy2(src, dst)
                paths.append(dst)
                idx += 1
            if paths:
                return paths

            # Fallback: slice full-page screenshot into viewport regions
            full_screenshot = page._cache_dir / "screenshot.png"
            if full_screenshot.exists():
                paths = self._slice_screenshot_into_regions(
                    full_screenshot, output_dir, page._cache_dir,
                )
                if paths:
                    return paths

            logger.warning("[Cache] No cached screenshots for %s", page.url)

        # Live capture
        if self._real is None:
            raise RuntimeError("No cached screenshots and no real browser")
        return await self._real.capture_regions(page, output_dir)

    def _slice_screenshot_into_regions(
        self, screenshot_path: Path, output_dir: Path, cache_dir: Path,
        max_regions: int = 4,
    ) -> list[Path]:
        """Slice a full-page screenshot into viewport-height region PNGs.
        
        Caps at max_regions to avoid excessive VLM calls on tall pages.
        """
        from PIL import Image

        try:
            img = Image.open(screenshot_path)
        except Exception as exc:
            logger.warning("[Cache] Cannot open screenshot %s: %s", screenshot_path, exc)
            return []

        img_width, img_height = img.size
        region_height = self.viewport_height
        paths = []
        idx = 0

        y = 0
        while y < img_height:
            box = (0, y, img_width, min(y + region_height, img_height))
            region_img = img.crop(box)

            # Save to cache dir (so next time it's a direct hit)
            cache_dst = cache_dir / f"region_{idx}.png"
            region_img.save(cache_dst)

            # Also copy to output dir
            out_dst = output_dir / f"region_{idx}.png"
            shutil.copy2(cache_dst, out_dst)
            paths.append(out_dst)

            y += region_height
            idx += 1

        logger.info(
            "[Cache] Sliced screenshot → %d regions (%dx%d viewport)",
            len(paths), img_width, region_height,
        )
        img.close()
        return paths

    async def inject_bounding_boxes(
        self, page: CachedPage | Page, elements_js_selector: str,
        tag_type: str = "img"
    ) -> str:
        """Bounding box injection.

        For CachedPage: purely offline — no browser.  The pipeline will
        detect `_is_cached` and fall back to text-based element selection.
        """
        if isinstance(page, CachedPage):
            # No-op: pipeline detects cached page and uses text-based selection
            logger.debug("[Cache] inject_bounding_boxes: offline no-op for %s", page.url[:60])
            return ""
        real = await self._ensure_real_browser()
        return await real.inject_bounding_boxes(
            page, elements_js_selector, tag_type
        )

    async def remove_bounding_boxes(self, page: CachedPage | Page):
        if isinstance(page, CachedPage):
            return  # No-op for cached pages
        real = await self._ensure_real_browser()
        await real.remove_bounding_boxes(page)

    async def execute_xpath(
        self, page: CachedPage | Page, xpath: str
    ) -> list[str]:
        """Execute XPath offline via lxml (cached) or in-browser (live)."""
        if isinstance(page, CachedPage):
            return self._lxml_execute_xpath(page._html, xpath, page.url)

        # Live execution
        real = await self._ensure_real_browser()
        return await real.execute_xpath(page, xpath)

    @staticmethod
    def _lxml_execute_xpath(html: str, xpath: str, url: str = "") -> list[str]:
        """Execute XPath on static HTML via lxml, matching browser behavior."""
        tree = etree.HTML(html)
        if tree is None:
            return []
        try:
            matches = tree.xpath(xpath)
        except etree.XPathError:
            return []

        values = []
        for node in matches:
            if isinstance(node, str):
                # Attribute node (e.g. @href)
                values.append(node)
            elif isinstance(node, etree._Element):
                tag = etree.QName(node.tag).localname if isinstance(
                    node.tag, str
                ) else ""
                if tag == "img":
                    src = node.get("src", "")
                    # Resolve relative URLs
                    if src and not src.startswith(("http://", "https://", "//")):
                        if url:
                            from urllib.parse import urljoin
                            src = urljoin(url, src)
                    values.append(src)
                elif tag == "a":
                    href = node.get("href", "")
                    if href and not href.startswith(
                        ("http://", "https://", "//", "#", "javascript:")
                    ):
                        if url:
                            from urllib.parse import urljoin
                            href = urljoin(url, href)
                    values.append(href)
                else:
                    # Text content — concatenate all descendant text
                    # Use lxml's itertext() which handles nested elements
                    full_text = "".join(node.itertext()).strip()
                    values.append(full_text)
            elif isinstance(node, etree._ElementUnicodeResult):
                values.append(str(node))
        return values

    async def scroll_to_load_all(self, page: CachedPage | Page, max_scrolls: int = 20):
        """No-op for cached pages (HTML already has full content)."""
        if isinstance(page, CachedPage):
            return
        real = await self._ensure_real_browser()
        await real.scroll_to_load_all(page, max_scrolls)

    async def is_spa(self, page: CachedPage | Page) -> bool:
        if isinstance(page, CachedPage):
            # Detect SPA markers from cached HTML
            html_lower = page._html[:5000].lower()
            spa_markers = [
                'id="root"', 'id="__next"', 'data-reactroot',
                'data-v-app', '__vue_app__', 'ng-version',
            ]
            return any(m in html_lower for m in spa_markers)
        real = await self._ensure_real_browser()
        return await real.is_spa(page)

    # ── Cache management utilities ──

    def cache_stats(self) -> dict:
        """Return cache statistics."""
        cached = list(self.cache_dir.iterdir())
        total_size = sum(
            f.stat().st_size
            for d in cached if d.is_dir()
            for f in d.iterdir() if f.is_file()
        )
        return {
            "cached_urls": len(cached),
            "total_size_mb": round(total_size / 1024 / 1024, 1),
            "cache_dir": str(self.cache_dir),
        }

    def clear_cache(self):
        """Delete all cached pages."""
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[Cache] Cleared all cached pages")

    def evict(self, url: str):
        """Remove a single URL from cache."""
        cache = self._cache_path(url)
        if cache.exists():
            shutil.rmtree(cache)
            logger.info("[Cache] Evicted %s", url[:60])
