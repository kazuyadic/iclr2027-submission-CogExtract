"""Playwright-based browser manager for web page rendering and screenshots."""
from __future__ import annotations

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright, Page, Browser


class BrowserManager:
    """Manage a headless Chromium browser for page rendering."""

    def __init__(self, viewport_width: int = 1280, viewport_height: int = 1100,
                 headless: bool = True, timeout_ms: int = 30_000,
                 proxy: str | None = None):
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.proxy = proxy
        self._playwright = None
        self._browser: Browser | None = None

    async def start(self):
        self._playwright = await async_playwright().start()
        launch_kwargs = {"headless": self.headless}
        if self.proxy:
            # Convert socks5h:// to playwright proxy format
            proxy_url = self.proxy
            if proxy_url.startswith("socks5h://"):
                proxy_url = proxy_url.replace("socks5h://", "socks5://")
            launch_kwargs["proxy"] = {"server": proxy_url}
        self._browser = await self._playwright.chromium.launch(**launch_kwargs)

    async def stop(self):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def load_page(self, url: str) -> Page:
        context = await self._browser.new_context(
            viewport={"width": self.viewport_width, "height": self.viewport_height},
        )
        page = await context.new_page()
        # Hard timeout wrapper to prevent hanging on problematic pages
        # Use config timeout + buffer for SPA rendering
        hard_timeout = self.timeout_ms / 1000 + 30
        try:
            await asyncio.wait_for(self._do_load(page, url), timeout=hard_timeout)
        except asyncio.TimeoutError:
            pass  # page may be partially loaded, still usable
        return page

    async def _do_load(self, page: Page, url: str):
        """Internal: load page with SPA-friendly waits."""
        # Use domcontentloaded for local/offline HTML to avoid timeout on dead external resources
        wait_event = "domcontentloaded" if url.startswith("file://") else "load"
        await page.goto(url, wait_until=wait_event, timeout=self.timeout_ms)
        try:
            await page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass
        # Wait for SPA frameworks to finish rendering
        await self._wait_for_spa_render(page)
        await self._close_cookie_dialogs(page)

    async def _wait_for_spa_render(self, page: Page, timeout: float = 8.0):
        """Wait for SPA (React/Vue/Angular) to complete hydration/render."""
        # Phase 1: wait for DOM to stabilize (no new mutations for 500ms)
        stability_js = """
        () => new Promise((resolve) => {
            let timer = null;
            const observer = new MutationObserver(() => {
                if (timer) clearTimeout(timer);
                timer = setTimeout(() => { observer.disconnect(); resolve(true); }, 500);
            });
            observer.observe(document.body, {childList: true, subtree: true});
            // Fallback: if no mutations at all, resolve after 1s
            setTimeout(() => { observer.disconnect(); resolve(true); }, 1000);
        })
        """
        try:
            await page.evaluate(stability_js, timeout=timeout * 1000)
        except Exception:
            pass
        # Phase 2: extra wait for React/Next.js hydration markers
        hydration_js = """
        () => {
            // React 18+ hydration
            const root = document.getElementById('root') || document.getElementById('__next');
            if (root && root._reactRootContainer) return true;
            // Check for React fiber
            const el = document.querySelector('[data-reactroot]');
            if (el) return true;
            // Vue app
            if (document.querySelector('[data-v-app]') || document.querySelector('.__vue_app__')) return true;
            return false;
        }
        """
        try:
            is_spa = await page.evaluate(hydration_js)
            if is_spa:
                await asyncio.sleep(1.0)
        except Exception:
            pass

    async def _close_cookie_dialogs(self, page: Page):
        """Try to close common cookie consent dialogs."""
        # Try clicking common "Accept" / "OK" / "Close" buttons
        selectors = [
            'button:has-text("Accept")',
            'button:has-text("Accept All")',
            'button:has-text("OK")',
            'button[aria-label*="close" i]',
            'button[aria-label*="Accept" i]',
            '.cookie-accept',
            '#cookie-accept',
            '[class*="cookie"] button',
            '[class*="consent"] button',
        ]
        for selector in selectors:
            try:
                btn = await page.query_selector(selector)
                if btn:
                    await btn.click(timeout=1000)
                    await asyncio.sleep(0.5)
                    break
            except Exception:
                continue

    async def is_spa(self, page: Page) -> bool:
        """Detect if the page is a Single Page Application (React/Vue/Angular/Next.js)."""
        detect_js = """
        () => {
            // React markers
            if (document.getElementById('root') || document.getElementById('__next')) return true;
            if (document.querySelector('[data-reactroot]')) return true;
            if (document.querySelector('[id="app"]') && document.querySelector('script[src*="chunk"]')) return true;
            // Vue markers
            if (document.querySelector('[data-v-app]') || document.querySelector('.__vue_app__')) return true;
            // Angular markers
            if (document.querySelector('[ng-version]') || document.querySelector('app-root')) return true;
            // Generic SPA indicators: heavy JS bundle loading
            const scripts = document.querySelectorAll('script[src]');
            let bundleCount = 0;
            for (const s of scripts) {
                if (s.src.includes('chunk') || s.src.includes('bundle') || s.src.includes('webpack') || s.src.includes('_next')) {
                    bundleCount++;
                }
            }
            if (bundleCount >= 2) return true;
            return false;
        }
        """
        try:
            return await page.evaluate(detect_js)
        except Exception:
            return False

    async def scroll_to_load_all(self, page: Page, max_scrolls: int = 20):
        """Scroll to bottom to trigger lazy loading / infinite scroll.

        Useful for list pages that load content on scroll.
        Stops when page height stops growing or max_scrolls reached.
        """
        prev_height = 0
        for _ in range(max_scrolls):
            height = await page.evaluate("document.documentElement.scrollHeight")
            if height == prev_height:
                break
            prev_height = height
            await page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
            await asyncio.sleep(1.0)
            # Wait for any new content to render
            try:
                await page.wait_for_load_state("networkidle", timeout=2000)
            except Exception:
                pass
        # Scroll back to top
        await page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(0.3)

    async def get_full_html(self, page: Page) -> str:
        return await page.content()

    async def capture_regions(self, page: Page, output_dir: Path) -> list[Path]:
        """Capture vertical region screenshots of the full page.

        The page is divided into non-overlapping regions of viewport height.
        Returns a list of saved screenshot paths.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        total_height = await page.evaluate("document.documentElement.scrollHeight")
        region_height = self.viewport_height
        region_paths: list[Path] = []
        region_index = 0

        scroll_y = 0
        while scroll_y < total_height:
            await page.evaluate(f"window.scrollTo(0, {scroll_y})")
            await asyncio.sleep(0.3)

            screenshot_path = output_dir / f"region_{region_index}.png"
            await page.screenshot(path=str(screenshot_path))
            region_paths.append(screenshot_path)

            region_index += 1
            scroll_y += region_height

        return region_paths

    async def inject_bounding_boxes(self, page: Page, elements_js_selector: str,
                                    tag_type: str = "img") -> str:
        """Inject Set-of-Mark style bounding boxes onto candidate elements.

        Returns the JS used (for potential cleanup) and modifies the page
        in-place so a screenshot will show labelled boxes.
        """
        inject_script = f"""
        (() => {{
            const elements = document.querySelectorAll('{elements_js_selector}');
            const colors = ['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7',
                            '#DDA0DD','#98D8C8','#F7DC6F','#BB8FCE','#85C1E9'];
            elements.forEach((el, i) => {{
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) return;

                const color = colors[i % colors.length];

                // bounding box
                const box = document.createElement('div');
                box.className = 'vgs-som-box';
                box.style.cssText = `position:absolute;left:${{rect.left+window.scrollX}}px;`
                    + `top:${{rect.top+window.scrollY}}px;width:${{rect.width}}px;`
                    + `height:${{rect.height}}px;border:3px solid ${{color}};`
                    + `pointer-events:none;z-index:99999;box-sizing:border-box;`;

                // label
                const label = document.createElement('span');
                label.textContent = i;
                label.style.cssText = `position:absolute;top:0;right:0;`
                    + `background:${{color}};color:#fff;font-size:14px;font-weight:bold;`
                    + `padding:2px 6px;z-index:100000;`;
                box.appendChild(label);
                document.body.appendChild(box);
            }});
        }})();
        """
        await page.evaluate(inject_script)
        return inject_script

    async def remove_bounding_boxes(self, page: Page):
        await page.evaluate("""
            document.querySelectorAll('.vgs-som-box').forEach(el => el.remove());
        """)

    async def execute_xpath(self, page: Page, xpath: str) -> list[str]:
        """Execute an XPath on the page and return matched element texts/attrs."""
        results = await page.evaluate(f"""
            (() => {{
                const result = document.evaluate(
                    `{xpath}`, document, null,
                    XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null
                );
                const values = [];
                for (let i = 0; i < result.snapshotLength; i++) {{
                    const node = result.snapshotItem(i);
                    if (node.nodeType === Node.ATTRIBUTE_NODE) {{
                        values.push(node.value);
                    }} else if (node.tagName === 'IMG') {{
                        values.push(node.src || '');
                    }} else if (node.tagName === 'A') {{
                        values.push(node.href || '');
                    }} else {{
                        // Extract direct text content, skipping descriptor/label children
                        let text = '';
                        for (const child of node.childNodes) {{
                            if (child.nodeType === Node.TEXT_NODE) {{
                                text += child.textContent;
                            }} else if (child.nodeType === Node.ELEMENT_NODE) {{
                                const cls = child.className || '';
                                if (cls.includes('descriptor') || cls.includes('label')) {{
                                    continue;
                                }}
                                text += child.textContent || '';
                            }}
                        }}
                        values.push(text.trim() || node.textContent?.trim() || '');
                    }}
                }}
                return values;
            }})()
        """)
        return results
