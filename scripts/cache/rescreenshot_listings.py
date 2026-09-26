"""Re-screenshot conference listing pages from cached HTML.

The cached HTML has poster content inside <noscript> blocks. When loaded in
Playwright (JS enabled), noscript is hidden. This script:
1. Reads cached HTML
2. Removes <noscript> tags (keeps content)
3. Injects CSS to ensure all content is visible
4. Takes a full-page screenshot
5. Saves to cache (replacing the broken 1-viewport screenshot)

No network access needed - uses local cached HTML only.
"""
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "pages"

INJECT_CSS = """<style id="fix-screenshot">
* { overflow: visible !important; max-height: none !important; }
body, html { height: auto !important; }
</style>"""


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


async def main():
    # Load conference listing URLs
    results_path = Path(__file__).resolve().parent.parent / "output" / "cog_plus_v7_1" / "results.json"
    with open(results_path) as f:
        results = json.load(f)

    listing_urls = set()
    for t in results:
        w = t["sample_id"].split("_")[1]
        if w in ("001", "002", "003"):
            url = t["url"]
            if "papers.html" in url:
                listing_urls.add(url)

    urls = sorted(listing_urls)
    logger.info("Re-screenshotting %d listing URLs from cached HTML", len(urls))

    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 1100},
        )

        success = 0
        failed = 0
        for i, url in enumerate(urls):
            h = url_hash(url)
            cache_path = CACHE_DIR / h
            html_path = cache_path / "page.html"

            if not html_path.exists():
                logger.warning("[%d/%d] No cached HTML, skipping: %s", i+1, len(urls), url[:60])
                failed += 1
                continue

            logger.info("[%d/%d] %s", i+1, len(urls), url[:70])

            # Read and fix HTML
            html = html_path.read_text(encoding="utf-8")
            # Remove <noscript> tags but keep their content
            html_fixed = re.sub(r'<noscript[^>]*>', '', html)
            html_fixed = html_fixed.replace('</noscript>', '')
            # Inject CSS for visibility
            html_fixed = html_fixed.replace('</head>', INJECT_CSS + '</head>')

            page = await context.new_page()
            try:
                await page.set_content(html_fixed, wait_until="load", timeout=15000)
                await page.wait_for_timeout(2000)

                doc_h = await page.evaluate("document.documentElement.scrollHeight")
                poster_count = await page.evaluate(
                    'document.querySelectorAll("a[href*=poster]").length'
                    + ' + document.querySelectorAll("a[href*=paper]").length'
                )
                logger.info("  Doc height: %d, content links: %d", doc_h, poster_count)

                # Take full-page screenshot
                ss_path = cache_path / "screenshot.png"
                await page.screenshot(path=str(ss_path), full_page=True, timeout=60000)
                ss_size = os.path.getsize(ss_path)
                logger.info("  Screenshot: %dKB", ss_size // 1024)

                # Delete old region files so they get re-sliced from new screenshot
                deleted = 0
                for region_file in cache_path.glob("region_*.png"):
                    region_file.unlink()
                    deleted += 1
                if deleted:
                    logger.info("  Deleted %d old region files", deleted)

                success += 1

            except Exception as e:
                logger.error("  Failed: %s", e)
                failed += 1
            finally:
                await page.close()

        await browser.close()

    logger.info("Done! Success: %d, Failed: %d", success, failed)


if __name__ == "__main__":
    asyncio.run(main())
