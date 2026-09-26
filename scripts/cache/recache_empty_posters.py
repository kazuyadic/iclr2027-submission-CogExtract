#!/usr/bin/env python3
"""Re-crawl conference poster pages that have empty JS-shell HTML caches.

These pages were cached before JS finished rendering, resulting in ~9KB
skeleton HTML with no actual content (authors, titles, etc.).

Strategy (from retry_missing_cache.py):
- domcontentloaded (faster than load)
- 10s wait for JS rendering
- short networkidle
- No proxy needed
"""
import asyncio
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from playwright.async_api import async_playwright
from utils.cached_browser import CachedBrowserManager
from configs.config import VGSConfig

VIEWPORT_W = 1280
VIEWPORT_H = 1100
GOTO_TIMEOUT = 120_000  # 2 minutes
WAIT_AFTER_LOAD = 10    # seconds to wait for JS
EMPTY_THRESHOLD = 15_000  # bytes - below this is considered empty shell

CONFERENCE_DOMAINS = [
    "iclr.cc/virtual",
    "neurips.cc/virtual",
    "icml.cc/virtual",
]


async def main():
    config = VGSConfig()

    # Find all poster URLs with empty HTML from results
    results_path = config.project_root / "output" / "cog_plus_v7_3" / "results.json"
    with open(results_path) as f:
        results = json.load(f)

    poster_urls = set()
    for r in results:
        url = r.get("url", "")
        if "/poster/" in url and any(d in url for d in CONFERENCE_DOMAINS):
            poster_urls.add(url)

    print(f"Total conference poster URLs: {len(poster_urls)}")

    # Check which have empty HTML
    empty_urls = []
    for url in sorted(poster_urls):
        h = hashlib.sha256(url.encode()).hexdigest()[:16]
        html_path = Path("cache/pages") / h / "page.html"
        if html_path.exists():
            size = html_path.stat().st_size
            if size < EMPTY_THRESHOLD:
                empty_urls.append(url)
        else:
            empty_urls.append(url)  # missing = also needs recrawl

    print(f"Empty HTML (<{EMPTY_THRESHOLD} bytes) or missing: {len(empty_urls)}")
    if not empty_urls:
        print("Nothing to do!")
        return

    # Group by site for logging
    by_site = {}
    for url in empty_urls:
        for d in CONFERENCE_DOMAINS:
            if d in url:
                by_site.setdefault(d, []).append(url)
                break
    for site, urls in sorted(by_site.items()):
        print(f"  {site}: {len(urls)} pages")

    cache_mgr = CachedBrowserManager(cache_dir=config.cache_dir)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)

        success = 0
        failed = 0
        still_empty = 0

        for i, url in enumerate(empty_urls):
            print(f"\n[{i+1}/{len(empty_urls)}] {url}")
            context = await browser.new_context(
                viewport={"width": VIEWPORT_W, "height": VIEWPORT_H}
            )
            page = await context.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=GOTO_TIMEOUT)
                # Wait for JS to render content
                await asyncio.sleep(WAIT_AFTER_LOAD)
                # Also try networkidle with short timeout
                try:
                    await page.wait_for_load_state("networkidle", timeout=15000)
                except Exception:
                    pass

                # Save cache (HTML + screenshot + meta)
                await cache_mgr._write_cache(url, page)
                await context.close()

                # Verify HTML size
                h = hashlib.sha256(url.encode()).hexdigest()[:16]
                html_path = Path("cache/pages") / h / "page.html"
                html_size = html_path.stat().st_size
                if html_size < EMPTY_THRESHOLD:
                    print(f"  ⚠ Still empty ({html_size:,} bytes HTML)")
                    still_empty += 1
                else:
                    print(f"  ✓ Fixed ({html_size:,} bytes HTML)")
                    success += 1

            except Exception as exc:
                failed += 1
                print(f"  ✗ Failed: {str(exc)[:100]}")
                try:
                    await context.close()
                except Exception:
                    pass

        await browser.close()

    print(f"\n{'='*50}")
    print(f"Done: ✓{success} fixed, ⚠{still_empty} still empty, ✗{failed} failed")
    print(f"Total processed: {len(empty_urls)}")


if __name__ == "__main__":
    asyncio.run(main())
