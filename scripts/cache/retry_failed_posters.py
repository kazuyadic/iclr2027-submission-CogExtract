#!/usr/bin/env python3
"""Retry failed poster pages with longer delays to avoid rate limiting."""
import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from playwright.async_api import async_playwright
from utils.cached_browser import CachedBrowserManager
from configs.config import VGSConfig

VIEWPORT_W = 1280
VIEWPORT_H = 1100
GOTO_TIMEOUT = 180_000  # 3 minutes
WAIT_AFTER_LOAD = 15    # more JS wait time
DELAY_BETWEEN = 10      # seconds between requests
EMPTY_THRESHOLD = 15_000


async def main():
    config = VGSConfig()

    # Parse failed URLs from log
    log_path = Path("/tmp/recache_posters.log")
    log = log_path.read_text()
    failed_urls = []
    for line in log.split("\n"):
        if "Failed" in line:
            m = re.search(r"https://\S+", line)
            if m:
                url = m.group(0)
                if url not in failed_urls:
                    failed_urls.append(url)

    print(f"Retrying {len(failed_urls)} failed URLs with {DELAY_BETWEEN}s delay")

    cache_mgr = CachedBrowserManager(cache_dir=config.cache_dir)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        success = 0
        failed = 0

        for i, url in enumerate(failed_urls):
            print(f"\n[{i+1}/{len(failed_urls)}] {url}")

            # Delay between requests to avoid rate limiting
            if i > 0:
                print(f"  Waiting {DELAY_BETWEEN}s...")
                await asyncio.sleep(DELAY_BETWEEN)

            context = await browser.new_context(
                viewport={"width": VIEWPORT_W, "height": VIEWPORT_H}
            )
            page = await context.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=GOTO_TIMEOUT)
                await asyncio.sleep(WAIT_AFTER_LOAD)
                try:
                    await page.wait_for_load_state("networkidle", timeout=20000)
                except Exception:
                    pass

                await cache_mgr._write_cache(url, page)
                await context.close()

                h = hashlib.sha256(url.encode()).hexdigest()[:16]
                html_path = Path("cache/pages") / h / "page.html"
                html_size = html_path.stat().st_size
                if html_size < EMPTY_THRESHOLD:
                    print(f"  ⚠ Still empty ({html_size:,} bytes)")
                    failed += 1
                else:
                    print(f"  ✓ Fixed ({html_size:,} bytes)")
                    success += 1
            except Exception as exc:
                failed += 1
                print(f"  ✗ Failed: {str(exc)[:120]}")
                try:
                    await context.close()
                except Exception:
                    pass

        await browser.close()

    print(f"\n{'='*50}")
    print(f"Retry done: ✓{success} fixed, ✗{failed} still failed")


if __name__ == "__main__":
    asyncio.run(main())
