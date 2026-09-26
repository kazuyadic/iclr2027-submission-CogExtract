#!/usr/bin/env python3
"""Retry caching slow conference pages with domcontentloaded strategy."""
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
WAIT_AFTER_LOAD = 10  # seconds to wait for JS after domcontentloaded


async def main():
    config = VGSConfig()
    
    # Find missing URLs
    data_root = config.data_dir / "LiveWeb_IE"
    missing = []
    for site in ['iclr', 'neurips', 'icml']:
        group_file = data_root / site / 'group' / 'group.jsonl'
        with open(group_file) as f:
            for line in f:
                entry = json.loads(line)
                for gid, gdata in entry.items():
                    for url in gdata.get('group_url', []):
                        h = hashlib.sha256(url.encode()).hexdigest()[:16]
                        p = Path(f"cache/pages/{h}/page.html")
                        if not p.exists():
                            missing.append(url)
    
    print(f"Missing URLs: {len(missing)}")
    if not missing:
        print("Nothing to do!")
        return

    cache_mgr = CachedBrowserManager(cache_dir=config.cache_dir)
    
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        
        success = 0
        failed = 0
        for i, url in enumerate(missing):
            print(f"\n[{i+1}/{len(missing)}] {url}")
            context = await browser.new_context(
                viewport={"width": VIEWPORT_W, "height": VIEWPORT_H}
            )
            page = await context.new_page()
            try:
                # Use domcontentloaded (faster than load)
                await page.goto(url, wait_until="domcontentloaded", timeout=GOTO_TIMEOUT)
                # Wait for JS to render content
                await asyncio.sleep(WAIT_AFTER_LOAD)
                # Also try networkidle with short timeout
                try:
                    await page.wait_for_load_state("networkidle", timeout=15000)
                except Exception:
                    pass
                
                await cache_mgr._write_cache(url, page)
                await context.close()
                success += 1
                
                # Check HTML size
                h = hashlib.sha256(url.encode()).hexdigest()[:16]
                html_size = (Path(f"cache/pages/{h}/page.html")).stat().st_size
                print(f"  ✓ Saved ({html_size:,} bytes HTML)")
            except Exception as exc:
                failed += 1
                print(f"  ✗ Failed: {str(exc)[:100]}")
                try:
                    await context.close()
                except Exception:
                    pass
        
        await browser.close()
    
    print(f"\nDone: ✓{success} ✗{failed}")


if __name__ == "__main__":
    asyncio.run(main())
