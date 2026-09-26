#!/usr/bin/env python3
"""Fetch website pages with progress bar, anti-crawl detection, and auto-stop.

Usage:
    python scripts/fetch_website.py <website_name> [--concurrency N] [--refetch]

Examples:
    python scripts/fetch_website.py fueleconomy
    python scripts/fetch_website.py quotestoscrape --concurrency 3
    python scripts/fetch_website.py booktoscrape --refetch
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from configs.config import VGSConfig
from utils.cached_browser import CachedBrowserManager
from utils.browser import BrowserManager

# Strong signals: any single match is likely anti-crawl
ANTI_CRAWL_STRONG = [
    "access denied", "403 forbidden", "429 too many requests",
    "rate limit exceeded", "challenge-platform", "cf-browser-verification",
    "please verify you are human", "captcha",
]

# Weak signals: need 2+ matches to trigger (avoid false positives from page content)
ANTI_CRAWL_WEAK = [
    "forbidden", "blocked", "banned", "cloudflare", "verify",
]


def collect_urls(config: VGSConfig, website_name: str) -> list[str]:
    """Collect unique URLs for a website from the dataset."""
    data_dir = Path(config.data_dir) / "LiveWeb_IE_repo" / "LiveWeb_IE"
    candidates = [website_name, website_name.replace(".", "")]
    site_dir = None
    for candidate in candidates:
        directory = data_dir / candidate
        if directory.is_dir():
            site_dir = directory
            break

    if site_dir is None:
        available = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
        raise ValueError(
            f"Website '{website_name}' not found.\nAvailable: {available}"
        )

    group_file = site_dir / "group" / "group.jsonl"
    urls: list[str] = []
    with open(group_file) as fh:
        for line in fh:
            obj = json.loads(line)
            for group_value in obj.values():
                urls.extend(group_value.get("group_url", []))
    return sorted(set(urls))


async def fetch_website(
    website_name: str,
    config: VGSConfig,
    concurrency: int = 5,
    refetch: bool = False,
    proxy: str | None = None,
    delay: float = 0,
    headed: bool = False,
):
    urls = collect_urls(config, website_name)
    print(f"\n📚 {website_name}: {len(urls)} unique URLs")

    browser = CachedBrowserManager(
        cache_dir=config.cache_dir,
        viewport_width=config.viewport_width,
        viewport_height=config.viewport_height,
        headless=config.headless,
        timeout_ms=config.page_load_timeout_ms,
    )

    if refetch:
        uncached = urls
        print(f"   Refetching all {len(uncached)} URLs")
    else:
        uncached = [u for u in urls if not browser._is_cached(u)]
        cached_count = len(urls) - len(uncached)
        print(f"   Uncached: {len(uncached)}, Already cached: {cached_count}")

    if not uncached:
        print("✅ All cached! Nothing to do.")
        return

    real = BrowserManager(
        viewport_width=config.viewport_width,
        viewport_height=config.viewport_height,
        headless=not headed,
        timeout_ms=90_000 if headed else config.page_load_timeout_ms,
        proxy=proxy,
    )
    await real.start()
    browser._real = real
    browser._real_started = True

    stop_event = asyncio.Event()
    success_count = 0
    error_count = 0
    consecutive_errors = 0
    max_consecutive_errors = 5
    lock = asyncio.Lock()

    from tqdm import tqdm
    progress_bar = tqdm(
        total=len(uncached),
        desc=f"Fetching {website_name}",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )

    async def fetch_one(url: str):
        nonlocal success_count, error_count, consecutive_errors
        if stop_event.is_set():
            progress_bar.update(1)
            return

        try:
            async def _do_fetch():
                nonlocal success_count, error_count, consecutive_errors

                # Pre-check HTTP status with lightweight request (no Playwright)
                import urllib.request
                import urllib.error
                pre_check_status = None
                try:
                    req = urllib.request.Request(url, method="HEAD")
                    req.add_header("User-Agent", "Mozilla/5.0")
                    resp = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: urllib.request.urlopen(req, timeout=10)
                    )
                    pre_check_status = resp.status
                    resp.close()
                except urllib.error.HTTPError as http_err:
                    pre_check_status = http_err.code
                except Exception:
                    pass  # Network error, let Playwright handle it

                if pre_check_status and pre_check_status >= 400:
                    cache_dir = browser._cache_path(url)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    import time as _time
                    meta = {
                        "url": url,
                        "status": str(pre_check_status),
                        "cached_at": _time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "error": True,
                    }
                    (cache_dir / "meta.json").write_text(
                        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
                    (cache_dir / "page.html").write_text("", encoding="utf-8")
                    print(f"\n  ⚠ HTTP {pre_check_status}: {url[:80]}")
                    return

                page = await real.load_page(url)
                content = (await page.content()).lower()
                title = (await page.title()).lower()
                status_text = f"{title} {content[:2000]}"

                triggered_strong = [kw for kw in ANTI_CRAWL_STRONG if kw in status_text]
                triggered_weak = [kw for kw in ANTI_CRAWL_WEAK if kw in status_text]
                is_anti_crawl = bool(triggered_strong) or len(triggered_weak) >= 2

                if is_anti_crawl:
                    if headed:
                        print(f"\n⚠️  Anti-crawl detected on {url[:80]}")
                        if triggered_strong:
                            print(f"   Strong signals: {triggered_strong}")
                        print("   👉 Please solve the captcha in the browser window manually.")
                        print("   ⏳ Waiting up to 120s for you to resolve...")
                        # Wait for user to solve captcha, check every 5s
                        resolved = False
                        for _ in range(24):
                            await asyncio.sleep(5)
                            try:
                                new_content = (await page.content()).lower()
                                new_title = (await page.title()).lower()
                                new_status = f"{new_title} {new_content[:2000]}"
                                still_blocked = (
                                    any(kw in new_status for kw in ANTI_CRAWL_STRONG)
                                    or sum(1 for kw in ANTI_CRAWL_WEAK if kw in new_status) >= 2
                                )
                                if not still_blocked:
                                    resolved = True
                                    print("   ✅ Captcha resolved! Continuing...")
                                    break
                            except Exception:
                                pass
                        if not resolved:
                            print("   ⏰ Timeout waiting for captcha resolution, skipping this page.")
                            await page.context.close()
                            progress_bar.update(1)
                            return
                        # Re-read content after captcha resolution
                        content = (await page.content()).lower()
                        title = (await page.title()).lower()
                        status_text = f"{title} {content[:2000]}"
                    else:
                        print(f"\n🚫 ANTI-CRAWL DETECTED on {url}")
                        if triggered_strong:
                            print(f"   Strong signals: {triggered_strong}")
                        if triggered_weak:
                            print(f"   Weak signals ({len(triggered_weak)}): {triggered_weak}")
                        await page.context.close()
                        stop_event.set()
                        progress_bar.update(1)
                        return

                await real.scroll_to_load_all(page, max_scrolls=15)
                await browser._write_cache(url, page)
                await page.context.close()

            await asyncio.wait_for(_do_fetch(), timeout=180)

            async with lock:
                success_count += 1
                consecutive_errors = 0
            progress_bar.update(1)
            progress_bar.set_postfix_str(f"✓{success_count} ✗{error_count}")

        except Exception as exc:
            error_message = str(exc).lower()
            print(f"\n  ✗ {url[:80]}")
            print(f"    {exc}")
            async with lock:
                error_count += 1
                consecutive_errors += 1
                should_stop = (
                    any(kw in error_message for kw in ANTI_CRAWL_STRONG)
                    or consecutive_errors >= max_consecutive_errors
                )
                if should_stop:
                    if any(kw in error_message for kw in ANTI_CRAWL_STRONG):
                        print(f"\n🚫 Anti-crawl error detected, stopping.")
                    else:
                        print(f"\n🛑 {max_consecutive_errors} consecutive errors, stopping.")
                    stop_event.set()
            progress_bar.update(1)
            progress_bar.set_postfix_str(f"✓{success_count} ✗{error_count}")

    semaphore = asyncio.Semaphore(concurrency)

    async def guarded_fetch(url: str):
        async with semaphore:
            await fetch_one(url)
            if delay > 0:
                await asyncio.sleep(delay)

    tasks = [guarded_fetch(u) for u in uncached]
    await asyncio.gather(*tasks)
    progress_bar.close()

    await real.stop()
    print(f"\n📊 Done: ✓{success_count} ✗{error_count} / {len(uncached)}")


def main():
    parser = argparse.ArgumentParser(description="Fetch website pages to local cache")
    parser.add_argument("website", help="Website name (e.g. fueleconomy, arxiv, booktoscrape)")
    parser.add_argument("--concurrency", type=int, default=5, help="Parallel fetch count (default: 5)")
    parser.add_argument("--refetch", action="store_true", help="Re-fetch even if already cached")
    parser.add_argument("--proxy", type=str, default=None, help="Proxy URL (e.g. socks5h://127.0.0.1:13659)")
    parser.add_argument("--delay", type=float, default=0, help="Delay in seconds between each page fetch (default: 0)")
    parser.add_argument("--headed", action="store_true", help="Use headed browser (visible window, better for anti-crawl sites)")
    args = parser.parse_args()

    config = VGSConfig()
    asyncio.run(fetch_website(
        website_name=args.website,
        config=config,
        concurrency=args.concurrency,
        refetch=args.refetch,
        proxy=args.proxy,
        delay=args.delay,
        headed=args.headed,
    ))


if __name__ == "__main__":
    main()
