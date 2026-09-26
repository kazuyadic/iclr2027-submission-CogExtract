#!/usr/bin/env python3
"""Pre-fetch and cache all dataset URLs for offline evaluation.

Usage:
    python scripts/prefetch_cache.py                    # all reachable websites
    python scripts/prefetch_cache.py --websites iclr neurips  # specific websites
    python scripts/prefetch_cache.py --dry-run           # show what would be fetched
    python scripts/prefetch_cache.py --stats             # show cache stats
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from configs.config import VGSConfig
from utils.cached_browser import CachedBrowserManager, _url_hash
from utils.data_loader import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EXCLUDED_WEBSITES = {
    "marinespecies", "thesportsdb", "themealdb",
    "thecocktaildb", "scrapethissite", "dp", "huggingface",
}


def collect_urls(config: VGSConfig, websites: list[str] | None = None) -> list[str]:
    """Collect all unique URLs from the dataset."""
    loader = DataLoader(config.data_dir)
    groups = loader.build_grouped_samples()

    all_urls = set()
    for group in groups:
        website = group.get("website", "")
        if website in EXCLUDED_WEBSITES:
            continue
        if websites and website not in websites:
            continue
        for url in group.get("urls", []):
            all_urls.add(url)

    return sorted(all_urls)


async def prefetch(
    urls: list[str],
    config: VGSConfig,
    concurrency: int = 3,
    skip_cached: bool = True,
):
    """Fetch and cache all URLs."""
    browser = CachedBrowserManager(
        cache_dir=config.cache_dir,
        viewport_width=config.viewport_width,
        viewport_height=config.viewport_height,
        headless=config.headless,
        timeout_ms=config.page_load_timeout_ms,
    )

    if skip_cached:
        uncached = [u for u in urls if not browser._is_cached(u)]
        cached_count = len(urls) - len(uncached)
        logger.info(
            "Total: %d URLs, already cached: %d, to fetch: %d",
            len(urls), cached_count, len(uncached),
        )
        urls = uncached
    else:
        logger.info("Total: %d URLs (re-fetching all)", len(urls))

    if not urls:
        logger.info("Nothing to fetch — all URLs are cached!")
        return

    # Start real browser for fetching
    from utils.browser import BrowserManager
    real_browser = BrowserManager(
        viewport_width=config.viewport_width,
        viewport_height=config.viewport_height,
        headless=config.headless,
        timeout_ms=config.page_load_timeout_ms,
    )
    await real_browser.start()
    browser._real = real_browser
    browser._real_started = True

    semaphore = asyncio.Semaphore(concurrency)
    success_count = 0
    error_count = 0

    from tqdm import tqdm
    pbar = tqdm(total=len(urls), desc="Prefetching",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")

    async def fetch_one(url: str):
        nonlocal success_count, error_count
        async with semaphore:
            try:
                page = await real_browser.load_page(url)
                # Scroll to load all content
                await real_browser.scroll_to_load_all(page, max_scrolls=15)
                # Write to cache
                await browser._write_cache(url, page)
                await page.context.close()
                success_count += 1
            except Exception as exc:
                logger.warning("Failed: %s — %s", url[:60], exc)
                error_count += 1
            finally:
                pbar.update(1)
                pbar.set_postfix_str(f"✓{success_count} ✗{error_count}")

    tasks = [fetch_one(url) for url in urls]
    try:
        await asyncio.gather(*tasks)
    finally:
        pbar.close()
        await real_browser.stop()

    logger.info(
        "Done! Fetched %d URLs: ✓%d ✗%d",
        len(urls), success_count, error_count,
    )

    # Print stats
    stats = browser.cache_stats()
    logger.info(
        "Cache: %d URLs, %.1f MB in %s",
        stats["cached_urls"], stats["total_size_mb"], stats["cache_dir"],
    )


def show_stats(config: VGSConfig):
    """Show cache statistics."""
    browser = CachedBrowserManager(cache_dir=config.cache_dir)
    stats = browser.cache_stats()

    all_urls = collect_urls(config)
    cached_count = sum(1 for u in all_urls if browser._is_cached(u))

    print(f"\n📦 Cache Statistics")
    print(f"   Directory:    {stats['cache_dir']}")
    print(f"   Cached URLs:  {stats['cached_urls']}")
    print(f"   Total size:   {stats['total_size_mb']} MB")
    print(f"   Coverage:     {cached_count}/{len(all_urls)} dataset URLs ({cached_count/len(all_urls)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Pre-fetch dataset pages to local cache")
    parser.add_argument("--websites", nargs="+", help="Only fetch specific websites")
    parser.add_argument("--concurrency", type=int, default=3, help="Parallel fetch count")
    parser.add_argument("--dry-run", action="store_true", help="Show URLs without fetching")
    parser.add_argument("--stats", action="store_true", help="Show cache stats")
    parser.add_argument("--refetch", action="store_true", help="Re-fetch even if cached")
    parser.add_argument("--cache-dir", type=str, help="Override cache directory")
    args = parser.parse_args()

    config = VGSConfig()
    if args.cache_dir:
        config.cache_dir = Path(args.cache_dir)

    if args.stats:
        show_stats(config)
        return

    urls = collect_urls(config, args.websites)
    logger.info("Collected %d URLs from dataset", len(urls))

    if args.dry_run:
        browser = CachedBrowserManager(cache_dir=config.cache_dir)
        for url in urls:
            status = "✓ cached" if browser._is_cached(url) else "✗ miss"
            print(f"  {status}  {url}")
        cached = sum(1 for u in urls if browser._is_cached(u))
        print(f"\n{cached}/{len(urls)} cached")
        return

    asyncio.run(prefetch(
        urls, config,
        concurrency=args.concurrency,
        skip_cached=not args.refetch,
    ))


if __name__ == "__main__":
    main()
