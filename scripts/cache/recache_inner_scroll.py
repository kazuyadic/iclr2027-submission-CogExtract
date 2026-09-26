"""Re-cache inner-scroll pages: detect inner scroll containers and capture
multiple region screenshots for pages that have scrollHeight ≈ viewportHeight.

Usage: python3 scripts/recache_inner_scroll.py
"""
import asyncio
import hashlib
import json
import shutil
import time
from pathlib import Path

from playwright.async_api import async_playwright


VIEWPORT_W = 1280
VIEWPORT_H = 1100
CACHE_DIR = Path("cache/pages")

# All conference papers.html URLs (type_3 groups: g_001)
URLS = [
    # ICLR
    "https://iclr.cc/virtual/2025/papers.html?filter=title&search=game",
    "https://iclr.cc/virtual/2025/papers.html?filter=title&search=llm+agent",
    "https://iclr.cc/virtual/2025/papers.html?filter=title&search=document",
    "https://iclr.cc/virtual/2025/papers.html?filter=title&search=tool",
    "https://iclr.cc/virtual/2025/papers.html?filter=title&search=expert",
    "https://iclr.cc/virtual/2025/papers.html?filter=title&search=vlm",
    "https://iclr.cc/virtual/2025/papers.html?filter=title&search=text-to",
    "https://iclr.cc/virtual/2025/papers.html?filter=title&search=score",
    "https://iclr.cc/virtual/2025/papers.html?filter=title&search=synthesis",
    "https://iclr.cc/virtual/2025/papers.html?filter=title&search=editing",
    "https://iclr.cc/virtual/2025/papers.html?filter=title&search=play",
    # NeurIPS
    "https://neurips.cc/virtual/2024/papers.html?filter=title&search=game",
    "https://neurips.cc/virtual/2024/papers.html?filter=title&search=llm+agent",
    "https://neurips.cc/virtual/2024/papers.html?filter=title&search=document",
    "https://neurips.cc/virtual/2024/papers.html?filter=title&search=tool",
    "https://neurips.cc/virtual/2024/papers.html?filter=title&search=expert",
    "https://neurips.cc/virtual/2024/papers.html?filter=title&search=vlm",
    "https://neurips.cc/virtual/2024/papers.html?filter=title&search=text-to",
    "https://neurips.cc/virtual/2024/papers.html?filter=title&search=score",
    "https://neurips.cc/virtual/2024/papers.html?filter=title&search=synthesis",
    "https://neurips.cc/virtual/2024/papers.html?filter=title&search=editing",
    "https://neurips.cc/virtual/2024/papers.html?filter=title&search=play",
    # ICML
    "https://icml.cc/virtual/2025/papers.html?filter=title&search=game",
    "https://icml.cc/virtual/2025/papers.html?filter=title&search=llm+agent",
    "https://icml.cc/virtual/2025/papers.html?filter=title&search=document",
    "https://icml.cc/virtual/2025/papers.html?filter=title&search=tool",
    "https://icml.cc/virtual/2025/papers.html?filter=title&search=expert",
    "https://icml.cc/virtual/2025/papers.html?filter=title&search=vlm",
    "https://icml.cc/virtual/2025/papers.html?filter=title&search=text-to",
    "https://icml.cc/virtual/2025/papers.html?filter=title&search=score",
    "https://icml.cc/virtual/2025/papers.html?filter=title&search=synthesis",
    "https://icml.cc/virtual/2025/papers.html?filter=title&search=editing",
    "https://icml.cc/virtual/2025/papers.html?filter=title&search=play",
]


def cache_path(url: str) -> Path:
    h = hashlib.sha256(url.encode()).hexdigest()[:16]
    return CACHE_DIR / h


async def recache_one(page, url: str):
    cache = cache_path(url)
    if not cache.exists():
        print(f"  SKIP (no cache dir): {url}")
        return

    # Navigate
    print(f"  Loading {url}...")
    await page.goto(url, wait_until="domcontentloaded", timeout=60000)
    await asyncio.sleep(5)  # Extra wait for dynamic content

    # Re-save HTML (in case it changed)
    html = await page.content()
    (cache / "page.html").write_text(html, encoding="utf-8")

    total_height = await page.evaluate("document.documentElement.scrollHeight")
    print(f"  total_height={total_height}, HTML={len(html):,} bytes")

    # Find inner scrollable container
    find_js = """
    (() => {
        const candidates = [];
        const all = document.querySelectorAll('*');
        for (const el of all) {
            const sh = el.scrollHeight;
            const ch = el.clientHeight;
            if (sh > ch + 100 && ch > 200) {
                candidates.push({sh, ch, tag: el.tagName,
                    cls: (el.className || '').toString().slice(0, 80)});
            }
        }
        if (candidates.length === 0) return null;
        candidates.sort((a, b) => b.sh - a.sh);
        const best = candidates[0];
        window.__inner_scroll_el = best.el;
        return best;
    })()
    """
    info = await page.evaluate(find_js)
    if not info:
        print(f"  No inner scroll container found")
        return

    inner_h = info["sh"]
    print(f"  Inner scroll: <{info['tag']} class='{info['cls'][:40]}'> "
          f"scrollHeight={inner_h} clientHeight={info['ch']}")

    # Delete old region files
    for old in cache.glob("region_*.png"):
        old.unlink()

    # Scroll and capture
    num_regions = min((inner_h + VIEWPORT_H - 1) // VIEWPORT_H, 60)
    print(f"  Capturing {num_regions} regions...")

    for idx in range(num_regions):
        scroll_y = idx * VIEWPORT_H
        await page.evaluate(f"""
            (() => {{
                const el = window.__inner_scroll_el;
                if (el) el.scrollTop = {scroll_y};
            }})()
        """)
        await asyncio.sleep(0.4)
        region_path = cache / f"region_{idx}.png"
        await page.screenshot(path=str(region_path), timeout=10000)

    # Reset scroll
    await page.evaluate(
        "if (window.__inner_scroll_el) window.__inner_scroll_el.scrollTop = 0"
    )

    # Update meta
    meta = {
        "url": url,
        "viewport_width": VIEWPORT_W,
        "viewport_height": VIEWPORT_H,
        "total_height": total_height,
        "inner_scroll_height": inner_h,
        "inner_scroll_regions": num_regions,
        "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (cache / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Done: {num_regions} regions saved")


async def main():
    import tempfile, os
    # Use project-local temp dir to avoid sandbox permission issues
    tmp_dir = Path("output/.tmp_playwright")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)
    
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": VIEWPORT_W, "height": VIEWPORT_H}
        )
        page = await context.new_page()

        for i, url in enumerate(URLS):
            print(f"\n[{i+1}/{len(URLS)}] {url.split('/')[2]} "
                  f"search={url.split('search=')[-1]}")
            try:
                await recache_one(page, url)
            except Exception as e:
                print(f"  ERROR: {e}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
