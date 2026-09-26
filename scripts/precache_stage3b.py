#!/usr/bin/env python3
"""Pre-cache VGS screenshots (full-page + Stage 3b Set-of-Mark) from cached HTML.

Serves each cached page.html from a localhost HTTP server, renders it in headless
Chromium, and writes the images the VGS/cog pipelines read from the cache:
  - screenshot.png            clean full-page capture (base image; capture_regions
                              slices region_*.png from it at runtime)
  - marked_{text,link,image}.png  Set-of-Mark annotated captures (Stage 3b)

This is the OFFLINE cache-image regenerator: it needs only the bundled page.html
(no internet). Note the render is localhost-only (external CSS/JS/images are not
fetched), so regenerated images are lower-fidelity than the original online
captures — fine for a machinery-check quick run, documented in README.

Usage:
    python3 scripts/precache_stage3b.py
    python3 scripts/precache_stage3b.py --concurrency 8 --port 18080
    python3 scripts/precache_stage3b.py --only-liveweb   # skip SWDE pages
    python3 scripts/precache_stage3b.py --force          # re-render even if present
"""
import argparse, asyncio, hashlib, json, logging, sys, time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from threading import Thread

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache/pages")
VIEWPORT_W, VIEWPORT_H = 1280, 1100

MODALITIES = {
    "text": "p, span, h1, h2, h3, h4, h5, h6, li, td, th, dd, dt, label",
    "link": "a[href]",
    "image": "img",
}

INJECT_JS = """
((selector) => {{
    const elements = document.querySelectorAll('{selector}');
    const colors = ['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7',
                    '#DDA0DD','#98D8C8','#F7DC6F','#BB8FCE','#85C1E9'];
    let count = 0;
    elements.forEach((el, i) => {{
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;
        count++;
        const color = colors[i % colors.length];
        const box = document.createElement('div');
        box.className = 'vgs-som-box';
        box.style.cssText = `position:absolute;left:${{rect.left+window.scrollX}}px;`
            + `top:${{rect.top+window.scrollY}}px;width:${{rect.width}}px;`
            + `height:${{rect.height}}px;border:3px solid ${{color}};`
            + `pointer-events:none;z-index:99999;box-sizing:border-box;`;
        const label = document.createElement('span');
        label.textContent = i;
        label.style.cssText = `position:absolute;top:0;right:0;`
            + `background:${{color}};color:#fff;font-size:14px;font-weight:bold;`
            + `padding:2px 6px;z-index:100000;`;
        box.appendChild(label);
        document.body.appendChild(box);
    }});
    return count;
}})('{selector}')
"""

REMOVE_JS = "document.querySelectorAll('.vgs-som-box').forEach(el => el.remove());"


def get_liveweb_hashes() -> set[str]:
    """Get all URL hashes that belong to LiveWeb-IE dataset."""
    root = Path("data/LiveWeb_IE")
    if not (root / "dataset.json").exists():
        return set()
    dataset = json.load(open(root / "dataset.json"))
    hashes = set()
    for entries in dataset.values():
        for entry in entries:
            gf = entry["group_url"].lstrip("./")
            fp = root / gf
            if not fp.exists():
                continue
            for line in fp.read_text().strip().split("\n"):
                obj = json.loads(line)
                for gid, gdata in obj.items():
                    for url in gdata.get("group_url", []):
                        hashes.add(hashlib.sha256(url.encode()).hexdigest()[:16])
    return hashes


# ── HTTP Server ──
class CacheHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        url_hash = self.path.strip("/").split("/")[0]
        html_file = CACHE_DIR / url_hash / "page.html"
        if html_file.exists():
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html_file.read_bytes())
        else:
            self.send_error(404)

    def log_message(self, *a):
        pass


def start_server(port: int) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", port), CacheHandler)
    t = Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


async def process_one(browser, url_hash: str, port: int, sem: asyncio.Semaphore,
                      force: bool = False) -> dict:
    """Process one cached page: load locally, capture clean full-page screenshot,
    then inject bbox for 3 modalities and capture each marked screenshot."""
    cache_path = CACHE_DIR / url_hash

    # Skip if already done (all marked + the clean base screenshot present)
    done = ((cache_path / "screenshot.png").exists()
            and all((cache_path / f"marked_{m}.png").exists() for m in MODALITIES))
    if done and not force:
        return {"status": "skip"}

    async with sem:
        ctx = None
        try:
            ctx = await browser.new_context(
                viewport={"width": VIEWPORT_W, "height": VIEWPORT_H}
            )
            page = await ctx.new_page()
            local_url = f"http://127.0.0.1:{port}/{url_hash}"
            await page.goto(local_url, wait_until="domcontentloaded", timeout=15000)
            await asyncio.sleep(0.3)

            # Clean full-page base screenshot (no Set-of-Mark boxes). capture_regions
            # slices region_*.png from this at runtime, so we don't ship region crops.
            await page.evaluate(REMOVE_JS)
            await page.screenshot(path=str(cache_path / "screenshot.png"), full_page=True)

            counts = {}
            for modality, selector in MODALITIES.items():
                await page.evaluate(REMOVE_JS)
                js = INJECT_JS.format(selector=selector)
                count = await page.evaluate(js)
                counts[modality] = count or 0
                out_path = cache_path / f"marked_{modality}.png"
                await page.screenshot(path=str(out_path))

            await page.close()
            await ctx.close()
            return {"status": "ok", "counts": counts}


        except Exception as e:
            if ctx:
                try:
                    await ctx.close()
                except Exception:
                    pass
            return {"status": "err", "error": str(e)[:80]}


async def main():
    parser = argparse.ArgumentParser(description="Pre-cache VGS Stage 3b marked screenshots")
    parser.add_argument("--concurrency", type=int, default=5, help="Browser concurrency")
    parser.add_argument("--port", type=int, default=18080, help="Local HTTP server port")
    parser.add_argument("--only-liveweb", action="store_true", help="Only process LiveWeb-IE pages")
    parser.add_argument("--force", action="store_true", help="Re-render even if images already exist")
    args = parser.parse_args()

    # Discover pages to process
    all_dirs = sorted(d.name for d in CACHE_DIR.iterdir() if d.is_dir())
    if args.only_liveweb:
        liveweb = get_liveweb_hashes()
        targets = [d for d in all_dirs if d in liveweb]
    else:
        targets = all_dirs

    # Filter out already-done (needs clean screenshot.png + all 3 marked images)
    def _done(d: str) -> bool:
        p = CACHE_DIR / d
        return ((p / "screenshot.png").exists()
                and all((p / f"marked_{m}.png").exists() for m in MODALITIES))
    todo = targets if args.force else [d for d in targets if not _done(d)]


    logger.info(f"Cached pages: {len(all_dirs)} | LiveWeb-IE: {len(get_liveweb_hashes())} | "
                f"Todo: {len(todo)}")
    logger.info(f"Concurrency: {args.concurrency} | Port: {args.port}")

    if not todo:
        logger.info("Nothing to do!")
        return

    # Start local HTTP server
    server = start_server(args.port)
    logger.info(f"HTTP server started on :{args.port}")

    # Launch Playwright
    from playwright.async_api import async_playwright
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(headless=True)

    sem = asyncio.Semaphore(args.concurrency)
    stats = {"ok": 0, "skip": 0, "err": 0}
    t0 = time.time()

    # Progress tracking
    from tqdm import tqdm
    pbar = tqdm(total=len(todo), desc="Pre-caching marked screenshots",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    # Process in batches for stable progress
    BATCH = min(50, len(todo))
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        tasks = [process_one(browser, h, args.port, sem, force=args.force) for h in batch]
        results = await asyncio.gather(*tasks)
        for r in results:
            stats[r["status"]] = stats.get(r["status"], 0) + 1
        pbar.update(len(batch))
        elapsed = time.time() - t0
        rate = (stats["ok"] + stats["skip"]) / max(elapsed, 1)
        pbar.set_postfix_str(
            f"ok={stats['ok']} err={stats['err']} {rate:.1f} pages/s")

    pbar.close()
    await browser.close()
    await pw.stop()
    server.shutdown()

    elapsed = time.time() - t0
    logger.info(f"\nDone in {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    logger.info(f"OK: {stats['ok']} | Skip: {stats['skip']} | Errors: {stats['err']}")
    if stats["err"] > 0:
        logger.warning(f"{stats['err']} pages failed — check logs above")


if __name__ == "__main__":
    asyncio.run(main())
