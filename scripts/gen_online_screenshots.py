"""Generate online screenshots for books.toscrape.com case study."""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

OUT_DIR = Path("output/screenshots/books_case")
OUT_DIR.mkdir(parents=True, exist_ok=True)

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 900})

        # Collect book URLs from main page
        page = await ctx.new_page()
        await page.goto("https://books.toscrape.com/", wait_until="load", timeout=30000)
        await page.wait_for_timeout(1500)
        book_urls = []
        links = await page.query_selector_all("article.product_pod h3 a")
        for link in links[:5]:
            href = await link.get_attribute("href")
            book_urls.append(f"https://books.toscrape.com/{href}")
        await page.close()
        print(f"URLs: {book_urls[:2]}")

        # ===== Page A =====
        pageA = await ctx.new_page()
        await pageA.goto(book_urls[0], wait_until="load", timeout=30000)
        await pageA.wait_for_timeout(2000)
        title_a = await pageA.title()
        print(f"Page A: {title_a}")

        # Clean screenshot
        await pageA.screenshot(path=str(OUT_DIR / "page_a.png"))
        print("  -> page_a.png")

        # SoM annotations
        som_data = await pageA.evaluate("""() => {
            const colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'];
            const allImgs = Array.from(document.querySelectorAll('img'));
            const annotations = [];
            allImgs.forEach((el, idx) => {
                const rect = el.getBoundingClientRect();
                if (rect.width < 5 || rect.height < 5) return;
                const color = colors[idx % colors.length];
                el.style.outline = '4px solid ' + color;
                el.style.outlineOffset = '3px';
                const label = document.createElement('div');
                label.textContent = '#' + idx;
                label.style.cssText = 'position:absolute;top:' + (rect.top + window.scrollY - 4) + 'px;left:' + (rect.left + window.scrollX - 4) + 'px;background:' + color + ';color:white;font-size:14px;font-weight:bold;padding:3px 8px;border-radius:4px;z-index:99999;font-family:monospace;box-shadow:0 2px 4px rgba(0,0,0,0.3);';
                document.body.appendChild(label);
                annotations.push({idx, w: Math.round(rect.width), h: Math.round(rect.height)});
            });
            return annotations;
        }""")
        print(f"  SoM: {len(som_data)} marks")
        await pageA.wait_for_timeout(300)
        await pageA.screenshot(path=str(OUT_DIR / "som_screenshot.png"))
        print("  -> som_screenshot.png")

        # VLM selection overlay
        await pageA.evaluate("""() => {
            const mainImg = document.querySelector('div#product_gallery img');
            if (!mainImg) return;
            const rect = mainImg.getBoundingClientRect();
            mainImg.style.outline = '6px solid #27ae60';
            mainImg.style.outlineOffset = '4px';
            const badge = document.createElement('div');
            badge.innerHTML = '&#10003; VLM Selected #0';
            badge.style.cssText = 'position:absolute;top:' + (rect.top + window.scrollY + 8) + 'px;left:' + (rect.left + window.scrollX + 8) + 'px;background:#27ae60;color:white;font-size:18px;font-weight:bold;padding:8px 16px;border-radius:6px;z-index:999999;font-family:Arial,sans-serif;box-shadow:0 3px 10px rgba(0,0,0,0.3);border:3px solid white;';
            document.body.appendChild(badge);
            const jsonBox = document.createElement('div');
            jsonBox.innerHTML = 'VLM Output: { "selected_ids": [0] }';
            jsonBox.style.cssText = 'position:fixed;bottom:20px;right:20px;background:#2c3e50;color:#2ecc71;font-size:14px;font-weight:bold;padding:12px 20px;border-radius:8px;z-index:999999;font-family:monospace;box-shadow:0 4px 12px rgba(0,0,0,0.4);border:2px solid #2ecc71;';
            document.body.appendChild(jsonBox);
        }""")
        await pageA.wait_for_timeout(300)
        await pageA.screenshot(path=str(OUT_DIR / "som_selected.png"))
        print("  -> som_selected.png")
        await pageA.close()

        # ===== Page B =====
        pageB = await ctx.new_page()
        await pageB.goto(book_urls[1], wait_until="load", timeout=30000)
        await pageB.wait_for_timeout(2000)
        title_b = await pageB.title()
        print(f"Page B: {title_b}")
        await pageB.screenshot(path=str(OUT_DIR / "page_b.png"))
        print("  -> page_b.png")
        await pageB.close()
        await browser.close()

        print(f"\nDone! Page A={title_a}, Page B={title_b}, SoM marks={len(som_data)}")

asyncio.run(main())
