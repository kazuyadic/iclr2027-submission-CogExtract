"""
Generate online screenshots for books.toscrape.com case study.
- Navigate to live pages via main page
- Take clean screenshots (Page A, Page B)
- Inject SoM annotations on Page A
- Show which mark the VLM selected (#0)
"""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

OUT_DIR = Path("output/screenshots/books_case")
OUT_DIR.mkdir(parents=True, exist_ok=True)

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 900})

        # --- Collect book URLs from main page ---
        page = await ctx.new_page()
        await page.goto("https://books.toscrape.com/", wait_until="load", timeout=30000)
        await page.wait_for_timeout(1500)

        book_urls = []
        links = await page.query_selector_all("article.product_pod h3 a")
        for link in links[:5]:
            href = await link.get_attribute("href")
            book_urls.append(f"https://books.toscrape.com/{href}")
        await page.close()

        print(f"Collected {len(book_urls)} book URLs:")
        for i, u in enumerate(book_urls):
            print(f"  [{i}] {u}")

        # ========== Page A (Book 0) ==========
        pageA = await ctx.new_page()
        print(f"\n--- Page A ---")
        await pageA.goto(book_urls[0], wait_until="load", timeout=30000)
        await pageA.wait_for_timeout(2000)
        title_a = await pageA.title()
        print(f"Title: {title_a}")

        # Verify images
        imgs_a = await pageA.query_selector_all("img")
        print(f"Images: {len(imgs_a)}")
        for i, img in enumerate(imgs_a):
            src = await img.get_attribute("src")
            box = await img.bounding_box()
            print(f"  #{i}: {box['width']}x{box['height']} src={src[:80] if src else 'None'}")

        # Clean screenshot Page A
        await pageA.screenshot(path=str(OUT_DIR / "page_a.png"))
        print("Saved page_a.png")

        # ========== SoM Annotation on Page A ==========
        # Annotate all <img> elements with colored borders + labels
        som_data = await pageA.evaluate("""
        () => {
            const colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22'];
            const allImgs = Array.from(document.querySelectorAll('img'));
            const annotations = [];

            allImgs.forEach((el, idx) => {
                const rect = el.getBoundingClientRect();
                if (rect.width < 5 || rect.height < 5) return;

                const color = colors[idx % colors.length];

                // Colored border
                el.style.outline = `4px solid ${color}`;
                el.style.outlineOffset = '3px';

                // ID label (top-left)
                const label = document.createElement('div');
                label.textContent = `#${idx}`;
                label.style.cssText = `
                    position: absolute;
                    top: ${rect.top + window.scrollY - 4}px;
                    left: ${rect.left + window.scrollX - 4}px;
                    background: ${color};
                    color: white;
                    font-size: 14px;
                    font-weight: bold;
                    padding: 3px 8px;
                    border-radius: 4px;
                    z-index: 99999;
                    font-family: 'Courier New', monospace;
                    line-height: 1;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                `;
                document.body.appendChild(label);

                annotations.push({
                    idx, color,
                    src: (el.getAttribute('src') || '').substring(0, 100),
                    w: Math.round(rect.width),
                    h: Math.round(rect.height),
                    x: Math.round(rect.left),
                    y: Math.round(rect.top)
                });
            });
            return annotations;
        }
        """)

        print(f"\nSoM annotations ({len(som_data)}):")
        for a in som_data:
            print(f"  #{a['idx']}: {a['w']}x{a['h']} at ({a['x']},{a['y']}) color={a['color']}")

        await pageA.wait_for_timeout(500)

        # Take SoM screenshot
        await pageA.screenshot(path=str(OUT_DIR / "som_screenshot.png"))
        print("Saved som_screenshot.png")

        # ========== VLM Selection Overlay ==========
        # VLM selected #0 (the main product cover image)
        sel_data = await pageA.evaluate("""
        () => {
            const mainImg = document.querySelector('div#product_gallery img');
            if (!mainImg) return null;
            const rect = mainImg.getBoundingClientRect();

            // Thick green selected border
            mainImg.style.outline = '6px solid #27ae60';
            mainImg.style.outlineOffset = '4px';

            // Selection badge
            const badge = document.createElement('div');
            badge.innerHTML = '✓ VLM Selected #0';
            badge.style.cssText = `
                position: absolute;
                top: ${rect.top + window.scrollY + 8}px;
                left: ${rect.left + window.scrollX + 8}px;
                background: #27ae60;
                color: white;
                font-size: 18px;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 6px;
                z-index: 999999;
                font-family: Arial, sans-serif;
                box-shadow: 0 3px 10px rgba(0,0,0,0.3);
                border: 3px solid white;
            `;
            document.body.appendChild(badge);

            // Also add selected_ids JSON display at bottom
            const jsonBox = document.createElement('div');
            jsonBox.innerHTML = 'VLM Output: { "selected_ids": [0] }';
            jsonBox.style.cssText = `
                position: fixed;
                bottom: 20px;
                right: 20px;
                background: #2c3e50;
                color: #2ecc71;
                font-size: 14px;
                font-weight: bold;
                padding: 12px 20px;
                border-radius: 8px;
                z-index: 999999;
                font-family: 'Courier New', monospace;
                box-shadow: 0 4px 12px rgba(0,0,0,0.4);
                border: 2px solid #2ecc71;
            `;
            document.body.appendChild(jsonBox);

            return {
                w: Math.round(rect.width),
                h: Math.round(rect.height),
                x: Math.round(rect.left),
                y: Math.round(rect.top)
            };
        }
        """)

        if sel_data:
            print(f"\nVLM selection: #0 at ({sel_data['x']},{sel_data['y']}) {sel_data['w']}x{sel_data['h']}")

        await pageA.wait_for_timeout(300)
        await pageA.screenshot(path=str(OUT_DIR / "som_selected.png"))
        print("Saved som_selected.png")
        await pageA.close()

        # ========== Page B (Book 1) ==========
        pageB = await ctx.new_page()
        print(f"\n--- Page B ---")
        await pageB.goto(book_urls[1], wait_until="load", timeout=30000)
        await pageB.wait_for_timeout(2000)
        title_b = await pageB.title()
        print(f"Title: {title_b}")

        imgs_b = await pageB.query_selector_all("img")
        print(f"Images: {len(imgs_b)}")
        for i, img in enumerate(imgs_b):
            src = await img.get_attribute("src")
            box = await img.bounding_box()
            print(f"  #{i}: {box['width']}x{box['height']} src={src[:80] if src else 'None'}")

        await pageB.screenshot(path=str(OUT_DIR / "page_b.png"))
        print("Saved page_b.png")
        await pageB.close()

        await browser.close()

        # Print summary for paper
        print("\n" + "="*60)
        print("SUMMARY FOR PAPER:")
        print(f"  Page A: {title_a}")
        print(f"  Page B: {title_b}")
        print(f"  SoM: {len(som_data)} image(s) annotated")
        print(f"  VLM selected: #0 (main product cover)")
        print("="*60)

asyncio.run(main())
"""
Generate online screenshots for books.toscrape.com case study.
1. Navigate to live pages
2. Take clean screenshots (Page A, Page B)
3. Inject SoM annotations on Page A and take annotated screenshot
4. Mark which element the VLM selected (#0 = main product image)
"""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

# books.toscrape.com book detail pages
URL_A = "https://books.toscrape.com/catalogue/10-day-green-smoothie-cleanse-lose-up-to-15-pounds-in-10-days_950/index.html"
URL_B = "https://books.toscrape.com/catalogue/10-happier-how-i-tamed-the-voice-in-my-head-reduced-stress-without-losing-my-edge-and-found-self-help-that-actually-works-a-true-story_949/index.html"

OUT_DIR = Path("output/screenshots/books_case")
OUT_DIR.mkdir(parents=True, exist_ok=True)

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 900})
        
        # --- Page A ---
        page = await ctx.new_page()
        print(f"Loading Page A: {URL_A}")
        try:
            await page.goto(URL_A, wait_until="networkidle", timeout=30000)
        except Exception as e:
            print(f"  networkidle timeout, trying load: {e}")
            await page.goto(URL_A, wait_until="load", timeout=30000)
        
        await page.wait_for_timeout(2000)
        title_a = await page.title()
        print(f"  Title: {title_a}")
        
        # Check what images are on the page
        imgs = await page.query_selector_all("img")
        print(f"  Found {len(imgs)} <img> elements")
        for i, img in enumerate(imgs):
            src = await img.get_attribute("src")
            alt = await img.get_attribute("alt") or ""
            box = await img.bounding_box()
            print(f"    #{i}: src={src[:80] if src else 'None'}, alt={alt[:40]}, box={box}")
        
        # Take clean Page A screenshot
        await page.screenshot(path=str(OUT_DIR / "page_a.png"))
        print(f"  Saved page_a.png")
        
        # --- SoM annotation on Page A ---
        # Find all <img> elements in the product_gallery div (target region)
        # For image_url attribute, the VLM would see SoM on the product gallery area
        target_imgs = await page.query_selector_all("div#product_gallery img, div.carousel img, article.product_page img")
        if not target_imgs:
            target_imgs = imgs  # fallback to all images
        
        print(f"\n  SoM targets: {len(target_imgs)} images")
        
        # Inject SoM annotations via JS
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#e91e63", "#00bcd4", "#ff5722"]
        
        som_result = await page.evaluate("""
        (colors) => {
            // Find all img elements in the main product area (not footer/nav)
            const allImgs = Array.from(document.querySelectorAll('article.product_page img, div#product_gallery img, div.carousel img'));
            // Deduplicate by src
            const seen = new Set();
            const uniqueImgs = [];
            for (const img of allImgs) {
                const src = img.getAttribute('src') || '';
                if (!seen.has(src)) {
                    seen.add(src);
                    uniqueImgs.push(img);
                }
            }
            
            const annotations = [];
            uniqueImgs.forEach((el, idx) => {
                const rect = el.getBoundingClientRect();
                if (rect.width < 5 || rect.height < 5) return;  // skip tiny elements
                
                const color = colors[idx % colors.length];
                
                // Colored border
                el.style.outline = `3px solid ${color}`;
                el.style.outlineOffset = '2px';
                
                // ID label (top-left corner)
                const label = document.createElement('div');
                label.textContent = `#${idx}`;
                label.style.cssText = `
                    position: fixed;
                    top: ${rect.top + window.scrollY - 2}px;
                    left: ${rect.left + window.scrollX - 2}px;
                    background: ${color};
                    color: white;
                    font-size: 12px;
                    font-weight: bold;
                    padding: 2px 6px;
                    border-radius: 3px;
                    z-index: 99999;
                    font-family: monospace;
                    line-height: 1;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
                `;
                document.body.appendChild(label);
                
                annotations.push({
                    idx,
                    color,
                    src: (el.getAttribute('src') || '').substring(0, 80),
                    alt: (el.getAttribute('alt') || '').substring(0, 40),
                    x: Math.round(rect.left),
                    y: Math.round(rect.top),
                    w: Math.round(rect.width),
                    h: Math.round(rect.height)
                });
            });
            return annotations;
        }
        """, colors)
        
        print(f"  Injected {len(som_result)} SoM annotations:")
        for a in som_result:
            print(f"    #{a['idx']}: {a['w']}x{a['h']} at ({a['x']},{a['y']}) src={a['src']}")
        
        await page.wait_for_timeout(500)
        
        # Take SoM screenshot (full page to capture all annotations)
        await page.screenshot(path=str(OUT_DIR / "som_screenshot.png"), full_page=True)
        print(f"  Saved som_screenshot.png (full page)")
        
        # Now create a version showing VLM selected mark #0
        # The VLM selects #0 (main product cover image) for image_url
        selected_js = """
        () => {
            // Add a big "VLM SELECTED #0" indicator on the main product image
            const mainImg = document.querySelector('div#product_gallery img, div.carousel-inner .item.active img, article.product_page .carousel img');
            if (!mainImg) return 'no main img found';
            
            const rect = mainImg.getBoundingClientRect();
            
            // Big green selection indicator
            const sel = document.createElement('div');
            sel.innerHTML = '✓ VLM Selected #0';
            sel.style.cssText = `
                position: fixed;
                top: ${rect.top + window.scrollY + rect.height/2 - 20}px;
                left: ${rect.left + window.scrollX + rect.width/2 - 80}px;
                background: #27ae60;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 6px;
                z-index: 999999;
                font-family: sans-serif;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                border: 3px solid white;
            `;
            document.body.appendChild(sel);
            
            // Extra thick green border on selected element
            mainImg.style.outline = '5px solid #27ae60';
            mainImg.style.outlineOffset = '3px';
            
            return `selected #0 at (${Math.round(rect.left)},${Math.round(rect.top)}) ${Math.round(rect.width)}x${Math.round(rect.height)}`;
        }
        """
        sel_result = await page.evaluate(selected_js)
        print(f"  VLM selection: {sel_result}")
        
        await page.wait_for_timeout(300)
        await page.screenshot(path=str(OUT_DIR / "som_selected.png"), full_page=True)
        print(f"  Saved som_selected.png (with VLM selection)")
        
        await page.close()
        
        # --- Page B ---
        page2 = await ctx.new_page()
        print(f"\nLoading Page B: {URL_B}")
        try:
            await page2.goto(URL_B, wait_until="networkidle", timeout=30000)
        except Exception as e:
            print(f"  networkidle timeout, trying load: {e}")
            await page2.goto(URL_B, wait_until="load", timeout=30000)
        
        await page2.wait_for_timeout(2000)
        title_b = await page2.title()
        print(f"  Title: {title_b}")
        
        imgs_b = await page2.query_selector_all("img")
        print(f"  Found {len(imgs_b)} <img> elements")
        for i, img in enumerate(imgs_b):
            src = await img.get_attribute("src")
            box = await img.bounding_box()
            print(f"    #{i}: src={src[:80] if src else 'None'}, box={box}")
        
        await page2.screenshot(path=str(OUT_DIR / "page_b.png"))
        print(f"  Saved page_b.png")
        
        await page2.close()
        await browser.close()
        
        print("\nDone! All screenshots saved to", OUT_DIR)

asyncio.run(main())
