"""VGS Smoke Test — runs ONE page through the full VGS pipeline with verbose output.

Shows all intermediate results:
  Stage 1: Attribute Identification
  Stage 2: Visual Grounding (region matching)
  Stage 3: Element Pinpointing (bounding box selection)
  Stage 4: XPath Synthesis (final XPath generation)
  Stage 5: XPath Execution (extract values)
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%H:%M:%S")

from configs.config import VGSConfig
from utils.llm_client import LLMClient
from utils.browser import BrowserManager
from vgs.attribute_identification import AttributeIdentifier
from vgs.visual_grounding import VisualGrounder
from vgs.element_pinpointing import ElementPinpointer
from vgs.xpath_synthesis import XPathSynthesizer


SWDE_ROOT = Path(__file__).resolve().parent.parent / "data" / "swde" / "swde" / "sourceCode" / "sourceCode"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "smoke_test_vgs"


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ── Pick test page ──
    html_path = SWDE_ROOT / "auto_extracted" / "auto" / "auto-aol(2000)" / "0000.htm"
    if not html_path.exists():
        print(f"ERROR: {html_path} not found")
        return
    
    url = f"file://{html_path.resolve()}"
    query = "Extract the car model name"
    
    print("=" * 70)
    print("VGS SMOKE TEST — Full Pipeline with Intermediate Results")
    print("=" * 70)
    print(f"  URL: {url}")
    print(f"  Query: {query}")
    print(f"  HTML size: {html_path.stat().st_size} bytes")
    print(f"  Output dir: {OUTPUT_DIR}")
    print("=" * 70)
    
    # ── Setup ──
    config = VGSConfig()
    config.model_name = "qwen3-vl-32b-instruct"
    config.screenshot_dir = OUTPUT_DIR / "screenshots"
    config.screenshot_dir.mkdir(parents=True, exist_ok=True)
    config.headless = False  # Show browser for debugging
    
    llm = LLMClient(
        api_key=config.api_key,
        api_base=config.api_base,
        model=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        enable_monitor=False,
    )
    browser = BrowserManager(
        viewport_width=config.viewport_width,
        viewport_height=config.viewport_height,
        headless=config.headless,
        timeout_ms=config.page_load_timeout_ms,
    )
    
    identifier = AttributeIdentifier(llm)
    grounder = VisualGrounder(llm)
    pinpointer = ElementPinpointer(llm, browser)
    synthesizer = XPathSynthesizer(llm, config.neighbor_distance)
    
    await browser.start()
    
    try:
        # ══════════════════════════════════════════════════════════════════
        # STAGE 1: Attribute Identification
        # ══════════════════════════════════════════════════════════════════
        print("\n" + "━" * 70)
        print("▶ STAGE 1: Attribute Identification")
        print("━" * 70)
        print(f"  Input query: '{query}'")
        
        attributes = identifier.identify(query)
        
        print(f"  ✅ Identified attributes: {attributes}")
        
        # Save Stage 1 result
        stage1_result = {"query": query, "attributes": attributes}
        (OUTPUT_DIR / "stage1_attributes.json").write_text(
            json.dumps(stage1_result, indent=2, ensure_ascii=False)
        )
        
        # ══════════════════════════════════════════════════════════════════
        # LOAD PAGE & CAPTURE REGIONS
        # ══════════════════════════════════════════════════════════════════
        print("\n" + "━" * 70)
        print("▶ LOADING PAGE & CAPTURING REGIONS")
        print("━" * 70)
        
        page = await browser.load_page(url)
        print(f"  ✅ Page loaded: {url[-60:]}")
        
        # Get page title/content
        title = await page.title()
        print(f"  Page title: '{title}'")
        
        # Full HTML
        full_html = await page.content()
        print(f"  Full HTML length: {len(full_html)} chars")
        (OUTPUT_DIR / "full_html.html").write_text(full_html, encoding='utf-8')
        print(f"  → Saved to {OUTPUT_DIR / 'full_html.html'}")
        
        # Capture region screenshots
        sample_id = "smoke_test"
        screenshot_dir = config.screenshot_dir / sample_id
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        region_paths = await browser.capture_regions(page, screenshot_dir)
        print(f"  ✅ Captured {len(region_paths)} region screenshots")
        for i, rp in enumerate(region_paths):
            print(f"    Region {i}: {rp.name} ({rp.stat().st_size // 1024} KB)")
        
        # Process each attribute
        all_xpaths = {}
        all_values = {}
        
        for attr_idx, attribute in enumerate(attributes):
            print(f"\n{'─' * 70}")
            print(f"  Processing attribute [{attr_idx+1}/{len(attributes)}]: '{attribute}'")
            print(f"{'─' * 70}")
            
            # ══════════════════════════════════════════════════════════════
            # STAGE 2: Visual Grounding
            # ══════════════════════════════════════════════════════════════
            print(f"\n  ▶ STAGE 2: Visual Grounding for '{attribute}'")
            print(f"    Input: {len(region_paths)} region screenshots")
            
            matched_region = grounder.ground(attribute, region_paths)
            region_index = int(matched_region.stem.split("_")[-1])
            
            print(f"    ✅ Matched region: {matched_region.name} (index={region_index})")
            print(f"    → Region screenshot: {matched_region}")
            
            # Scroll to matched region
            scroll_y = region_index * config.viewport_height
            await page.evaluate(f"window.scrollTo(0, {scroll_y})")
            await asyncio.sleep(0.3)
            print(f"    Scrolled to y={scroll_y}")
            
            # ══════════════════════════════════════════════════════════════
            # STAGE 3: Element Pinpointing
            # ══════════════════════════════════════════════════════════════
            print(f"\n  ▶ STAGE 3: Element Pinpointing for '{attribute}'")
            
            modality = pinpointer.classify_modality(attribute)
            css_selector = pinpointer._modality_to_selector(modality)
            print(f"    Modality: {modality}")
            print(f"    CSS selector: {css_selector}")
            
            selected_ids = await pinpointer.pinpoint(
                page, attribute, region_index, screenshot_dir
            )
            print(f"    ✅ Selected element IDs: {selected_ids}")
            
            # Check marked screenshot
            marked_path = screenshot_dir / f"region_{region_index}_marked.png"
            if marked_path.exists():
                print(f"    → Marked screenshot: {marked_path} ({marked_path.stat().st_size // 1024} KB)")
            else:
                print(f"    ⚠️ Marked screenshot NOT found: {marked_path}")
            
            # ══════════════════════════════════════════════════════════════
            # STAGE 4: XPath Synthesis
            # ══════════════════════════════════════════════════════════════
            print(f"\n  ▶ STAGE 4: XPath Synthesis for '{attribute}'")
            print(f"    Input: selected_ids={selected_ids}, css='{css_selector}'")
            
            xpath = await synthesizer.synthesize(
                page, attribute, selected_ids, css_selector, marked_path
            )
            
            print(f"    ✅ Generated XPath: {xpath}")
            all_xpaths[attribute] = xpath
            
            # ══════════════════════════════════════════════════════════════
            # STAGE 5: XPath Execution
            # ══════════════════════════════════════════════════════════════
            print(f"\n  ▶ STAGE 5: Execute XPath on page")
            print(f"    XPath: {xpath}")
            
            if xpath:
                extracted = await browser.execute_xpath(page, xpath)
                all_values[attribute] = extracted
                print(f"    ✅ Extracted values: {extracted}")
                if not extracted:
                    print(f"    ⚠️ XPath returned EMPTY! This is the failure point!")
                    # Try to diagnose: check if xpath elements exist
                    import re
                    cls_matches = re.findall(r"@class='([^']+)'", xpath)
                    for cls in cls_matches:
                        exists = cls in full_html
                        print(f"       Class '{cls}' in HTML: {exists}")
            else:
                all_values[attribute] = []
                print(f"    ⚠️ No XPath generated!")
        
        # ══════════════════════════════════════════════════════════════════
        # FINAL SUMMARY
        # ══════════════════════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print("FINAL RESULTS SUMMARY")
        print("=" * 70)
        for attr in attributes:
            xpath = all_xpaths.get(attr, "")
            vals = all_values.get(attr, [])
            status = "✅" if vals else "❌"
            print(f"  {status} {attr}")
            print(f"     XPath: {xpath}")
            print(f"     Values: {vals}")
        
        # Save final results
        final_result = {
            "url": url,
            "query": query,
            "attributes": attributes,
            "xpaths": all_xpaths,
            "values": all_values,
        }
        (OUTPUT_DIR / "final_result.json").write_text(
            json.dumps(final_result, indent=2, ensure_ascii=False)
        )
        print(f"\n  Results saved to: {OUTPUT_DIR / 'final_result.json'}")
        
        # Also check GT
        print(f"\n  Ground truth for this page:")
        gt_dir = SWDE_ROOT.parent.parent / "groundtruth" / "groundtruth" / "auto" / "auto-aol(2000)"
        if gt_dir.exists():
            for gt_file in sorted(gt_dir.glob("*.txt"))[:5]:
                lines = gt_file.read_text(encoding='utf-8', errors='replace').splitlines()
                if len(lines) > 0:
                    # First line = first page value
                    print(f"    {gt_file.stem}: '{lines[0][:60]}'")
        
    finally:
        try:
            await page.context.close()
        except:
            pass
        await browser.stop()


if __name__ == "__main__":
    asyncio.run(main())
