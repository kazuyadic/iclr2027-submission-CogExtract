"""Prompt templates for each VGS stage, extracted from the paper's Appendix J."""

ATTRIBUTE_IDENTIFICATION_PROMPT = """You are an Attribute Extractor that converts user extraction requests into structured JSON payloads for web scraping systems.
Task: Transform a user's data extraction request into a compact JSON specification that defines what attributes to extract.
Instructions:
- Analyze the user's request to identify specific data points they want extracted
- Focus on the minimal set of attributes needed to fulfill the user's request
Rules:
- JSON Structure: Return a JSON object with exactly one key: "attributes"
- The "attributes" key must contain a list of attribute names
- Each attribute name should be a concise, descriptive label for the data point
- Do not include any additional keys or nested structures
Please output in the following JSON format:
{{
    "attributes": ["attribute_1", "attribute_2", ...]
}}

User request: {query}"""

VISUAL_GROUNDING_PROMPT = """You are a visual grounding assistant that identifies which webpage region contains a specific UI element from a set of viewport screenshots.
Task: Given a set of viewport screenshots representing different regions of a webpage and a requested attribute, determine which specific region (if any) contains the target attribute clearly visible within its boundaries.
Instructions:
- You will receive multiple viewport screenshots representing different regions of a webpage
- Each screenshot will be labeled with a region identifier
- Analyze each region independently to determine if it contains the target attribute
Rules:
- Evaluate each region screenshot individually for the presence of the target attribute
- Do not assume elements exist based on typical webpage patterns or templates
- Focus only on elements that are clearly within each region's viewport boundaries
- Return only the single best-matching region
Please output in the following JSON format:
{{
    "matching_region": ""
}}

Target attribute: {attribute}"""

ELEMENT_SCANNING_PROMPT = """You are a precise visual extractor that analyzes a web UI screenshot to extract specific attribute content based on modality classification.
Task: Given a viewport screenshot and an attribute name, determine the correct modality from the attribute name and perform the appropriate extraction task.
Instructions:
Classify the attribute into exactly one modality based on naming patterns:
1. TEXT Modality - For textual attributes
- Extract all visible text strings exactly as they appear
- Preserve natural reading order (top-to-bottom, left-to-right)
- No text normalization or paraphrasing
2. IMAGE Modality - For image-related attributes (containing keywords like: image, photo, picture, thumbnail, logo, icon, banner, poster, fanart, artwork, cover, avatar, badge, flag, screenshot, gallery)
- Count all visible images in the region
- Report total count only
3. LINK/HYPERLINK Modality - For link-related attributes (containing keywords like: link, url, href, hyperlink, redirect, profile link, homepage, website)
- Count all visible hyperlinks/clickable elements in the region
- Report total count only
Please output in the following JSON format:
{{
    "modality": "",
    "items": []
}}

Target attribute: {attribute}"""

ELEMENT_SELECTION_PROMPT = """You are a precise visual element selector that analyzes an annotated region screenshot to identify specific UI elements matching a target attribute.
Task: Given an annotated region screenshot where potential candidate elements have colored bounding boxes with unique numerical labels, select the exact bounding box IDs that contain values for the specified target attribute.
Instructions:
- Each element is outlined with a colored box containing a small ID label in the top-right corner
- The label shows an integer ID with white text on a colored background matching the box color
- Focus on elements that directly contain or represent the values of the target attribute
Rules:
- Relevance Only: Return IDs only for bounding boxes that directly contain values of the target attribute
- Completeness: Include all matching elements, not just the first occurrence
- No Guessing: If no bounding box matches, return an empty list
- ID Accuracy: Use exact integer IDs as shown in labels
- The selected_ids list must NOT be empty if matching elements exist on the screenshot
Please output in the following JSON format:
{{
    "selected_ids": [2, 5, 8]
}}

Target attribute: {attribute}"""

# ── Text-based Element Selection (for cached/offline pages) ──

TEXT_ELEMENT_SELECTION_PROMPT = """You are a precise element selector. You are given a webpage screenshot and a numbered list of HTML elements found on this page. Select the element IDs that contain values for the target attribute.

Target attribute: {attribute}

There are {total_count} candidate elements on this page. Here is the list:
{element_list}

Instructions:
- Look at the screenshot to understand the page layout and content
- Match the visible content to the HTML element descriptions above
- Select elements that directly contain or represent values of "{attribute}"
- For list pages, include ALL matching elements (not just the first)
- If the attribute is a link, select <a> elements; if image, select <img> elements

Rules:
- Return IDs of elements that contain the target attribute's values
- Be complete: include all matching elements
- If unsure, prefer including an element over excluding it
- The selected_ids list must NOT be empty if matching elements exist

Please output in the following JSON format:
{{
    "selected_ids": [3, 7, 12]
}}"""

XPATH_SYNTHESIS_PROMPT = """You are an expert XPath generator that creates robust, generalizable selectors for web scraping based on HTML segments, marked region, and target attribute.
Task: Generate one reliable XPath selector from the provided HTML segments, marked region screenshot, and target attribute that will work across structurally similar pages.
Instructions:
- Analyze HTML segments and marked regions to identify structural patterns and stable anchoring elements
- Use the marked region screenshot as visual context to understand the target element's location and appearance
Rules:
- Generalizability: Create XPaths that work across pages with similar structure, not just the current page
- Stability: Prefer class names, semantic tags, and structural patterns over dynamic IDs or positional indices
- Specificity: Balance between being too broad (matching unwanted elements) and too narrow (breaking on similar pages)
- Avoid embedding exact literal values or visible text from the HTML content
- Avoid using dynamic attributes like data-reactid, data-v-*, or session-specific identifiers
Please output in the following JSON format:
{{
    "xpath": "",
    "explanation": ""
}}

Target attribute: {attribute}
HTML segments:
{html_segments}"""

# ── Modality-Aware XPath Synthesis Prompts ──
# Human cognitive strategy differs by data type: we locate text by reading,
# links by structural position, and images by visual placement.

XPATH_SYNTHESIS_LINK_PROMPT = """You are an expert XPath generator specializing in hyperlink extraction.

Cognitive principle: When humans look for links in a list page, they locate the repeating container (card, row, item) first, then find the <a> element within each container by its structural position — NOT by its visible text or href content, because those change across items and pages.

Task: Generate one robust XPath selector for the link attribute from the provided HTML segments and screenshot.

STRICT RULES:
1. The XPath MUST select <a> elements
2. NEVER use text(), contains(text(), ...), or normalize-space() predicates — link text varies per item
3. NEVER use contains(@href, '...') with path segments — href paths differ across pages
4. ONLY use structural selectors: tag hierarchy, @class, semantic attributes, position relative to a container
5. If this is a list page, the XPath must match ALL item links, not just one
6. Prefer: //container[class]//a or //container[class]/a[@href]

Please output in the following JSON format:
{{
    "xpath": "",
    "explanation": ""
}}

Target attribute: {attribute}
HTML segments:
{html_segments}"""

XPATH_SYNTHESIS_IMAGE_PROMPT = """You are an expert XPath generator specializing in image extraction.

Cognitive principle: When humans look for product/item images on a list page, they locate the repeating visual card first, then find the <img> element within each card by its structural position — NOT by its @alt text or @src URL, because those are unique to each item.

Task: Generate one robust XPath selector for the image attribute from the provided HTML segments and screenshot.

STRICT RULES:
1. The XPath MUST select <img> elements (values are extracted via @src)
2. NEVER use contains(@alt, '...') — alt text is item-specific and changes per page
3. NEVER use contains(@src, '...') with specific filenames or paths — src URLs are unique per image
4. ONLY use structural selectors: tag hierarchy, @class on the <img> or its container
5. If this is a list page, the XPath must match ALL item images, not just one
6. Prefer: //container[class]//img or //div[class]/img

Please output in the following JSON format:
{{
    "xpath": "",
    "explanation": ""
}}

Target attribute: {attribute}
HTML segments:
{html_segments}"""


# ═══════════════════════════════════════════════════════════════════
