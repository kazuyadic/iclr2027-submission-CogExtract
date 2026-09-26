"""Prompt templates for the Cog (Reflection-guided XPath Refinement) framework."""

# ── Stage 4: Failure Diagnosis ──

FAILURE_DIAGNOSIS_PROMPT = """You are an XPath debugging expert. An XPath was generated on Page A and extracts values correctly there, but it produces unexpected results on Page B (a structurally similar page from the same website).

Your task: diagnose the failure type and explain why.

Target attribute: {attribute}
XPath: {xpath}

Page A results ({count_a} matches):
{values_a}

Page B results ({count_b} matches):
{values_b}

Page B HTML (relevant section):
{html_b_section}

Classify the failure into exactly ONE of these types:

1. "empty" — XPath matches 0 nodes on Page B. Common cause: content-specific predicates (text(), @alt, @src with literal values) that only exist on Page A.

2. "wrong" — XPath matches nodes on Page B, but the extracted values are clearly NOT the target attribute. For example, extracting a navigation link instead of a product price.

3. "over_extraction" — XPath matches far too many nodes (e.g., expected ~1-5, got 50+). The selector is too broad.

4. "format_mismatch" — XPath matches nodes but the value type is wrong (e.g., expected text price, got an image URL; expected a URL, got plain text).

Output JSON:
{{
    "failure_type": "empty|wrong|over_extraction|format_mismatch",
    "diagnosis": "one-sentence explanation of WHY the XPath fails on Page B",
    "failing_predicate": "the specific part of the XPath that causes the failure (e.g., contains(@alt, 'Zelda'))"
}}"""


# ── Stage 5: Reflection & Revision ──

REFLECTION_PROMPT = """You are an XPath repair agent. An XPath works on Page A but fails on Page B. You have been given a structured diagnosis. Now reflect on the failure and produce a fixed XPath.

Target attribute: {attribute}
Original XPath: {xpath}

=== Diagnosis ===
Failure type: {failure_type}
Diagnosis: {diagnosis}
Failing predicate: {failing_predicate}

=== Evidence ===
Page A ({count_a} matches):
{values_a}

Page B ({count_b} matches):
{values_b}

Page A HTML (relevant section):
{html_a_section}

Page B HTML (relevant section):
{html_b_section}

=== Repair Strategies by Failure Type ===

For "empty":
- Remove content-specific predicates (text(), @alt, @src with literal values)
- Replace with structural selectors (@class, tag hierarchy, position relative to landmarks)
- Keep the XPath specific enough to avoid over-extraction

For "wrong":
- The XPath selects the wrong element type — adjust the tag or container
- Look for distinguishing @class or @data-* attributes on the correct element
- Use the Page A successful extraction as reference for what the right element looks like

For "over_extraction":
- Add a more specific container or @class filter
- Narrow with positional or structural constraints
- Ensure the XPath targets the specific repeating unit, not a broader container

For "format_mismatch":
- Adjust the selected tag (e.g., <img> for images, <a> for links, <span> for text)
- Check if the value should come from an attribute (@src, @href) vs text content

=== Conservative Repair Rules (MUST follow) ===
These rules prevent over-generalization. Apply them IN ORDER before making any other changes:

R1. Simplify compound class predicates: @class='active item card' → contains(@class,'active')
    Only keep the FIRST token of multi-word class values. Never invent new class names.

R2. Remove positional predicates: div[3]/span[1] → div/span
    Strip all [N] numeric indices — element order varies across pages.

R3. Shorten deeply nested paths: if path has ≥5 steps, keep only the last 3 with // prefix
    e.g., /html/body/div/main/section/div/span → //section/div/span

R4. Exact class → contains: [@class='title'] → [contains(@class,'title')]
    Handles dynamically appended classes.

CRITICAL CONSTRAINTS:
- The revised XPath MUST extract a SIMILAR NUMBER of values on Page A as the original (within ±50%).
  If original extracts 1 value, revised must extract 1-2. If original extracts 5, revised must extract 3-8.
  NEVER produce an XPath that extracts 10x more values than the original — that means over-generalization.
- Make the MINIMAL change needed to fix the failure. Do NOT rewrite the entire XPath.
- If the failing predicate is a class name, try contains(@class,'first-token') BEFORE trying broader structural selectors.
- Prefer keeping the original XPath structure and only modifying the failing predicate.

=== Instructions ===
1. First apply R1-R4 to the original XPath to get a baseline candidate
2. Check if the baseline candidate fixes the failure on Page B
3. Only if R1-R4 are insufficient, make additional targeted changes
4. Explain your reasoning step by step
5. The fixed XPath MUST work on BOTH Page A and Page B
6. Use ONLY basic XPath 1.0 syntax: //, /, [], @, text(), contains(), tag names
7. Do NOT use translate(), normalize-space(), string-length() or other advanced functions
8. Do NOT use text content, product names, or page-specific literal values as predicates

Output JSON:
{{
    "reasoning": "step-by-step explanation of the fix",
    "revision_strategy": "one-line summary of what was changed",
    "revised_xpath": "the fixed XPath expression"
}}"""


# ── Stage 5 (retry): Reflection with history ──

REFLECTION_WITH_HISTORY_PROMPT = """You are an XPath repair agent. Previous repair attempts have failed. Learn from past failures and try a different approach.

Target attribute: {attribute}

=== Repair History ===
{repair_history}

=== Current State ===
Latest XPath: {xpath}
Failure type: {failure_type}
Diagnosis: {diagnosis}

Page A HTML (relevant section):
{html_a_section}

Page B HTML (relevant section):
{html_b_section}

=== Conservative Repair Rules (MUST follow) ===
These rules prevent over-generalization. Apply them IN ORDER before making any other changes:

R1. Simplify compound class predicates: @class='active item card' → contains(@class,'active')
    Only keep the FIRST token of multi-word class values. Never invent new class names.

R2. Remove positional predicates: div[3]/span[1] → div/span
    Strip all [N] numeric indices — element order varies across pages.

R3. Shorten deeply nested paths: if path has ≥5 steps, keep only the last 3 with // prefix
    e.g., /html/body/div/main/section/div/span → //section/div/span

R4. Exact class → contains: [@class='title'] → [contains(@class,'title')]
    Handles dynamically appended classes.

CRITICAL CONSTRAINTS:
- The revised XPath MUST extract a SIMILAR NUMBER of values on Page A as the original (within ±50%).
  If original extracts 1 value, revised must extract 1-2. If original extracts 5, revised must extract 3-8.
  NEVER produce an XPath that extracts 10x more values than the original — that means over-generalization.
- Make the MINIMAL change needed to fix the failure. Do NOT rewrite the entire XPath.
- If the failing predicate is a class name, try contains(@class,'first-token') BEFORE trying broader structural selectors.
- Prefer keeping the original XPath structure and only modifying the failing predicate.

=== Instructions ===
1. First apply R1-R4 to the latest XPath to get a baseline candidate
2. Check if the baseline candidate fixes the failure on Page B
3. Only if R1-R4 are insufficient, make additional targeted changes
4. Do NOT repeat any previously attempted XPath
5. Try a fundamentally different selection strategy if previous attempts failed
6. Focus on structural patterns that are stable across both pages
7. If class-based selectors failed, try tag hierarchy or landmark-relative paths
8. If positional selectors failed, try semantic attributes
9. Use ONLY basic XPath 1.0 syntax: //, /, [], @, text(), contains(), tag names
10. Do NOT use translate(), normalize-space(), string-length() or other advanced functions
11. Do NOT use text content, product names, or page-specific literal values as predicates

Output JSON:
{{
    "reasoning": "what went wrong before and what new approach you are trying",
    "revision_strategy": "one-line summary",
    "revised_xpath": "the new XPath expression"
}}"""


# ═══════════════════════════════════════════════════════════════════
# Cog — Multi-Candidate XPath Synthesis with Structural Context
# ═══════════════════════════════════════════════════════════════════

MULTI_CANDIDATE_SYNTHESIS_PROMPT = """You are an expert XPath engineer. Your task: extract the VALUE of "{attribute}" from web pages.

You are given a screenshot showing the page and HTML segments around candidate elements. Generate THREE different XPath expressions that could correctly extract the target attribute's VALUE.

=== Structural Context (from DOM analysis) ===
{structural_context}

=== CRITICAL: What to Extract ===
- You must extract the ACTUAL VALUE of "{attribute}", NOT labels/headers/descriptors
- For example, if attribute is "title", extract the actual title TEXT, not a label that says "Title:"
- If attribute is "price", extract "$29.99", not a label that says "Price:"
- Look at the SCREENSHOT to identify where the actual value appears visually

=== Diversity Requirement ===
Generate 3 XPaths using DIFFERENT structural paths to reach the same target value:
- They should all aim to extract the SAME content (the actual value)
- But use different DOM navigation strategies to get there
- This ensures at least one will generalize to other pages from the same site

=== Rules ===
1. Target the element containing the ACTUAL VALUE, not labels/descriptors
2. Prefer SHORT paths (2-3 steps with //) over deeply nested ones
3. NEVER use positional indices [N]
4. NEVER embed literal text values or page-specific content as predicates
5. Use contains(@class, 'token') for compound class values
6. For repeating content (lists), match ALL items
7. Each candidate must be independently valid
8. Prefer selectors within the MAIN CONTENT area — avoid elements inside navigation bars, sidebars, footers, or other UI chrome
9. If the attribute appears to be a list of items, ensure at least one candidate targets the repeating container structure

Output JSON:
{{
    "xpath_1": "first XPath to extract {attribute} value",
    "xpath_2": "second alternative XPath",
    "xpath_3": "third alternative XPath",
    "target_description": "brief description of what the actual value looks like"
}}

Target attribute: {attribute}
HTML segments:
{html_segments}"""

MULTI_CANDIDATE_LINK_PROMPT = """You are an expert XPath engineer. Your task: extract the actual URL/link for "{attribute}" from web pages.

Look at the screenshot to identify which links contain the target attribute's VALUE. Generate THREE different XPaths.

=== Structural Context ===
{structural_context}

=== CRITICAL ===
- Extract the ACTUAL target links, NOT navigation/menu links
- The XPath must select <a> elements (values extracted via @href)
- Look at the screenshot to identify the correct link visually

=== Rules ===
1. ALL candidates MUST select <a> elements
2. NEVER use text() or contains(text(), ...) — link text varies
3. NEVER embed specific URL paths in predicates
4. For lists of links, match ALL items
5. Use contains(@class, 'token') for classes
6. Keep paths short (2-3 steps)

Output JSON:
{{
    "xpath_1": "first XPath for the target link",
    "xpath_2": "second alternative XPath",
    "xpath_3": "third alternative XPath",
    "target_description": "what link we are looking for"
}}

Target attribute: {attribute}
HTML segments:
{html_segments}"""

MULTI_CANDIDATE_IMAGE_PROMPT = """You are an expert XPath engineer. Your task: extract the actual image for "{attribute}" from web pages.

Look at the screenshot to identify which images represent the target attribute. Generate THREE different XPaths.

=== Structural Context ===
{structural_context}

=== CRITICAL ===
- Extract the ACTUAL target images, NOT icons/logos/decorative images
- The XPath must select <img> elements (values extracted via @src)
- Look at the screenshot to identify the correct images visually

=== Rules ===
1. ALL candidates MUST select <img> elements
2. NEVER use specific file paths or names in predicates
3. NEVER use contains(@alt, ...) with specific text
4. For galleries, match ALL images
5. Use contains(@class, 'token') for classes
6. Keep paths short (2-3 steps)

Output JSON:
{{
    "xpath_1": "first XPath for target images",
    "xpath_2": "second alternative XPath",
    "xpath_3": "third alternative XPath",
    "target_description": "what images we are looking for"
}}

Target attribute: {attribute}
HTML segments:
{html_segments}"""


# ═══════════════════════════════════════════════════════════════════
# Cog — Structure-Aware Reflection Prompt
# ═══════════════════════════════════════════════════════════════════

ZERO_HIT_REGENERATION_PROMPT = """You are an XPath engineer. Your initial XPath candidates ALL matched ZERO elements on the target page. This means the class names or structure you assumed DO NOT EXIST in the actual HTML.

Below is the REAL HTML structure around where the "{attribute}" value should be. Generate 3 new XPath candidates based on ACTUAL elements you can see.

Target attribute: {attribute}

=== Your Failed Candidates (all match 0 nodes) ===
{failed_candidates}

=== REAL HTML around target value ===
{real_dom_context}

=== Instructions ===
1. Look at the REAL HTML above — use ONLY class names, tags, and attributes that actually appear there
2. DO NOT invent or assume any class name — every class/id you reference must be visible in the HTML above
3. The target value should be somewhere in the provided HTML
4. Generate 3 XPaths using different strategies (class-based, tag-hierarchy, semantic attributes)
5. Keep paths short (2-3 steps with //)
6. NEVER use positional indices [N]
7. NEVER embed literal text values as predicates

Output JSON:
{{
    "xpath_1": "first XPath based on real DOM",
    "xpath_2": "second alternative XPath",
    "xpath_3": "third alternative XPath",
    "reasoning": "brief explanation of what you found in the real HTML"
}}"""


STRUCTURAL_REFLECTION_PROMPT = """You are an XPath repair expert. Previous XPath candidates ALL failed to generalize across pages. You must now generate a FUNDAMENTALLY DIFFERENT XPath.

Target attribute: {attribute}

=== Failed Candidates ===
{failed_candidates}

=== Structural Analysis ===
Page A extraction results: {values_a_summary}
Page B extraction results: {values_b_summary}

Structural differences between pages:
- Page A DOM depth of target area: {depth_a}
- Page B DOM depth of target area: {depth_b}
- Common ancestor classes: {common_classes}

=== Current Diagnosis ===
Failure type: {failure_type}
Diagnosis: {diagnosis}

=== Page HTML ===
Page A (relevant section):
{html_a_section}

Page B (relevant section):
{html_b_section}

=== Instructions ===
1. ALL previous strategies have FAILED. You MUST try something completely different.
2. Analyze what structural patterns are SHARED between Page A and Page B
3. Look for semantic landmarks (main, article, section, header) that exist on BOTH pages
4. If class-based approaches failed, try tag-only hierarchy (e.g., //main//h1)
5. If hierarchy failed, try attribute presence (e.g., //h1[@class])
6. Consider that the target element might have different class names across pages but the SAME tag/position
7. The XPath MUST work on BOTH pages simultaneously

Output JSON:
{{
    "reasoning": "analysis of why previous attempts failed and new strategy",
    "revision_strategy": "one-line summary of fundamentally new approach",
    "revised_xpath": "the new XPath"
}}"""


# ═══════════════════════════════════════════════════════════════════
# v8 — Unified Reflection Prompt (single loop, no branching)
# ═══════════════════════════════════════════════════════════════════

UNIFIED_REFLECTION_PROMPT = """You are an XPath generation expert. Your task is to extract the value of "{attribute}" from web pages. Previous attempts have failed. You must analyze WHY they failed and generate new XPath candidates.

Target attribute: {attribute}
Original query: {query}

=== Previous Attempts ===
{failed_candidates}

=== Extraction Results ===
Page A (seed): {values_a_summary}
Page B (validation): {values_b_summary}

=== Structural Context ===
- Common classes between pages: {common_classes}
- Page A DOM depth: {depth_a}  |  Page B DOM depth: {depth_b}

=== Page HTML ===
Page A:
{html_a_section}

Page B:
{html_b_section}

=== Analysis Required ===
Before generating new XPaths, diagnose the failure:
- If BOTH pages extracted 0 values: the class names or structure you assumed do not exist. Look at the ACTUAL HTML above.
- If Page A extracted values but Page B did not: the XPath is too page-specific. Find patterns SHARED across both pages.
- If extracted values look like navigation/UI elements instead of content: target the main content area instead.
- If extracted too many values: the selector is too broad, add specificity.

=== Generation Rules ===
1. Use ONLY class names, tags, and attributes that you can SEE in the HTML above
2. The XPath MUST extract values on BOTH pages
3. Target CONTENT elements, not navigation bars, sidebars, footers, or UI chrome
4. Prefer SHORT paths (2-3 steps with //)
5. NEVER use positional indices [N]
6. NEVER embed literal text values as predicates
7. Generate 3 candidates using fundamentally DIFFERENT strategies from all previous attempts

Output JSON:
{{
    "diagnosis": "what specifically went wrong and why",
    "strategy": "your new approach, different from all previous attempts",
    "xpath_1": "first new XPath candidate",
    "xpath_2": "second new XPath candidate",
    "xpath_3": "third new XPath candidate"
}}"""
