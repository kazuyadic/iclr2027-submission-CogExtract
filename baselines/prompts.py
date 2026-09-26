"""Prompt templates for CoT and Reflexion baselines (from LiveWeb-IE paper Appendix J)."""

# ────────────────────────────────────────────────────────────────────
# CoT / Reflexion: Top-Down Operation (generate XPaths from HTML)
# ────────────────────────────────────────────────────────────────────
TOP_DOWN_PROMPT = """You are a web parser that is good at reading and understanding the HTML code and can give clear executable code on the browser.

Please read the following HTML code, and then return XPaths grouped by field names that directly match the elements in the page, satisfying the instructions below.

Instruction: {instruction}

Rules:
- Field Determination:
  - Use explicitly listed field names from the instruction
  - Otherwise, infer minimal relevant fields present in the DOM without inventing unsupported fields
- XPath Construction:
  - Avoid embedding exact literal values or visible text from the HTML content
  - Create structurally robust selectors using attribute and structural patterns
  - Prefer stable element attributes and hierarchical relationships over brittle identifiers
- Field Organization:
  - Generate separate XPaths for each target node when multiple nodes exist for a field
  - Maintain distinct XPaths that satisfy the instruction requirements for each field
- Data Structure Requirements:
  - Ensure xpath and value dictionaries have identical key sets
  - Align the XPath and value lists so each position corresponds to the same target node
  - Extract text from single nodes without concatenating content from multiple elements
- Missing Information Handling:
  - Return empty lists for explicitly enumerated fields when HTML lacks suitable content
  - Return empty objects for both xpath and value when no instruction fields are specified and no suitable content is found

Please output in the following JSON format:
{{
  "thought": "",
  "value": {{}},
  "xpath": {{}}
}}

Here's the HTML code:
```
{html}
```"""

# ────────────────────────────────────────────────────────────────────
# Reflexion: Self-Reflection Operation
# ────────────────────────────────────────────────────────────────────
SELF_REFLECTION_PROMPT = """Please read the following HTML code, and then return all possible XPaths that can recognize the elements in the HTML matching the instructions below.

Instruction: {instruction}

Rules:
- History Analysis:
  - Evaluate consistency between extraction results and expected values
  - Identify irrelevant elements that were incorrectly captured
  - Check for empty results indicating failed extraction
  - Accept raw values with redundant separators as consistent, since post-processing will handle them
- Value Assessment:
  - Re-examine expected values in the context of the HTML structure
  - Determine optimal XPath strategies for locating target content
  - Consider structural patterns and element relationships for reliable targeting
- XPath Refinement:
  - Generate new XPaths when current ones fail to meet requirements
  - Retain existing XPaths when they demonstrate accurate extraction
  - Base decisions on analysis findings and extraction quality assessment
- XPath Construction Constraints:
  - Avoid embedding exact literal values or specific HTML element content
  - Prevent overly broad selectors that match multiple nodes with different meanings
  - Use specific class attributes and positional indicators to differentiate target nodes
  - Maintain precision through structural anchors and attribute-based targeting
- Missing Content Handling:
  - Return empty outputs when HTML lacks information matching the instruction
  - Acknowledge extraction limitations rather than forcing inappropriate matches

Please output in the following JSON format:
{{
  "thought": "",
  "consistent": "",
  "value": {{}},
  "xpath": {{}}
}}

And here's the history about the thoughts, XPaths, and results extracted by the crawler:
{history}

Here's the HTML code:
```
{html}
```"""

# ────────────────────────────────────────────────────────────────────
# Synthesis Operation (shared by CoT and Reflexion)
# ────────────────────────────────────────────────────────────────────
SYNTHESIS_PROMPT = """You are a perfect discriminator, which is good at HTML understanding as well.

Following the instruction, there are some action sequence written from several HTML and the corresponding result extracted from several HTML. Please choose one that can potentially be best adapted to the same extraction task on other webpages on the same website.

Here are the instructions for the task:
Instructions: {instruction}

The action sequences and the corresponding extracted results with different sequences on different webpages are as follows:
{candidates}

Please output in the following JSON format:
{{
  "thought": "",
  "number": ""
}}"""
