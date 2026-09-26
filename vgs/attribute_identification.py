"""Stage 1: Attribute Identification — decompose query into target attributes."""
import re
import logging

from utils.llm_client import LLMClient
from vgs.prompts import ATTRIBUTE_IDENTIFICATION_PROMPT

logger = logging.getLogger(__name__)


class AttributeIdentifier:
    """Use an LLM to parse a natural language query into structured attributes."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def identify(self, query: str) -> list[str]:
        prompt = ATTRIBUTE_IDENTIFICATION_PROMPT.format(query=query)
        result = self.llm.text_query(prompt, label="Stage1·AttrIdent")
        attributes = result.get("attributes", [])
        if not attributes:
            # Fallback: extract noun-phrase attributes from query itself
            attributes = self._fallback_extract(query)
            if attributes:
                logger.info("  [AttrIdent] LLM returned empty, fallback inferred: %s", attributes)
            else:
                raise ValueError(f"No attributes identified from query: {query}")
        return attributes

    @staticmethod
    def _fallback_extract(query: str) -> list[str]:
        """Infer attributes from query when LLM fails.

        Uses lightweight NP-chunking heuristics on common extraction phrases
        to derive plausible attribute names without requiring an LLM call.
        """
        query_lower = query.lower().strip()
        attributes = []

        # ── Possessive pattern: "each game's title" → title ──
        possessive = re.findall(
            r"(?:each|every|all)\s+\w+['\u2019]s\s+([\w\s]+?)(?:\s+and\s+|\s*,|\s*$)",
            query_lower,
        )
        if possessive:
            for attr in possessive:
                attributes.append(attr.strip().replace(" ", "_"))

        # ── "Extract all <entity> <attr-noun>" → attr-noun ──
        all_noun = re.findall(
            r"(?:extract|get|fetch|collect)\s+(?:all|each|every)\s+(\w+)\s+(\w+s?)(?:\s|$|\.)",
            query_lower,
        )
        if all_noun and not attributes:
            for entity, attr in all_noun:
                # Singularize: "prices" → "price"
                canonical = attr.rstrip("s") if attr.endswith("s") and len(attr) > 3 else attr
                attributes.append(canonical)

        # ── "names of all authors" / "title of every paper" ──
        of_pattern = re.findall(
            r"((?:[\w]+\s+){0,2}[\w]+)\s+of\s+(?:all|each|every)\s+\w+",
            query_lower,
        )
        if of_pattern and not attributes:
            for phrase in of_pattern:
                # Take last meaningful noun: "the names" → "name"
                tokens = phrase.strip().split()
                noun = tokens[-1]
                canonical = noun.rstrip("s") if noun.endswith("s") and len(noun) > 3 else noun
                if canonical not in ("the", "a", "an", "set"):
                    attributes.append(canonical)

        # ── "provide each author's name and the link" → name, link ──
        and_pattern = re.findall(
            r"(\w+(?:\s+\w+)?)\s+and\s+(?:the\s+|their\s+)?(\w+(?:\s+\w+)?)",
            query_lower,
        )
        if and_pattern and not attributes:
            for left, right in and_pattern:
                for part in (left, right):
                    tokens = part.strip().split()
                    noun = tokens[-1]
                    if noun not in ("the", "a", "its", "their", "to"):
                        attributes.append(noun.replace(" ", "_"))

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for attr in attributes:
            if attr not in seen:
                seen.add(attr)
                unique.append(attr)
        return unique
