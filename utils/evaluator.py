"""Evaluation metrics for LiveWeb-IE: Precision, Recall, F1."""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Evaluator:
    """Compute P / R / F1 by comparing predicted values against ground truth."""

    # Unicode confusable characters → canonical form
    # Visually identical separators that appear in author lists, etc.
    _CONFUSABLE_MAP = str.maketrans({
        '\u22C5': '\u00B7',  # DOT OPERATOR → MIDDLE DOT
        '\u2027': '\u00B7',  # HYPHENATION POINT → MIDDLE DOT
        '\u0387': '\u00B7',  # GREEK ANO TELEIA → MIDDLE DOT
        '\u2022': '\u00B7',  # BULLET → MIDDLE DOT
        '\u2024': '.',       # ONE DOT LEADER → period
        '\u2025': '..',      # TWO DOT LEADER → ..
        '\uFE52': '.',       # SMALL FULL STOP → period
        '\uFF0E': '.',       # FULLWIDTH FULL STOP → period
        '\u00A0': ' ',       # NO-BREAK SPACE → space
        '\u2009': ' ',       # THIN SPACE → space
        '\u200A': ' ',       # HAIR SPACE → space
        '\u202F': ' ',       # NARROW NO-BREAK SPACE → space
        '\u205F': ' ',       # MEDIUM MATHEMATICAL SPACE → space
        '\u3000': ' ',       # IDEOGRAPHIC SPACE → space
        '\u2018': "'",       # LEFT SINGLE QUOTATION MARK → apostrophe
        '\u2019': "'",       # RIGHT SINGLE QUOTATION MARK → apostrophe
        '\u201C': '"',       # LEFT DOUBLE QUOTATION MARK → quote
        '\u201D': '"',       # RIGHT DOUBLE QUOTATION MARK → quote
        '\uFF0D': '-',       # FULLWIDTH HYPHEN-MINUS → hyphen
        '\u2010': '-',       # HYPHEN → hyphen
        '\u2011': '-',       # NON-BREAKING HYPHEN → hyphen
        '\u2012': '-',       # FIGURE DASH → hyphen
        '\u2013': '-',       # EN DASH → hyphen
        '\u2014': '-',       # EM DASH → hyphen
    })

    @staticmethod
    def normalize(value: str) -> str:
        """Normalize a value for comparison.

        Handles:
        - Whitespace/case normalization
        - URL prefix stripping (absolute → relative path matching)
        - Relative path prefix stripping (../ sequences)
        - Leading slash normalization
        - Leading 'Developer:', 'Platform:' label stripping
        - Unicode confusable character normalization (dot separators, spaces, quotes)
        """
        import re
        import unicodedata
        v = value.strip().lower()
        # Unicode NFKC: decomposes compatibility characters
        v = unicodedata.normalize('NFKC', v)
        # Map visually confusable characters to canonical forms
        v = v.translate(Evaluator._CONFUSABLE_MAP)
        # Strip common URL prefixes for relative path matching
        v = re.sub(r'^https?://[^/]+', '', v)
        # Strip leading ../ sequences (relative paths)
        v = re.sub(r'^(\.\./)+', '', v)
        # Normalize leading slashes
        v = v.lstrip('/')
        # Strip leading label prefixes like "Developer: ", "Title: "
        # (common pattern where GT includes descriptive labels before values)
        v = re.sub(r'^[a-z]+:\s*', '', v)
        # Collapse multiple whitespace
        v = re.sub(r'\s+', ' ', v)
        return v.strip()

    @classmethod
    def compute_attribute_f1(cls, predicted: list[str], ground_truth: list[str]) -> dict:
        pred_set = {cls.normalize(v) for v in predicted if v}
        gt_set = {cls.normalize(v) for v in ground_truth if v}

        if not gt_set:
            return {"precision": 1.0 if not pred_set else 0.0,
                    "recall": 1.0, "f1": 1.0 if not pred_set else 0.0}
        if not pred_set:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Exact matching
        true_positives = len(pred_set & gt_set)

        # Fuzzy matching for truncated GT values (ending with "...")
        remaining_gt = gt_set - pred_set
        remaining_pred = pred_set - gt_set
        if remaining_gt and remaining_pred:
            for gt_val in list(remaining_gt):
                if gt_val.endswith("..."):
                    prefix = gt_val[:-3].rstrip()
                    if len(prefix) < 5:
                        continue
                    for pred_val in list(remaining_pred):
                        if pred_val.startswith(prefix):
                            true_positives += 1
                            remaining_gt.discard(gt_val)
                            remaining_pred.discard(pred_val)
                            break

        precision = true_positives / len(pred_set)
        recall = true_positives / len(gt_set)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    @classmethod
    def _fuzzy_match_attrs(cls, pred_attrs: set, gt_attrs: set) -> dict:
        """Fuzzy-match predicted attribute names to ground truth names.

        Handles cases like 'author_profile_url' vs 'author_profile_link'.
        Returns a mapping: pred_attr → gt_attr for best matches.
        """
        import re

        exact = pred_attrs & gt_attrs
        mapping = {a: a for a in exact}

        unmatched_pred = pred_attrs - exact
        unmatched_gt = gt_attrs - exact

        if not unmatched_pred or not unmatched_gt:
            return mapping

        # Normalize: lowercase, remove separators, common synonyms
        synonyms = {
            "url": "link", "urls": "links", "href": "link",
            "img": "image", "pic": "image", "photo": "image",
        }

        def normalize_attr(attr: str) -> str:
            tokens = re.split(r"[_\-\s]+", attr.lower())
            normalized = [synonyms.get(t, t) for t in tokens]
            # Singularize common plurals for matching
            singularized = []
            for t in normalized:
                if t.endswith("s") and len(t) > 3:
                    singularized.append(t[:-1])
                else:
                    singularized.append(t)
            return " ".join(singularized)

        # Greedy best-match by token overlap
        for pred_attr in list(unmatched_pred):
            pred_norm = normalize_attr(pred_attr)
            pred_tokens = set(pred_norm.split())

            best_score = 0.0
            best_gt = None
            for gt_attr in unmatched_gt:
                gt_norm = normalize_attr(gt_attr)
                gt_tokens = set(gt_norm.split())
                overlap = len(pred_tokens & gt_tokens)
                total = max(len(pred_tokens), len(gt_tokens))
                score = overlap / total if total > 0 else 0.0
                if score > best_score:
                    best_score = score
                    best_gt = gt_attr

            # Also check if one is a substring of the other (e.g. 'subject' in 'top_level_subject')
            if best_score < 0.8:
                for gt_attr in unmatched_gt:
                    gt_lower = gt_attr.lower().replace("_", " ").replace("-", " ")
                    pred_lower = pred_attr.lower().replace("_", " ").replace("-", " ")
                    if gt_lower in pred_lower or pred_lower in gt_lower:
                        best_score = 0.8
                        best_gt = gt_attr
                        break

            if best_score >= 0.8 and best_gt:
                mapping[pred_attr] = best_gt
                unmatched_gt.discard(best_gt)

        return mapping

    @classmethod
    def evaluate_sample(cls, prediction: dict, ground_truth: dict) -> dict:
        """Evaluate a single sample with fuzzy attribute name matching.

        Parameters
        ----------
        prediction : dict with attribute → list[str] values
        ground_truth : dict with attribute → list[str] values
        """
        pred_attrs = set(prediction.keys())
        gt_attrs = set(ground_truth.keys())

        # Fuzzy match attribute names to handle synonyms
        attr_mapping = cls._fuzzy_match_attrs(pred_attrs, gt_attrs)

        all_scores = []
        matched_gt_attrs = set()

        for pred_attr, gt_attr in attr_mapping.items():
            score = cls.compute_attribute_f1(prediction[pred_attr], ground_truth[gt_attr])
            all_scores.append(score)
            matched_gt_attrs.add(gt_attr)

        # Penalise unmatched predicted attributes (false positives)
        unmatched_pred = pred_attrs - set(attr_mapping.keys())
        for _ in unmatched_pred:
            all_scores.append({"precision": 0.0, "recall": 0.0, "f1": 0.0})

        # Penalise missing ground-truth attributes (false negatives)
        unmatched_gt = gt_attrs - matched_gt_attrs
        for _ in unmatched_gt:
            all_scores.append({"precision": 0.0, "recall": 0.0, "f1": 0.0})

        if not all_scores:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        avg = lambda key: sum(s[key] for s in all_scores) / len(all_scores)
        return {"precision": avg("precision"), "recall": avg("recall"), "f1": avg("f1")}

    @classmethod
    def evaluate_batch(cls, predictions: list[dict], ground_truths: list[dict],
                       task_types: list[str] | None = None) -> dict:
        """Evaluate a batch and return overall + per-type metrics."""
        overall_scores = []
        type_scores: dict[str, list] = {}

        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            score = cls.evaluate_sample(pred, gt)
            overall_scores.append(score)

            if task_types:
                ttype = task_types[i]
                type_scores.setdefault(ttype, []).append(score)

        def aggregate(scores_list):
            n = len(scores_list)
            if n == 0:
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "count": 0}
            return {
                "precision": sum(s["precision"] for s in scores_list) / n * 100,
                "recall": sum(s["recall"] for s in scores_list) / n * 100,
                "f1": sum(s["f1"] for s in scores_list) / n * 100,
                "count": n,
            }

        result = {"overall": aggregate(overall_scores)}
        for ttype, scores in sorted(type_scores.items()):
            result[ttype] = aggregate(scores)
        return result

    @classmethod
    def load_ground_truth(cls, label_paths: list[str], data_root: Path,
                          url: str = "") -> dict[str, list[str]]:
        """Load ground truth from label file paths.

        Ground truth file format:
            Line 1: header (domain group_id attribute_name)
            Line 2: stats (total_urls total_values ...)
            Line 3+: "url" count value1
                     (if count > 1, subsequent lines are additional values)

        When *url* is provided, only values matching that URL are returned.
        """
        gt = {}
        for label_path in label_paths:
            full_path = data_root / label_path
            if not full_path.exists():
                logger.warning("Ground truth file not found: %s", full_path)
                continue
            # Derive attribute name from filename: {website}-{group_id}-{attribute}.txt
            stem = full_path.stem
            parts = stem.rsplit("-", 1)
            attribute = parts[-1] if len(parts) > 1 else stem

            lines = full_path.read_text(encoding="utf-8").strip().splitlines()
            if len(lines) < 3:
                gt[attribute] = []
                continue

            # Parse data lines (skip header + stats)
            values = []
            for line in lines[2:]:
                line = line.strip()
                if not line:
                    continue
                # Format: "url" count value1\tvalue2\t...
                if line.startswith('"'):
                    end_quote = line.index('"', 1)
                    line_url = line[1:end_quote]
                    remainder = line[end_quote + 1:].strip()
                    # remainder is "count value1\tvalue2\t..."
                    parts_r = remainder.split(" ", 1)
                    value_str = parts_r[1] if len(parts_r) > 1 else ""

                    if url and line_url != url:
                        continue
                    if value_str:
                        # Multiple values are tab-separated
                        for v in value_str.split("\t"):
                            v = v.strip()
                            if v:
                                values.append(v)
                else:
                    # Continuation line (additional value for multi-value entries)
                    if not url:
                        for v in line.split("\t"):
                            v = v.strip()
                            if v:
                                values.append(v)

            gt[attribute] = values
        return gt
