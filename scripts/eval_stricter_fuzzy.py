"""Re-eval with stricter fuzzy matching: no substring fallback, threshold 0.67."""
import json, re, sys, unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_loader import DataLoader
from utils.evaluator import Evaluator
from configs.config import VGSConfig


class StricterFuzzyEval:
    """Stricter fuzzy: higher threshold (0.67), no substring fallback."""

    _CONFUSABLE_MAP = Evaluator._CONFUSABLE_MAP

    @classmethod
    def normalize_value(cls, value):
        v = value.strip().lower()
        v = unicodedata.normalize('NFKC', v)
        v = v.translate(cls._CONFUSABLE_MAP)
        v = re.sub(r'^https?://[^/]+', '', v)
        v = re.sub(r'^(\.\./)+', '', v)
        v = v.lstrip('/')
        v = re.sub(r'^[a-z]+:\s*', '', v)
        v = re.sub(r'\s+', ' ', v)
        return v.strip()

    @classmethod
    def compute_attr_f1(cls, predicted, ground_truth):
        pred_set = {cls.normalize_value(v) for v in predicted if v}
        gt_set = {cls.normalize_value(v) for v in ground_truth if v}
        if not gt_set:
            return 1.0 if not pred_set else 0.0
        if not pred_set:
            return 0.0
        tp = len(pred_set & gt_set)
        # Truncation fuzzy (keep this - it's about values not attrs)
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
                            tp += 1
                            remaining_gt.discard(gt_val)
                            remaining_pred.discard(pred_val)
                            break
        p = tp / len(pred_set)
        r = tp / len(gt_set)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @classmethod
    def _fuzzy_match_attrs_strict(cls, pred_attrs, gt_attrs):
        """Stricter fuzzy: threshold=0.67, NO substring fallback."""
        exact = pred_attrs & gt_attrs
        mapping = {a: a for a in exact}
        unmatched_pred = pred_attrs - exact
        unmatched_gt = gt_attrs - exact
        if not unmatched_pred or not unmatched_gt:
            return mapping

        synonyms = {"url": "link", "urls": "link", "href": "link"}

        def normalize_attr(attr):
            tokens = re.split(r"[_\-\s]+", attr.lower())
            normalized = [synonyms.get(t, t) for t in tokens]
            singularized = []
            for t in normalized:
                if t.endswith("s") and len(t) > 3:
                    singularized.append(t[:-1])
                else:
                    singularized.append(t)
            return " ".join(singularized)

        THRESHOLD = 0.67  # Stricter: need 2/3 overlap

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

            # NO substring fallback!
            if best_score >= THRESHOLD and best_gt:
                mapping[pred_attr] = best_gt
                unmatched_gt.discard(best_gt)

        return mapping

    @classmethod
    def eval_sample(cls, pred, gt):
        pa = set(pred.keys())
        ga = set(gt.keys())
        mapping = cls._fuzzy_match_attrs_strict(pa, ga)
        all_scores = []
        matched_gt = set()
        for pred_attr, gt_attr in mapping.items():
            score = cls.compute_attr_f1(pred[pred_attr], gt[gt_attr])
            all_scores.append(score)
            matched_gt.add(gt_attr)
        for _ in (pa - set(mapping.keys())):
            all_scores.append(0.0)
        for _ in (ga - matched_gt):
            all_scores.append(0.0)
        return sum(all_scores) / len(all_scores) if all_scores else 0.0


def main():
    config = VGSConfig()
    loader = DataLoader(config.data_dir)
    groups = loader.build_grouped_samples()
    data_root = config.data_dir / "LiveWeb_IE"

    print("Pre-loading ground truths...", flush=True)
    gt_cache = {}
    for g in groups:
        for url in g["urls"]:
            gt = Evaluator.load_ground_truth(g.get("label_paths", []), data_root, url=url)
            gt_cache[url] = gt
    print(f"Loaded GT for {len(gt_cache)} URLs", flush=True)

    def eval_exp(dirname):
        d = Path("experiments") / dirname
        rf = d / "results_checkpoint.json"
        if not rf.exists():
            return None
        results = json.loads(rf.read_text())
        total_f1 = 0.0
        count = 0
        for r in results:
            url = r.get("url", "")
            gt = gt_cache.get(url)
            if gt is None:
                continue
            pred = {} if "error" in r else r.get("values", {})
            total_f1 += StricterFuzzyEval.eval_sample(pred, gt)
            count += 1
        return total_f1 / count * 100 if count else None

    models = [
        "qwen3-vl-8b-instruct", "qwen3-vl-8b-thinking",
        "qwen3-vl-30b-a3b-instruct", "qwen3-vl-30b-a3b-thinking",
        "qwen3-vl-32b-instruct", "qwen3-vl-32b-thinking",
        "qwen3.5-27b", "qwen3.5-35b-a3b", "qwen3.5-122b-a10b",
        "qwen3.7-plus", "qwen3.7-max-2026-06-08",
        "kimi-k2.5", "MiniMax-M2.5",
    ]
    cog_map = {"qwen3.5-27b": "main_qwen3.5-27b_no_thinking"}

    print(f"\n{'Model':<30} {'CoT':>7} {'Reflex':>7} {'VGS':>7} {'Cog+':>7}", flush=True)
    print("-" * 62, flush=True)

    for m in models:
        row = []
        for method in ["cot", "reflexion", "vgs", "main"]:
            if method == "main":
                dirname = cog_map.get(m, f"main_{m}")
            else:
                dirname = f"{method}_{m}"
            v = eval_exp(dirname)
            row.append(f"{v:.1f}" if v else "-")
        print(f"{m:<30} {row[0]:>7} {row[1]:>7} {row[2]:>7} {row[3]:>7}", flush=True)


if __name__ == "__main__":
    main()
