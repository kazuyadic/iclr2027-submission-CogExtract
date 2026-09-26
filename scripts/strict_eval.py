"""Re-evaluate all experiments with strict evaluator (no fuzzy matching)."""
import json, sys, re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.data_loader import DataLoader
from configs.config import VGSConfig


class StrictEvaluator:
    """Strict evaluator: exact attribute names, minimal normalization, no truncation fuzzy."""

    @staticmethod
    def normalize(value: str) -> str:
        """Minimal normalization: strip + lowercase + collapse whitespace only."""
        v = value.strip().lower()
        v = re.sub(r'\s+', ' ', v)
        return v

    @classmethod
    def compute_attribute_f1(cls, predicted: list[str], ground_truth: list[str]) -> dict:
        pred_set = {cls.normalize(v) for v in predicted if v}
        gt_set = {cls.normalize(v) for v in ground_truth if v}

        if not gt_set:
            return {"precision": 1.0 if not pred_set else 0.0,
                    "recall": 1.0, "f1": 1.0 if not pred_set else 0.0}
        if not pred_set:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Strict exact matching only - no truncation fuzzy
        true_positives = len(pred_set & gt_set)

        precision = true_positives / len(pred_set)
        recall = true_positives / len(gt_set)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    @classmethod
    def evaluate_sample(cls, prediction: dict, ground_truth: dict) -> dict:
        """Strict: exact attribute name matching only."""
        pred_attrs = set(prediction.keys())
        gt_attrs = set(ground_truth.keys())

        # Exact match only - no fuzzy attribute matching
        common_attrs = pred_attrs & gt_attrs

        all_scores = []
        for attr in common_attrs:
            score = cls.compute_attribute_f1(prediction[attr], ground_truth[attr])
            all_scores.append(score)

        # Penalise unmatched predicted attributes
        for _ in (pred_attrs - common_attrs):
            all_scores.append({"precision": 0.0, "recall": 0.0, "f1": 0.0})

        # Penalise missing ground-truth attributes
        for _ in (gt_attrs - common_attrs):
            all_scores.append({"precision": 0.0, "recall": 0.0, "f1": 0.0})

        if not all_scores:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        avg = lambda key: sum(s[key] for s in all_scores) / len(all_scores)
        return {"precision": avg("precision"), "recall": avg("recall"), "f1": avg("f1")}


def load_ground_truth_for_results(results, groups, data_root):
    """Match results to groups and load GT."""
    # Build lookup
    lookup = {}
    for g in groups:
        for url in g["urls"]:
            lookup[url] = g

    scored = []
    for r in results:
        url = r.get("url", "")
        g = lookup.get(url, {})
        from utils.evaluator import Evaluator
        gt = Evaluator.load_ground_truth(g.get("label_paths", []), data_root, url=url)
        pred = {} if "error" in r else r.get("values", {})
        score = StrictEvaluator.evaluate_sample(pred, gt)
        scored.append(score)
    return scored


def main():
    config = VGSConfig()
    loader = DataLoader(config.data_dir)
    groups = loader.build_grouped_samples()
    data_root = config.data_dir / "LiveWeb_IE"

    exp_dir = Path("experiments")

    models = [
        "qwen3-vl-8b-instruct", "qwen3-vl-8b-thinking",
        "qwen3-vl-30b-a3b-instruct", "qwen3-vl-30b-a3b-thinking",
        "qwen3-vl-32b-instruct", "qwen3-vl-32b-thinking",
        "qwen3.5-27b", "qwen3.5-35b-a3b", "qwen3.5-122b-a10b",
        "qwen3.7-plus", "qwen3.7-max-2026-06-08",
        "kimi-k2.5", "MiniMax-M2.5",
    ]
    methods = ["cot", "reflexion", "vgs", "main"]
    method_labels = {"cot": "CoT", "reflexion": "Reflexion", "vgs": "VGS", "main": "Cog+"}

    cog_override = {"qwen3.5-27b": "main_qwen3.5-27b_no_thinking"}

    print(f"{'Model':<30} {'CoT':>8} {'Reflexion':>9} {'VGS':>8} {'Cog+':>8}")
    print("-" * 67)

    for model in models:
        row = []
        for method in methods:
            if method == "main":
                dirname = cog_override.get(model, f"main_{model}")
            else:
                dirname = f"{method}_{model}"

            d = exp_dir / dirname
            # Find results checkpoint
            results_file = d / "results_checkpoint.json"
            if not results_file.exists():
                row.append("-")
                continue

            results = json.loads(results_file.read_text())
            scores = load_ground_truth_for_results(results, groups, data_root)

            if scores:
                avg_f1 = sum(s["f1"] for s in scores) / len(scores) * 100
                row.append(f"{avg_f1:.2f}")
            else:
                row.append("-")

        print(f"{model:<30} {row[0]:>8} {row[1]:>9} {row[2]:>8} {row[3]:>8}")


if __name__ == "__main__":
    main()
