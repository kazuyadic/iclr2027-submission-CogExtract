"""Re-eval: keep full normalization + truncation fuzzy, only remove attribute fuzzy matching."""
import json, re, sys, unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_loader import DataLoader
from utils.evaluator import Evaluator
from configs.config import VGSConfig


class NoFuzzyAttrEval:
    """Same as original evaluator but WITHOUT fuzzy attribute name matching."""

    _CONFUSABLE_MAP = str.maketrans({
        '\u22C5': '\u00B7', '\u2027': '\u00B7', '\u0387': '\u00B7', '\u2022': '\u00B7',
        '\u2024': '.', '\u2025': '..', '\uFE52': '.', '\uFF0E': '.',
        '\u00A0': ' ', '\u2009': ' ', '\u200A': ' ', '\u202F': ' ', '\u205F': ' ', '\u3000': ' ',
        '\u2018': "'", '\u2019': "'", '\u201C': '"', '\u201D': '"',
        '\uFF0D': '-', '\u2010': '-', '\u2011': '-', '\u2012': '-', '\u2013': '-', '\u2014': '-',
    })

    @classmethod
    def normalize(cls, value):
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
        pred_set = {cls.normalize(v) for v in predicted if v}
        gt_set = {cls.normalize(v) for v in ground_truth if v}
        if not gt_set:
            return 1.0 if not pred_set else 0.0
        if not pred_set:
            return 0.0
        # Exact + truncation fuzzy (same as original)
        tp = len(pred_set & gt_set)
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
    def eval_sample(cls, pred, gt):
        """EXACT attribute name matching only (no fuzzy)."""
        pa = set(pred.keys())
        ga = set(gt.keys())
        common = pa & ga
        scores = [cls.compute_attr_f1(pred[a], gt[a]) for a in common]
        scores += [0.0] * len(pa - common)
        scores += [0.0] * len(ga - common)
        return sum(scores) / len(scores) if scores else 0.0


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
            total_f1 += NoFuzzyAttrEval.eval_sample(pred, gt)
            count += 1
        return total_f1 / count * 100 if count else None

    models = [
        "qwen3-vl-8b-instruct",
        "qwen3-vl-8b-thinking",
        "qwen3-vl-30b-a3b-instruct",
        "qwen3-vl-30b-a3b-thinking",
        "qwen3-vl-32b-instruct",
        "qwen3-vl-32b-thinking",
        "qwen3.5-27b",
        "qwen3.5-35b-a3b",
        "qwen3.5-122b-a10b",
        "qwen3.7-plus",
        "qwen3.7-max-2026-06-08",
        "kimi-k2.5",
        "MiniMax-M2.5",
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
