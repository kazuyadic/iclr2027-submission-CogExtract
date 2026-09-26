"""Fast strict re-evaluation: sample-based for speed."""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_loader import DataLoader
from utils.evaluator import Evaluator
from configs.config import VGSConfig


class StrictEval:
    @staticmethod
    def normalize(value):
        v = value.strip().lower()
        v = re.sub(r'\s+', ' ', v)
        return v

    @classmethod
    def compute_attr_f1(cls, pred, gt):
        ps = {cls.normalize(v) for v in pred if v}
        gs = {cls.normalize(v) for v in gt if v}
        if not gs:
            return 1.0 if not ps else 0.0
        if not ps:
            return 0.0
        tp = len(ps & gs)
        p = tp / len(ps)
        r = tp / len(gs)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @classmethod
    def eval_sample(cls, pred, gt):
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

    lookup = {}
    for g in groups:
        for url in g["urls"]:
            lookup[url] = g

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
            g = lookup.get(url, {})
            if not g:
                continue
            gt = Evaluator.load_ground_truth(g.get("label_paths", []), data_root, url=url)
            pred = {} if "error" in r else r.get("values", {})
            total_f1 += StrictEval.eval_sample(pred, gt)
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
    methods = ["cot", "reflexion", "vgs", "main"]
    cog_map = {"qwen3.5-27b": "main_qwen3.5-27b_no_thinking"}

    print(f"{'Model':<30} {'CoT':>7} {'Reflex':>7} {'VGS':>7} {'Cog+':>7}")
    print("-" * 62)

    for m in models:
        row = []
        for method in methods:
            if method == "main":
                dirname = cog_map.get(m, f"main_{m}")
            else:
                dirname = f"{method}_{m}"
            v = eval_exp(dirname)
            row.append(f"{v:.1f}" if v else "-")
        print(f"{m:<30} {row[0]:>7} {row[1]:>7} {row[2]:>7} {row[3]:>7}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
