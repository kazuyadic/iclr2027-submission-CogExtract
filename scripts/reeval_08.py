"""Re-evaluate all experiments with current evaluator (0.8 threshold).

For Cog (main_*): re-scores from embedded values+ground_truth in _results.json.
For VGS/CoT/Reflexion: reloads GT from label files via DataLoader, then re-scores.
"""
import json, sys, glob
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from utils.evaluator import Evaluator
from utils.data_loader import DataLoader
from configs.config import VGSConfig


def load_gt_cache():
    """Pre-load all ground truths keyed by URL."""
    config = VGSConfig()
    loader = DataLoader(config.data_dir)
    groups = loader.build_grouped_samples()
    data_root = config.data_dir / "LiveWeb_IE"

    print("Pre-loading ground truths...", flush=True)
    gt_cache = {}
    group_cache = {}
    for g in groups:
        for url in g["urls"]:
            gt = Evaluator.load_ground_truth(g.get("label_paths", []), data_root, url=url)
            gt_cache[url] = gt
            group_cache[url] = g
    print(f"Loaded GT for {len(gt_cache)} URLs", flush=True)
    return gt_cache, group_cache


def reeval_with_embedded(results):
    """Re-eval from results with embedded ground_truth."""
    all_scores, type_scores = [], {}
    for r in results:
        pred = r.get("values", {})
        gt = r.get("ground_truth", {})
        if not gt and not pred:
            continue
        score = Evaluator.evaluate_sample(pred, gt)
        all_scores.append(score)
        sid = r.get("sample_id", "")
        for p in sid.split("_"):
            if p.startswith("type"):
                type_scores.setdefault(p, []).append(score)
                break
    return all_scores, type_scores


def reeval_with_cache(results, gt_cache, group_cache):
    """Re-eval from results using pre-loaded GT cache."""
    all_scores, type_scores = [], {}
    for r in results:
        url = r.get("url", "")
        gt = gt_cache.get(url)
        if gt is None:
            continue
        pred = {} if r.get("error") else r.get("values", {})
        score = Evaluator.evaluate_sample(pred, gt)
        all_scores.append(score)
        g = group_cache.get(url, {})
        ttype = g.get("task_type", "")
        if ttype:
            type_scores.setdefault(ttype, []).append(score)
    return all_scores, type_scores


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


def build_metrics(all_scores, type_scores):
    metrics = {"overall": aggregate(all_scores)}
    for ttype, scores in sorted(type_scores.items()):
        metrics[ttype] = aggregate(scores)
    return metrics


def find_results_file(exp_dir):
    """Find the full_eval results file."""
    for f in sorted(exp_dir.glob("full_eval_*_results.json")):
        if "checkpoint" not in f.name:
            return f
    for f in sorted(exp_dir.glob("*_results.json")):
        if "checkpoint" not in f.name:
            return f
    return None


def main():
    gt_cache, group_cache = load_gt_cache()
    exp_root = _PROJECT_ROOT / "experiments"

    models = [
        "qwen3.7-plus", "qwen3.7-max-2026-06-08",
        "qwen3.5-122b-a10b", "qwen3.5-35b-a3b", "qwen3.5-27b",
        "kimi-k2.5", "MiniMax-M2.5",
        "qwen3-vl-32b-thinking", "qwen3-vl-32b-instruct",
        "qwen3-vl-30b-a3b-thinking", "qwen3-vl-30b-a3b-instruct",
        "qwen3-vl-8b-thinking", "qwen3-vl-8b-instruct",
    ]
    methods = {"main": "CogExtract", "vgs": "VGS", "cot": "CoT", "reflexion": "Reflexion"}
    cog_override = {"qwen3.5-27b": "main_qwen3.5-27b_no_thinking"}

    print(f"\n{'Model':<26} {'CogExtract':>10} {'VGS':>10} {'CoT':>10} {'Reflexion':>10}")
    print("-" * 70)

    all_data = {}
    for model in models:
        row = {}
        for mk in methods:
            dirname = cog_override.get(model, f"main_{model}") if mk == "main" else f"{mk}_{model}"
            exp_dir = exp_root / dirname
            rf = find_results_file(exp_dir)
            if rf is None:
                continue

            results = json.loads(rf.read_text())
            # Check if GT is embedded
            has_gt = any("ground_truth" in r and r["ground_truth"] for r in results[:5])

            if has_gt:
                scores, tscores = reeval_with_embedded(results)
            else:
                scores, tscores = reeval_with_cache(results, gt_cache, group_cache)

            if scores:
                metrics = build_metrics(scores, tscores)
                row[mk] = metrics
                (exp_dir / "metrics_08.json").write_text(json.dumps(metrics, indent=2))

        all_data[model] = row
        vals = []
        for mk in methods:
            m = row.get(mk, {}).get("overall", {})
            vals.append(f"{m['f1']:.1f}" if m else "-")
        print(f"{model:<26} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")

    # ── Ablation ──
    print(f"\n{'Ablation':<26} {'VGS':>8} {'+Multi':>8} {'+Verify':>8} {'Full':>8} {'Δ':>8}")
    print("-" * 70)

    abl_models = ["qwen3.7-plus", "qwen3.5-122b-a10b", "qwen3-vl-32b-instruct",
                  "qwen3-vl-8b-instruct", "kimi-k2.5"]
    abl_data = []
    for model in abl_models:
        row = {}
        for prefix, key in [("vgs", "vgs"), ("ablation_multi_only", "multi"),
                            ("ablation_verify_no_reflect", "verify"), ("main", "full")]:
            dirname = f"{prefix}_{model}"
            exp_dir = exp_root / dirname
            rf = find_results_file(exp_dir)
            if rf is None:
                continue
            results = json.loads(rf.read_text())
            has_gt = any("ground_truth" in r and r["ground_truth"] for r in results[:5])
            if has_gt:
                scores, tscores = reeval_with_embedded(results)
            else:
                scores, tscores = reeval_with_cache(results, gt_cache, group_cache)
            if scores:
                m = build_metrics(scores, tscores)
                row[key] = m["overall"]["f1"]
                (exp_dir / "metrics_08.json").write_text(json.dumps(m, indent=2))

        v, mu, ve, fu = row.get("vgs",0), row.get("multi",0), row.get("verify",0), row.get("full",0)
        print(f"{model:<26} {v:>8.1f} {mu:>8.1f} {ve:>8.1f} {fu:>8.1f} {fu-v:>+8.1f}")
        abl_data.append({"model": model, "vgs": round(v,1), "multi": round(mu,1),
                         "verify": round(ve,1), "full": round(fu,1)})

    # ── JS output ──
    print("\n\n// === JS: models ===")
    for model in models:
        row = all_data.get(model, {})
        parts = [f"name:'{model}'"]
        for mk, jk in [("main","cog"),("vgs","vgs"),("cot","cot"),("reflexion","ref")]:
            m = row.get(mk,{}).get("overall")
            if m: parts.append(f"{jk}:{{f1:{m['f1']:.1f},p:{m['precision']:.1f},r:{m['recall']:.1f}}}")
            else: parts.append(f"{jk}:null")
        print(f"  {{{', '.join(parts)}}},")

    print("\n// === JS: ablation ===")
    for a in abl_data:
        print(f"  {{model:'{a['model']}', vgs:{a['vgs']}, multi:{a['multi']}, "
              f"verify:{a['verify']}, full:{a['full']}}},")


if __name__ == "__main__":
    main()
