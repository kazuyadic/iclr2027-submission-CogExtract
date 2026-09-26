"""Experiment Registry — track and log all experiment runs.

Every experiment run is registered with:
  - experiment_type: ablation / multi_model / cross_dataset / method_comparison
  - model: backbone model name
  - dataset: dataset name
  - ablation: ablation ID (if applicable)
  - method: method name (if applicable)
  - timestamp: auto-generated
  - exp_id: unique identifier (e.g. ablation_A1_qwen2.5-7b_20260625_215000)

Registry is stored in experiments/registry.json (append-only).
"""
import json
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REGISTRY_PATH = _PROJECT_ROOT / "experiments" / "registry.json"


def _load_registry() -> list[dict]:
    if _REGISTRY_PATH.exists():
        with open(_REGISTRY_PATH) as f:
            return json.load(f)
    return []


def _save_registry(entries: list[dict]):
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_REGISTRY_PATH, "w") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def register_experiment(
    experiment_type: str,
    model: str,
    dataset: str = "LiveWeb-IE",
    page_source: str = "cached",
    cache_policy: str = "exclude_no_cache",
    ablation: str | None = None,
    method: str | None = None,
    result_files: dict | None = None,
    config=None,
    extra: dict | None = None,
    dry_run: bool = False,
) -> str:
    """Register an experiment and return its unique ID.

    Parameters
    ----------
    experiment_type : str
        One of: ablation, multi_model, cross_dataset, method_comparison, full_eval
    model : str
        Backbone model name (e.g. qwen3.5-27b)
    dataset : str
        Dataset name (default: LiveWeb-IE)
    page_source : str
        How pages are served: "cached" (local cache, anti-crawl) or "live" (direct URL)
    cache_policy : str
        What to do when cache is missing: "exclude_no_cache" (skip) or "fetch_live" (download)
    ablation : str, optional
        Ablation ID: A0, A1, A2, A3
    method : str, optional
        Method name: CogExtract, VGS, CoT, Reflexion, AutoScraper
    config : VGSConfig, optional
        Full config for reproducibility
    extra : dict, optional
        Any additional metadata
    dry_run : bool
        If True, print config and return without saving

    Returns
    -------
    str
        Experiment ID, e.g. "ablation_A1_qwen2.5-7b_20260625_215000"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build exp_id
    parts = [experiment_type]
    if ablation:
        parts.append(ablation)
    if method:
        parts.append(method)
    parts.append(model)
    parts.append(timestamp)
    exp_id = "_".join(parts)

    # Build entry
    entry = {
        "exp_id": exp_id,
        "experiment_type": experiment_type,
        "model": model,
        "dataset": dataset,
        "page_source": page_source,
        "cache_policy": cache_policy,
        "timestamp": datetime.now().isoformat(),
    }
    if result_files:
        entry["result_files"] = result_files
    if ablation:
        entry["ablation"] = ablation
    if method:
        entry["method"] = method
    if config:
        entry["config"] = {
            "model_name": config.model_name,
            "api_base": config.api_base,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "neighbor_distance": config.neighbor_distance,
            "viewport": f"{config.viewport_width}x{config.viewport_height}",
        }
    if extra:
        entry["extra"] = extra

    if dry_run:
        print("=" * 60)
        print("DRY RUN — Experiment Registration")
        print("=" * 60)
        for k, v in entry.items():
            print(f"  {k:20s}: {v}")
        print("=" * 60)
        return exp_id

    # Append to registry
    registry = _load_registry()
    registry.append(entry)
    _save_registry(registry)

    return exp_id


def list_experiments(
    experiment_type: str | None = None,
    model: str | None = None,
) -> list[dict]:
    """List registered experiments, optionally filtered."""
    registry = _load_registry()
    results = registry
    if experiment_type:
        results = [e for e in results if e.get("experiment_type") == experiment_type]
    if model:
        results = [e for e in results if e.get("model") == model]
    return results


def update_result_files(exp_id: str, result_files: dict):
    """Update result file paths for a registered experiment.

    Call this after an experiment completes to record where metrics/results are saved.

    Parameters
    ----------
    exp_id : str
        Experiment ID returned by register_experiment
    result_files : dict
        Mapping of file type to relative path, e.g.
        {"metrics": "experiments/ablation_A1/xxx/metrics.json",
         "results": "experiments/ablation_A1/xxx/results.json"}
    """
    registry = _load_registry()
    for entry in registry:
        if entry["exp_id"] == exp_id:
            entry["result_files"] = result_files
            _save_registry(registry)
            return
    print(f"⚠  Experiment '{exp_id}' not found in registry")


def print_registry():
    """Pretty-print the experiment registry."""
    entries = _load_registry()
    if not entries:
        print("No experiments registered.")
        return

    print(f"\n{'=' * 80}")
    print(f"Experiment Registry ({len(entries)} entries)")
    print(f"{'=' * 80}")
    print(f"{'exp_id':<50s} {'type':<18s} {'model':<16s} {'source':<8s} {'dataset'}")
    print(f"{'-' * 50} {'-' * 18} {'-' * 16} {'-' * 8} {'-' * 14}")
    for e in entries:
        print(
            f"{e['exp_id']:<50s} "
            f"{e.get('experiment_type', ''):<18s} "
            f"{e.get('model', ''):<16s} "
            f"{e.get('page_source', ''):<8s} "
            f"{e.get('dataset', '')}"
        )


if __name__ == "__main__":
    print_registry()
