"""Aggregate ablation results into a comparison table.

Usage:
    python scripts/ablation/aggregate_results.py
    python scripts/ablation/aggregate_results.py --latest

Reads metrics.json from each ablation output directory and prints
a LaTeX-ready comparison table.
"""
import argparse
import json
import sys
from pathlib import Path

# scripts/ablation/ → project root (2 levels up to scripts/, then 1 more to root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

ABLATION_NAMES = {
    "A0": "Full CogExtract",
    "A1": "w/o MHG",
    "A2": "w/o CPSV",
    "A3": "w/o UIR",
}

TYPES = ["type_1", "type_2", "type_3", "type_4", "overall"]
TYPE_LABELS = {
    "type_1": "T1 (Detail)",
    "type_2": "T2 (Listing)",
    "type_3": "T3 (Struct.)",
    "type_4": "T4 (Media)",
    "overall": "Overall",
}


def find_latest_metrics(ablation_dir: Path) -> dict | None:
    """Find the most recent *_metrics.json in an ablation directory.

    Supports both naming conventions:
    - Old: experiments/ablation_A1/metrics.json
    - New: experiments/ablation_A1/{exp_id}/{exp_id}_metrics.json
    """
    if not ablation_dir.exists():
        return None

    # Check for *_metrics.json directly
    direct_matches = sorted(ablation_dir.glob("*_metrics.json"), reverse=True)
    if direct_matches:
        with open(direct_matches[0]) as f:
            return json.load(f)

    # Legacy: metrics.json directly
    direct = ablation_dir / "metrics.json"
    if direct.exists():
        with open(direct) as f:
            return json.load(f)

    # Check subdirectories for timestamped runs
    subdirs = sorted(
        [d for d in ablation_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    for subdir in subdirs:
        # New naming: {exp_id}_metrics.json
        new_matches = sorted(subdir.glob("*_metrics.json"), reverse=True)
        if new_matches:
            with open(new_matches[0]) as f:
                return json.load(f)
        # Legacy naming: metrics.json
        legacy = subdir / "metrics.json"
        if legacy.exists():
            with open(legacy) as f:
                return json.load(f)
    return None


def fmt(val: float) -> str:
    return f"{val:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Filter by model name in exp_id")
    args = parser.parse_args()

    experiments_dir = _PROJECT_ROOT / "experiments"

    # A0 uses cog_v8 (the full system baseline)
    a0_path = experiments_dir / "cog_v8"

    all_metrics = {}

    # Load A0
    a0_metrics = find_latest_metrics(a0_path)
    if a0_metrics is not None:
        all_metrics["A0"] = a0_metrics
    else:
        print("⚠  A0 (Full CogExtract): metrics not found at", a0_path)

    # Load A1-A3
    for ablation_id in ["A1", "A2", "A3"]:
        ablation_dir = experiments_dir / f"ablation_{ablation_id}"
        metrics = find_latest_metrics(ablation_dir)
        if metrics is None:
            print(f"⚠  {ablation_id} ({ABLATION_NAMES[ablation_id]}): no metrics found")
        else:
            all_metrics[ablation_id] = metrics

    if not all_metrics:
        print("No ablation results found. Run ablations first:")
        print("  python scripts/ablation/run_ablation.py --ablation A1")
        return

    a0_overall_f1 = all_metrics.get("A0", {}).get("overall", {}).get("f1", 0)

    # ── Print Markdown Table ──
    print("\n## Ablation Study Results\n")
    print("| Method | P (%) | R (%) | F1 (%) | ΔF1 |")
    print("|--------|-------|-------|--------|-----|")

    for ablation_id in ["A0", "A1", "A2", "A3"]:
        name = ABLATION_NAMES[ablation_id]
        if ablation_id not in all_metrics:
            print(f"| {name} | — | — | — | — |")
            continue
        m = all_metrics[ablation_id]["overall"]
        delta = m["f1"] - a0_overall_f1 if ablation_id != "A0" else 0
        delta_str = "—" if ablation_id == "A0" else f"{delta:+.2f}"
        print(
            f"| {name} | {fmt(m['precision'])} | {fmt(m['recall'])} "
            f"| {fmt(m['f1'])} | {delta_str} |"
        )

    # ── Per-type breakdown ──
    print("\n## Per-Type F1 Breakdown\n")
    header = "| Type | " + " | ".join(
        ABLATION_NAMES[aid] for aid in ["A0", "A1", "A2", "A3"]
    ) + " |"
    print(header)
    print("|------" + "|-------" * 4 + "|")

    for type_key in TYPES:
        row = f"| {TYPE_LABELS[type_key]} |"
        for ablation_id in ["A0", "A1", "A2", "A3"]:
            if ablation_id not in all_metrics:
                row += " — |"
            else:
                f1 = all_metrics[ablation_id].get(type_key, {}).get("f1", 0)
                row += f" {fmt(f1)} |"
        print(row)

    # ── LaTeX Table ──
    print("\n## LaTeX Table (for paper)\n")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Ablation study results.}")
    print("\\label{tab:ablation}")
    print("\\small")
    print("\\begin{tabular}{@{}lcccc@{}}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{P (\\%)} & \\textbf{R (\\%)} & \\textbf{F1 (\\%)} & $\\boldsymbol{\\Delta}$\\textbf{F1} \\\\")
    print("\\midrule")

    for ablation_id in ["A0", "A1", "A2", "A3"]:
        name = ABLATION_NAMES[ablation_id]
        if ablation_id not in all_metrics:
            print(f"{name} & — & — & — & — \\\\")
            continue
        m = all_metrics[ablation_id]["overall"]
        delta = m["f1"] - a0_overall_f1 if ablation_id != "A0" else 0
        delta_str = "—" if ablation_id == "A0" else f"{delta:+.2f}"
        bold = "\\textbf{" if ablation_id == "A0" else ""
        bold_end = "}" if ablation_id == "A0" else ""
        print(
            f"{bold}{name}{bold_end} & {fmt(m['precision'])} & {fmt(m['recall'])} "
            f"& {fmt(m['f1'])} & {delta_str} \\\\"
        )

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    main()
