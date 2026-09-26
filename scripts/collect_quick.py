#!/usr/bin/env python3
"""Aggregate per-type quick-run metrics and compare against expected/subset_expected.json.

Usage: python3 scripts/collect_quick.py <model>
Reads: quick_out/<model>/{cog,vgs}_type_{1..4}.json
Writes: quick_out/<model>/summary.json  (+ printed comparison table)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    m = json.loads(path.read_text())
    # main.py writes {"overall": {...}, "type_X": {...}, ...}; take overall
    return m.get("overall", m)


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3.7-plus"
    qout = ROOT / "quick_out" / model
    types = ["type_1", "type_2", "type_3", "type_4"]

    # expected (archived full-run, filtered to the same 24 samples)
    exp_all = json.loads((ROOT / "expected" / "subset_expected.json").read_text())
    exp_main = exp_all.get(f"main_{model}", {})
    exp_vgs = exp_all.get(f"vgs_{model}", {})

    summary: dict = {"model": model, "cog": {}, "vgs": {}}
    for method, exp in (("cog", exp_main), ("vgs", exp_vgs)):
        rows = []
        for t in types:
            got = load_metrics(qout / f"{method}_{t}.json")
            want = exp.get(t, {})
            rows.append({
                "type": t,
                "f1_got": round(got.get("f1", 0.0), 2),
                "f1_expected": round(want.get("f1", 0.0), 2),
                "n": got.get("count", 0),
            })
            summary[method][t] = rows[-1]

    (qout / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Quick-run comparison ({model}) ===")
    print("F1 is macro-avg over 6 samples/type; LLM sampling is non-deterministic,")
    print("so treat +/-5-10 pts on 6 samples as consistent.\n")
    for method in ("cog", "vgs"):
        print(f"[{method}]")
        print(f"  {'type':8} {'got F1':>8} {'expected F1':>12} {'n':>4}")
        for t in types:
            r = summary[method][t]
            print(f"  {r['type']:8} {r['f1_got']:>8.2f} {r['f1_expected']:>12.2f} {r['n']:>4}")
        print()
    print(f"Saved -> {qout/'summary.json'}")


if __name__ == "__main__":
    main()
