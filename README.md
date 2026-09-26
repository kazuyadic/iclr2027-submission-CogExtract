# CogExtract — Slim Reproduction Package

This repository reproduces the **main experiment (Cog)** and the **ablation
experiments** of **"CogExtract: Credit Assignment for Visual-Language Web
Information Extraction"** (ICLR 2027 submission). It ships as a self-contained
package of code + data + archived page snapshots.

## Directory Structure

```
CogExtract/
├── run_all.sh            # entry point: quick | full | smoke
├── main.py               # single-run CLI
├── configs/              # VGSConfig; cache_dir defaults to <root>/cache/pages
├── utils/                # DataLoader, CachedBrowserManager, Evaluator, LLM client
├── cog/                  # Cog pipeline (multi-candidate generation + cross-page
│                         #   verification + failure-driven regeneration)
├── vgs/                  # VGS baseline (visual-grounding extractor)
├── baselines/            # CoT / Reflexion prompt baselines
├── ablations/            # 5 ablation variants + batch driver
├── scripts/              # multi-model batch runs, strict eval, pre-caching, SWDE, etc.
├── data/LiveWeb_IE/      # full LiveWeb-IE dataset (~5.6 MB)
└── cache/pages/          # 8 archived pages (page.html + screenshots) for offline quick
```

## Environment Setup

```bash
pip install -r requirements.txt
playwright install chromium        # needed for offline image regeneration / online fetch
export DASHSCOPE_API_KEY=sk-...    # your own key; the repo contains no credentials
```

Requires Python 3.10+. The `smoke` mode needs no network. `configs/config.py`
reads the API key from the environment variable `DASHSCOPE_API_KEY` (optional
`_2` / `_3` suffixes form a rate-limit rotation pool).

## Run Modes

### 0. Offline self-check (no API key) — ~5 seconds

```bash
./run_all.sh smoke
```

Validates dataset loading, the 8-page cache coverage, and that every cached page
ships with Set-of-Mark screenshots.

### 1. Small-scale reproduction (uses the 8-page cache, needs API key) — ~10-15 min

```bash
./run_all.sh quick                        # default model: qwen3-vl-8b-instruct
MODEL=qwen3.7-plus ./run_all.sh quick     # the open-weight model used in the paper
```

Runs **Cog** and the **VGS** baseline with 3 groups × 2 URLs across 4 task types
(24 samples per method). All pages are read from `cache/pages/` — no online
browsing. Per-type metrics are written to `quick_out/<model>/`.

### 2. Full reproduction (needs API key, several hours per model)

```bash
./run_all.sh full
```

Across the model list in `scripts/run_multi_model.py`, runs Cog (main), the VGS
baseline, the CoT / Reflexion baselines, and all ablations. The archived cache
covers only the 8 quick pages; the full run fetches pages online (requires
playwright + chromium). Sites may have drifted since self-archiving — which is
precisely the phenomenon the paper studies.

Per-family commands:

```bash
python3 scripts/run_multi_model.py --models qwen3.7-plus --concurrency 3   # main (Cog)
python3 scripts/run_vgs_multi_model.py --models qwen3.7-plus               # VGS
python3 scripts/run_baselines.py --models qwen3.7-plus --methods cot reflexion
python3 ablations/run_ablation.py multi_only --model qwen3.7-plus          # ablations
```

## Caveats

- **Checkpoint warning**: `main.py eval` always writes to the same directory
  (`experiments/cog_v8`), and checkpoints are **not isolated by model / method /
  task type** — `results_checkpoint.json` (Cog, keyed by `(group, url)`) and
  `phase1_checkpoint.json` (VGS, keyed by group index 0/1/2, which collides
  across task types). Stale checkpoints are **silently reused and re-scored**,
  distorting F1. For this reason `run_all.sh quick` clears the entire output
  directory before each run. If you invoke `main.py eval` manually, always run
  `rm -rf experiments/cog_v8` between different models / methods / types.
- **Offline image regeneration**: the 8 quick pages already ship with
  screenshots; to regenerate images offline for other (online-fetched) pages,
  run `python3 scripts/precache_stage3b.py`.
- **Metric convention**: all P/R/F1 are macro-averages (%) of per-sample scores,
  computed by `utils/evaluator.py`.
