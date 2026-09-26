"""
Configuration for the evaluation system.
Set your DashScope API key here or via environment variable DASHSCOPE_API_KEY.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

# API keys are read from the environment. NO credentials are bundled with this
# reproduction package. Set DASHSCOPE_API_KEY before running; optionally add
# DASHSCOPE_API_KEY_2 / _3 to enlarge the rate-limit rotation pool (utils/llm_client.py
# cycles over all provided keys). For other providers, edit api_base/model_name below.
DASHSCOPE_API_KEYS = [
    k for k in (
        os.environ.get("DASHSCOPE_API_KEY", ""),
        os.environ.get("DASHSCOPE_API_KEY_2", ""),
        os.environ.get("DASHSCOPE_API_KEY_3", ""),
    ) if k
] or [""]
DASHSCOPE_API_KEY = DASHSCOPE_API_KEYS[0]


@dataclass
class VGSConfig:
    # ── Model ──
    model_name: str = "qwen3.5-27b"
    api_key: Union[str, list] = field(default_factory=lambda: DASHSCOPE_API_KEYS)
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    temperature: float = 0.0
    max_tokens: int = 8192

    # ── Browser ──
    viewport_width: int = 1280
    viewport_height: int = 1100
    page_load_timeout_ms: int = 60_000
    headless: bool = True

    # ── VGS hyper-parameters ──
    neighbor_distance: int = 2          # d in paper §4.4

    # ── Paths ──
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    experiments_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "experiments")
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "experiments" / "cog_v8")
    screenshot_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "experiments" / "screenshots")
    cache_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "cache" / "pages")

    # ── Cache ──
    use_cache: bool = True              # Enable transparent page caching

    def __post_init__(self):
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
