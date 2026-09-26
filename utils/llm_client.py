"""Unified LLM/VLM client wrapping OpenAI-compatible APIs."""
from __future__ import annotations

import base64
import itertools
import json
import logging
import re
import time
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

from openai import OpenAI, RateLimitError, APIStatusError


class LLMClient:
    """Thin wrapper around the OpenAI chat-completion API with multi-key round-robin."""

    # Models known to have thinking mode enabled by default
    THINKING_MODELS = {"qwen3.5-35b-a3b", "qwen3.5-122b-a10b"}

    def __init__(self, api_key, api_base: str, model: str,
                 temperature: float = 0.0, max_tokens: int = 8192,
                 enable_monitor: bool = False):
        # Support single key (str) or multiple keys (list)
        if isinstance(api_key, list):
            self._clients = [OpenAI(api_key=k, base_url=api_base) for k in api_key]
        else:
            self._clients = [OpenAI(api_key=api_key, base_url=api_base)]
        self._client_cycle = itertools.cycle(self._clients)
        self._lock = threading.Lock()
        self.client = self._clients[0]  # default for backward compat
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_monitor = enable_monitor
        # Disable thinking mode for models accessed via DashScope
        # MiniMax requires enable_thinking=True, so we skip it
        # VL models: "-instruct" variants disable thinking; "-thinking" variants keep it
        if "thinking" in model:
            self._needs_no_thinking = False
        elif "instruct" in model:
            self._needs_no_thinking = True
        else:
            self._needs_no_thinking = any(t in model for t in ("qwen3.5", "qwen3.7", "kimi-k2"))

    def _next_client(self) -> OpenAI:
        """Thread-safe round-robin client selection."""
        with self._lock:
            return next(self._client_cycle)

    def _build_kwargs(self, messages: list, timeout: int = 120) -> dict:
        """Build common kwargs for chat.completions.create, disabling thinking if needed."""
        kwargs = dict(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=messages,
            timeout=timeout,
        )
        if self._needs_no_thinking:
            kwargs["extra_body"] = {"enable_thinking": False}
        return kwargs

    # ── public helpers ──────────────────────────────────────────────

    def text_query(self, prompt: str, label: str = "") -> dict:
        """Send a text-only prompt and return the parsed JSON response."""
        tag = f"[LLM·{label}]" if label else "[LLM]"
        print(f"\n{'='*60}")
        print(f"{tag} INPUT prompt ({len(prompt)} chars):")
        print(f"{'─'*60}")
        print(prompt[:2000] + ("…(truncated)" if len(prompt) > 2000 else ""))
        print(f"{'─'*60}")

        record = None
        if self.enable_monitor:
            from monitor.collector import StageCollector
            collector = StageCollector.get()
            record = collector.start_record(stage="LLM", label=label, prompt=prompt)

        kwargs = self._build_kwargs([{"role": "user", "content": prompt}], timeout=120)
        response = self._call_with_retry(kwargs, tag)
        if response is None:
            if record and self.enable_monitor:
                from monitor.collector import StageCollector
                StageCollector.get().finish_record(record, "ERROR: max retries", {})
            return {}
        raw_output = response.choices[0].message.content
        print(f"{tag} RAW OUTPUT:")
        print(raw_output)
        print(f"{'='*60}\n")

        parsed = self._parse_json(raw_output)
        if record and self.enable_monitor:
            from monitor.collector import StageCollector
            StageCollector.get().finish_record(record, raw_output, parsed)
        return parsed

    def vision_query(self, prompt: str, images: list[Path | str],
                     image_labels: list[str] | None = None,
                     label: str = "") -> dict:
        """Send a prompt together with one or more images."""
        tag = f"[VLM·{label}]" if label else "[VLM]"
        print(f"\n{'='*60}")
        print(f"{tag} INPUT prompt ({len(prompt)} chars):")
        print(f"{'─'*60}")
        print(prompt[:2000] + ("…(truncated)" if len(prompt) > 2000 else ""))
        img_info = []
        for i, img in enumerate(images):
            lbl = image_labels[i] if image_labels else f"Image {i}"
            size = Path(img).stat().st_size // 1024 if isinstance(img, (str, Path)) and Path(img).exists() else 0
            img_info.append(f"{lbl} ({size} KB)")
        print(f"{tag} Images: {img_info}")
        print(f"{'─'*60}")

        record = None
        if self.enable_monitor:
            from monitor.collector import StageCollector
            collector = StageCollector.get()
            record = collector.start_record(
                stage="VLM", label=label, prompt=prompt, images=images
            )

        content: list[dict] = []
        for index, img in enumerate(images):
            if image_labels:
                content.append({"type": "text", "text": image_labels[index]})
            content.append({
                "type": "image_url",
                "image_url": {"url": self._to_data_url(img), "detail": "high"},
            })
        content.append({"type": "text", "text": prompt})

        kwargs = self._build_kwargs([{"role": "user", "content": content}], timeout=180)
        response = self._call_with_retry(kwargs, tag)
        if response is None:
            if record and self.enable_monitor:
                from monitor.collector import StageCollector
                StageCollector.get().finish_record(record, "ERROR: max retries", {})
            return {}
        raw_output = response.choices[0].message.content
        print(f"{tag} RAW OUTPUT:")
        print(raw_output)
        print(f"{'='*60}\n")

        parsed = self._parse_json(raw_output)
        if record and self.enable_monitor:
            from monitor.collector import StageCollector
            StageCollector.get().finish_record(record, raw_output, parsed)
        return parsed

    # ── retry logic ────────────────────────────────────────────────

    def _call_with_retry(self, kwargs: dict, tag: str = "",
                         max_retries: int = 5, base_delay: float = 2.0):
        """Call API with exponential backoff on rate limit (429) errors."""
        for attempt in range(max_retries):
            client = self._next_client()
            try:
                return client.chat.completions.create(**kwargs)
            except RateLimitError as e:
                delay = base_delay * (2 ** attempt)  # 2, 4, 8, 16, 32s
                logger.warning(
                    "%s 429 rate limited (attempt %d/%d), retrying in %.1fs...",
                    tag, attempt + 1, max_retries, delay
                )
                time.sleep(delay)
            except APIStatusError as e:
                if e.status_code == 429:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "%s 429 quota exceeded (attempt %d/%d), retrying in %.1fs...",
                        tag, attempt + 1, max_retries, delay
                    )
                    time.sleep(delay)
                else:
                    print(f"{tag} ERROR: {e}")
                    return None
            except Exception as e:
                print(f"{tag} ERROR: {e}")
                return None
        logger.error("%s max retries (%d) exceeded", tag, max_retries)
        return None

    # ── internals ───────────────────────────────────────────────────

    @staticmethod
    def _to_data_url(image: Path | str) -> str:
        if isinstance(image, Path) or (isinstance(image, str) and not image.startswith("data:")):
            path = Path(image)
            raw = path.read_bytes()
            b64 = base64.b64encode(raw).decode()
            return f"data:image/png;base64,{b64}"
        return image

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Best-effort extraction of a JSON object from LLM output."""
        if not text or not text.strip():
            return {}
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        candidate = match.group(1).strip() if match else text.strip()
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            inner = re.search(r"\{[\s\S]*\}", candidate)
            if inner:
                try:
                    return json.loads(inner.group(), strict=False)
                except json.JSONDecodeError:
                    pass
            # Graceful fallback: try regex extraction for common fields
            result = {}
            xp = re.search(r'"xpath"\s*:\s*"((?:[^"\\]|\\.)*)"', candidate)
            if xp:
                result["xpath"] = xp.group(1).replace('\\"', '"').replace('\\n', '\n')
            sel = re.search(r'"selected_ids"\s*:\s*\[([^\]]*)\]', candidate)
            if sel:
                try:
                    result["selected_ids"] = [int(x.strip()) for x in sel.group(1).split(',') if x.strip().isdigit()]
                except ValueError:
                    pass
            attr = re.search(r'"attributes"\s*:\s*\[([^\]]*)\]', candidate)
            if attr:
                try:
                    result["attributes"] = [x.strip().strip('"') for x in attr.group(1).split(',') if x.strip().strip('"')]
                except ValueError:
                    pass
            if result:
                logger.info("JSON parse failed, regex fallback extracted: %s", list(result.keys()))
                return result
            logger.warning("Failed to parse JSON from LLM output (len=%d)", len(text))
            return {}
