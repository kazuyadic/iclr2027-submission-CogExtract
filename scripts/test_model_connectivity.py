"""Test model connectivity on each API key.

Usage:
    python3 scripts/test_model_connectivity.py

Tests each candidate model with a simple text query and vision query
on every configured API key.
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from openai import OpenAI
from configs.config import DASHSCOPE_API_KEYS

# ── Models to test ──
CANDIDATE_MODELS = [
    "qwen3.5-122b-a10b",
    "qwen3.5-35b-a3b",
    "qwen3.7-plus",
    "qwen3.7-max-2026-06-08",
]

# ── Test image (a small local PNG, or skip if not available) ──
TEST_IMAGE = _PROJECT_ROOT / "paper" / "aaai" / "figures" / "architecture.png"

API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_text(client: OpenAI, model: str) -> tuple[bool, str]:
    """Simple text query test."""
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=64,
            messages=[{"role": "user", "content": "Reply with exactly: {\"ok\": true}"}],
            timeout=30,
        )
        content = resp.choices[0].message.content or ""
        return True, content.strip()[:120]
    except Exception as e:
        return False, str(e)[:120]


def test_vision(client: OpenAI, model: str) -> tuple[bool, str]:
    """Simple vision query test."""
    if not TEST_IMAGE.exists():
        return True, "(skipped: no test image)"
    try:
        import base64
        raw = TEST_IMAGE.read_bytes()
        b64 = base64.b64encode(raw).decode()
        data_url = f"data:image/png;base64,{b64}"

        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=64,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
                    {"type": "text", "text": "Describe this image in 5 words. Reply as JSON: {\"desc\": \"...\"}"},
                ],
            }],
            timeout=60,
        )
        content = resp.choices[0].message.content or ""
        return True, content.strip()[:120]
    except Exception as e:
        return False, str(e)[:120]


def main():
    # Deduplicate active keys
    active_keys = [k for k in DASHSCOPE_API_KEYS if k and not k.startswith("#")]
    if not active_keys:
        print("No active API keys found!")
        return

    print(f"Active API keys: {len(active_keys)}")
    print(f"Candidate models: {len(CANDIDATE_MODELS)}")
    print(f"{'='*80}\n")

    results = []

    for key_idx, key in enumerate(active_keys):
        key_short = key[:8] + "..." + key[-4:]
        client = OpenAI(api_key=key, base_url=API_BASE)

        for model in CANDIDATE_MODELS:
            print(f"Key {key_idx} ({key_short}) × {model}")

            # Text test
            text_ok, text_msg = test_text(client, model)
            text_status = "✓" if text_ok else "✗"
            print(f"  Text:   {text_status} {text_msg}")

            # Vision test
            vis_ok, vis_msg = test_vision(client, model)
            vis_status = "✓" if vis_ok else "✗"
            print(f"  Vision: {vis_status} {vis_msg}")

            results.append({
                "key_idx": key_idx,
                "key_short": key_short,
                "model": model,
                "text_ok": text_ok,
                "vision_ok": vis_ok,
            })
            print()

    # ── Summary ──
    print(f"{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<30s} {'Text':<8s} {'Vision':<8s} {'Key'}")
    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*16}")

    all_pass = []
    for r in results:
        status_t = "✓" if r["text_ok"] else "✗"
        status_v = "✓" if r["vision_ok"] else "✗"
        print(f"{r['model']:<30s} {status_t:<8s} {status_v:<8s} {r['key_short']}")
        if r["text_ok"] and r["vision_ok"]:
            all_pass.append(r["model"])

    print(f"\n{'='*80}")
    if all_pass:
        unique_pass = sorted(set(all_pass))
        print(f"Models passing BOTH text + vision: {unique_pass}")
    else:
        print("No models passed both tests!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
