#!/usr/bin/env python3
"""LiveWeb-IE Trace Viewer — backend API + static frontend server."""
import json, glob, os, sys, asyncio
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CACHE = {}

def load_experiments():
    """Discover all experiments and their result files."""
    exps = {}
    for rf in sorted(glob.glob(str(PROJECT_ROOT / "experiments/**/*results.json"), recursive=True)):
        p = Path(rf)
        exp_name = p.parent.name
        # Parse method and model
        parts = exp_name.split("_", 1)
        method = parts[0] if len(parts) == 2 else "unknown"
        model = parts[1] if len(parts) == 2 else exp_name
        # Clean up
        model = model.replace("_run1","").replace("_run2","").strip("_")
        exps[exp_name] = {"path": rf, "method": method, "model": model}
    return exps

def get_sample(exp_name, idx):
    """Get a single sample from an experiment's results."""
    if exp_name not in RESULTS_CACHE:
        exps = load_experiments()
        if exp_name not in exps:
            return None
        with open(exps[exp_name]["path"]) as f:
            RESULTS_CACHE[exp_name] = json.load(f)
    data = RESULTS_CACHE[exp_name]
    if idx < 0 or idx >= len(data):
        return None
    return data[idx]

def get_sample_list(exp_name, limit=100, offset=0, filter_score=None):
    """Get a list of samples (summary only) from an experiment."""
    if exp_name not in RESULTS_CACHE:
        exps = load_experiments()
        if exp_name not in exps:
            return [], 0
        with open(exps[exp_name]["path"]) as f:
            RESULTS_CACHE[exp_name] = json.load(f)
    data = RESULTS_CACHE[exp_name]
    items = []
    total = 0
    for i, r in enumerate(data):
        score = r.get("eval_score", {}).get("f1", 0)
        if filter_score is not None and score >= filter_score:
            continue
        if total >= offset and len(items) < limit:
            items.append({
                "idx": i,
                "url": r.get("url", ""),
                "query": r.get("query", ""),
                "attributes": r.get("attributes", []),
                "f1": score,
                "has_values": bool(r.get("values") and any(v for v in r.get("values", {}).values())),
            })
        total += 1
    return items, total

_vgs_pipeline = None

def run_vgs_trace(url, query, model="qwen3-vl-8b-instruct"):
    """Run VGS pipeline on a URL and return result with stages."""
    global _vgs_pipeline
    sys.path.insert(0, str(PROJECT_ROOT))
    from configs.config import VGSConfig
    from vgs.pipeline import VGSPipeline

    if _vgs_pipeline is None:
        config = VGSConfig(model_name=model)
        _vgs_pipeline = VGSPipeline(config, enable_monitor=False)

    async def _run():
        return await _vgs_pipeline.run(url, query)

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(_run())
    finally:
        loop.close()
    return result

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT / "output"), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/experiments":
            self._json_response(load_experiments())
        elif parsed.path == "/api/samples":
            qs = parse_qs(parsed.query)
            exp = qs.get("exp", [""])[0]
            limit = int(qs.get("limit", [100])[0])
            offset = int(qs.get("offset", [0])[0])
            filt = qs.get("filter", [None])[0]
            filter_score = float(filt) if filt else None
            items, total = get_sample_list(exp, limit, offset, filter_score)
            self._json_response({"items": items, "total": total})
        elif parsed.path == "/api/sample":
            qs = parse_qs(parsed.query)
            exp = qs.get("exp", [""])[0]
            idx = int(qs.get("idx", [0])[0])
            sample = get_sample(exp, idx)
            self._json_response(sample or {})
        elif parsed.path == "/api/run_vgs":
            qs = parse_qs(parsed.query)
            url = qs.get("url", [""])[0]
            query = qs.get("query", [""])[0]
            model = qs.get("model", ["qwen3-vl-8b-instruct"])[0]
            result = run_vgs_trace(url, query, model)
            self._json_response(result)
        elif parsed.path.startswith("/cache_screenshot/"):
            # Serve screenshot files from screenshot dirs
            rel = parsed.path[len("/cache_screenshot/"):]
            fpath = PROJECT_ROOT / rel
            if fpath.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                self.wfile.write(fpath.read_bytes())
            else:
                self.send_error(404)
        elif parsed.path == "/" or parsed.path == "":
            self.path = "/liveweb.html"
            super().do_GET()
        else:
            super().do_GET()

    def _json_response(self, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # Suppress logs

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8769
    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"LiveWeb-IE Viewer: http://localhost:{port}/liveweb.html")
    server.serve_forever()
