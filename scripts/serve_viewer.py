#!/usr/bin/env python3
"""Serve the local experiment replay UI and recorded run data."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIEWER_ROOT = PROJECT_ROOT / "viewer"
RUNS_ROOT = PROJECT_ROOT / "results" / "runs"
RUN_FILES = (
    "config.json",
    "episode-summary.json",
    "events.jsonl",
    "artifacts.jsonl",
    "calls.json",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def safe_run_dir(name: str) -> Path:
    candidate = (RUNS_ROOT / name).resolve()
    if candidate.parent != RUNS_ROOT.resolve() or not candidate.is_dir():
        raise ValueError("unknown run")
    return candidate


def list_runs() -> list[dict[str, Any]]:
    if not RUNS_ROOT.exists():
        return []
    runs = []
    for directory in sorted(RUNS_ROOT.iterdir(), reverse=True):
        summary_path = directory / "episode-summary.json"
        if not directory.is_dir() or not summary_path.exists():
            continue
        try:
            summary = read_json(summary_path)
        except (OSError, json.JSONDecodeError):
            continue
        runs.append(
            {
                "name": directory.name,
                "map_id": summary.get("map_id"),
                "mode": summary.get("mode"),
                "planning_granularity": summary.get("planning_granularity"),
                "success": summary.get("success"),
                "fault_id": summary.get("fault_id"),
                "llm_calls": summary.get("llm_calls"),
                "total_tokens": summary.get("usage", {}).get("total_tokens"),
            }
        )
    return runs


def load_run(name: str) -> dict[str, Any]:
    directory = safe_run_dir(name)
    missing = [filename for filename in RUN_FILES if not (directory / filename).exists()]
    if missing:
        raise ValueError("run is missing: " + ", ".join(missing))
    config = read_json(directory / "config.json")
    map_snapshot = config.get("map_snapshot")
    if map_snapshot is None:
        map_path = Path(config["map_path"]).resolve()
        if PROJECT_ROOT.resolve() not in map_path.parents:
            raise ValueError("map path is outside the project")
        map_snapshot = read_json(map_path)
    return {
        "name": name,
        "config": config,
        "summary": read_json(directory / "episode-summary.json"),
        "map": map_snapshot,
        "events": read_jsonl(directory / "events.jsonl"),
        "artifacts": read_jsonl(directory / "artifacts.jsonl"),
        "calls": read_json(directory / "calls.json"),
    }


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "MASRecoveryReplay/0.1"

    def send_json(self, value: Any, status: int = 200) -> None:
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path == "/api/runs":
                self.send_json({"runs": list_runs(), "default_run": self.server.default_run})
                return
            if parsed.path == "/api/run":
                query = urllib.parse.parse_qs(parsed.query)
                name = query.get("name", [self.server.default_run])[0]
                if not name:
                    raise ValueError("no run selected")
                self.send_json(load_run(name))
                return
            self.serve_static(parsed.path)
        except (ValueError, OSError, json.JSONDecodeError, KeyError) as exc:
            self.send_json({"error": str(exc)}, status=400)

    def serve_static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        path = (VIEWER_ROOT / relative).resolve()
        if VIEWER_ROOT.resolve() not in path.parents or not path.is_file():
            self.send_error(404)
            return
        body = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if mime.startswith("text/") or mime in {"application/javascript", "application/json"}:
            mime += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"[viewer] {self.address_string()} {format_string % args}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="run directory name under results/runs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open the viewer in the default browser")
    args = parser.parse_args()

    if args.run:
        safe_run_dir(args.run)
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    server.default_run = args.run  # type: ignore[attr-defined]
    url = f"http://{args.host}:{args.port}/"
    print(f"Replay viewer: {url}", flush=True)
    if args.open:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
