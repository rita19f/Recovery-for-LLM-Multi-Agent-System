#!/usr/bin/env python3
"""Run a reproducible batch of recovery experiments.

The command prints the complete run plan by default. Pass --execute to make API
calls. Every run gets an isolated output directory, and failed or timed-out
runs remain in the batch summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "scripts" / "run_experiment.py"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def read_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data.get("runs"), list) or not data["runs"]:
        raise ValueError("batch manifest must contain a non-empty runs list")
    repetitions = data.get("repetitions", 1)
    if not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    return data


def resolve_repo_path(raw_path: str) -> Path:
    path = (REPO_ROOT / raw_path).resolve()
    if REPO_ROOT.resolve() not in path.parents:
        raise ValueError(f"path escapes repository: {raw_path}")
    if not path.is_file():
        raise ValueError(f"missing file: {raw_path}")
    return path


def build_command(
    spec: dict[str, Any],
    defaults: dict[str, Any],
    *,
    backend: str,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        str(resolve_repo_path(spec["map"])),
        "--backend",
        backend,
        "--mode",
        spec["mode"],
        "--planning-granularity",
        str(spec.get("planning_granularity", defaults.get("planning_granularity", "chunk"))),
        "--chunk-size",
        str(spec.get("chunk_size", defaults.get("chunk_size", 3))),
        "--timeout",
        str(spec.get("request_timeout", defaults.get("request_timeout", 180))),
        "--output",
        str(output_dir),
    ]
    fault = spec.get("fault")
    if fault:
        command.extend(["--fault", str(fault)])
    detection_delay = spec.get(
        "detection_delay_ticks", defaults.get("detection_delay_ticks")
    )
    if detection_delay is not None and spec["mode"] != "clean":
        command.extend(["--detection-delay-ticks", str(detection_delay)])
    return command


def parse_child_output(stdout: str) -> dict[str, Any]:
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        return {"status": "error", "error": "runner returned non-JSON stdout"}
    return parsed if isinstance(parsed, dict) else {
        "status": "error",
        "error": "runner returned a non-object JSON value",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_revision() -> dict[str, Any]:
    try:
        top_level = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--show-toplevel"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        if Path(top_level).resolve() != REPO_ROOT.resolve():
            return {"commit": None, "dirty": None}
        commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--backend", choices=("llm", "scripted", "dry-run"), default="llm"
    )
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--process-timeout", type=int, default=900)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute the plan; without this flag, only print commands",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = read_manifest(manifest_path)
    repetitions = args.repetitions or manifest.get("repetitions", 1)
    if repetitions < 1:
        parser.error("--repetitions must be positive")
    defaults = manifest.get("defaults", {})
    batch_name = manifest.get("name", manifest_path.stem)
    if not isinstance(batch_name, str) or not SAFE_ID.fullmatch(batch_name):
        raise ValueError("batch name must contain only letters, digits, '_' or '-'")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else REPO_ROOT / "results" / "batches" / f"{timestamp}-{batch_name}"
    )

    planned: list[tuple[str, list[str], Path, dict[str, Any]]] = []
    for spec in manifest["runs"]:
        run_id = spec.get("id")
        if not isinstance(run_id, str) or not SAFE_ID.fullmatch(run_id):
            raise ValueError(f"invalid run id: {run_id!r}")
        for repetition in range(1, repetitions + 1):
            instance_id = f"{run_id}-r{repetition:02d}"
            output_dir = output_root / instance_id
            map_path = resolve_repo_path(spec["map"])
            map_data = json.loads(map_path.read_text())
            condition = {
                "condition_id": run_id,
                "repetition": repetition,
                "map": spec["map"],
                "map_id": map_data.get("id"),
                "mode": spec["mode"],
                "fault_id": spec.get("fault"),
                "fault_class": next(
                    (
                        fault.get("class", "unspecified")
                        for fault in map_data.get("faults", [])
                        if fault.get("id") == spec.get("fault")
                    ),
                    None,
                ),
                "planning_granularity": str(
                    spec.get(
                        "planning_granularity",
                        defaults.get("planning_granularity", "chunk"),
                    )
                ),
                "chunk_size": int(spec.get("chunk_size", defaults.get("chunk_size", 3))),
                "detection_delay_ticks": spec.get(
                    "detection_delay_ticks", defaults.get("detection_delay_ticks")
                ),
            }
            planned.append(
                (
                    instance_id,
                    build_command(
                        spec, defaults, backend=args.backend, output_dir=output_dir
                    ),
                    output_dir,
                    condition,
                )
            )

    print(f"Batch: {batch_name}")
    print(f"Runs: {len(planned)}")
    for instance_id, command, _, _ in planned:
        print(f"[{instance_id}] {shlex.join(command)}")
    if not args.execute:
        print("Dry run only. Add --execute to start the batch.")
        return 0

    if args.backend == "llm":
        missing = [
            name
            for name in ("LLM_CHAT_COMPLETIONS_URL", "LLM_API_KEY", "LLM_MODEL")
            if not os.environ.get(name)
        ]
        if missing:
            raise ValueError("missing environment variables: " + ", ".join(missing))
        expected_model = manifest.get("expected_model")
        if expected_model and os.environ["LLM_MODEL"] != expected_model:
            raise ValueError(
                f"manifest requires LLM_MODEL={expected_model!r}, got "
                f"{os.environ['LLM_MODEL']!r}"
            )
        expected_host = manifest.get("expected_endpoint_host")
        actual_host = urllib.parse.urlparse(
            os.environ["LLM_CHAT_COMPLETIONS_URL"]
        ).hostname
        if expected_host and actual_host != expected_host:
            raise ValueError(
                f"manifest requires endpoint host {expected_host!r}, got "
                f"{actual_host!r}"
            )

    output_root.mkdir(parents=True, exist_ok=False)
    outcomes: list[dict[str, Any]] = []
    for index, (instance_id, command, output_dir, condition) in enumerate(
        planned, start=1
    ):
        print(f"Running {index}/{len(planned)}: {instance_id}", flush=True)
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                timeout=args.process_timeout,
                check=False,
            )
            result = parse_child_output(completed.stdout)
            outcomes.append(
                {
                    "run_id": instance_id,
                    "return_code": completed.returncode,
                    "output_dir": str(output_dir),
                    "condition": condition,
                    "result": result,
                    "stderr": completed.stderr[-4000:],
                }
            )
        except subprocess.TimeoutExpired as exc:
            outcomes.append(
                {
                    "run_id": instance_id,
                    "return_code": None,
                    "output_dir": str(output_dir),
                    "condition": condition,
                    "result": {
                        "status": "timeout",
                        "error": f"process exceeded {args.process_timeout} seconds",
                    },
                    "stderr": (exc.stderr or "")[-4000:]
                    if isinstance(exc.stderr, str)
                    else "",
                }
            )

    endpoint_host = urllib.parse.urlparse(
        os.environ.get("LLM_CHAT_COMPLETIONS_URL", "")
    ).hostname
    summary = {
        "batch_name": batch_name,
        "created_at_utc": timestamp,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "repository": repository_revision(),
        "python_version": sys.version,
        "map_sha256": {
            spec["map"]: sha256_file(resolve_repo_path(spec["map"]))
            for spec in manifest["runs"]
        },
        "backend": args.backend,
        "model": os.environ.get("LLM_MODEL") if args.backend == "llm" else None,
        "endpoint_host": endpoint_host,
        "planned_runs": len(planned),
        "completed_runs": len(outcomes),
        "status_counts": {
            status: sum(
                outcome["result"].get("status") == status for outcome in outcomes
            )
            for status in ("ok", "error", "timeout")
        },
        "outcomes": outcomes,
    }
    summary_path = output_root / "batch-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "complete", "summary": str(summary_path)}, indent=2))
    return 0 if summary["status_counts"]["ok"] == len(outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
