#!/usr/bin/env python3
"""Run the shared-world propagation and recovery experiment."""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

from recovery_harness import (
    DryRunBackend,
    EpisodeError,
    LLMBackend,
    ScriptedBackend,
    read_json,
    redact_secrets,
    run_experiment,
    run_scheduler_preflight,
    validate_map,
    write_run,
)


def default_output_dir(
    project_root: Path,
    map_id: str,
    mode: str,
    backend: str,
    fault_id: str | None = None,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe_map_id = "".join(char if char.isalnum() or char in "-_" else "-" for char in map_id)
    safe_fault_id = (
        "".join(char if char.isalnum() or char in "-_" else "-" for char in fault_id)
        if fault_id
        else ""
    )
    suffix = f"-{safe_fault_id}" if safe_fault_id else ""
    return (
        project_root
        / "results"
        / "runs"
        / f"{timestamp}-{safe_map_id}-{mode}-{backend}{suffix}"
    )


def build_backend(args: argparse.Namespace):
    if args.backend == "scripted":
        return ScriptedBackend()
    if args.backend == "dry-run":
        return DryRunBackend()

    url = os.environ.get("LLM_CHAT_COMPLETIONS_URL")
    key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL")
    missing = [
        name
        for name, value in (
            ("LLM_CHAT_COMPLETIONS_URL", url),
            ("LLM_API_KEY", key),
            ("LLM_MODEL", model),
        )
        if not value
    ]
    if missing:
        raise ValueError("missing environment variables: " + ", ".join(missing))
    return LLMBackend(
        url=url,
        key=key,
        model=model,
        timeout=args.timeout,
        reasoning_effort=os.environ.get("LLM_REASONING_EFFORT", "none"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("map", type=Path)
    parser.add_argument(
        "--mode",
        choices=("clean", "no-recovery", "full-restart", "always-verify"),
        default="clean",
    )
    parser.add_argument("--fault")
    parser.add_argument(
        "--backend", choices=("scripted", "dry-run", "llm"), default="scripted"
    )
    parser.add_argument(
        "--planning-granularity",
        choices=("route", "chunk", "step", "junction"),
        default="step",
        help=(
            "step asks the LLM for one adjacent coordinate per environment turn; chunk asks "
            "for a short sequence of primitive actions; route asks for a complete coordinate "
            "path; junction retains the low-cost branch baseline"
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=3,
        help="maximum primitive actions in each chunk plan (default: 3)",
    )
    parser.add_argument(
        "--detection-delay-ticks",
        type=int,
        default=None,
        help=(
            "controlled detection delay after fault injection; 0 detects before "
            "the next plan, N waits N scheduler ticks, and an episode that ends "
            "first records outcome-time detection"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--list-faults", action="store_true")
    args = parser.parse_args()

    if args.chunk_size < 1:
        parser.error("--chunk-size must be at least 1")
    if args.detection_delay_ticks is not None and args.detection_delay_ticks < 0:
        parser.error("--detection-delay-ticks must be non-negative")

    project_root = Path(__file__).resolve().parent.parent
    data = read_json(args.map)
    validation = validate_map(data)

    if args.list_faults:
        print(
            json.dumps(
                [
                    {
                        "id": fault["id"],
                        "class": fault.get("class", "unspecified"),
                        "kind": fault["kind"],
                        "phase": fault["phase"],
                        "transmission_occurrence": fault.get(
                            "transmission_occurrence", 1
                        ),
                        "expected_effect": fault.get("expected_effect"),
                    }
                    for fault in data.get("faults", [])
                ],
                indent=2,
            )
        )
        return 0
    if args.validate_only:
        validation["scheduler_preflight"] = run_scheduler_preflight(
            data,
            planning_granularity=args.planning_granularity,
            chunk_size=args.chunk_size,
        )
        print(json.dumps(validation, indent=2))
        return 0

    if args.mode != "clean" and not args.fault:
        parser.error(f"--mode {args.mode} requires --fault")
    if args.mode == "clean" and args.fault:
        parser.error("--fault is not used in clean mode")

    output_dir = args.output or default_output_dir(
        project_root, data["id"], args.mode, args.backend, args.fault
    )
    endpoint = os.environ.get("LLM_CHAT_COMPLETIONS_URL", "")
    endpoint_host = (urllib.parse.urlparse(endpoint).hostname or "").lower()
    config = {
        "map_path": str(args.map.resolve()),
        "map_id": data["id"],
        "map_snapshot": data,
        "mode": args.mode,
        "fault_id": args.fault,
        "backend": args.backend,
        "planning_granularity": args.planning_granularity,
        "chunk_size": args.chunk_size,
        "detection_delay_ticks": args.detection_delay_ticks,
        "model": os.environ.get("LLM_MODEL") if args.backend == "llm" else None,
        "reasoning_effort": (
            os.environ.get("LLM_REASONING_EFFORT", "none")
            if args.backend == "llm"
            else None
        ),
        "temperature": (
            None
            if args.backend != "llm" or endpoint_host == "api.openai.com"
            else 0
        ),
        "validation": validation,
    }

    backend = None
    execution_started_ns = perf_counter_ns()
    try:
        backend = build_backend(args)
        result = run_experiment(
            data,
            backend,
            mode=args.mode,
            fault_id=args.fault,
            planning_granularity=args.planning_granularity,
            chunk_size=args.chunk_size,
            detection_delay_ticks=args.detection_delay_ticks,
        )
        write_run(output_dir, config, result)
    except (EpisodeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        failure_latency_ms = round(
            (perf_counter_ns() - execution_started_ns) / 1_000_000, 3
        )
        error_text = redact_secrets(str(exc), [os.environ.get("LLM_API_KEY", "")])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        if backend is not None:
            (output_dir / "calls.json").write_text(
                json.dumps(backend.call_records, indent=2) + "\n"
            )
        (output_dir / "run-error.json").write_text(
            json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "error": error_text,
                    "latency_ms": {"end_to_end": failure_latency_ms},
                },
                indent=2,
            )
            + "\n"
        )
        print(
            json.dumps(
                {
                    "status": "error",
                    "output": str(output_dir),
                    "error": error_text,
                    "latency_ms": {"end_to_end": failure_latency_ms},
                },
                indent=2,
            )
        )
        return 1

    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(output_dir),
                "map_id": result["map_id"],
                "mode": result["mode"],
                "fault_id": result["fault_id"],
                "fault_class": result["fault_class"],
                "planning_granularity": result["planning_granularity"],
                "chunk_size": result["chunk_size"],
                "detection_delay_ticks": result["detection_delay_ticks"],
                "initial_success": result["initial_success"],
                "final_success": result["final_success"],
                "recovered_success": result["recovered_success"],
                "clean_episode_success": result["clean_episode_success"],
                "ever_failed": result["ever_failed"],
                "attempt_count": result["attempt_count"],
                "trap_entries": result["trap_entries"],
                "safety_violations": result["safety_violations"],
                "fault_metrics": result["fault_metrics"],
                "dependency_quality": result["dependency_quality"],
                "verification_checks": result["verification_checks"],
                "verification_rejections": result["verification_rejections"],
                "detection": result["detection"],
                "paired_baselines": result["paired_baselines"],
                "recovery_triggered": result["recovery_triggered"],
                "backend_calls": result["backend_calls"],
                "llm_calls": result["llm_calls"],
                "usage": result["usage"],
                "latency_ms": result["latency_ms"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
