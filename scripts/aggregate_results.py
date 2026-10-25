#!/usr/bin/env python3
"""Aggregate batch outputs into reviewable Phase 1 CSV/JSON summaries.

The aggregator intentionally keeps errors and timeouts in the run-level table
and in all denominators. Successful child runs are enriched from their
``episode-summary.json`` when it is available; otherwise the structured result
embedded in ``batch-summary.json`` is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


RUN_FIELDS = [
    "batch_name",
    "run_id",
    "run_status",
    "return_code",
    "backend",
    "model",
    "map_id",
    "mode",
    "fault_id",
    "fault_class",
    "fault_selection_rule_type",
    "fault_declared_phase",
    "fault_injection_phase",
    "fault_injection_tick",
    "detection_phase",
    "detection_tick",
    "observed_detection_delay_ticks",
    "planning_granularity",
    "chunk_size",
    "detection_delay_ticks",
    "final_success",
    "initial_success",
    "clean_episode_success",
    "recovered_success",
    "ever_failed",
    "recovery_triggered",
    "attempt_count",
    "trap_entries",
    "safety_violations",
    "backend_calls",
    "llm_calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "end_to_end_ms",
    "initial_execution_ms",
    "recovery_ms",
    "llm_ms",
    "dependency_tracking_ms",
    "fault_injected",
    "fault_delivered_to_recipient",
    "fault_exposed_to_planner",
    "fault_explicitly_consumed",
    "fault_structurally_consumed",
    "fault_counterfactually_relevant_plan",
    "fault_counterfactually_relevant_executed",
    "fault_caused_failure",
    "affected_stage_count",
    "fault_affected_stages",
    "unauthorized_actions",
    "blocked_unsafe_action_attempts",
    "wasted_executed_actions",
    "wasted_backend_calls",
    "wasted_llm_calls",
    "wasted_total_tokens",
    "declared_tp",
    "declared_fp",
    "declared_fn",
    "declared_tn",
    "declared_precision",
    "declared_recall",
    "structured_tp",
    "structured_fp",
    "structured_fn",
    "structured_tn",
    "structured_precision",
    "structured_recall",
    "conservative_tp",
    "conservative_fp",
    "conservative_fn",
    "conservative_tn",
    "conservative_precision",
    "conservative_recall",
    "verification_checks",
    "verification_rejections",
    "error_category",
    "error_type",
    "error",
    "output_dir",
    "batch_summary",
]

GROUP_FIELDS = [
    "backend",
    "model",
    "map_id",
    "mode",
    "fault_id",
    "fault_class",
    "planning_granularity",
    "chunk_size",
    "detection_delay_ticks",
]

METRIC_FIELDS = [
    "trap_entries",
    "safety_violations",
    "backend_calls",
    "llm_calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "end_to_end_ms",
    "initial_execution_ms",
    "recovery_ms",
    "llm_ms",
    "dependency_tracking_ms",
    "verification_checks",
    "verification_rejections",
    "affected_stage_count",
    "unauthorized_actions",
    "blocked_unsafe_action_attempts",
    "wasted_executed_actions",
    "wasted_backend_calls",
    "wasted_llm_calls",
    "wasted_total_tokens",
]

PAIR_FIELDS = [
    "batch_name",
    "run_id",
    "backend",
    "model",
    "map_id",
    "fault_id",
    "fault_class",
    "planning_granularity",
    "chunk_size",
    "detection_delay_ticks",
    "fault_declared_phase",
    "fault_injection_phase",
    "fault_injection_tick",
    "detection_phase",
    "detection_tick",
    "observed_detection_delay_ticks",
    "shared_faulty_trajectory",
    "no_recovery_success",
    "no_recovery_trap_entries",
    "no_recovery_backend_calls",
    "no_recovery_llm_calls",
    "no_recovery_total_tokens",
    "no_recovery_latency_ms",
    "full_restart_success",
    "full_restart_trap_entries",
    "full_restart_backend_calls",
    "full_restart_llm_calls",
    "full_restart_total_tokens",
    "full_restart_latency_ms",
    "full_restart_wasted_executed_actions",
    "full_restart_wasted_backend_calls",
    "full_restart_wasted_llm_calls",
    "full_restart_wasted_total_tokens",
    "delta_trap_entries",
    "delta_backend_calls",
    "delta_llm_calls",
    "delta_total_tokens",
    "delta_latency_ms",
]

PAIR_GROUP_FIELDS = [
    "backend",
    "model",
    "map_id",
    "fault_id",
    "fault_class",
    "planning_granularity",
    "chunk_size",
    "detection_delay_ticks",
]

PAIR_METRIC_FIELDS = [
    "no_recovery_trap_entries",
    "no_recovery_backend_calls",
    "no_recovery_llm_calls",
    "no_recovery_total_tokens",
    "no_recovery_latency_ms",
    "full_restart_trap_entries",
    "full_restart_backend_calls",
    "full_restart_llm_calls",
    "full_restart_total_tokens",
    "full_restart_latency_ms",
    "full_restart_wasted_executed_actions",
    "full_restart_wasted_backend_calls",
    "full_restart_wasted_llm_calls",
    "full_restart_wasted_total_tokens",
    "delta_trap_entries",
    "delta_backend_calls",
    "delta_llm_calls",
    "delta_total_tokens",
    "delta_latency_ms",
]

FAULT_BOOLEAN_FIELDS = [
    "fault_injected",
    "fault_delivered_to_recipient",
    "fault_exposed_to_planner",
    "fault_explicitly_consumed",
    "fault_structurally_consumed",
    "fault_counterfactually_relevant_plan",
    "fault_counterfactually_relevant_executed",
    "fault_caused_failure",
]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def discover_batch_summaries(inputs: Iterable[Path]) -> list[Path]:
    found: set[Path] = set()
    for raw_path in inputs:
        path = raw_path.resolve()
        if path.is_file():
            if path.name != "batch-summary.json":
                raise ValueError(f"expected batch-summary.json, got: {path}")
            found.add(path)
        elif path.is_dir():
            found.update(item.resolve() for item in path.rglob("batch-summary.json"))
        else:
            raise ValueError(f"input does not exist: {path}")
    if not found:
        raise ValueError("no batch-summary.json files found")
    return sorted(found)


def nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def load_episode(output_dir: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    episode_path = output_dir / "episode-summary.json"
    if episode_path.is_file():
        try:
            return read_json(episode_path)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return fallback


def load_error(output_dir: Path) -> dict[str, Any]:
    error_path = output_dir / "run-error.json"
    if not error_path.is_file():
        return {}
    try:
        return read_json(error_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def error_category(status: str, error_type: Any, error: Any) -> str:
    if status == "timeout":
        return "timeout"
    if status == "ok":
        return "none"
    text = f"{error_type or ''} {error or ''}".lower()
    if "json" in text or "schema" in text or "validation" in text:
        return "malformed_output"
    if "loop" in text:
        return "navigation_loop"
    if "http" in text or "endpoint" in text or "connection" in text:
        return "provider_error"
    if "length" in text or "token" in text:
        return "generation_limit"
    return "execution_error"


def dependency_columns(episode: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    quality = episode.get("dependency_quality", {})
    for method in ("declared", "structured", "conservative"):
        values = quality.get(method, {}) if isinstance(quality, dict) else {}
        for metric in ("tp", "fp", "fn", "tn", "precision", "recall"):
            output[f"{method}_{metric}"] = (
                values.get(metric) if isinstance(values, dict) else None
            )
    return output


def outcome_to_row(
    batch: dict[str, Any], batch_path: Path, outcome: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    result = outcome.get("result", {})
    if not isinstance(result, dict):
        result = {"status": "error", "error": "invalid batch result object"}
    status = str(result.get("status", "error"))
    output_dir = Path(str(outcome.get("output_dir", "")))
    if not output_dir.is_absolute():
        output_dir = (batch_path.parent / output_dir).resolve()
    episode = load_episode(output_dir, result) if status == "ok" else result
    condition = outcome.get("condition", {})
    if not isinstance(condition, dict):
        condition = {}
    error_record = load_error(output_dir) if status != "ok" else {}
    error_text = error_record.get("error", result.get("error"))
    error_type = error_record.get("error_type")
    usage = episode.get("usage", {}) if isinstance(episode, dict) else {}
    latency = episode.get("latency_ms", {}) if isinstance(episode, dict) else {}
    fault_metrics = (
        episode.get("fault_metrics", {}) if isinstance(episode, dict) else {}
    )
    detection = episode.get("detection", {}) if isinstance(episode, dict) else {}
    if not isinstance(detection, dict):
        detection = {}
    paired_baselines = (
        episode.get("paired_baselines") if isinstance(episode, dict) else None
    )
    restart_pair = (
        paired_baselines.get("full_restart", {})
        if isinstance(paired_baselines, dict)
        else {}
    )
    row: dict[str, Any] = {
        "batch_name": batch.get("batch_name"),
        "run_id": outcome.get("run_id"),
        "run_status": status,
        "return_code": outcome.get("return_code"),
        "backend": batch.get("backend"),
        "model": batch.get("model"),
        "map_id": episode.get("map_id", condition.get("map_id")),
        "mode": episode.get("mode", condition.get("mode")),
        "fault_id": episode.get("fault_id", condition.get("fault_id")),
        "fault_class": episode.get("fault_class", condition.get("fault_class")),
        "fault_selection_rule_type": nested(
            episode, "fault_selection_rule", "type"
        ),
        "fault_declared_phase": episode.get("fault_declared_phase"),
        "fault_injection_phase": episode.get("fault_injection_phase"),
        "fault_injection_tick": episode.get("fault_injection_tick"),
        "detection_phase": detection.get("phase"),
        "detection_tick": detection.get("tick"),
        "observed_detection_delay_ticks": detection.get("observed_delay_ticks"),
        "planning_granularity": episode.get(
            "planning_granularity", condition.get("planning_granularity")
        ),
        "chunk_size": episode.get("chunk_size", condition.get("chunk_size")),
        "detection_delay_ticks": episode.get(
            "detection_delay_ticks", condition.get("detection_delay_ticks")
        ),
        "final_success": episode.get("final_success", episode.get("success")),
        "initial_success": episode.get("initial_success"),
        "clean_episode_success": episode.get("clean_episode_success"),
        "recovered_success": episode.get("recovered_success"),
        "ever_failed": episode.get("ever_failed"),
        "recovery_triggered": episode.get("recovery_triggered"),
        "attempt_count": episode.get("attempt_count"),
        "trap_entries": episode.get("trap_entries"),
        "safety_violations": episode.get("safety_violations"),
        "backend_calls": episode.get("backend_calls"),
        "llm_calls": episode.get("llm_calls"),
        "prompt_tokens": nested(usage, "prompt_tokens"),
        "completion_tokens": nested(usage, "completion_tokens"),
        "total_tokens": nested(usage, "total_tokens"),
        "end_to_end_ms": nested(latency, "end_to_end"),
        "initial_execution_ms": nested(latency, "initial_execution"),
        "recovery_ms": nested(latency, "recovery"),
        "llm_ms": nested(latency, "llm"),
        "dependency_tracking_ms": nested(latency, "dependency_tracking"),
        "affected_stage_count": fault_metrics.get("affected_stage_count"),
        "fault_affected_stages": json.dumps(
            episode.get("fault_affected_stages", []), separators=(",", ":")
        ),
        "unauthorized_actions": fault_metrics.get("unauthorized_actions"),
        "blocked_unsafe_action_attempts": fault_metrics.get(
            "blocked_unsafe_action_attempts"
        ),
        "wasted_executed_actions": restart_pair.get("wasted_executed_actions"),
        "wasted_backend_calls": restart_pair.get("wasted_backend_calls"),
        "wasted_llm_calls": restart_pair.get("wasted_llm_calls"),
        "wasted_total_tokens": nested(restart_pair, "wasted_usage", "total_tokens"),
        "verification_checks": episode.get("verification_checks"),
        "verification_rejections": episode.get("verification_rejections"),
        "error_category": error_category(status, error_type, error_text),
        "error_type": error_type,
        "error": error_text,
        "output_dir": (
            str(output_dir.relative_to(batch_path.parent))
            if batch_path.parent in output_dir.parents
            else output_dir.name
        ),
        "batch_summary": f"{batch_path.parent.name}/{batch_path.name}",
    }
    for key in FAULT_BOOLEAN_FIELDS:
        row[key] = fault_metrics.get(key) if isinstance(fault_metrics, dict) else None
    row.update(dependency_columns(episode))
    return row, paired_baselines if status == "ok" else None


def collect_rows(
    batch_paths: Iterable[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for batch_path in batch_paths:
        batch = read_json(batch_path)
        outcomes = batch.get("outcomes", [])
        if not isinstance(outcomes, list):
            raise ValueError(f"batch outcomes must be a list: {batch_path}")
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                continue
            row, paired = outcome_to_row(batch, batch_path, outcome)
            rows.append(row)
            if isinstance(paired, dict):
                paired_rows.append(make_paired_row(row, paired))
    return rows, paired_rows


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def mean_std(values: Iterable[Any]) -> tuple[float | None, float | None]:
    clean = [value for item in values if (value := numeric(item)) is not None]
    if not clean:
        return None, None
    mean = statistics.fmean(clean)
    std = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return round(mean, 4), round(std, 4)


def rate(values: Iterable[Any]) -> float | None:
    clean = [value for value in values if isinstance(value, bool)]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 4)


def condition_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in GROUP_FIELDS)].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: repr(item[0])):
        summary = dict(zip(GROUP_FIELDS, key))
        summary.update(
            {
                "n_planned": len(group),
                "n_ok": sum(item["run_status"] == "ok" for item in group),
                "n_error": sum(item["run_status"] == "error" for item in group),
                "n_timeout": sum(item["run_status"] == "timeout" for item in group),
                "execution_success_rate": round(
                    sum(item["run_status"] == "ok" for item in group) / len(group), 4
                ),
                # Errors/timeouts count as unsuccessful in the primary task rate.
                "task_success_rate_all_runs": round(
                    sum(item.get("final_success") is True for item in group)
                    / len(group),
                    4,
                ),
                "task_success_rate_completed": rate(
                    item.get("final_success")
                    for item in group
                    if item["run_status"] == "ok"
                ),
                "clean_episode_success_rate_all_runs": round(
                    sum(item.get("clean_episode_success") is True for item in group)
                    / len(group),
                    4,
                ),
                "initial_success_rate_all_runs": round(
                    sum(item.get("initial_success") is True for item in group)
                    / len(group),
                    4,
                ),
                "recovered_success_rate_all_runs": round(
                    sum(item.get("recovered_success") is True for item in group)
                    / len(group),
                    4,
                ),
                "ever_failed_rate_all_runs": round(
                    sum(item.get("ever_failed") is True for item in group)
                    / len(group),
                    4,
                ),
                "recovery_triggered_rate_all_runs": round(
                    sum(item.get("recovery_triggered") is True for item in group)
                    / len(group),
                    4,
                ),
            }
        )
        fault_condition = summary.get("fault_id") is not None
        for field in FAULT_BOOLEAN_FIELDS:
            summary[f"{field}_rate_completed"] = rate(
                item.get(field)
                for item in group
                if item["run_status"] == "ok"
            )
            summary[f"{field}_rate_all_runs"] = (
                None
                if not fault_condition
                else round(
                    sum(item.get(field) is True for item in group) / len(group),
                    4,
                )
            )
        for field in METRIC_FIELDS:
            mean, std = mean_std(item.get(field) for item in group)
            summary[f"{field}_mean"] = mean
            summary[f"{field}_std"] = std
        for method in ("declared", "structured", "conservative"):
            totals = {
                metric: sum(
                    int(item.get(f"{method}_{metric}") or 0)
                    for item in group
                    if item["run_status"] == "ok"
                )
                for metric in ("tp", "fp", "fn", "tn")
            }
            summary.update(
                {f"{method}_{metric}_sum": value for metric, value in totals.items()}
            )
            predicted_positive = totals["tp"] + totals["fp"]
            actual_positive = totals["tp"] + totals["fn"]
            summary[f"{method}_pooled_precision"] = (
                None
                if predicted_positive == 0
                else round(totals["tp"] / predicted_positive, 4)
            )
            summary[f"{method}_pooled_recall"] = (
                None
                if actual_positive == 0
                else round(totals["tp"] / actual_positive, 4)
            )
        output.append(summary)
    return output


def make_paired_row(row: dict[str, Any], paired: dict[str, Any]) -> dict[str, Any]:
    no_recovery = paired.get("no_recovery", {})
    restart = paired.get("full_restart", {})
    output = {
        field: row.get(field)
        for field in (
            "batch_name",
            "run_id",
            "backend",
            "model",
            "map_id",
            "fault_id",
            "fault_class",
            "planning_granularity",
            "chunk_size",
            "detection_delay_ticks",
            "fault_declared_phase",
            "fault_injection_phase",
            "fault_injection_tick",
            "detection_phase",
            "detection_tick",
            "observed_detection_delay_ticks",
        )
    }
    output["shared_faulty_trajectory"] = paired.get("shared_faulty_trajectory")
    for prefix, values in (("no_recovery", no_recovery), ("full_restart", restart)):
        output[f"{prefix}_success"] = values.get("success")
        output[f"{prefix}_trap_entries"] = values.get("trap_entries")
        output[f"{prefix}_backend_calls"] = values.get("backend_calls")
        output[f"{prefix}_llm_calls"] = values.get("llm_calls")
        output[f"{prefix}_total_tokens"] = nested(values, "usage", "total_tokens")
        output[f"{prefix}_latency_ms"] = values.get("latency_ms")
    output["full_restart_wasted_executed_actions"] = restart.get(
        "wasted_executed_actions"
    )
    output["full_restart_wasted_backend_calls"] = restart.get(
        "wasted_backend_calls"
    )
    output["full_restart_wasted_llm_calls"] = restart.get("wasted_llm_calls")
    output["full_restart_wasted_total_tokens"] = nested(
        restart, "wasted_usage", "total_tokens"
    )
    for metric in ("trap_entries", "backend_calls", "llm_calls", "total_tokens", "latency_ms"):
        left = numeric(output.get(f"full_restart_{metric}"))
        right = numeric(output.get(f"no_recovery_{metric}"))
        output[f"delta_{metric}"] = None if left is None or right is None else round(left - right, 4)
    return output


def paired_condition_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate paired no-recovery/full-restart outcomes by condition."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in PAIR_GROUP_FIELDS)].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: repr(item[0])):
        summary = dict(zip(PAIR_GROUP_FIELDS, key))
        summary.update(
            {
                "n_pairs": len(group),
                "no_recovery_success_rate": rate(
                    item.get("no_recovery_success") for item in group
                ),
                "full_restart_success_rate": rate(
                    item.get("full_restart_success") for item in group
                ),
            }
        )
        for field in PAIR_METRIC_FIELDS:
            mean, std = mean_std(item.get(field) for item in group)
            summary[f"{field}_mean"] = mean
            summary[f"{field}_std"] = std
        output.append(summary)
    return output


def failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: Counter[tuple[Any, ...]] = Counter()
    for row in rows:
        if row["run_status"] != "ok":
            outcome = row["error_category"]
        elif row.get("trap_entries", 0):
            outcome = "trap_entry"
        elif row.get("final_success") is not True:
            outcome = "task_failure"
        else:
            outcome = "success"
        grouped[(row.get("map_id"), row.get("mode"), row.get("fault_id"), outcome)] += 1
    return [
        {
            "map_id": key[0],
            "mode": key[1],
            "fault_id": key[2],
            "outcome": key[3],
            "count": count,
        }
        for key, count in sorted(grouped.items(), key=lambda item: repr(item[0]))
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="batch-summary.json files or directories searched recursively",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    batch_paths = discover_batch_summaries(args.inputs)
    rows, paired = collect_rows(batch_paths)
    conditions = condition_rows(rows)
    paired_conditions = paired_condition_rows(paired)
    failures = failure_rows(rows)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "run_level.csv", rows, RUN_FIELDS)
    write_csv(output_dir / "condition_summary.csv", conditions)
    write_csv(output_dir / "paired_recovery.csv", paired, PAIR_FIELDS)
    write_csv(output_dir / "paired_condition_summary.csv", paired_conditions)
    write_csv(output_dir / "failure_summary.csv", failures)

    status_counts = Counter(row["run_status"] for row in rows)
    overview = {
        "batch_summaries": [
            f"{path.parent.name}/{path.name}" for path in batch_paths
        ],
        "planned_run_records": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "execution_success_rate": (
            None
            if not rows
            else round(status_counts.get("ok", 0) / len(rows), 4)
        ),
        "task_success_rate_all_runs": (
            None
            if not rows
            else round(sum(row.get("final_success") is True for row in rows) / len(rows), 4)
        ),
        "initial_success_rate_all_runs": (
            None
            if not rows
            else round(
                sum(row.get("initial_success") is True for row in rows) / len(rows),
                4,
            )
        ),
        "clean_episode_success_rate_all_runs": (
            None
            if not rows
            else round(
                sum(row.get("clean_episode_success") is True for row in rows)
                / len(rows),
                4,
            )
        ),
        "condition_count": len(conditions),
        "paired_recovery_count": len(paired),
        "paired_condition_count": len(paired_conditions),
        "outputs": {
            "run_level": "run_level.csv",
            "condition_summary": "condition_summary.csv",
            "paired_recovery": "paired_recovery.csv",
            "paired_condition_summary": "paired_condition_summary.csv",
            "failure_summary": "failure_summary.csv",
        },
    }
    (output_dir / "phase1_summary.json").write_text(json.dumps(overview, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output": str(output_dir), **overview}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
