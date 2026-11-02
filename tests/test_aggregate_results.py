from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGGREGATOR = PROJECT_ROOT / "scripts" / "aggregate_results.py"
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from aggregate_results import (  # noqa: E402
    collect_rows,
    condition_rows,
    discover_batch_summaries,
    error_category,
    paired_condition_rows,
)


class AggregateResultsTest(unittest.TestCase):
    def build_fixture(self, root: Path) -> Path:
        successful_dir = root / "clean-r01"
        successful_dir.mkdir()
        (successful_dir / "episode-summary.json").write_text(
            json.dumps(
                {
                    "map_id": "held-out-map-v1",
                    "mode": "clean",
                    "fault_id": None,
                    "fault_class": None,
                    "planning_granularity": "chunk",
                    "chunk_size": 3,
                    "detection_delay_ticks": None,
                    "final_success": True,
                    "initial_success": True,
                    "clean_episode_success": True,
                    "recovered_success": False,
                    "ever_failed": False,
                    "recovery_triggered": False,
                    "attempt_count": 1,
                    "trap_entries": 0,
                    "safety_violations": 0,
                    "backend_calls": 4,
                    "llm_calls": 4,
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "total_tokens": 120,
                    },
                    "latency_ms": {
                        "end_to_end": 1000,
                        "initial_execution": 1000,
                        "recovery": None,
                        "llm": 900,
                        "dependency_tracking": 1,
                    },
                    "fault_metrics": {},
                    "dependency_quality": {
                        "declared": {
                            "tp": 1,
                            "fp": 0,
                            "fn": 0,
                            "tn": 1,
                            "precision": 1.0,
                            "recall": 1.0,
                        }
                    },
                    "verification_checks": 0,
                    "verification_rejections": 0,
                    "paired_baselines": None,
                }
            )
            + "\n"
        )
        timeout_dir = root / "clean-r02"
        timeout_dir.mkdir()
        batch = {
            "batch_name": "fixture",
            "backend": "llm",
            "model": "test-model",
            "outcomes": [
                {
                    "run_id": "clean-r01",
                    "return_code": 0,
                    "output_dir": str(successful_dir),
                    "result": {"status": "ok", "final_success": True},
                    "stderr": "",
                },
                {
                    "run_id": "clean-r02",
                    "return_code": None,
                    "output_dir": str(timeout_dir),
                    "condition": {
                        "map_id": "held-out-map-v1",
                        "mode": "clean",
                        "fault_id": None,
                        "fault_class": None,
                        "planning_granularity": "chunk",
                        "chunk_size": 3,
                        "detection_delay_ticks": 0,
                    },
                    "result": {"status": "timeout", "error": "too slow"},
                    "stderr": "",
                },
            ],
        }
        batch_path = root / "batch-summary.json"
        batch_path.write_text(json.dumps(batch) + "\n")
        return batch_path

    def test_timeout_remains_in_primary_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            batch_path = self.build_fixture(Path(temporary))
            rows, paired = collect_rows([batch_path])
            self.assertEqual(len(rows), 2)
            self.assertEqual(paired, [])
            summary = condition_rows(rows)
            self.assertEqual(len(summary), 1)
            self.assertEqual(summary[0]["n_planned"], 2)
            self.assertEqual(summary[0]["n_timeout"], 1)
            self.assertEqual(summary[0]["task_success_rate_all_runs"], 0.5)

    def test_non_json_child_timeout_is_recovered_from_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = {
                "batch_name": "fixture",
                "backend": "llm",
                "model": "test-model",
                "outcomes": [
                    {
                        "run_id": "timeout-r01",
                        "return_code": 1,
                        "output_dir": str(root / "timeout-r01"),
                        "condition": {
                            "condition_id": "timeout-condition",
                            "repetition": 1,
                            "map_id": "map",
                            "mode": "full-restart",
                            "fault_id": "fault",
                            "fault_class": "adversarial",
                            "planning_granularity": "chunk",
                            "chunk_size": 3,
                            "detection_delay_ticks": 0,
                        },
                        "result": {
                            "status": "error",
                            "error": "runner returned non-JSON stdout",
                        },
                        "stderr": "traceback omitted\nTimeoutError: read timed out",
                    }
                ],
            }
            batch_path = root / "batch-summary.json"
            batch_path.write_text(json.dumps(batch))
            rows, _ = collect_rows([batch_path])
        self.assertEqual(rows[0]["error_type"], "TimeoutError")
        self.assertEqual(rows[0]["error_category"], "timeout")
        self.assertEqual(rows[0]["error"], "TimeoutError: read timed out")
        self.assertEqual(error_category("error", "TimeoutError", "read timed out"), "timeout")

    def test_cli_writes_all_phase1_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.build_fixture(root)
            output = root / "summaries"
            completed = subprocess.run(
                [sys.executable, str(AGGREGATOR), str(root), "--output", str(output)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            for name in (
                "run_level.csv",
                "condition_summary.csv",
                "paired_recovery.csv",
                "paired_condition_summary.csv",
                "failure_summary.csv",
                "phase1_summary.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            with (output / "run_level.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["run_status"] for row in rows}, {"ok", "timeout"})
            overview = json.loads((output / "phase1_summary.json").read_text())
            self.assertEqual(overview["planned_run_records"], 2)
            self.assertEqual(overview["task_success_rate_all_runs"], 0.5)

    def test_discovery_rejects_non_batch_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "other.json"
            path.write_text("{}")
            with self.assertRaises(ValueError):
                discover_batch_summaries([path])

    def test_fault_rates_name_both_denominators(self) -> None:
        base = {
            "backend": "llm",
            "model": "test-model",
            "map_id": "map",
            "mode": "full-restart",
            "fault_id": "fault",
            "fault_class": "accidental",
            "planning_granularity": "chunk",
            "chunk_size": 3,
            "detection_delay_ticks": 0,
        }
        rows = [
            {**base, "run_status": "ok", "fault_injected": True},
            {**base, "run_status": "error", "fault_injected": None},
        ]
        summary = condition_rows(rows)[0]
        self.assertEqual(summary["fault_injected_rate_completed"], 1.0)
        self.assertEqual(summary["fault_injected_rate_all_runs"], 0.5)
        self.assertNotIn("fault_injected_rate", summary)

    def test_paired_condition_summary_aggregates_repetitions(self) -> None:
        base = {
            "backend": "llm",
            "model": "test-model",
            "map_id": "map",
            "fault_id": "fault",
            "fault_class": "accidental",
            "planning_granularity": "chunk",
            "chunk_size": 3,
            "detection_delay_ticks": 0,
            "no_recovery_success": False,
            "full_restart_success": True,
            "delta_total_tokens": 20,
            "full_restart_wasted_total_tokens": 10,
        }
        summary = paired_condition_rows([base, {**base, "delta_total_tokens": 40}])[0]
        self.assertEqual(summary["n_pairs"], 2)
        self.assertEqual(summary["no_recovery_success_rate"], 0.0)
        self.assertEqual(summary["full_restart_success_rate"], 1.0)
        self.assertEqual(summary["delta_total_tokens_mean"], 30.0)
        self.assertEqual(summary["full_restart_wasted_total_tokens_mean"], 10.0)

    def test_verifier_confusion_is_pooled_across_runs(self) -> None:
        base = {
            "backend": "llm",
            "model": "test-model",
            "map_id": "map",
            "mode": "llm-verify",
            "fault_id": "fault",
            "fault_class": "accidental",
            "planning_granularity": "chunk",
            "chunk_size": 3,
            "detection_delay_ticks": 0,
            "run_status": "ok",
        }
        rows = [
            {
                **base,
                "verification_tp": 1,
                "verification_fp": 0,
                "verification_fn": 0,
                "verification_tn": 3,
            },
            {
                **base,
                "verification_tp": 0,
                "verification_fp": 1,
                "verification_fn": 1,
                "verification_tn": 2,
            },
        ]
        summary = condition_rows(rows)[0]
        self.assertEqual(summary["verification_tp_sum"], 1)
        self.assertEqual(summary["verification_fp_sum"], 1)
        self.assertEqual(summary["verification_fn_sum"], 1)
        self.assertEqual(summary["verification_tn_sum"], 5)
        self.assertEqual(summary["verification_pooled_precision"], 0.5)
        self.assertEqual(summary["verification_pooled_recall"], 0.5)
        self.assertEqual(summary["verification_accuracy"], 0.75)

    def test_historical_route_relevance_fields_are_migrated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "legacy-r01"
            run_dir.mkdir()
            (run_dir / "episode-summary.json").write_text(
                json.dumps(
                    {
                        "map_id": "legacy-map",
                        "mode": "no-recovery",
                        "fault_id": "legacy-fault",
                        "fault_metrics": {
                            "fault_counterfactually_relevant_plan": True,
                            "fault_counterfactually_relevant_executed": False,
                        },
                    }
                )
            )
            batch = {
                "batch_name": "legacy",
                "backend": "llm",
                "model": "test-model",
                "outcomes": [
                    {
                        "run_id": "legacy-r01",
                        "return_code": 0,
                        "output_dir": str(run_dir),
                        "condition": {
                            "condition_id": "legacy-condition",
                            "repetition": 1,
                        },
                        "result": {"status": "ok"},
                    }
                ],
            }
            batch_path = root / "batch-summary.json"
            batch_path.write_text(json.dumps(batch))
            rows, _ = collect_rows([batch_path])
        self.assertTrue(rows[0]["fault_structured_route_relevant_plan"])
        self.assertFalse(rows[0]["fault_structured_route_relevant_executed"])
        self.assertEqual(rows[0]["condition_id"], "legacy-condition")
        self.assertEqual(rows[0]["repetition"], 1)


if __name__ == "__main__":
    unittest.main()
