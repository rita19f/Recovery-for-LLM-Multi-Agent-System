from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from recovery_harness import run_scheduler_preflight, validate_map  # noqa: E402


EXPECTED_CLASSIFICATIONS = {
    "coincident_detours.json": {
        "f-water-lower-branch-swap": "outcome_relevant",
        "f-water-upper-side-drop-once": "non_propagating",
        "a-fire-upper-branch-swap": "outcome_relevant",
    },
    "three_portal_exchange.json": {
        "f-water-center-first-message-drop": "exposed_without_relevant_plan",
        "a-fire-center-relocated-right": "outcome_relevant",
    },
    "staged_crossing.json": {
        "f-water-approach-center-to-outer": "outcome_relevant",
        "f-water-exchange-delayed-shift": "exposed_without_relevant_plan",
        "a-fire-approach-center-to-outer": "outcome_relevant",
        "a-fire-exchange-delayed-shift": "exposed_without_relevant_plan",
    },
}


class EvaluationMapTest(unittest.TestCase):
    def test_frozen_map_geometry_and_fault_preflight(self) -> None:
        evaluation_dir = PROJECT_ROOT / "maps" / "evaluation"
        for filename, expected in EXPECTED_CLASSIFICATIONS.items():
            with self.subTest(map=filename):
                data = json.loads((evaluation_dir / filename).read_text())
                validation = validate_map(data)
                self.assertTrue(validation["passed"])
                self.assertLessEqual(data["dimensions"]["width"], 11)
                self.assertLessEqual(data["dimensions"]["height"], 9)
                preflight = run_scheduler_preflight(
                    data, planning_granularity="chunk", chunk_size=3
                )
                self.assertTrue(preflight["passed"])
                self.assertTrue(preflight["clean_success"])
                self.assertTrue(
                    all(item["phase_matches"] for item in preflight["fault_checks"])
                )
                self.assertTrue(
                    all(item["actual_tick"] is not None for item in preflight["fault_checks"])
                )
                observed = {
                    item["fault_id"]: item["operational_classification"]
                    for item in preflight["fault_checks"]
                }
                self.assertEqual(observed, expected)
                for fault in data["faults"]:
                    rule_type = fault["selection_rule"]["type"]
                    if fault["kind"] == "omission":
                        self.assertEqual(rule_type, "drop_once")
                    elif fault["class"] == "accidental":
                        self.assertEqual(rule_type, "fixed_offset")
                    else:
                        self.assertEqual(rule_type, "route_decoy")

    def test_base_map_has_coincident_start_and_goal(self) -> None:
        path = PROJECT_ROOT / "maps" / "evaluation" / "coincident_detours.json"
        data = json.loads(path.read_text())
        self.assertEqual(data["agents"]["P1"]["start"], data["agents"]["P2"]["start"])
        self.assertEqual(data["agents"]["P1"]["goal"], data["agents"]["P2"]["goal"])


if __name__ == "__main__":
    unittest.main()
