from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import generate_evaluation_maps as generator  # noqa: E402
from recovery_harness import run_scheduler_preflight, validate_map  # noqa: E402


EVALUATION_DIR = PROJECT_ROOT / "maps" / "evaluation"
DEVELOPMENT_DIR = PROJECT_ROOT / "maps" / "development"
EVALUATION_MANIFEST = PROJECT_ROOT / "configs" / "evaluation_batch.json"

# Phase 1 held-out maps whose LLM outcomes have been inspected; they are now
# development maps and are kept only as scheduler regression fixtures.
PHASE_ONE_CLASSIFICATIONS = {
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


def load_evaluation_maps() -> dict[str, dict]:
    return {
        path.name: json.loads(path.read_text())
        for path in sorted(EVALUATION_DIR.glob("*.json"))
    }


class EvaluationMapTest(unittest.TestCase):
    """Deterministic, model-free acceptance of the frozen held-out maps."""

    def setUp(self) -> None:
        self.maps = load_evaluation_maps()
        self.templates = {
            generator.map_filename(template): template
            for template in generator.TEMPLATES
        }

    def test_suite_has_five_basic_and_five_extended_maps(self) -> None:
        self.assertEqual(set(self.maps), set(self.templates))
        suites = [data["generator"]["suite"] for data in self.maps.values()]
        self.assertEqual(suites.count("basic"), 5)
        self.assertEqual(suites.count("extended"), 5)
        for data in self.maps.values():
            self.assertEqual(
                data["schema_version"],
                5 if data["generator"]["suite"] == "extended" else 4,
            )
        grids = ["\n".join(data["public_rows"]) for data in self.maps.values()]
        self.assertEqual(len(set(grids)), len(grids))

    def test_maps_share_the_fixed_interface(self) -> None:
        for name, data in self.maps.items():
            with self.subTest(map=name):
                self.assertTrue(validate_map(data)["passed"])
                self.assertLessEqual(data["dimensions"]["width"], 11)
                self.assertLessEqual(data["dimensions"]["height"], 9)
                self.assertEqual(data["observation_radius"], 2)
                self.assertEqual(data["message_fact_limit"], 2)
                # Stage ids reach the prompt, so they must not describe routes.
                self.assertEqual(
                    [phase["id"] for phase in data["phases"]],
                    [f"stage-{index + 1}" for index in range(len(data["phases"]))],
                )

    def test_each_map_has_relevant_faults_and_an_irrelevant_control(self) -> None:
        for name, data in self.maps.items():
            with self.subTest(map=name):
                roles = sorted(
                    "omission" if fault["kind"] == "omission" else fault["class"]
                    for fault in data["faults"]
                )
                self.assertEqual(roles, ["accidental", "adversarial", "omission"])
                for fault in data["faults"]:
                    rule = fault["selection_rule"]["type"]
                    if fault["kind"] == "omission":
                        self.assertEqual(rule, "drop_once")
                    elif fault["class"] == "accidental":
                        self.assertEqual(rule, "fixed_offset")
                    else:
                        self.assertEqual(rule, "route_decoy")

    def test_scheduler_preflight_matches_frozen_classifications(self) -> None:
        for name, data in self.maps.items():
            with self.subTest(map=name):
                preflight = run_scheduler_preflight(
                    data, planning_granularity="chunk", chunk_size=3
                )
                self.assertTrue(preflight["passed"])
                self.assertTrue(preflight["clean_success"])
                self.assertEqual(preflight["clean_trap_entries"], 0)
                observed = {
                    item["fault_id"]: item["operational_classification"]
                    for item in preflight["fault_checks"]
                }
                self.assertEqual(
                    observed, data["generator"]["expected_classifications"]
                )
                harmful = (
                    "outcome_relevant"
                    if data["generator"]["suite"] == "basic"
                    else "executed_with_harm"
                )
                for fault in data["faults"]:
                    expected = observed[fault["id"]]
                    if fault["kind"] == "omission":
                        self.assertEqual(expected, "non_propagating")
                    elif fault["class"] == "adversarial":
                        self.assertEqual(expected, harmful)
                    else:
                        self.assertIn(
                            expected,
                            self.templates[name].accidental_classes
                            if data["generator"]["suite"] == "extended"
                            else (harmful,),
                        )

    def test_structure_and_forked_restarts_pass_acceptance(self) -> None:
        for name, data in self.maps.items():
            with self.subTest(map=name):
                template = self.templates[name]
                self.assertIsNone(generator.structural_checks(template, data))
                self.assertIsNone(generator.restart_checks(data, data["faults"]))

    def test_after_harm_conditions_have_an_observable_harm_trigger(self) -> None:
        manifest = json.loads(EVALUATION_MANIFEST.read_text())
        preflights = {
            name: {
                item["fault_id"]: item
                for item in run_scheduler_preflight(
                    data, planning_granularity="chunk", chunk_size=3
                )["fault_checks"]
            }
            for name, data in self.maps.items()
        }
        for condition in manifest["runs"]:
            if condition.get("detection_trigger") != "after-fault-caused-harm":
                continue
            map_name = Path(condition["map"]).name
            with self.subTest(condition=condition["id"]):
                fault_check = preflights[map_name][condition["fault"]]
                self.assertTrue(
                    fault_check["fault_caused_harm"],
                    "after-harm conditions must use a fault that reaches the harm event",
                )

    def test_formal_restart_baseline_preserves_the_physical_environment(self) -> None:
        manifest = json.loads(EVALUATION_MANIFEST.read_text())
        default_scope = manifest.get("defaults", {}).get("recovery_scope")
        restart_conditions = [
            condition
            for condition in manifest["runs"]
            if condition["mode"] == "full-restart"
        ]
        self.assertTrue(restart_conditions)
        self.assertEqual(default_scope, "agent-only")
        self.assertTrue(
            all(
                condition.get("recovery_scope", default_scope) == "agent-only"
                for condition in restart_conditions
            )
        )

    def test_generator_reproduces_frozen_maps(self) -> None:
        for name, template in self.templates.items():
            with self.subTest(map=name):
                data, _ = generator.generate(template)
                self.assertIsNotNone(data)
                self.assertEqual(
                    generator.serialize(data), (EVALUATION_DIR / name).read_text()
                )


class PhaseOneMapRegressionTest(unittest.TestCase):
    def test_phase_one_maps_keep_their_scheduler_classifications(self) -> None:
        for filename, expected in PHASE_ONE_CLASSIFICATIONS.items():
            with self.subTest(map=filename):
                data = json.loads((DEVELOPMENT_DIR / filename).read_text())
                self.assertTrue(validate_map(data)["passed"])
                preflight = run_scheduler_preflight(
                    data, planning_granularity="chunk", chunk_size=3
                )
                self.assertTrue(preflight["passed"])
                observed = {
                    item["fault_id"]: item["operational_classification"]
                    for item in preflight["fault_checks"]
                }
                self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
