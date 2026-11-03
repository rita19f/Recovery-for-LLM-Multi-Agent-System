from __future__ import annotations

import http.client
import json
import io
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from recovery_harness import (  # noqa: E402
    assess_step_plan_output,
    AttemptTrace,
    detect_navigation_loop,
    DryRunBackend,
    EpisodeError,
    execute_segment,
    HTTP_USER_AGENT,
    initial_runtime_state,
    LLMBackend,
    LLMResponseParseError,
    MOVES,
    ScriptedBackend,
    inject_fault,
    planning_messages,
    parse_sse_chat_response,
    redact_secrets,
    decision_error,
    run_experiment,
    run_scheduler_preflight,
    validate_map,
    validate_message_output,
    validate_chunk_plan_output,
    validate_plan_output,
    validate_route_plan_output,
    visible_public_cells,
)


MAP_PATH = PROJECT_ROOT / "maps" / "development" / "shared_static_traps_small.json"
ASYMMETRIC_MAP_PATH = (
    PROJECT_ROOT / "maps" / "development" / "shared_asymmetric_crossing.json"
)
STATEFUL_MAP_PATH = (
    PROJECT_ROOT / "maps" / "development" / "stateful_weapon_crossing.json"
)
RELEVANT_FAULT = "f-water-lower-coordinate-shift"
IRRELEVANT_FAULT = "f-water-lower-irrelevant-omission"


class SimulatedLLMVerifierBackend(ScriptedBackend):
    """Scripted navigation with an LLM-shaped verifier call record."""

    def __init__(self, *, always_accept: bool = False) -> None:
        super().__init__()
        self.always_accept = always_accept

    def verify(
        self,
        *,
        sender: str,
        recipient: str,
        phase: str,
        message_facts: list[dict],
        trusted_source_facts: list[dict],
    ) -> dict:
        valid = self.always_accept or message_facts == trusted_source_facts
        self.call_records.append(
            {
                "backend": "llm",
                "kind": "verify",
                "agent": "verifier",
                "sender": sender,
                "recipient": recipient,
                "phase": phase,
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "total_tokens": 25,
                },
                "latency_ms": 2.0,
            }
        )
        return {
            "valid": valid,
            "reason": "simulated verifier verdict",
        }


class RecoveryHarnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data = json.loads(MAP_PATH.read_text())

    def test_map_validates(self) -> None:
        result = validate_map(self.data)
        self.assertTrue(result["passed"])
        self.assertEqual(result["world_model"], "shared")
        self.assertEqual(result["starts"], {"P1": [3, 6], "P2": [1, 5]})
        self.assertEqual(result["goals"], {"P1": [3, 0], "P2": [3, 1]})
        self.assertEqual(len(result["phases"]), 2)
        self.assertEqual([phase["steps"] for phase in result["phases"]], [6, 6])

    def test_shared_world_allows_coincident_starts_and_goals(self) -> None:
        variant = json.loads(json.dumps(self.data))
        variant["agents"]["P2"]["start"] = [3, 6]
        variant["agents"]["P2"]["goal"] = [3, 0]
        variant["prelude_actions"]["P2"] = ["UP"]
        variant["finale_actions"]["P2"] = ["UP"]
        result = validate_map(variant)
        self.assertTrue(result["passed"])
        self.assertEqual(result["starts"]["P1"], result["starts"]["P2"])
        self.assertEqual(result["goals"]["P1"], result["goals"]["P2"])

    def test_asymmetric_map_has_independent_phase_targets(self) -> None:
        data = json.loads(ASYMMETRIC_MAP_PATH.read_text())
        result = validate_map(data)
        self.assertTrue(result["passed"])
        self.assertEqual(result["starts"], {"P1": [4, 7], "P2": [6, 7]})
        self.assertEqual(result["goals"], {"P1": [2, 1], "P2": [8, 1]})
        self.assertEqual(result["phases"][0]["steps"], {"P1": 7, "P2": 7})
        self.assertEqual(
            result["phases"][0]["agents"]["P1"]["target"], [9, 5]
        )
        self.assertEqual(
            result["phases"][0]["agents"]["P2"]["target"], [1, 5]
        )

    def test_stateful_weapon_map_validates(self) -> None:
        data = json.loads(STATEFUL_MAP_PATH.read_text())
        result = validate_map(data)
        self.assertTrue(result["passed"])
        self.assertEqual(result["stateful_world"]["max_hp"], 3)
        self.assertEqual(
            result["stateful_world"]["weapon_ids"],
            ["fire-sword-west", "ice-sword-east"],
        )
        self.assertEqual(
            result["stateful_world"]["monster_ids"],
            ["fire-monster-west", "ice-monster-east"],
        )

    def test_stateful_objects_may_not_occupy_start(self) -> None:
        data = json.loads(STATEFUL_MAP_PATH.read_text())
        data["stateful_world"]["weapons"][0]["cell"] = data["agents"]["P1"]["start"]
        with self.assertRaisesRegex(Exception, "start/goal"):
            validate_map(data)

    def test_stateful_sensing_agent_cannot_pick_up_and_weapons_are_single_use(
        self,
    ) -> None:
        data = json.loads(STATEFUL_MAP_PATH.read_text())
        runtime = initial_runtime_state(data)
        trace = AttemptTrace("unit")
        step_state = {"P1": 0, "P2": 0}

        def step(agent: str, position: list[int], action: str) -> None:
            dx, dy = {"UP": (0, -1), "DOWN": (0, 1)}[action]
            execute_segment(
                data=data,
                agent=agent,
                position=tuple(position),
                actions=[action],
                target=(position[0] + dx, position[1] + dy),
                plan_id="unit-plan",
                trace=trace,
                step_state=step_state,
                runtime=runtime,
            )

        # P2 senses the ice sword, so stepping onto it does not collect it.
        step("P2", [6, 5], "DOWN")
        self.assertEqual(runtime["inventory"]["P2"], [])
        self.assertIn("ice-sword-east", runtime["available_weapons"])

        # P1 can collect it; the weapon then leaves the shared world.
        step("P1", [6, 5], "DOWN")
        self.assertEqual(runtime["inventory"]["P1"], ["ice_sword"])
        self.assertNotIn("ice-sword-east", runtime["available_weapons"])

        # Defeating the matching monster consumes the weapon without damage.
        step("P1", [7, 4], "UP")
        self.assertEqual(runtime["inventory"]["P1"], [])
        self.assertNotIn("ice-monster-east", runtime["alive_monsters"])
        self.assertEqual(runtime["health"]["P1"], 3)
        defeat = next(
            event for event in trace.events if event["event_type"] == "monster_defeated"
        )
        self.assertEqual(defeat["consumed_weapon"], "ice_sword")

    def test_stateful_clean_run_consumes_both_weapons_at_full_health(self) -> None:
        data = json.loads(STATEFUL_MAP_PATH.read_text())
        result = run_experiment(
            data,
            ScriptedBackend(),
            mode="clean",
            fault_id=None,
            planning_granularity="chunk",
            chunk_size=3,
        )
        self.assertTrue(result["full_health_success"])
        self.assertEqual(result["weapon_pickups"], 2)
        self.assertEqual(result["monsters_defeated"], 2)
        self.assertEqual(result["final_inventory"], {"P1": [], "P2": []})
        self.assertEqual(result["alive_monster_ids"], [])

    def test_stateful_relevant_faults_cause_damage_without_blocking_goal(self) -> None:
        data = json.loads(STATEFUL_MAP_PATH.read_text())
        for fault_id in ("f-ice-sword-east-offset", "a-ice-sword-monster-decoy"):
            with self.subTest(fault_id=fault_id):
                result = run_experiment(
                    data,
                    ScriptedBackend(),
                    mode="no-recovery",
                    fault_id=fault_id,
                    planning_granularity="chunk",
                    chunk_size=3,
                )
                metrics = result["fault_metrics"]
                self.assertTrue(result["final_success"])
                self.assertFalse(result["full_health_success"])
                self.assertEqual(result["irreversible_damage"], 1)
                self.assertEqual(result["final_health"], {"P1": 2, "P2": 3})
                self.assertTrue(metrics["fault_structured_route_relevant_plan"])
                self.assertTrue(metrics["fault_caused_harm"])
                self.assertFalse(metrics["fault_caused_failure"])

    def test_stateful_irrelevant_omission_causes_no_harm(self) -> None:
        data = json.loads(STATEFUL_MAP_PATH.read_text())
        result = run_experiment(
            data,
            ScriptedBackend(),
            mode="no-recovery",
            fault_id="f-fire-sword-west-drop-once",
            planning_granularity="chunk",
            chunk_size=3,
        )
        metrics = result["fault_metrics"]
        self.assertTrue(metrics["fault_delivered_to_recipient"])
        self.assertTrue(result["full_health_success"])
        self.assertFalse(metrics["fault_structured_route_relevant_executed"])
        self.assertFalse(metrics["fault_caused_harm"])

    def test_stateful_blocked_fault_is_not_labeled_as_executed(self) -> None:
        data = json.loads(STATEFUL_MAP_PATH.read_text())
        result = run_experiment(
            data,
            ScriptedBackend(),
            mode="always-verify",
            fault_id="f-ice-sword-east-offset",
            planning_granularity="chunk",
            chunk_size=3,
        )
        metrics = result["fault_metrics"]
        self.assertFalse(metrics["fault_delivered_to_recipient"])
        self.assertFalse(metrics["fault_structured_route_relevant_executed"])
        self.assertFalse(metrics["fault_caused_harm"])
        self.assertTrue(result["full_health_success"])

    def test_stateful_full_restart_keeps_pre_detection_damage(self) -> None:
        data = json.loads(STATEFUL_MAP_PATH.read_text())
        expectations = {
            0: {
                "pre_damage": 0,
                "full_health": True,
                "health": {"P1": 3, "P2": 3},
                "available_weapons": ["fire-sword-west", "ice-sword-east"],
                "alive_monsters": ["fire-monster-west", "ice-monster-east"],
            },
            9: {
                "pre_damage": 1,
                "full_health": False,
                "health": {"P1": 2, "P2": 3},
                "available_weapons": ["ice-sword-east"],
                "alive_monsters": ["ice-monster-east"],
            },
        }
        for delay, expected in expectations.items():
            with self.subTest(detection_delay_ticks=delay):
                result = run_experiment(
                    data,
                    ScriptedBackend(),
                    mode="full-restart",
                    fault_id="f-ice-sword-east-offset",
                    planning_granularity="chunk",
                    chunk_size=3,
                    detection_delay_ticks=delay,
                    recovery_scope="agent-only",
                )
                restart = result["paired_baselines"]["full_restart_by_scope"][
                    "agent-only"
                ]
                resumed = result["attempts"][1]
                self.assertTrue(restart["success"])
                self.assertEqual(
                    result["recovery_scope"], "agent-only"
                )
                self.assertTrue(resumed["resumed_from_physical_snapshot"])
                self.assertEqual(
                    resumed["initial_physical_state"],
                    result["detection"]["physical_state_snapshot"],
                )
                self.assertEqual(
                    restart["pre_detection_damage"], expected["pre_damage"]
                )
                self.assertEqual(
                    restart["full_health_success"], expected["full_health"]
                )
                self.assertEqual(
                    result["full_health_success"], expected["full_health"]
                )
                self.assertEqual(
                    result["irreversible_damage"], expected["pre_damage"]
                )
                self.assertEqual(
                    resumed["initial_physical_state"]["health"],
                    expected["health"],
                )
                self.assertEqual(resumed["final_health"], expected["health"])
                self.assertEqual(
                    resumed["initial_physical_state"]["available_weapon_ids"],
                    expected["available_weapons"],
                )
                self.assertEqual(
                    resumed["initial_physical_state"]["alive_monster_ids"],
                    expected["alive_monsters"],
                )

    def test_recovery_scope_is_a_matched_experimental_factor(self) -> None:
        data = json.loads(STATEFUL_MAP_PATH.read_text())
        result = run_experiment(
            data,
            ScriptedBackend(),
            mode="full-restart",
            fault_id="f-ice-sword-east-offset",
            planning_granularity="chunk",
            chunk_size=3,
            detection_delay_ticks=9,
            recovery_scope="both",
        )

        # Both restarts fork from the single shared faulty prefix.
        self.assertEqual(result["recovery_scopes"], ["environment-reset", "agent-only"])
        self.assertEqual(
            [attempt["attempt"] for attempt in result["attempts"]],
            [
                "attempt-1",
                "attempt-2-environment-reset",
                "attempt-2-agent-only",
            ],
        )
        self.assertEqual(
            [attempt["recovery_scope"] for attempt in result["attempts"]],
            [None, "environment-reset", "agent-only"],
        )
        # A multi-scope run has no single selected outcome.
        self.assertFalse(result["scope_outcome_available"])
        self.assertIsNone(result["final_success"])
        self.assertIsNone(result["full_health_success"])
        self.assertIsNone(result["outcome_path"])

        by_scope = result["paired_baselines"]["full_restart_by_scope"]
        environment_reset = by_scope["environment-reset"]
        agent_only = by_scope["agent-only"]
        self.assertTrue(environment_reset["success"])
        self.assertTrue(agent_only["success"])
        self.assertTrue(environment_reset["full_health_success"])
        self.assertFalse(agent_only["full_health_success"])
        self.assertEqual(environment_reset["damage_taken"], 0)
        self.assertEqual(agent_only["damage_taken"], 1)
        self.assertEqual(environment_reset["final_health"], {"P1": 3, "P2": 3})
        self.assertEqual(agent_only["final_health"], {"P1": 2, "P2": 3})
        self.assertTrue(environment_reset["recovery_improved_safe_outcome"])
        self.assertFalse(agent_only["recovery_improved_safe_outcome"])
        restarts = {attempt["recovery_scope"]: attempt for attempt in result["attempts"][1:]}
        self.assertFalse(
            restarts["environment-reset"]["resumed_from_physical_snapshot"]
        )
        self.assertTrue(restarts["agent-only"]["resumed_from_physical_snapshot"])
        self.assertEqual(
            restarts["agent-only"]["initial_physical_state"],
            result["detection"]["physical_state_snapshot"],
        )

        # Prefix actions are redone after an environment reset but their
        # physical effects are kept by an agent-only restart.
        prefix_actions = result["detection"]["executed_action_count"]
        self.assertGreater(prefix_actions, 0)
        for scope, wasted_prefix in (
            ("environment-reset", True),
            ("agent-only", False),
        ):
            restart = by_scope[scope]
            self.assertEqual(
                restart["wasted_executed_actions"],
                prefix_actions if wasted_prefix else 0,
            )
            self.assertEqual(
                restart["retained_executed_actions"],
                0 if wasted_prefix else prefix_actions,
            )
            self.assertEqual(
                restart["executed_actions"],
                prefix_actions + restarts[scope]["executed_actions"],
            )
            self.assertEqual(
                restart["wasted_backend_calls"],
                result["detection"]["backend_call_count"],
            )
        # Each scope is charged the shared prefix plus only its own restart.
        self.assertEqual(
            environment_reset["backend_calls"] + agent_only["backend_calls"],
            2 * result["detection"]["backend_call_count"]
            + (
                result["backend_calls"]
                - result["paired_baselines"]["no_recovery"]["backend_calls"]
            ),
        )

        single = run_experiment(
            data,
            ScriptedBackend(),
            mode="full-restart",
            fault_id="f-ice-sword-east-offset",
            planning_granularity="chunk",
            chunk_size=3,
            detection_delay_ticks=9,
            recovery_scope="agent-only",
        )
        self.assertTrue(single["scope_outcome_available"])
        self.assertEqual(
            single["full_health_success"],
            single["paired_baselines"]["full_restart_by_scope"]["agent-only"][
                "full_health_success"
            ],
        )
        self.assertEqual(single["irreversible_damage"], 1)

    def test_unobserved_event_trigger_falls_back_to_outcome(self) -> None:
        data = json.loads(STATEFUL_MAP_PATH.read_text())
        result = run_experiment(
            data,
            ScriptedBackend(),
            mode="full-restart",
            fault_id="f-fire-sword-west-drop-once",
            planning_granularity="chunk",
            chunk_size=3,
            detection_trigger="after-fault-caused-harm",
            recovery_scope="agent-only",
        )

        self.assertFalse(result["fault_metrics"]["fault_caused_harm"])
        self.assertTrue(result["recovery_triggered"])
        self.assertEqual(
            result["detection"]["reason"],
            "trigger_not_observed_before_outcome:after-fault-caused-harm",
        )
        self.assertFalse(result["detection"]["trigger_observed"])

    def test_event_trigger_snapshots_immediately_after_fault_caused_harm(
        self,
    ) -> None:
        data = json.loads(STATEFUL_MAP_PATH.read_text())
        result = run_experiment(
            data,
            ScriptedBackend(),
            mode="full-restart",
            fault_id="a-ice-sword-monster-decoy",
            planning_granularity="chunk",
            chunk_size=3,
            detection_trigger="after-fault-caused-harm",
            recovery_scope="agent-only",
        )

        self.assertEqual(result["detection_trigger"], "after-fault-caused-harm")
        self.assertEqual(result["detection"]["reason"], "after_fault_caused_harm")
        self.assertTrue(result["detection"]["trigger_observed"])
        self.assertEqual(result["detection"]["damage_taken"], 1)
        self.assertEqual(
            result["detection"]["physical_state_snapshot"]["health"],
            {"P1": 2, "P2": 3},
        )
        self.assertTrue(result["recovery_completed_successfully"])
        self.assertFalse(result["recovery_improved_goal_outcome"])
        self.assertFalse(result["recovery_improved_safe_outcome"])
        self.assertFalse(result["recovered_success"])
        self.assertFalse(result["full_health_success"])
        self.assertEqual(result["outcome_path"]["damage_taken"], 1)
        self.assertGreater(
            result["actual_harness_work"]["executed_actions"],
            result["outcome_path"]["executed_actions"],
        )

    def test_environment_reset_improves_safe_outcome_after_harm(self) -> None:
        data = json.loads(STATEFUL_MAP_PATH.read_text())
        result = run_experiment(
            data,
            ScriptedBackend(),
            mode="full-restart",
            fault_id="a-ice-sword-monster-decoy",
            planning_granularity="chunk",
            chunk_size=3,
            detection_trigger="after-fault-caused-harm",
            recovery_scope="environment-reset",
        )

        self.assertFalse(result["initial_full_health_success"])
        self.assertTrue(result["final_success"])
        self.assertTrue(result["full_health_success"])
        self.assertTrue(result["recovery_completed_successfully"])
        self.assertFalse(result["recovery_improved_goal_outcome"])
        self.assertTrue(result["recovery_improved_safe_outcome"])
        self.assertIsNone(result["clean_episode_success"])

    def test_stateful_planner_prompt_hides_true_cell_and_states_hp_objective(
        self,
    ) -> None:
        data = json.loads(STATEFUL_MAP_PATH.read_text())
        backend = DryRunBackend()
        run_experiment(
            data,
            backend,
            mode="no-recovery",
            fault_id="a-ice-sword-monster-decoy",
            planning_granularity="chunk",
            chunk_size=3,
        )
        prompts = [
            "".join(message["content"] for message in record["prompt"])
            for record in backend.call_records
            if record.get("agent") == "P1"
            and record.get("kind") == "chunk_plan"
            and "prompt" in record
        ]
        self.assertTrue(prompts)
        for prompt in prompts:
            # The true ice-sword cell [6,6] must not reach the misled planner.
            self.assertNotIn("6:6", prompt)
            self.assertNotIn("semantic_fact_id", prompt)
            self.assertIn("permanent safety violation", prompt)

    def test_hazard_map_uses_common_safe_status_prompt_without_source_id_leak(
        self,
    ) -> None:
        data = json.loads(ASYMMETRIC_MAP_PATH.read_text())
        backend = DryRunBackend()
        run_experiment(
            data,
            backend,
            mode="no-recovery",
            fault_id=RELEVANT_FAULT,
            planning_granularity="chunk",
            chunk_size=3,
        )
        prompts = [
            "".join(message["content"] for message in record["prompt"])
            for record in backend.call_records
            if record.get("kind") == "chunk_plan" and "prompt" in record
        ]
        self.assertTrue(prompts)
        self.assertFalse(any("semantic_fact_id" in prompt for prompt in prompts))
        self.assertTrue(
            all("permanent safety violation" in prompt for prompt in prompts)
        )
        self.assertTrue(all('"hp": 3' in prompt for prompt in prompts))
        self.assertTrue(
            all('"shared_objects_enabled": false' in prompt for prompt in prompts)
        )

    def test_basic_and_extended_maps_share_the_same_system_prompt(self) -> None:
        def first_chunk_system_prompt(data: dict) -> str:
            backend = DryRunBackend()
            run_experiment(
                data,
                backend,
                mode="clean",
                fault_id=None,
                planning_granularity="chunk",
                chunk_size=3,
            )
            record = next(
                item
                for item in backend.call_records
                if item.get("kind") == "chunk_plan" and "prompt" in item
            )
            return record["prompt"][0]["content"]

        basic = json.loads(MAP_PATH.read_text())
        extended = json.loads(STATEFUL_MAP_PATH.read_text())
        self.assertEqual(
            first_chunk_system_prompt(basic),
            first_chunk_system_prompt(extended),
        )

    def test_stateful_missing_weapon_at_visited_cell_does_not_livelock(self) -> None:
        data = json.loads(STATEFUL_MAP_PATH.read_text())
        fault = next(
            item for item in data["faults"] if item["id"] == "f-ice-sword-east-offset"
        )
        # [6,5] is on P1's route, so the believed pickup is visited and absent.
        fault["selection_rule"]["delta"] = [0, -1]
        result = run_experiment(
            data,
            ScriptedBackend(),
            mode="no-recovery",
            fault_id="f-ice-sword-east-offset",
            planning_granularity="chunk",
            chunk_size=3,
        )
        self.assertTrue(result["final_success"])
        self.assertEqual(result["final_health"], {"P1": 2, "P2": 3})

    def test_asymmetric_chunk_smoke_and_relevant_fault(self) -> None:
        data = json.loads(ASYMMETRIC_MAP_PATH.read_text())
        clean = run_experiment(
            data,
            ScriptedBackend(),
            mode="clean",
            fault_id=None,
            planning_granularity="chunk",
            chunk_size=3,
        )
        faulty = run_experiment(
            data,
            ScriptedBackend(),
            mode="no-recovery",
            fault_id="f-water-lower-coordinate-shift",
            planning_granularity="chunk",
            chunk_size=3,
        )
        self.assertTrue(clean["success"])
        self.assertEqual(clean["attempts"][0]["steps"], {"P1": 18, "P2": 18})
        self.assertFalse(faulty["success"])
        self.assertTrue(faulty["attempts"][0]["fault_consumed"])

    def test_scheduler_preflight_classifies_registered_faults(self) -> None:
        data = json.loads(ASYMMETRIC_MAP_PATH.read_text())
        preflight = run_scheduler_preflight(
            data, planning_granularity="chunk", chunk_size=3
        )
        self.assertTrue(preflight["passed"])
        self.assertTrue(preflight["clean_success"])
        checks = {item["fault_id"]: item for item in preflight["fault_checks"]}
        self.assertEqual(
            checks["f-water-lower-coordinate-shift"]["operational_classification"],
            "outcome_relevant",
        )
        self.assertEqual(
            checks["f-water-lower-irrelevant-omission"]["operational_classification"],
            "non_propagating",
        )

    def test_dynamic_sensor_refresh_exposes_new_water_before_next_chunk(self) -> None:
        data = json.loads(ASYMMETRIC_MAP_PATH.read_text())
        result = run_experiment(
            data,
            DryRunBackend(),
            mode="clean",
            fault_id=None,
            planning_granularity="chunk",
            chunk_size=3,
        )
        self.assertTrue(result["success"])
        p2_followup = next(
            record
            for record in result["call_records"]
            if record.get("kind") == "chunk_plan"
            and record.get("agent") == "P2"
            and json.loads(record["prompt"][1]["content"])["current_cell"] == [4, 6]
        )
        payload = json.loads(p2_followup["prompt"][1]["content"])
        self.assertIn([3, 4], payload["known_forbidden_cells"])

    def test_newly_visible_hazard_interrupts_remaining_chunk(self) -> None:
        data = json.loads(ASYMMETRIC_MAP_PATH.read_text())
        # P1 can publicly see [3,5] but cannot sense water. P2 initially cannot
        # see it; after P2's first move, the new report must interrupt P1's
        # already queued chunk before P1 enters the trap.
        data["hazards"]["water"].append([3, 5])

        class InitiallyUnsafeChunkBackend(ScriptedBackend):
            def __init__(self) -> None:
                super().__init__()
                self.forced = False

            def plan(self, **kwargs):
                if (
                    not self.forced
                    and kwargs["planning_granularity"] == "chunk"
                    and kwargs["agent"] == "P1"
                    and kwargs["phase"]["id"] == "lower-divergence"
                    and kwargs["position"] == (4, 7)
                ):
                    self.forced = True
                    self.call_records.append(
                        {
                            "backend": "scripted",
                            "kind": "chunk_plan",
                            "agent": "P1",
                            "phase": "lower-divergence",
                        }
                    )
                    return {
                        "actions": ["LEFT", "UP", "UP"],
                        "waypoint": [3, 5],
                        "declared_dependency_ids": [],
                        "reason": "water is not visible to P1's fire sensor",
                    }
                return super().plan(**kwargs)

        result = run_experiment(
            data,
            InitiallyUnsafeChunkBackend(),
            mode="clean",
            fault_id=None,
            planning_granularity="chunk",
            chunk_size=3,
        )
        self.assertTrue(result["success"])
        events = result["attempts"][0]["events"]
        interruptions = [
            event for event in events if event["event_type"] == "plan_interrupted"
        ]
        self.assertEqual(len(interruptions), 1)
        self.assertEqual(interruptions[0]["agent"], "P1")
        self.assertIn([3, 5], interruptions[0]["hazard_cells"])
        self.assertFalse(
            any(event["event_type"] == "trap_triggered" for event in events)
        )

    def test_fact_ids_are_stable_and_reobservations_do_not_duplicate_facts(self) -> None:
        data = json.loads(ASYMMETRIC_MAP_PATH.read_text())
        result = run_experiment(
            data,
            ScriptedBackend(),
            mode="clean",
            fault_id=None,
            planning_granularity="chunk",
            chunk_size=3,
        )
        artifacts = result["attempts"][0]["artifacts"]
        sensor_facts = [
            artifact
            for artifact in artifacts
            if artifact.get("artifact_type") == "sensor_fact"
        ]
        fact_ids = [fact["artifact_id"] for fact in sensor_facts]
        self.assertEqual(len(fact_ids), len(set(fact_ids)))
        self.assertTrue(all(":lower-" not in fact_id for fact_id in fact_ids))
        self.assertTrue(all(":upper-" not in fact_id for fact_id in fact_ids))
        self.assertTrue(
            any(
                event["event_type"] == "sensor_fact_reobserved"
                for event in result["attempts"][0]["events"]
            )
        )

    def test_pending_queue_drains_across_message_limit(self) -> None:
        data = json.loads(ASYMMETRIC_MAP_PATH.read_text())
        data["hazards"]["water"].extend([[5, 6], [7, 6]])
        data["message_fact_limit"] = 1
        result = run_experiment(
            data,
            ScriptedBackend(),
            mode="clean",
            fault_id=None,
            planning_granularity="chunk",
            chunk_size=3,
        )
        self.assertTrue(result["success"])
        p2_messages = [
            artifact
            for artifact in result["attempts"][0]["artifacts"]
            if artifact.get("artifact_type") == "message"
            and artifact.get("agent") == "P2"
        ]
        delivered_source_ids = {
            source_id
            for message in p2_messages
            for source_id in message["source_ids"]
        }
        expected = {
            "attempt-1:fact:P2:water:5:6",
            "attempt-1:fact:P2:water:6:5",
            "attempt-1:fact:P2:water:7:6",
        }
        self.assertTrue(expected.issubset(delivered_source_ids))
        self.assertTrue(all(len(message["source_ids"]) <= 1 for message in p2_messages))
        self.assertEqual(
            result["attempts"][0]["communication_state"]["P2"]["pending"], 0
        )

    def test_scripted_clean_succeeds(self) -> None:
        result = run_experiment(
            self.data, ScriptedBackend(), mode="clean", fault_id=None
        )
        self.assertTrue(result["success"])
        self.assertTrue(result["initial_success"])
        self.assertTrue(result["full_health_success"])
        self.assertEqual(result["llm_calls"], 0)
        self.assertEqual(
            set(result["latency_ms"]),
            {
                "end_to_end",
                "initial_execution",
                "recovery",
                "llm",
                "dependency_tracking",
                "verification",
            },
        )
        self.assertGreaterEqual(result["latency_ms"]["end_to_end"], 0)
        self.assertGreaterEqual(result["latency_ms"]["initial_execution"], 0)
        self.assertIsNone(result["latency_ms"]["recovery"])
        self.assertEqual(result["latency_ms"]["llm"], 0)
        self.assertEqual(result["latency_ms"]["verification"], 0)
        self.assertGreaterEqual(result["latency_ms"]["dependency_tracking"], 0)
        self.assertIn("latency_ms", result["attempts"][0])

    def test_relevant_fault_propagates_without_recovery(self) -> None:
        result = run_experiment(
            self.data,
            ScriptedBackend(),
            mode="no-recovery",
            fault_id=RELEVANT_FAULT,
        )
        self.assertFalse(result["success"])
        attempt = result["attempts"][0]
        self.assertTrue(attempt["fault_injected"])
        self.assertTrue(attempt["fault_consumed"])
        self.assertEqual(attempt["failure"]["agent"], "P1")
        self.assertIn("water trap", attempt["failure"]["error"])

    def test_full_restart_recovers_relevant_fault(self) -> None:
        result = run_experiment(
            self.data,
            ScriptedBackend(),
            mode="full-restart",
            fault_id=RELEVANT_FAULT,
        )
        self.assertFalse(result["initial_success"])
        self.assertTrue(result["recovery_triggered"])
        self.assertTrue(result["success"])
        self.assertTrue(result["final_success"])
        self.assertTrue(result["recovered_success"])
        self.assertIsNone(result["clean_episode_success"])
        self.assertTrue(result["ever_failed"])
        self.assertEqual(result["attempt_count"], 2)
        self.assertGreaterEqual(result["trap_entries"], 1)
        self.assertEqual(len(result["attempts"]), 2)
        self.assertIsNotNone(result["latency_ms"]["recovery"])
        self.assertGreaterEqual(result["latency_ms"]["recovery"], 0)

    def test_basic_map_agent_only_restart_cannot_revive_terminal_failure(
        self,
    ) -> None:
        result = run_experiment(
            self.data,
            ScriptedBackend(),
            mode="full-restart",
            fault_id=RELEVANT_FAULT,
            planning_granularity="chunk",
            chunk_size=3,
            recovery_scope="agent-only",
        )
        self.assertFalse(result["initial_success"])
        self.assertFalse(result["final_success"])
        self.assertFalse(result["full_health_success"])
        self.assertEqual(result["final_health"], {"P1": 0, "P2": 3})
        self.assertEqual(result["irreversible_damage"], 3)
        self.assertEqual(
            result["attempts"][1]["failure"]["phase"], "recovery_start"
        )

    def test_full_restart_uses_same_faulty_attempt_for_paired_baseline(self) -> None:
        result = run_experiment(
            self.data,
            ScriptedBackend(),
            mode="full-restart",
            fault_id=RELEVANT_FAULT,
            detection_delay_ticks=0,
        )
        paired = result["paired_baselines"]
        restart = paired["full_restart_by_scope"]["environment-reset"]
        self.assertEqual(paired["shared_faulty_trajectory"], "attempt-1")
        self.assertEqual(
            paired["no_recovery"]["success"], result["attempts"][0]["success"]
        )
        self.assertLess(
            paired["detection"]["executed_action_count"],
            sum(result["attempts"][0]["steps"].values()),
        )
        self.assertEqual(paired["detection"]["safety_violations"], 0)
        self.assertTrue(restart["success"])
        self.assertEqual(restart["trap_entries"], 0)
        self.assertGreater(restart["diagnostic_suffix_excluded"], 0)
        self.assertEqual(
            result["fault_injection_phase"], result["fault_declared_phase"]
        )
        self.assertEqual(result["detection"]["observed_delay_ticks"], 0)
        self.assertGreaterEqual(
            restart["wasted_executed_actions"], 1
        )
        self.assertGreaterEqual(restart["wasted_backend_calls"], 1)
        self.assertIn("total_tokens", restart["wasted_usage"])

    def test_fault_metrics_separate_exposure_plan_action_and_harm(self) -> None:
        result = run_experiment(
            self.data,
            ScriptedBackend(),
            mode="no-recovery",
            fault_id=RELEVANT_FAULT,
        )
        metrics = result["fault_metrics"]
        self.assertTrue(metrics["fault_delivered_to_recipient"])
        self.assertTrue(metrics["fault_exposed_to_planner"])
        self.assertTrue(metrics["fault_structured_route_relevant_plan"])
        self.assertTrue(metrics["fault_structured_route_relevant_executed"])
        self.assertTrue(metrics["fault_caused_failure"])
        self.assertGreaterEqual(result["dependency_quality"]["declared"]["tp"], 1)

    def test_always_verify_rejects_and_regenerates_faulty_message(self) -> None:
        result = run_experiment(
            self.data,
            ScriptedBackend(),
            mode="always-verify",
            fault_id=RELEVANT_FAULT,
            detection_delay_ticks=0,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["trap_entries"], 0)
        self.assertGreater(result["verification_checks"], 0)
        self.assertEqual(result["verification_rejections"], 1)
        self.assertEqual(result["verification_confusion"]["tp"], 1)
        self.assertEqual(result["verification_llm_calls"], 0)
        self.assertFalse(result["fault_metrics"]["fault_delivered_to_recipient"])
        self.assertTrue(
            any(
                event["event_type"] == "message_regenerated_after_verification"
                for event in result["attempts"][0]["events"]
            )
        )

    def test_llm_verify_can_reject_and_regenerate_faulty_message(self) -> None:
        result = run_experiment(
            self.data,
            SimulatedLLMVerifierBackend(),
            mode="llm-verify",
            fault_id=RELEVANT_FAULT,
            detection_delay_ticks=0,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["trap_entries"], 0)
        self.assertEqual(result["verification_rejections"], 1)
        self.assertEqual(result["verification_confusion"]["tp"], 1)
        self.assertGreater(result["verification_confusion"]["tn"], 0)
        self.assertEqual(
            result["verification_llm_calls"], result["verification_checks"]
        )
        self.assertEqual(
            result["verification_usage"]["total_tokens"],
            25 * result["verification_llm_calls"],
        )
        self.assertFalse(result["fault_metrics"]["fault_delivered_to_recipient"])

    def test_llm_verify_miss_allows_fault_to_propagate(self) -> None:
        result = run_experiment(
            self.data,
            SimulatedLLMVerifierBackend(always_accept=True),
            mode="llm-verify",
            fault_id=RELEVANT_FAULT,
            detection_delay_ticks=0,
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["verification_rejections"], 0)
        self.assertEqual(result["verification_confusion"]["fn"], 1)
        self.assertTrue(result["fault_metrics"]["fault_delivered_to_recipient"])
        self.assertTrue(result["fault_metrics"]["fault_caused_failure"])

    def test_asymmetric_map_includes_structured_adversarial_fault(self) -> None:
        data = json.loads(ASYMMETRIC_MAP_PATH.read_text())
        adversarial = [
            fault for fault in data["faults"] if fault.get("class") == "adversarial"
        ]
        self.assertEqual(len(adversarial), 1)
        self.assertEqual(adversarial[0]["kind"], "coordinate_shift")
        self.assertNotIn("advice", adversarial[0])

    def test_irrelevant_omission_does_not_break_saved_plan(self) -> None:
        result = run_experiment(
            self.data,
            ScriptedBackend(),
            mode="no-recovery",
            fault_id=IRRELEVANT_FAULT,
        )
        attempt = result["attempts"][0]
        self.assertTrue(attempt["fault_injected"])
        self.assertTrue(result["success"])
        self.assertEqual(
            attempt["fault_injection_phase"], attempt["fault_declared_phase"]
        )
        omitted_fact_id = next(
            artifact["artifact_id"]
            for artifact in attempt["artifacts"]
            if artifact.get("artifact_type") == "sensor_fact"
            and artifact.get("cell") == [5, 1]
        )
        transmissions = [
            artifact
            for artifact in attempt["artifacts"]
            if artifact.get("artifact_type") == "message"
            and omitted_fact_id in artifact.get("source_ids", [])
        ]
        self.assertGreaterEqual(len(transmissions), 2)
        self.assertEqual(transmissions[0]["content"]["message_facts"], [])
        self.assertEqual(
            transmissions[1]["content"]["message_facts"][0]["source_fact_id"],
            omitted_fact_id,
        )
        self.assertEqual(attempt["communication_state"]["P2"]["pending"], 0)

    def test_step_dry_run_records_every_decision(self) -> None:
        backend = DryRunBackend()
        result = run_experiment(
            self.data,
            backend,
            mode="clean",
            fault_id=None,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["planning_granularity"], "step")
        self.assertEqual(result["backend_calls"], 27)
        self.assertTrue(all("prompt" in record for record in result["call_records"]))
        plans = [
            artifact
            for artifact in result["attempts"][0]["artifacts"]
            if artifact.get("artifact_type") == "plan"
        ]
        self.assertEqual(len(plans), 24)
        self.assertTrue(all("next_cell" in plan for plan in plans))
        self.assertTrue(all(plan["decision_status"] == "accepted" for plan in plans))
        self.assertTrue(all(plan["planning_granularity"] == "step" for plan in plans))

    def test_step_invalid_coordinate_is_feedback_not_episode_error(self) -> None:
        class OneBadStepBackend(ScriptedBackend):
            def __init__(self) -> None:
                super().__init__()
                self.sent_bad_step = False
                self.feedbacks = []

            def plan(self, **kwargs):
                self.feedbacks.append(kwargs.get("last_action_feedback"))
                if (
                    not self.sent_bad_step
                    and kwargs["planning_granularity"] == "step"
                    and kwargs["agent"] == "P1"
                    and kwargs["phase"]["id"] == "lower-junction"
                ):
                    self.sent_bad_step = True
                    self.call_records.append(
                        {
                            "backend": "scripted",
                            "kind": "plan",
                            "agent": kwargs["agent"],
                            "phase": kwargs["phase"]["id"],
                        }
                    )
                    return {
                        "next_cell": [3, 3],
                        "declared_dependency_ids": [],
                        "reason": "deliberate non-adjacent jump",
                    }
                return super().plan(**kwargs)

        backend = OneBadStepBackend()
        result = run_experiment(
            self.data,
            backend,
            mode="clean",
            fault_id=None,
            planning_granularity="step",
        )
        self.assertTrue(result["success"])
        noop_events = [
            event
            for event in result["attempts"][0]["events"]
            if event["event_type"] == "action_noop"
        ]
        self.assertEqual(len(noop_events), 1)
        self.assertEqual(noop_events[0]["issue_code"], "non_adjacent_or_blocked")
        self.assertTrue(
            any(
                feedback
                and feedback.get("issue_code") == "non_adjacent_or_blocked"
                for feedback in backend.feedbacks
            )
        )

    def test_step_loop_is_classified_without_controller_rerouting(self) -> None:
        class LoopingBackend(ScriptedBackend):
            def plan(self, **kwargs):
                if (
                    kwargs["planning_granularity"] == "step"
                    and kwargs["agent"] == "P1"
                    and kwargs["phase"]["id"] == "lower-junction"
                    and kwargs["position"] in {(3, 5), (2, 5)}
                ):
                    destination = (
                        [2, 5] if kwargs["position"] == (3, 5) else [3, 5]
                    )
                    self.call_records.append(
                        {
                            "backend": "scripted",
                            "kind": "plan",
                            "agent": "P1",
                            "phase": "lower-junction",
                        }
                    )
                    return {
                        "next_cell": destination,
                        "declared_dependency_ids": [],
                        "reason": "deliberate two-cell oscillation",
                    }
                return super().plan(**kwargs)

        result = run_experiment(
            self.data,
            LoopingBackend(),
            mode="clean",
            fault_id=None,
            planning_granularity="step",
        )
        self.assertFalse(result["success"])
        self.assertIn("navigation loop detected", result["attempts"][0]["failure"]["error"])
        events = result["attempts"][0]["events"]
        loop_events = [
            event for event in events if event["event_type"] == "navigation_loop_detected"
        ]
        self.assertEqual(len(loop_events), 1)
        self.assertEqual(loop_events[0]["period"], 2)
        self.assertEqual(loop_events[0]["cycle_cells"], [[3, 5], [2, 5]])

    def test_known_hazard_choice_executes_and_triggers_trap(self) -> None:
        class HazardIgnoringBackend(ScriptedBackend):
            forced_path = {(3, 5): [2, 5], (2, 5): [1, 5], (1, 5): [1, 4]}

            def plan(self, **kwargs):
                if (
                    kwargs["planning_granularity"] == "step"
                    and kwargs["agent"] == "P1"
                    and kwargs["phase"]["id"] == "lower-junction"
                    and kwargs["position"] in self.forced_path
                ):
                    dependency_ids = sorted(
                        fact["dependency_id"] for fact in kwargs["available_facts"]
                    )
                    self.call_records.append(
                        {
                            "backend": "scripted",
                            "kind": "plan",
                            "agent": "P1",
                            "phase": "lower-junction",
                        }
                    )
                    return {
                        "next_cell": self.forced_path[kwargs["position"]],
                        "declared_dependency_ids": dependency_ids,
                        "reason": "deliberately ignore the reported water trap",
                    }
                return super().plan(**kwargs)

        result = run_experiment(
            self.data,
            HazardIgnoringBackend(),
            mode="clean",
            fault_id=None,
            planning_granularity="step",
        )
        self.assertFalse(result["success"])
        self.assertIn("water trap", result["attempts"][0]["failure"]["error"])
        event_types = {
            event["event_type"] for event in result["attempts"][0]["events"]
        }
        self.assertIn("known_hazard_selected", event_types)
        self.assertIn("trap_triggered", event_types)

    def test_route_dry_run_remains_available_and_compiles_primitive_actions(self) -> None:
        backend = DryRunBackend()
        result = run_experiment(
            self.data,
            backend,
            mode="clean",
            fault_id=None,
            planning_granularity="route",
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["planning_granularity"], "route")
        self.assertEqual(result["backend_calls"], 7)
        plans = [
            artifact
            for artifact in result["attempts"][0]["artifacts"]
            if artifact.get("artifact_type") == "plan"
        ]
        self.assertEqual(len(plans), 4)
        self.assertTrue(all(plan["planning_granularity"] == "route" for plan in plans))
        self.assertTrue(all(plan["compiled_actions"] for plan in plans))
        self.assertTrue(
            all(plan["compiled_cells"][-1] == self.data["phases"][index // 2]["target"]
                for index, plan in enumerate(plans))
        )

    def test_chunk_dry_run_uses_direct_action_sequences(self) -> None:
        backend = DryRunBackend()
        result = run_experiment(
            self.data,
            backend,
            mode="clean",
            fault_id=None,
            planning_granularity="chunk",
            chunk_size=3,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["planning_granularity"], "chunk")
        self.assertEqual(result["chunk_size"], 3)
        self.assertEqual(result["backend_calls"], 11)
        plans = [
            artifact
            for artifact in result["attempts"][0]["artifacts"]
            if artifact.get("artifact_type") == "plan"
        ]
        self.assertEqual(len(plans), 8)
        self.assertTrue(all(plan["planning_granularity"] == "chunk" for plan in plans))
        self.assertTrue(all(1 <= len(plan["direct_actions"]) <= 3 for plan in plans))
        self.assertTrue(all("compiled_actions" not in plan for plan in plans))

    def test_chunk_validator_checks_direct_action_endpoint(self) -> None:
        public_cells = visible_public_cells(self.data, (3, 5))
        plan = validate_chunk_plan_output(
            {
                "actions": ["RIGHT", "RIGHT", "UP"],
                "waypoint": [5, 4],
                "declared_dependency_ids": ["water-fact"],
                "reason": "take the safe right corridor",
            },
            allowed_dependency_ids={"water-fact"},
            current_cell=(3, 5),
            target=(3, 3),
            public_cells=public_cells,
            available_facts=[
                {
                    "cell": [1, 4],
                    "hazard": "water",
                    "dependency_id": "water-fact",
                }
            ],
            remaining_actions=14,
            chunk_size=3,
        )
        self.assertEqual(plan["actions"], ["RIGHT", "RIGHT", "UP"])
        self.assertEqual(plan["execution_cells"], [[4, 5], [5, 5], [5, 4]])
        self.assertEqual(plan["speculative_execution_cells"], [])

        frontier_plan = validate_chunk_plan_output(
            {
                "actions": ["RIGHT", "RIGHT", "RIGHT"],
                "waypoint": [6, 5],
                "declared_dependency_ids": [],
                "reason": "extend one step beyond the current view",
            },
            allowed_dependency_ids=set(),
            current_cell=(3, 5),
            target=(3, 3),
            public_cells=public_cells,
            available_facts=[],
            remaining_actions=14,
            chunk_size=3,
            grid_dimensions=self.data["dimensions"],
        )
        self.assertEqual(frontier_plan["speculative_from_step"], 3)
        self.assertEqual(
            frontier_plan["speculative_execution_cells"], [[6, 5]]
        )

        with self.assertRaises(EpisodeError):
            validate_chunk_plan_output(
                {
                    "actions": ["RIGHT", "RIGHT", "UP"],
                    "waypoint": [5, 5],
                    "declared_dependency_ids": [],
                    "reason": "incorrect endpoint",
                },
                allowed_dependency_ids=set(),
                current_cell=(3, 5),
                target=(3, 3),
                public_cells=public_cells,
                available_facts=[],
                remaining_actions=14,
                chunk_size=3,
            )

    def test_chunk_frontier_extension_is_revalidated_each_tick(self) -> None:
        data = json.loads(STATEFUL_MAP_PATH.read_text())

        class FrontierExtensionBackend(ScriptedBackend):
            def __init__(self) -> None:
                super().__init__()
                self.forced = False

            def plan(self, **kwargs):
                if (
                    not self.forced
                    and kwargs["planning_granularity"] == "chunk"
                    and kwargs["agent"] == "P1"
                    and kwargs["position"] == (2, 5)
                ):
                    self.forced = True
                    self.call_records.append(
                        {
                            "backend": "scripted",
                            "kind": "chunk_plan",
                            "agent": "P1",
                            "phase": kwargs["phase"]["id"],
                        }
                    )
                    return {
                        "actions": ["RIGHT", "RIGHT", "RIGHT"],
                        "waypoint": [5, 5],
                        "declared_dependency_ids": [],
                        "reason": "tentatively extend beyond the current view",
                    }
                return super().plan(**kwargs)

        result = run_experiment(
            data,
            FrontierExtensionBackend(),
            mode="clean",
            fault_id=None,
            planning_granularity="chunk",
            chunk_size=3,
        )
        self.assertTrue(result["success"])
        events = result["attempts"][0]["events"]
        extension = next(
            event
            for event in events
            if event["event_type"] == "chunk_frontier_extension_planned"
            and event["agent"] == "P1"
        )
        self.assertEqual(extension["speculative_from_step"], 3)
        self.assertEqual(extension["speculative_cells"], [[5, 5]])
        self.assertFalse(
            any(
                event["event_type"] == "plan_interrupted"
                and event.get("plan_artifact_id")
                == extension["plan_artifact_id"]
                for event in events
            )
        )

    def test_chunk_frontier_monster_interrupts_only_unforeseen_entry(
        self,
    ) -> None:
        data = json.loads(STATEFUL_MAP_PATH.read_text())
        # [5, 5] is outside P1's view from its start but visible from [3, 5].
        data["stateful_world"]["monsters"].append(
            {
                "id": "ice-monster-frontier",
                "monster_type": "ice_monster",
                "required_weapon": "ice_sword",
                "cell": [5, 5],
            }
        )

        class ForcedChunkBackend(ScriptedBackend):
            def __init__(self, forced: dict[tuple[int, int], list[str]]) -> None:
                super().__init__()
                self.forced = dict(forced)

            def plan(self, **kwargs):
                actions = (
                    self.forced.pop(kwargs["position"], None)
                    if kwargs["agent"] == "P1"
                    else None
                )
                if actions is None:
                    return super().plan(**kwargs)
                self.call_records.append(
                    {
                        "backend": "scripted",
                        "kind": "chunk_plan",
                        "agent": "P1",
                        "phase": kwargs["phase"]["id"],
                    }
                )
                cursor = kwargs["position"]
                for action in actions:
                    dx, dy = MOVES[action]
                    cursor = cursor[0] + dx, cursor[1] + dy
                return {
                    "actions": actions,
                    "waypoint": list(cursor),
                    "declared_dependency_ids": [],
                    "reason": "forced chunk",
                }

        def run(forced: dict[tuple[int, int], list[str]]) -> list[dict]:
            result = run_experiment(
                data,
                ForcedChunkBackend(forced),
                mode="clean",
                fault_id=None,
                planning_granularity="chunk",
                chunk_size=3,
            )
            return result["attempts"][0]["events"]

        def frontier_damage(events: list[dict]) -> list[dict]:
            return [
                event
                for event in events
                if event["event_type"] == "monster_damage_taken"
                and event["agent"] == "P1"
                and event["cell"] == [5, 5]
            ]

        with self.subTest("unforeseen monster on speculative suffix"):
            events = run({(2, 5): ["RIGHT", "RIGHT", "RIGHT"]})
            extension = next(
                event
                for event in events
                if event["event_type"] == "chunk_frontier_extension_planned"
                and event["agent"] == "P1"
            )
            self.assertEqual(extension["speculative_cells"], [[5, 5]])
            interruption = next(
                event
                for event in events
                if event["event_type"] == "plan_interrupted"
                and event["plan_artifact_id"] == extension["plan_artifact_id"]
            )
            self.assertEqual(
                interruption["reason"], "newly_observed_monster_on_remaining_path"
            )
            self.assertEqual(interruption["monster_cells"], [[5, 5]])
            self.assertEqual(interruption["discarded_actions"], ["RIGHT", "RIGHT"])
            self.assertEqual(frontier_damage(events), [])

        with self.subTest("monster known at planning time is a deliberate entry"):
            events = run({(2, 5): ["RIGHT"], (3, 5): ["RIGHT", "RIGHT"]})
            self.assertFalse(
                any(
                    event["event_type"] == "plan_interrupted"
                    and event.get("reason")
                    == "newly_observed_monster_on_remaining_path"
                    for event in events
                )
            )
            self.assertEqual(len(frontier_damage(events)), 1)

    def test_chunk_frontier_wall_interrupts_suffix_without_repair(self) -> None:
        class WallSeekingChunkBackend(ScriptedBackend):
            def __init__(self) -> None:
                super().__init__()
                self.forced = False

            def plan(self, **kwargs):
                if (
                    not self.forced
                    and kwargs["planning_granularity"] == "chunk"
                    and kwargs["agent"] == "P1"
                    and kwargs["position"] == (3, 5)
                ):
                    self.forced = True
                    self.call_records.append(
                        {
                            "backend": "scripted",
                            "kind": "chunk_plan",
                            "agent": "P1",
                            "phase": kwargs["phase"]["id"],
                        }
                    )
                    return {
                        "actions": ["RIGHT", "RIGHT", "RIGHT"],
                        "waypoint": [6, 5],
                        "declared_dependency_ids": [],
                        "reason": "the third cell is outside the current view",
                    }
                return super().plan(**kwargs)

        result = run_experiment(
            self.data,
            WallSeekingChunkBackend(),
            mode="clean",
            fault_id=None,
            planning_granularity="chunk",
            chunk_size=3,
        )
        self.assertTrue(result["success"])
        events = result["attempts"][0]["events"]
        interruption = next(
            event
            for event in events
            if event["event_type"] == "plan_interrupted"
            and event.get("reason") == "queued_action_revealed_wall"
        )
        self.assertEqual(interruption["destination"], [6, 5])
        self.assertEqual(interruption["discarded_actions"], ["RIGHT"])

    def test_coordinate_path_validator_uses_reported_hazards_not_hidden_truth(self) -> None:
        public_cells = visible_public_cells(self.data, (3, 5))
        faulty_known_facts = [
            {"cell": [5, 4], "hazard": "water", "dependency_id": "faulty-message"}
        ]
        result = validate_route_plan_output(
            {
                "waypoint": [1, 3],
                "path_cells": [
                    [3, 5], [2, 5], [1, 5], [1, 4], [1, 3], [2, 3], [3, 3]
                ],
                "subgoal": "take the apparently safe left corridor",
                "declared_dependency_ids": ["faulty-message"],
                "reason": "the reported hazard blocks the right corridor",
            },
            allowed_dependency_ids={"faulty-message"},
            current_cell=(3, 5),
            target=(3, 3),
            public_cells=public_cells,
            available_facts=faulty_known_facts,
            remaining_actions=14,
        )
        self.assertIn([1, 4], result["path_cells"])
        self.assertEqual(
            result["compiled_actions"],
            ["LEFT", "LEFT", "UP", "UP", "RIGHT", "RIGHT"],
        )

    def test_junction_dry_run_remains_low_cost_baseline(self) -> None:
        backend = DryRunBackend()
        result = run_experiment(
            self.data,
            backend,
            mode="clean",
            fault_id=None,
            planning_granularity="junction",
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["backend_calls"], 7)
        plans = [
            artifact
            for artifact in result["attempts"][0]["artifacts"]
            if artifact.get("artifact_type") == "plan"
        ]
        self.assertEqual(len(plans), 4)
        self.assertTrue(
            all(set(plan["candidate_moves"]) == {"LEFT", "RIGHT"} for plan in plans)
        )

    def test_message_cannot_mutate_cited_sensor_fact(self) -> None:
        source = {
            "artifact_id": "fact-1",
            "hazard": "water",
            "cell": [1, 4],
        }
        with self.assertRaises(EpisodeError):
            validate_message_output(
                {
                    "message_facts": [
                        {
                            "source_fact_id": "fact-1",
                            "hazard": "water",
                            "cell": [5, 4],
                        }
                    ]
                },
                [source],
                limit=1,
            )

    def test_coordinate_fault_does_not_leave_stale_coordinate_in_note(self) -> None:
        message, matched = inject_fault(
            {
                "message_facts": [
                    {
                        "source_fact_id": "fact-1",
                        "hazard": "water",
                        "cell": [1, 4],
                    }
                ],
                "note": "Water hazard at [1,4]",
            },
            {
                "id": "fault-1",
                "kind": "coordinate_shift",
                "hazard": "water",
                "true_cell": [1, 4],
                "faulty_cell": [5, 4],
            },
        )
        self.assertTrue(matched)
        self.assertEqual(message["message_facts"][0]["cell"], [5, 4])
        self.assertNotIn("[1,4]", message["note"])

    def test_plan_cannot_invent_dependency_id(self) -> None:
        with self.assertRaises(EpisodeError):
            validate_plan_output(
                {
                    "next_action": "RIGHT",
                    "declared_dependency_ids": ["invented"],
                },
                allowed_dependency_ids={"real"},
                allowed_actions={"RIGHT"},
            )

    def test_plan_cannot_repeat_dependency_id(self) -> None:
        with self.assertRaises(EpisodeError):
            validate_plan_output(
                {
                    "next_action": "RIGHT",
                    "declared_dependency_ids": ["real", "real"],
                },
                allowed_dependency_ids={"real"},
                allowed_actions={"RIGHT"},
            )

    def test_plan_accepts_primitive_action(self) -> None:
        plan = validate_plan_output(
            {
                "next_action": "RIGHT",
                "declared_dependency_ids": ["fact-1"],
            },
            allowed_dependency_ids={"fact-1"},
            allowed_actions={"LEFT", "RIGHT"},
        )
        self.assertEqual(plan["next_action"], "RIGHT")

    def test_plan_rejects_non_primitive_action(self) -> None:
        with self.assertRaises(EpisodeError):
            validate_plan_output(
                {
                    "next_action": "RIGHT_BRANCH",
                    "declared_dependency_ids": [],
                },
                allowed_dependency_ids=set(),
                allowed_actions={"LEFT", "RIGHT"},
            )

    def test_step_plan_accepts_one_adjacent_coordinate(self) -> None:
        plan = assess_step_plan_output(
            {
                "next_cell": [4, 5],
                "declared_dependency_ids": ["water-fact"],
                "reason": "take the right side around known water",
            },
            allowed_dependency_ids={"water-fact"},
            current_cell=(3, 5),
            candidate_moves={"LEFT": [2, 5], "RIGHT": [4, 5]},
            available_facts=[
                {
                    "cell": [1, 4],
                    "hazard": "water",
                    "dependency_id": "water-fact",
                }
            ],
        )
        self.assertTrue(plan["accepted"])
        self.assertEqual(plan["next_action"], "RIGHT")

    def test_step_plan_turns_coordinate_jump_into_feedback_noop(self) -> None:
        plan = assess_step_plan_output(
            {
                "next_cell": [3, 3],
                "declared_dependency_ids": [],
                "reason": "incorrect jump",
            },
            allowed_dependency_ids=set(),
            current_cell=(3, 6),
            candidate_moves={"UP": [3, 5]},
            available_facts=[],
        )
        self.assertFalse(plan["accepted"])
        self.assertIsNone(plan["next_action"])
        self.assertEqual(plan["issue_code"], "non_adjacent_or_blocked")
        self.assertIn("public adjacent cells", plan["feedback"])

    def test_step_plan_executes_known_hazard_for_environment_to_resolve(self) -> None:
        plan = assess_step_plan_output(
            {
                "next_cell": [1, 4],
                "declared_dependency_ids": ["water-fact"],
                "reason": "incorrectly enter reported water",
            },
            allowed_dependency_ids={"water-fact"},
            current_cell=(1, 5),
            candidate_moves={"UP": [1, 4], "RIGHT": [2, 5]},
            available_facts=[
                {
                    "cell": [1, 4],
                    "hazard": "water",
                    "dependency_id": "water-fact",
                }
            ],
        )
        self.assertTrue(plan["accepted"])
        self.assertEqual(plan["next_action"], "UP")
        self.assertTrue(plan["known_hazard_selected"])
        self.assertIsNone(plan["issue_code"])

    def test_navigation_loop_detector_only_flags_repeated_terminal_cycle(self) -> None:
        self.assertIsNone(
            detect_navigation_loop([[3, 5], [2, 5], [3, 5], [4, 5]])
        )
        loop = detect_navigation_loop(
            [[3, 5], [2, 5], [3, 5], [2, 5], [3, 5], [2, 5]]
        )
        self.assertEqual(
            loop,
            {
                "period": 2,
                "repetitions": 3,
                "cycle_cells": [[3, 5], [2, 5]],
            },
        )

    def test_step_prompt_exposes_absolute_motion_without_full_routes(self) -> None:
        messages = planning_messages(
            data=self.data,
            agent="P1",
            phase=self.data["phases"][0],
            position=(3, 5),
            public_cells=visible_public_cells(self.data, (3, 5)),
            available_facts=[
                {
                    "cell": [1, 4],
                    "hazard": "water",
                    "dependency_id": "water-fact",
                }
            ],
            candidate_moves={"LEFT": [2, 5], "RIGHT": [4, 5]},
            visited_cells=[],
            remaining_actions=8,
            planning_granularity="step",
            last_action_feedback={
                "success": False,
                "requested_cell": [3, 3],
                "current_cell": [3, 6],
                "issue_code": "non_adjacent_or_blocked",
                "reason": "requested cell was not adjacent",
            },
        )
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["coordinate_system"]["action_deltas"]["UP"], [0, -1])
        self.assertEqual(payload["annotated_local_map"]["rows"][0]["cells"], "..T..")
        self.assertEqual(payload["annotated_local_map"]["rows"][1]["cells"], "W###.")
        self.assertEqual(payload["annotated_local_map"]["rows"][2]["cells"], "..@..")
        self.assertNotIn("branch_options", payload)
        self.assertNotIn("candidate_moves", payload)
        self.assertNotIn("public_transition_graph", payload)
        self.assertEqual(payload["public_next_cells"], [[2, 5], [4, 5]])
        self.assertEqual(payload["known_forbidden_cells"], [[1, 4]])
        self.assertEqual(
            payload["last_action_feedback"]["issue_code"],
            "non_adjacent_or_blocked",
        )
        self.assertFalse(payload["coordinate_system"]["has_heading"])
        self.assertEqual(payload["coordinate_system"]["rotation_effect"], "none")
        self.assertNotIn("candidate_continuations", payload)
        self.assertIn("next_cell", messages[0]["content"])
        self.assertIn("Backtracking is allowed", messages[0]["content"])
        self.assertIn("no heading or facing direction", messages[0]["content"])

    def test_junction_prompt_includes_dynamic_continuations(self) -> None:
        messages = planning_messages(
            data=self.data,
            agent="P1",
            phase=self.data["phases"][0],
            position=(3, 5),
            public_cells=visible_public_cells(self.data, (3, 5)),
            available_facts=[],
            candidate_moves={"LEFT": [2, 5], "RIGHT": [4, 5]},
            visited_cells=[],
            remaining_actions=8,
            planning_granularity="junction",
        )
        payload = json.loads(messages[1]["content"])
        self.assertEqual(
            payload["candidate_continuations"]["LEFT"]["cells"],
            [[2, 5], [1, 5], [1, 4], [1, 3], [2, 3], [3, 3]],
        )

    def test_route_prompt_requires_llm_coordinate_path(self) -> None:
        messages = planning_messages(
            data=self.data,
            agent="P1",
            phase=self.data["phases"][0],
            position=(3, 5),
            public_cells=visible_public_cells(self.data, (3, 5)),
            available_facts=[
                {
                    "cell": [1, 4],
                    "hazard": "water",
                    "dependency_id": "water-fact",
                }
            ],
            candidate_moves={},
            visited_cells=[[3, 6], [3, 5]],
            remaining_actions=14,
            planning_granularity="route",
        )
        payload = json.loads(messages[1]["content"])
        self.assertNotIn("candidate_routes", payload)
        self.assertNotIn("candidate_moves", payload)
        self.assertIn("public_adjacency_graph", payload)
        self.assertNotIn("public_transition_graph", payload)
        self.assertIn("Choose the waypoint yourself", messages[0]["content"])
        self.assertIn("path_cells", messages[0]["content"])
        self.assertNotIn("selected_route_id", messages[0]["content"])

    def test_chunk_prompt_requires_direct_primitive_actions(self) -> None:
        messages = planning_messages(
            data=self.data,
            agent="P1",
            phase=self.data["phases"][0],
            position=(3, 5),
            public_cells=visible_public_cells(self.data, (3, 5)),
            available_facts=[],
            candidate_moves={"LEFT": [2, 5], "RIGHT": [4, 5]},
            visited_cells=[[3, 6], [3, 5]],
            remaining_actions=14,
            planning_granularity="chunk",
            chunk_size=3,
        )
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["chunk_action_limit"], 3)
        self.assertIn("public_transition_graph", payload)
        self.assertIn("output them directly", messages[0]["content"])
        self.assertIn('"actions"', messages[0]["content"])

    def test_route_plan_rejects_non_adjacent_coordinate_jump(self) -> None:
        with self.assertRaises(EpisodeError):
            validate_route_plan_output(
                {
                    "waypoint": [5, 3],
                    "path_cells": [
                        [3, 5], [5, 5], [5, 4], [5, 3], [4, 3], [3, 3]
                    ],
                    "declared_dependency_ids": [],
                },
                allowed_dependency_ids=set(),
                current_cell=(3, 5),
                target=(3, 3),
                public_cells=visible_public_cells(self.data, (3, 5)),
                available_facts=[],
                remaining_actions=14,
            )

    def test_llm_route_plan_repairs_invalid_coordinate_path(self) -> None:
        backend = LLMBackend(
            url="https://example.invalid/v1/chat/completions",
            key="test-key",
            model="test-model",
        )
        responses = iter(
            [
                {
                    "waypoint": [5, 3],
                    "path_cells": [
                        [3, 5], [5, 5], [5, 4], [5, 3], [4, 3], [3, 3]
                    ],
                    "subgoal": "right corridor",
                    "declared_dependency_ids": ["water-fact"],
                    "reason": "avoid known water",
                },
                {
                    "waypoint": [5, 3],
                    "path_cells": [
                        [3, 5], [4, 5], [5, 5], [5, 4], [5, 3], [4, 3], [3, 3]
                    ],
                    "subgoal": "right corridor",
                    "declared_dependency_ids": ["water-fact"],
                    "reason": "avoid known water",
                },
            ]
        )

        def fake_chat(**_kwargs):
            backend.call_records.append({"backend": "llm"})
            return next(responses)

        facts = [
            {"cell": [1, 4], "hazard": "water", "dependency_id": "water-fact"}
        ]
        with patch.object(backend, "_chat", side_effect=fake_chat) as mocked:
            result = backend.plan(
                data=self.data,
                agent="P1",
                phase=self.data["phases"][0],
                position=(3, 5),
                public_cells=visible_public_cells(self.data, (3, 5)),
                available_facts=facts,
                candidate_moves={},
                visited_cells=[[3, 6], [3, 5]],
                remaining_actions=14,
                planning_granularity="route",
            )

        self.assertEqual(result["path_cells"][:2], [[3, 5], [4, 5]])
        self.assertEqual(mocked.call_count, 2)
        self.assertIn("jumps", backend.call_records[0]["semantic_validation_error"])

    def test_llm_chunk_plan_repairs_invalid_direct_actions(self) -> None:
        backend = LLMBackend(
            url="https://example.invalid/v1/chat/completions",
            key="test-key",
            model="test-model",
        )
        responses = iter(
            [
                {
                    "actions": ["RIGHT", "RIGHT", "UP", "UP"],
                    "waypoint": [5, 3],
                    "declared_dependency_ids": ["water-fact"],
                    "reason": "too many actions",
                },
                {
                    "actions": ["RIGHT", "RIGHT", "UP"],
                    "waypoint": [5, 4],
                    "declared_dependency_ids": ["water-fact"],
                    "reason": "corrected direct-action chunk",
                },
            ]
        )

        def fake_chat(**_kwargs):
            backend.call_records.append({"backend": "llm"})
            return next(responses)

        facts = [
            {"cell": [1, 4], "hazard": "water", "dependency_id": "water-fact"}
        ]
        with patch.object(backend, "_chat", side_effect=fake_chat) as mocked:
            result = backend.plan(
                data=self.data,
                agent="P1",
                phase=self.data["phases"][0],
                position=(3, 5),
                public_cells=visible_public_cells(self.data, (3, 5)),
                available_facts=facts,
                candidate_moves={"LEFT": [2, 5], "RIGHT": [4, 5]},
                visited_cells=[[3, 6], [3, 5]],
                remaining_actions=14,
                planning_granularity="chunk",
                chunk_size=3,
            )

        self.assertEqual(result["actions"], ["RIGHT", "RIGHT", "UP"])
        self.assertEqual(mocked.call_count, 2)
        self.assertIn(
            "current limit is 3",
            backend.call_records[0]["semantic_validation_error"],
        )

    def test_decision_checker_uses_known_information_only(self) -> None:
        facts = [{"cell": [1, 4], "hazard": "water"}]
        illegal_error = decision_error(
            next_action="UP",
            candidate_moves={"LEFT": [2, 5], "RIGHT": [4, 5]},
            continuations={
                "LEFT": {"cells": [[2, 5], [1, 5], [1, 4]]},
                "RIGHT": {"cells": [[4, 5], [5, 5], [5, 4]]},
            },
            available_facts=facts,
            visited_cells=[],
        )
        hazard_error = decision_error(
            next_action="LEFT",
            candidate_moves={"LEFT": [2, 5], "RIGHT": [4, 5]},
            continuations={
                "LEFT": {"cells": [[2, 5], [1, 5], [1, 4]]},
                "RIGHT": {"cells": [[4, 5], [5, 5], [5, 4]]},
            },
            available_facts=facts,
            visited_cells=[],
        )
        safe = decision_error(
            next_action="RIGHT",
            candidate_moves={"LEFT": [2, 5], "RIGHT": [4, 5]},
            continuations={
                "LEFT": {"cells": [[2, 5], [1, 5], [1, 4]]},
                "RIGHT": {"cells": [[4, 5], [5, 5], [5, 4]]},
            },
            available_facts=facts,
            visited_cells=[],
        )
        self.assertIn("not a candidate", illegal_error or "")
        self.assertIn("known hazard", hazard_error or "")
        self.assertIsNone(safe)

    def test_decision_checker_rejects_avoidable_revisit(self) -> None:
        issue = decision_error(
            next_action="RIGHT",
            candidate_moves={"LEFT": [1, 5], "RIGHT": [3, 5]},
            continuations=None,
            available_facts=[{"cell": [1, 4], "hazard": "water"}],
            visited_cells=[[3, 5], [2, 5]],
        )
        self.assertIn("revisits", issue or "")
        self.assertIn("LEFT", issue or "")

    def test_llm_decision_repairs_choice_with_known_downstream_hazard(self) -> None:
        backend = LLMBackend(
            url="https://example.invalid/v1/chat/completions",
            key="test-key",
            model="test-model",
        )
        responses = iter(
            [
                {
                    "next_action": "LEFT",
                    "declared_dependency_ids": ["water-fact"],
                    "reason": "incorrect choice",
                },
                {
                    "next_action": "RIGHT",
                    "declared_dependency_ids": ["water-fact"],
                    "reason": "corrected choice",
                },
            ]
        )

        def fake_chat(**_kwargs):
            backend.call_records.append({"backend": "llm"})
            return next(responses)

        with patch.object(backend, "_chat", side_effect=fake_chat) as mocked:
            result = backend.plan(
                data=self.data,
                agent="P2",
                phase=self.data["phases"][0],
                position=(3, 5),
                public_cells=visible_public_cells(self.data, (3, 5)),
                available_facts=[
                    {
                        "cell": [1, 4],
                        "hazard": "water",
                        "dependency_id": "water-fact",
                    }
                ],
                candidate_moves={"LEFT": [2, 5], "RIGHT": [4, 5]},
                visited_cells=[],
                remaining_actions=8,
                planning_granularity="junction",
            )

        self.assertEqual(result["next_action"], "RIGHT")
        self.assertEqual(mocked.call_count, 2)
        self.assertIn(
            "crosses known hazard",
            backend.call_records[0]["semantic_validation_error"],
        )

    def test_llm_step_parse_failure_returns_deterministic_stay(self) -> None:
        backend = LLMBackend(
            url="https://example.invalid/v1/chat/completions",
            key="test-key",
            model="test-model",
        )
        with patch.object(
            backend,
            "_chat",
            side_effect=LLMResponseParseError("invalid JSON"),
        ):
            result = backend.plan(
                data=self.data,
                agent="P1",
                phase=self.data["phases"][0],
                position=(3, 5),
                public_cells=visible_public_cells(self.data, (3, 5)),
                available_facts=[],
                candidate_moves={"LEFT": [2, 5], "RIGHT": [4, 5]},
                visited_cells=[],
                remaining_actions=8,
                planning_granularity="step",
            )
        self.assertEqual(result["next_cell"], [3, 5])
        self.assertTrue(result["_format_fallback"])

    def test_llm_backend_uses_gateway_safe_headers(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {"message_facts": [], "note": "ok"}
                                    )
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {},
                    }
                ).encode()

        backend = LLMBackend(
            url="https://api.together.ai/v1/chat/completions",
            key="test-key",
            model="Qwen/Qwen3.5-9B",
        )
        with patch(
            "recovery_harness.urllib.request.urlopen", return_value=FakeResponse()
        ) as mocked:
            backend._chat(
                kind="communicate",
                agent="P1",
                phase="test",
                messages=[{"role": "user", "content": "return JSON"}],
            )

        request = mocked.call_args.args[0]
        self.assertEqual(request.get_header("User-agent"), HTTP_USER_AGENT)
        self.assertEqual(request.get_header("Accept"), "application/json")
        payload = json.loads(request.data)
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertEqual(
            payload["response_format"]["json_schema"]["name"],
            "communication_output",
        )
        self.assertEqual(payload["reasoning"], {"enabled": False})
        self.assertEqual(payload["max_tokens"], 1200)

    def test_llm_backend_turns_socket_timeout_into_episode_error(self) -> None:
        backend = LLMBackend(
            url="https://api.together.ai/v1/chat/completions",
            key="test-key",
            model="Qwen/Qwen3.5-9B",
        )
        with patch(
            "recovery_harness.urllib.request.urlopen",
            side_effect=TimeoutError("read timed out"),
        ):
            with self.assertRaisesRegex(EpisodeError, "request failed"):
                backend._chat(
                    kind="communicate",
                    agent="P1",
                    phase="test",
                    messages=[{"role": "user", "content": "return JSON"}],
                )
        self.assertEqual(backend.call_records[-1]["error_type"], "transport_error")
        self.assertIn("timed out", backend.call_records[-1]["error"])

    def test_llm_verifier_uses_verification_schema_and_trusted_facts(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "valid": False,
                                            "reason": "coordinate mismatch",
                                        }
                                    )
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 20,
                            "completion_tokens": 5,
                            "total_tokens": 25,
                        },
                    }
                ).encode()

        backend = LLMBackend(
            url="https://api.together.ai/v1/chat/completions",
            key="test-key",
            model="Qwen/Qwen3.5-9B",
        )
        with patch(
            "recovery_harness.urllib.request.urlopen", return_value=FakeResponse()
        ) as mocked:
            verdict = backend.verify(
                sender="P2",
                recipient="P1",
                phase="lower-junction",
                message_facts=[
                    {
                        "source_fact_id": "fact-1",
                        "hazard": "water",
                        "cell": [5, 4],
                    }
                ],
                trusted_source_facts=[
                    {
                        "source_fact_id": "fact-1",
                        "hazard": "water",
                        "cell": [1, 4],
                    }
                ],
            )

        self.assertFalse(verdict["valid"])
        request = mocked.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(
            payload["response_format"]["json_schema"]["name"],
            "verification_output",
        )
        verifier_payload = json.loads(payload["messages"][1]["content"])
        self.assertEqual(
            verifier_payload["trusted_source_facts"][0]["cell"], [1, 4]
        )
        self.assertEqual(backend.call_records[-1]["kind"], "verify")

    def test_llm_backend_uses_openai_gpt54_parameters(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {"message_facts": [], "note": "ok"}
                                    )
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {},
                    }
                ).encode()

        backend = LLMBackend(
            url="https://api.openai.com/v1/chat/completions",
            key="test-key",
            model="gpt-5.4",
            reasoning_effort="low",
        )
        with patch(
            "recovery_harness.urllib.request.urlopen", return_value=FakeResponse()
        ) as mocked:
            backend._chat(
                kind="communicate",
                agent="P1",
                phase="test",
                messages=[{"role": "user", "content": "return JSON"}],
            )

        request = mocked.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "gpt-5.4")
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertFalse(payload["store"])
        self.assertNotIn("temperature", payload)
        self.assertNotIn("max_tokens", payload)
        self.assertEqual(payload["max_completion_tokens"], 1200)
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertEqual(
            payload["response_format"]["json_schema"]["name"],
            "communication_output",
        )

    def test_llm_backend_uses_tritonai_completion_token_parameter(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {"message_facts": [], "note": "ok"}
                                    )
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {},
                    }
                ).encode()

        backend = LLMBackend(
            url="https://tritonai-api.ucsd.edu/v1/chat/completions",
            key="test-key",
            model="api-glm-5.3",
        )
        with patch(
            "recovery_harness.urllib.request.urlopen", return_value=FakeResponse()
        ) as mocked:
            backend._chat(
                kind="communicate",
                agent="P1",
                phase="test",
                messages=[{"role": "system", "content": "return JSON"}],
            )

        payload = json.loads(mocked.call_args.args[0].data)
        self.assertEqual(payload["max_completion_tokens"], 1200)
        self.assertNotIn("max_tokens", payload)
        self.assertEqual(
            backend.call_records[0]["token_limit_parameter"],
            "max_completion_tokens",
        )

    def test_triton_glm_navigation_starts_with_larger_completion_budget(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "actions": ["RIGHT"],
                                            "waypoint": [4, 5],
                                            "declared_dependency_ids": [],
                                            "reason": "advance",
                                        }
                                    )
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {},
                    }
                ).encode()

        backend = LLMBackend(
            url="https://tritonai-api.ucsd.edu/v1/chat/completions",
            key="test-key",
            model="api-glm-5.3",
        )
        with patch(
            "recovery_harness.urllib.request.urlopen", return_value=FakeResponse()
        ) as mocked:
            backend._chat(
                kind="chunk_plan",
                agent="P1",
                phase="test",
                messages=[{"role": "system", "content": "return JSON"}],
            )

        payload = json.loads(mocked.call_args.args[0].data)
        self.assertEqual(payload["max_completion_tokens"], 4096)
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(backend.call_records[0]["max_tokens"], 4096)
        self.assertEqual(backend.call_records[0]["reasoning_effort"], "low")

    def test_triton_glm_honors_explicit_reasoning_effort(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {"message_facts": [], "note": "ok"}
                                    )
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {},
                    }
                ).encode()

        backend = LLMBackend(
            url="https://tritonai-api.ucsd.edu/v1/chat/completions",
            key="test-key",
            model="api-glm-5.3",
            reasoning_effort="medium",
        )
        with patch(
            "recovery_harness.urllib.request.urlopen", return_value=FakeResponse()
        ) as mocked:
            backend._chat(
                kind="communicate",
                agent="P1",
                phase="test",
                messages=[{"role": "system", "content": "return JSON"}],
            )

        payload = json.loads(mocked.call_args.args[0].data)
        self.assertEqual(payload["reasoning_effort"], "medium")

    def test_llm_backend_retries_reasoning_only_response(self) -> None:
        class FakeResponse:
            def __init__(self, message, finish_reason):
                self.message = message
                self.finish_reason = finish_reason

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [
                            {
                                "message": self.message,
                                "finish_reason": self.finish_reason,
                            }
                        ],
                        "usage": {},
                    }
                ).encode()

        backend = LLMBackend(
            url="https://tritonai-api.ucsd.edu/v1/chat/completions",
            key="test-key",
            model="api-glm-5.3",
        )
        responses = [
            FakeResponse(
                {"content": None, "reasoning_content": "private reasoning"},
                "length",
            ),
            FakeResponse(
                {
                    "content": json.dumps(
                        {"message_facts": [], "note": "recovered"}
                    )
                },
                "stop",
            ),
        ]
        with patch(
            "recovery_harness.urllib.request.urlopen", side_effect=responses
        ) as mocked:
            result = backend._chat(
                kind="communicate",
                agent="P1",
                phase="test",
                messages=[{"role": "system", "content": "return JSON"}],
            )

        self.assertEqual(result["note"], "recovered")
        self.assertEqual(mocked.call_count, 2)
        first = backend.call_records[0]
        self.assertEqual(first["error_type"], "missing_final_content")
        self.assertTrue(first["response_shape"]["has_reasoning_content"])
        self.assertNotIn("reasoning_content", first)
        second_payload = json.loads(mocked.call_args_list[1].args[0].data)
        self.assertEqual(second_payload["max_completion_tokens"], 2400)

    def test_llm_backend_retries_truncated_json_once(self) -> None:
        class FakeResponse:
            def __init__(self, content, finish_reason):
                self.content = content
                self.finish_reason = finish_reason

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [
                            {
                                "message": {"content": self.content},
                                "finish_reason": self.finish_reason,
                            }
                        ],
                        "usage": {},
                    }
                ).encode()

        backend = LLMBackend(
            url="https://api.together.ai/v1/chat/completions",
            key="test-key",
            model="Qwen/Qwen3.5-9B",
        )
        responses = [
            FakeResponse('{"message_facts": [], "note": "cut', "length"),
            FakeResponse('{"message_facts": [], "note": "ok"}', "stop"),
        ]
        with patch(
            "recovery_harness.urllib.request.urlopen", side_effect=responses
        ) as mocked:
            result = backend._chat(
                kind="communicate",
                agent="P1",
                phase="test",
                messages=[{"role": "system", "content": "return JSON"}],
            )

        self.assertEqual(result["note"], "ok")
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(backend.call_records[0]["finish_reason"], "length")
        self.assertIn("parse_error", backend.call_records[0])
        self.assertEqual(backend.call_records[1]["max_tokens"], 2400)
        self.assertGreaterEqual(backend.call_records[0]["latency_ms"], 0)
        self.assertGreaterEqual(backend.call_records[1]["latency_ms"], 0)
        self.assertEqual(backend.call_records[0]["transport_attempts"], 1)

    def test_sse_chat_response_is_collapsed(self) -> None:
        body = (
            'data: {"choices":[{"delta":{"content":"{\\"message_facts\\":"},'
            '"finish_reason":null}]}\n\n'
            'data: {"choices":[{"delta":{"content":"[],\\"note\\":\\"ok\\"}"},'
            '"finish_reason":null}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}\n\n'
            'data: [DONE]\n\n'
        ).encode()
        result = parse_sse_chat_response(body)
        self.assertEqual(
            json.loads(result["choices"][0]["message"]["content"]),
            {"message_facts": [], "note": "ok"},
        )
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
        self.assertEqual(result["usage"]["total_tokens"], 14)

    def test_llm_backend_falls_back_to_required_streaming(self) -> None:
        class FakeStreamingResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return (
                    'data: {"choices":[{"delta":{"content":"{\\"message_facts\\":[],'
                    '\\"note\\":\\"ok\\"}"},"finish_reason":"stop"}],'
                    '"usage":{"total_tokens":9}}\n\n'
                    'data: [DONE]\n\n'
                ).encode()

        streaming_error = urllib.error.HTTPError(
            url="https://api.together.ai/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=io.BytesIO(
                b'{"error":{"code":"streaming_required",'
                b'"message":"This model only supports streaming."}}'
            ),
        )
        backend = LLMBackend(
            url="https://api.together.ai/v1/chat/completions",
            key="test-key",
            model="Qwen/Qwen3.8-Flash",
        )
        with patch(
            "recovery_harness.urllib.request.urlopen",
            side_effect=[streaming_error, FakeStreamingResponse()],
        ) as mocked:
            result = backend._chat(
                kind="communicate",
                agent="P1",
                phase="test",
                messages=[{"role": "system", "content": "return JSON"}],
            )

        self.assertEqual(result["note"], "ok")
        self.assertEqual(mocked.call_count, 2)
        second_payload = json.loads(mocked.call_args_list[1].args[0].data)
        self.assertTrue(second_payload["stream"])
        self.assertTrue(second_payload["stream_options"]["include_usage"])
        self.assertTrue(backend.force_streaming)
        self.assertTrue(backend.call_records[0]["streaming_fallback_used"])
        self.assertEqual(backend.call_records[0]["transport_attempts"], 2)
        self.assertGreaterEqual(backend.call_records[0]["latency_ms"], 0)

    def test_llm_backend_retries_transient_http_error(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": '{"message_facts": [], "note": "ok"}'
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {},
                    }
                ).encode()

        rate_limit = urllib.error.HTTPError(
            url="https://example.invalid/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs={"Retry-After": "0"},
            fp=io.BytesIO(b'{"error":"temporary rate limit"}'),
        )
        backend = LLMBackend(
            url="https://example.invalid/v1/chat/completions",
            key="test-key",
            model="test-model",
        )
        with patch(
            "recovery_harness.urllib.request.urlopen",
            side_effect=[rate_limit, FakeResponse()],
        ) as mocked, patch("recovery_harness.time.sleep") as sleep:
            result = backend._chat(
                kind="communicate",
                agent="P1",
                phase="test",
                messages=[{"role": "system", "content": "return JSON"}],
            )

        self.assertEqual(result["note"], "ok")
        self.assertEqual(mocked.call_count, 2)
        sleep.assert_called_once_with(0.0)
        record = backend.call_records[0]
        self.assertEqual(record["transport_attempts"], 2)
        self.assertEqual(record["transport_retry_count"], 1)
        self.assertEqual(
            [item["status"] for item in record["transport_attempt_log"]],
            ["retry", "ok"],
        )

    def test_llm_backend_retries_connection_dropped_mid_response(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": '{"message_facts": [], "note": "ok"}'
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {},
                    }
                ).encode()

        class TruncatedResponse(FakeResponse):
            def read(self):
                raise http.client.IncompleteRead(b'{"choi')

        backend = LLMBackend(
            url="https://example.invalid/v1/chat/completions",
            key="test-key",
            model="test-model",
        )
        # urllib does not wrap these in URLError: the first is raised by
        # getresponse(), the second while the body is read.
        with patch(
            "recovery_harness.urllib.request.urlopen",
            side_effect=[
                http.client.RemoteDisconnected("closed without response"),
                TruncatedResponse(),
                FakeResponse(),
            ],
        ) as mocked, patch("recovery_harness.time.sleep") as sleep:
            result = backend._chat(
                kind="communicate",
                agent="P1",
                phase="test",
                messages=[{"role": "system", "content": "return JSON"}],
            )

        self.assertEqual(result["note"], "ok")
        self.assertEqual(mocked.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1.0, 2.0])
        record = backend.call_records[0]
        self.assertEqual(record["transport_retry_count"], 2)
        self.assertEqual(
            [item["status"] for item in record["transport_attempt_log"]],
            ["retry", "retry", "ok"],
        )

    def test_redact_secrets_removes_known_key_and_bearer_tokens(self) -> None:
        secret = "unit-test-secret-value-1234"
        text = f"Invalid header value b'Bearer {secret}\\n'"
        redacted = redact_secrets(text, [secret + "\n"])

        self.assertNotIn(secret, redacted)
        self.assertIn("Bearer [REDACTED]", redacted)
        self.assertEqual(
            redact_secrets("Bearer tritonai-token-xyz"), "Bearer [REDACTED]"
        )
        self.assertEqual(redact_secrets("no credentials here"), "no credentials here")

    def test_llm_backend_strips_pasted_key_whitespace(self) -> None:
        backend = LLMBackend(
            url="https://example.invalid/v1/chat/completions",
            key="  test-key\n",
            model="test-model",
        )
        self.assertEqual(backend.key, "test-key")
        for bad_key in ("", "test key", "test-\x00key"):
            with self.assertRaises(ValueError) as caught:
                LLMBackend(
                    url="https://example.invalid/v1/chat/completions",
                    key=bad_key,
                    model="test-model",
                )
            if bad_key:
                self.assertNotIn(bad_key, str(caught.exception))

    def test_llm_http_error_detail_does_not_leak_key(self) -> None:
        secret = "tritonai-secret-token-123"
        http_error = urllib.error.HTTPError(
            url="https://example.invalid/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(
                f'{{"error":"bad key {secret}","auth":"Bearer {secret}"}}'.encode()
            ),
        )
        backend = LLMBackend(
            url="https://example.invalid/v1/chat/completions",
            key=secret,
            model="test-model",
        )
        with patch(
            "recovery_harness.urllib.request.urlopen", side_effect=http_error
        ), self.assertRaises(EpisodeError) as caught:
            backend._chat(
                kind="communicate",
                agent="P1",
                phase="test",
                messages=[{"role": "system", "content": "return JSON"}],
            )

        self.assertNotIn(secret, str(caught.exception))
        self.assertNotIn(secret, json.dumps(backend.call_records))


if __name__ == "__main__":
    unittest.main()
