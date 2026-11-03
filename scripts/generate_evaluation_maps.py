#!/usr/bin/env python3
"""Generate the held-out evaluation maps without consulting an LLM.

Each topology is hand-designed so that a map has a fixed structural role. A
seeded sampler places traps or weapons in precommitted candidate cells, derives
the precommitted faults from the resulting routes, and accepts the first seed
whose map passes every deterministic acceptance check. All checks run the
production scheduler with the scripted planner; no model is called.

    python3 scripts/generate_evaluation_maps.py --output maps/evaluation
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recovery_harness import (  # noqa: E402
    AGENTS,
    MOVES,
    EpisodeError,
    MapValidationError,
    ScriptedBackend,
    run_experiment,
    run_scheduler_preflight,
    validate_map,
)

Cell = tuple[int, int]

GENERATOR_VERSION = 1
MAX_SEEDS = 500
OBSERVATION_RADIUS = 2
MESSAGE_FACT_LIMIT = 2
MAX_HP = 3
PROBE_BUDGET = 80
MAX_TRAP_DETOUR = 6
PARTNER = {"P1": "P2", "P2": "P1"}
SENSOR = {"P1": "fire", "P2": "water"}
# Sensing does not imply use: each agent carries only the weapon its partner senses.
CARRIED_WEAPON = {"P1": "ice_sword", "P2": "fire_sword"}
WEAPON_SENSOR = {"fire_sword": "P1", "ice_sword": "P2"}
MONSTER_REQUIRES = {"fire_monster": "fire_sword", "ice_monster": "ice_sword"}
ACTION_ORDER = ("UP", "RIGHT", "DOWN", "LEFT")
# Accidental corruption is a fixed single-axis coordinate offset.
BASIC_OFFSETS = [
    (dx * size, dy * size)
    for size in (2, 3, 4)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
]
# Weapon offsets keep the false cell out of the true weapon's neighbourhood so
# that a chance step onto the real weapon does not mask the fault.
WEAPON_OFFSETS = [offset for offset in BASIC_OFFSETS if max(map(abs, offset)) >= 3]


@dataclass(frozen=True)
class SlotGroup:
    """Candidate cells and the number of objects of one type sampled there."""

    kind: str
    cells: tuple[Cell, ...]
    count: int


@dataclass(frozen=True)
class Template:
    name: str
    suite: str
    size_class: str
    role: str
    rows: tuple[str, ...]
    starts: dict[str, Cell]
    stages: tuple[dict[str, Cell], ...]
    slot_groups: tuple[SlotGroup, ...]
    monsters: tuple[tuple[str, str, Cell], ...] = ()
    cross_stage_faults: bool = False
    hidden_monster: bool = False
    avoidable_monster: bool = False
    # Preflight classes accepted for the accidental fault, in preference order.
    accidental_classes: tuple[str, ...] = ("executed_with_harm",)
    # Basic maps: stages in which at least one agent needs a partner report.
    report_stages: tuple[int, ...] = ()

    @property
    def goals(self) -> dict[str, Cell]:
        return self.stages[-1]


# ---------------------------------------------------------------------------
# Topologies. Rows are indexed by y and columns by x; [x, y] uses a top-left
# origin. Every map uses observation radius 2 and two facts per message.
# ---------------------------------------------------------------------------

B1 = Template(
    name="b1-three-lane-detours",
    suite="basic",
    size_class="small",
    role="coincident start and goal; two short three-lane decisions; base case",
    rows=(
        "###.###",
        "#.....#",
        "#.#.#.#",
        "#.....#",
        "#.#.#.#",
        "#.....#",
        "###.###",
    ),
    starts={"P1": (3, 6), "P2": (3, 6)},
    stages=({"P1": (3, 3), "P2": (3, 3)}, {"P1": (3, 0), "P2": (3, 0)}),
    slot_groups=(
        SlotGroup("fire", ((1, 4), (3, 4), (5, 4), (1, 2), (3, 2), (5, 2)), 2),
        SlotGroup("water", ((1, 4), (3, 4), (5, 4), (1, 2), (3, 2), (5, 2)), 2),
    ),
    report_stages=(0, 1),
)

B2 = Template(
    name="b2-corner-swap",
    suite="basic",
    size_class="small",
    role="separate starts that swap corners in one stage; each agent starts "
    "where the partner's hazards are",
    rows=(
        "#########",
        "#.......#",
        "#.##.##.#",
        "#.......#",
        "#.##.##.#",
        "#.......#",
        "#########",
    ),
    starts={"P1": (1, 5), "P2": (7, 1)},
    stages=({"P1": (7, 1), "P2": (1, 5)},),
    slot_groups=(
        SlotGroup(
            "fire",
            ((1, 2), (4, 2), (7, 2), (1, 4), (4, 4), (7, 4), (3, 3), (5, 3)),
            2,
        ),
        SlotGroup(
            "water",
            ((1, 2), (4, 2), (7, 2), (1, 4), (4, 4), (7, 4), (3, 3), (5, 3)),
            2,
        ),
    ),
    report_stages=(0,),
)

B3 = Template(
    name="b3-zigzag-exchange",
    suite="basic",
    size_class="small",
    role="separate starts; cross to the partner's side and back in two stages",
    rows=(
        "#########",
        "#.......#",
        "#.#.#.#.#",
        "#.......#",
        "#.#.#.#.#",
        "#.......#",
        "#########",
    ),
    starts={"P1": (1, 5), "P2": (7, 5)},
    stages=({"P1": (7, 3), "P2": (1, 3)}, {"P1": (1, 1), "P2": (7, 1)}),
    slot_groups=(
        SlotGroup(
            "fire",
            ((1, 4), (3, 4), (5, 4), (7, 4), (1, 2), (3, 2), (5, 2), (7, 2)),
            2,
        ),
        SlotGroup(
            "water",
            ((1, 4), (3, 4), (5, 4), (7, 4), (1, 2), (3, 2), (5, 2), (7, 2)),
            2,
        ),
    ),
    report_stages=(0, 1),
)

_LEFT_LADDER_9 = ((1, 2), (3, 2), (1, 4), (3, 4), (1, 6), (3, 6))
_RIGHT_LADDER_9 = ((5, 2), (7, 2), (5, 4), (7, 4), (5, 6), (7, 6))

B4 = Template(
    name="b4-ladder-exchange",
    suite="basic",
    size_class="medium",
    role="descend one's own ladder, then climb the partner's; hazards reported "
    "in stage 1 are first needed in stage 2",
    rows=(
        "#########",
        "#...#...#",
        "#.#.#.#.#",
        "#...#...#",
        "#.#.#.#.#",
        "#...#...#",
        "#.#.#.#.#",
        "#.......#",
        "#########",
    ),
    starts={"P1": (1, 1), "P2": (7, 1)},
    stages=({"P1": (4, 7), "P2": (4, 7)}, {"P1": (7, 1), "P2": (1, 1)}),
    # The first visitor of each ladder senses its hazards, so the partner can
    # only learn them from a stage-1 report.
    slot_groups=(
        SlotGroup("fire", _LEFT_LADDER_9, 2),
        SlotGroup("water", _RIGHT_LADDER_9, 2),
    ),
    cross_stage_faults=True,
    report_stages=(1,),
)

B5 = Template(
    name="b5-staged-relay",
    suite="basic",
    size_class="large",
    role="three stages: descend, exchange through a shared zone, climb the "
    "partner's ladder; the longest restart prefix",
    rows=(
        "###########",
        "#...###...#",
        "#.#.###.#.#",
        "#...###...#",
        "#.#.###.#.#",
        "#.........#",
        "#.#.#.#.#.#",
        "#.........#",
        "###########",
    ),
    starts={"P1": (1, 1), "P2": (9, 1)},
    stages=(
        {"P1": (3, 5), "P2": (7, 5)},
        {"P1": (7, 5), "P2": (3, 5)},
        {"P1": (9, 1), "P2": (1, 1)},
    ),
    slot_groups=(
        SlotGroup("fire", ((1, 2), (3, 2), (1, 4), (3, 4)), 2),
        SlotGroup("water", ((7, 2), (9, 2), (7, 4), (9, 4)), 2),
        SlotGroup("fire", ((4, 5), (5, 5), (6, 5), (5, 6)), 1),
        SlotGroup("water", ((4, 5), (5, 5), (6, 5), (5, 6)), 1),
    ),
    cross_stage_faults=True,
    report_stages=(1, 2),
)

E1 = Template(
    name="e1-pocket-crossing",
    suite="extended",
    size_class="small",
    role="separate starts that cross; each goal is guarded by a monster and "
    "the matching sword lies in a pocket only the partner senses",
    rows=(
        "#.#####.#",
        "#.......#",
        "#.##.##.#",
        "#.......#",
        "#.##.##.#",
        "#.......#",
        "##.#.#.##",
    ),
    starts={"P1": (1, 5), "P2": (7, 5)},
    stages=({"P1": (7, 0), "P2": (1, 0)},),
    monsters=(
        ("fire-monster-west", "fire_monster", (1, 1)),
        ("ice-monster-east", "ice_monster", (7, 1)),
    ),
    slot_groups=(
        SlotGroup("ice_sword", ((2, 6), (4, 6), (6, 6)), 1),
        SlotGroup("fire_sword", ((2, 6), (4, 6), (6, 6)), 1),
    ),
)

E2 = Template(
    name="e2-shared-start-split",
    suite="extended",
    size_class="small",
    role="coincident start, separate guarded goals; both agents see the "
    "starting area, then split",
    rows=(
        "#.#####.#",
        "#.......#",
        "#.#.#.#.#",
        "#.......#",
        "#.#.#.#.#",
        "#.......#",
        "#.#.#.#.#",
    ),
    starts={"P1": (4, 5), "P2": (4, 5)},
    stages=({"P1": (7, 0), "P2": (1, 0)},),
    monsters=(
        ("fire-monster-west", "fire_monster", (1, 1)),
        ("ice-monster-east", "ice_monster", (7, 1)),
    ),
    # Inner pockets are visible from the shared start, keeping the map small.
    slot_groups=(
        SlotGroup("ice_sword", ((3, 6), (5, 6)), 1),
        SlotGroup("fire_sword", ((3, 6), (5, 6)), 1),
    ),
)

E3 = Template(
    name="e3-guarded-doors",
    suite="extended",
    size_class="small",
    role="each goal compartment has a short door guarded by a monster and a "
    "longer monster-free corridor; a reported sword makes the door worthwhile",
    rows=(
        "#########",
        "#.......#",
        "#.#####.#",
        "#.......#",
        "....#####",
        "#.......#",
        "....###.#",
        "#.......#",
        "#########",
    ),
    starts={"P1": (1, 4), "P2": (2, 4)},
    stages=({"P1": (7, 5), "P2": (7, 3)},),
    monsters=(
        ("fire-monster-upper-door", "fire_monster", (4, 3)),
        ("ice-monster-lower-door", "ice_monster", (4, 5)),
    ),
    slot_groups=(
        SlotGroup("ice_sword", ((0, 4), (0, 6)), 1),
        SlotGroup("fire_sword", ((0, 4), (0, 6)), 1),
    ),
    avoidable_monster=True,
    # Without a sword the safe corridor remains, so an accidental offset
    # usually wastes a planned detour instead of costing HP.
    accidental_classes=("executed_with_harm", "plan_relevant_without_execution"),
)

E4 = Template(
    name="e4-ladder-relay",
    suite="extended",
    size_class="medium",
    role="descend one's own ladder while the partner reports swords, then "
    "climb the partner's ladder to a goal monster first seen in stage 2",
    rows=(
        "##.###.##",
        "#...#...#",
        "#.#.#.#.#",
        "#...#...#",
        "#.#.#.#.#",
        "#...#...#",
        "#.#.#.#.#",
        "#.......#",
        "##.#.#.##",
    ),
    starts={"P1": (1, 1), "P2": (7, 1)},
    stages=({"P1": (4, 7), "P2": (4, 7)}, {"P1": (6, 0), "P2": (2, 0)}),
    monsters=(
        ("fire-monster-west", "fire_monster", (2, 1)),
        ("ice-monster-east", "ice_monster", (6, 1)),
    ),
    slot_groups=(
        SlotGroup("ice_sword", ((6, 8), (4, 8), (2, 8)), 1),
        SlotGroup("fire_sword", ((2, 8), (4, 8), (6, 8)), 1),
    ),
    cross_stage_faults=True,
    hidden_monster=True,
)

E5 = Template(
    name="e5-double-gate",
    suite="extended",
    size_class="medium",
    role="each agent passes a gate monster and then a goal monster, so it "
    "needs two single-use swords; a fault corrupts only one of four sword facts",
    rows=(
        "#.#######.#",
        "#.........#",
        "###.###.###",
        "#.........#",
        "#.#.#.#.#.#",
        "#.........#",
        "#.#.###.#.#",
    ),
    starts={"P1": (2, 5), "P2": (8, 5)},
    stages=({"P1": (9, 0), "P2": (1, 0)},),
    monsters=(
        ("fire-monster-gate", "fire_monster", (3, 2)),
        ("ice-monster-gate", "ice_monster", (7, 2)),
        ("fire-monster-goal", "fire_monster", (1, 1)),
        ("ice-monster-goal", "ice_monster", (9, 1)),
    ),
    slot_groups=(
        SlotGroup("ice_sword", ((1, 6), (3, 6), (7, 6), (9, 6)), 2),
        SlotGroup("fire_sword", ((1, 6), (3, 6), (7, 6), (9, 6)), 2),
    ),
)


TEMPLATES = (B1, B2, B3, B4, B5, E1, E2, E3, E4, E5)


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def open_cell(rows: tuple[str, ...], cell: Cell) -> bool:
    x, y = cell
    return 0 <= y < len(rows) and 0 <= x < len(rows[0]) and rows[y][x] != "#"


def shortest_path(
    rows: tuple[str, ...], start: Cell, goal: Cell, blocked: set[Cell]
) -> list[Cell] | None:
    """Deterministic breadth-first shortest path over open, unblocked cells."""
    previous: dict[Cell, Cell | None] = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if cell == goal:
            path = [cell]
            while previous[path[-1]] is not None:
                path.append(previous[path[-1]])
            return path[::-1]
        for action in ACTION_ORDER:
            dx, dy = MOVES[action]
            step = (cell[0] + dx, cell[1] + dy)
            if step in previous or not open_cell(rows, step) or step in blocked:
                continue
            previous[step] = cell
            queue.append(step)
    return None


def distance(rows: tuple[str, ...], start: Cell, goal: Cell, blocked: set[Cell]) -> int | None:
    path = shortest_path(rows, start, goal, blocked)
    return None if path is None else len(path) - 1


def on_some_shortest_path(
    rows: tuple[str, ...], start: Cell, goal: Cell, cell: Cell, blocked: set[Cell]
) -> bool:
    total = distance(rows, start, goal, blocked)
    head = distance(rows, start, cell, blocked)
    tail = distance(rows, cell, goal, blocked)
    return None not in (total, head, tail) and head + tail == total


def path_actions(path: list[Cell]) -> list[str]:
    delta_to_action = {delta: action for action, delta in MOVES.items()}
    return [
        delta_to_action[(b[0] - a[0], b[1] - a[1])] for a, b in zip(path, path[1:])
    ]


def chebyshev(a: Cell, b: Cell) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def stage_segments(template: Template) -> list[dict[str, tuple[Cell, Cell]]]:
    segments = []
    cursor = dict(template.starts)
    for targets in template.stages:
        segments.append({agent: (cursor[agent], targets[agent]) for agent in AGENTS})
        cursor = dict(targets)
    return segments


# ---------------------------------------------------------------------------
# Map construction
# ---------------------------------------------------------------------------


def sample_placement(template: Template, rng: random.Random) -> dict[str, list[Cell]]:
    placement: dict[str, list[Cell]] = {}
    used: set[Cell] = set()
    for group in template.slot_groups:
        free = [cell for cell in group.cells if cell not in used]
        if len(free) < group.count:
            raise ValueError("slot group exhausted")
        chosen = rng.sample(free, group.count)
        placement.setdefault(group.kind, []).extend(chosen)
        used.update(chosen)
    return placement


def build_map(
    template: Template,
    placement: dict[str, list[Cell]],
    *,
    budgets: dict[tuple[int, str], int] | None = None,
) -> dict[str, Any]:
    traps = {
        "fire": sorted(placement.get("fire", [])),
        "water": sorted(placement.get("water", [])),
    }
    all_traps = set(traps["fire"]) | set(traps["water"])
    phases = []
    for index, segment in enumerate(stage_segments(template)):
        starts, targets, routes, max_actions = {}, {}, {}, {}
        for agent in AGENTS:
            start, target = segment[agent]
            own = set(traps[SENSOR[agent]])
            safe = shortest_path(template.rows, start, target, all_traps)
            naive = shortest_path(template.rows, start, target, own)
            if safe is None or naive is None:
                raise MapValidationError(
                    f"stage {index + 1} has no trap-safe route for {agent}"
                )
            agent_routes = [path_actions(safe)]
            if naive != safe:
                agent_routes.append(path_actions(naive))
            starts[agent] = list(start)
            targets[agent] = list(target)
            routes[agent] = agent_routes
            max_actions[agent] = (
                PROBE_BUDGET if budgets is None else budgets[(index, agent)]
            )
        phases.append(
            {
                "id": f"stage-{index + 1}",
                "start": starts,
                "target": targets,
                "max_actions": max_actions,
                "reference_routes": routes,
            }
        )
    step_limit = max(
        sum(phase["max_actions"][agent] for phase in phases) for agent in AGENTS
    )
    data: dict[str, Any] = {
        "schema_version": 5 if template.suite == "extended" else 4,
        "id": f"{template.name}-eval-v2",
        "description": template.role,
        "world_model": "shared",
        "dimensions": {"width": len(template.rows[0]), "height": len(template.rows)},
        "coordinate_system": {"origin": "top-left", "cell": "[x, y]"},
        "public_rows": list(template.rows),
        "hazards": {kind: [list(cell) for cell in cells] for kind, cells in traps.items()},
        "agents": {
            agent: {
                "sensor": SENSOR[agent],
                "start": list(template.starts[agent]),
                "goal": list(template.goals[agent]),
            }
            for agent in AGENTS
        },
        "observation_radius": OBSERVATION_RADIUS,
        "message_fact_limit": MESSAGE_FACT_LIMIT,
        "step_limit": step_limit,
        "collision_policy": "non_blocking",
        "prelude_actions": {agent: [] for agent in AGENTS},
        "phases": phases,
        "finale_actions": {agent: [] for agent in AGENTS},
        "faults": [],
    }
    if template.suite == "extended":
        weapons = []
        for weapon_type in ("fire_sword", "ice_sword"):
            for index, cell in enumerate(sorted(placement.get(weapon_type, []))):
                weapons.append(
                    {
                        "id": f"{weapon_type.replace('_', '-')}-{index + 1}",
                        "weapon_type": weapon_type,
                        "cell": list(cell),
                        "sensed_by": WEAPON_SENSOR[weapon_type],
                    }
                )
        data["stateful_world"] = {
            "max_hp": MAX_HP,
            "weapons": weapons,
            "monsters": [
                {
                    "id": monster_id,
                    "monster_type": monster_type,
                    "required_weapon": MONSTER_REQUIRES[monster_type],
                    "cell": list(cell),
                }
                for monster_id, monster_type, cell in template.monsters
            ],
        }
    return data


def run_clean(data: dict[str, Any]) -> dict[str, Any]:
    return run_experiment(
        data,
        ScriptedBackend(),
        mode="clean",
        fault_id=None,
        planning_granularity="chunk",
        chunk_size=3,
    )


def stage_steps(result: dict[str, Any]) -> dict[tuple[int, str], int]:
    """Count executed actions per stage and agent in a scripted run."""
    counts: dict[tuple[int, str], int] = {}
    stage = 0
    for event in result["attempts"][0]["events"]:
        if event["event_type"] == "round_started":
            stage = int(str(event["phase"]).rsplit("-", 1)[1]) - 1
        elif event["event_type"] == "action_executed":
            key = (stage, event["agent"])
            counts[key] = counts.get(key, 0) + 1
    return counts


def first_transmission_stage(
    clean: dict[str, Any], source: str, fact_suffix: str
) -> str | None:
    for event in clean["attempts"][0]["events"]:
        if event["event_type"] == "message_delivered" and event["sender"] == source:
            if any(item.endswith(fact_suffix) for item in event["source_fact_ids"]):
                return str(event["phase"])
    return None


def classify(data: dict[str, Any], fault: dict[str, Any]) -> dict[str, Any] | None:
    """Return the deterministic preflight check for one candidate fault."""
    candidate = {**data, "faults": [fault]}
    try:
        validate_map(candidate)
        check = run_scheduler_preflight(
            candidate, planning_granularity="chunk", chunk_size=3
        )["fault_checks"][0]
    except (MapValidationError, EpisodeError):
        return None
    if not (check["fault_injected"] and check["phase_matches"]):
        return None
    return check


def harm_stage(data: dict[str, Any], fault: dict[str, Any]) -> str | None:
    result = run_experiment(
        {**data, "faults": [fault]},
        ScriptedBackend(),
        mode="no-recovery",
        fault_id=fault["id"],
        planning_granularity="chunk",
        chunk_size=3,
    )
    stage = None
    for event in result["attempts"][0]["events"]:
        if event["event_type"] == "round_started":
            stage = event["phase"]
        elif event["event_type"] in {"trap_triggered", "monster_damage_taken"}:
            return stage
    return None


# ---------------------------------------------------------------------------
# Fault derivation
# ---------------------------------------------------------------------------


def stage_index(stage_id: str) -> int:
    return int(stage_id.rsplit("-", 1)[1]) - 1


def hazard_faults(
    template: Template, data: dict[str, Any], clean: dict[str, Any]
) -> tuple[list[dict[str, Any]], str | None]:
    traps = {kind: {tuple(cell) for cell in cells} for kind, cells in data["hazards"].items()}
    all_traps = traps["fire"] | traps["water"]
    segments = stage_segments(template)

    def relevant_traps(recipient: str) -> list[tuple[int, Cell]]:
        """Partner-sensed traps on some shortest route that ignores them."""
        partner_kind = SENSOR[PARTNER[recipient]]
        own = traps[SENSOR[recipient]]
        found = []
        for index, segment in enumerate(segments):
            start, target = segment[recipient]
            for cell in sorted(traps[partner_kind]):
                if on_some_shortest_path(template.rows, start, target, cell, own):
                    found.append((index, cell))
        return found

    def source_stage(source: str, kind: str, cell: Cell) -> str | None:
        return first_transmission_stage(clean, source, f"{kind}:{cell[0]}:{cell[1]}")

    def ordered(recipient: str) -> list[tuple[str, Cell, int]]:
        source = PARTNER[recipient]
        kind = SENSOR[source]
        options = []
        for index, cell in relevant_traps(recipient):
            stage_id = source_stage(source, kind, cell)
            if stage_id is None:
                continue
            delay = index - stage_index(stage_id)
            if template.cross_stage_faults and delay < 1:
                continue
            options.append((stage_id, cell, index, delay))
        if template.cross_stage_faults:
            # Prefer the longest propagation from report to first use.
            options.sort(key=lambda item: -item[3])
        return [item[:3] for item in options]

    def derive_relevant(
        cls: str, recipient: str, avoid: set[Cell]
    ) -> dict[str, Any] | None:
        source = PARTNER[recipient]
        kind = SENSOR[source]
        for stage_id, cell, index in ordered(recipient):
            if cell in avoid:
                continue
            start, target = segments[index][recipient]
            safe = shortest_path(template.rows, start, target, all_traps) or []
            if cls == "accidental":
                rules = [{"type": "fixed_offset", "delta": list(delta)} for delta in BASIC_OFFSETS]
            else:
                naive = set(
                    shortest_path(template.rows, start, target, traps[SENSOR[recipient]]) or []
                )
                decoys = [c for c in safe[1:-1] if c not in naive and c != cell]
                rules = [{"type": "route_decoy", "cell": list(c)} for c in decoys]
            for rule in rules:
                prefix = "f" if cls == "accidental" else "a"
                label = "offset" if cls == "accidental" else "decoy"
                fault = {
                    "id": f"{prefix}-{kind}-{stage_id}-{label}",
                    "class": cls,
                    "phase": stage_id,
                    "source_agent": source,
                    "transmission_occurrence": 1,
                    "kind": "coordinate_shift",
                    "hazard": kind,
                    "true_cell": list(cell),
                    "selection_rule": rule,
                }
                check = classify(data, fault)
                if check is None or check["operational_classification"] != "outcome_relevant":
                    continue
                if template.cross_stage_faults and harm_stage(data, fault) == stage_id:
                    continue
                fault["expected_effect"] = (
                    f"{recipient} receives a corrupted {kind} report in {stage_id} and "
                    f"may enter the true {kind} trap at {list(cell)}"
                    + (" in a later stage." if template.cross_stage_faults else ".")
                )
                return fault
        return None

    def derive_irrelevant() -> dict[str, Any] | None:
        for recipient in AGENTS:
            source = PARTNER[recipient]
            kind = SENSOR[source]
            relevant = {cell for _, cell in relevant_traps(recipient)}
            for cell in sorted(traps[kind], key=lambda item: (item in relevant, item)):
                stage_id = source_stage(source, kind, cell)
                if stage_id is None:
                    continue
                fault = {
                    "id": f"f-{kind}-{stage_id}-drop-once",
                    "class": "accidental",
                    "phase": stage_id,
                    "source_agent": source,
                    "transmission_occurrence": 1,
                    "kind": "omission",
                    "hazard": kind,
                    "true_cell": list(cell),
                    "selection_rule": {"type": "drop_once"},
                }
                check = classify(data, fault)
                if check is None or check["operational_classification"] != "non_propagating":
                    continue
                fault["expected_effect"] = (
                    f"The first {kind} report about {list(cell)} is dropped and "
                    "retransmitted unchanged; no route should change."
                )
                return fault
        return None

    faults: list[dict[str, Any]] = []
    recipients = ["P1", "P2"]
    for cls in ("accidental", "adversarial"):
        # Prefer a different fact from the accidental fault when one exists.
        used = {tuple(item["true_cell"]) for item in faults}
        fault = next(
            (
                item
                for avoid in (used, set())
                for item in (derive_relevant(cls, r, avoid) for r in recipients)
                if item
            ),
            None,
        )
        if fault is None:
            return [], f"no {cls} relevant fault"
        faults.append(fault)
        # Prefer the opposite message direction for the adversarial fault:
        # the accidental fault's source becomes the preferred recipient.
        recipients = [fault["source_agent"], PARTNER[fault["source_agent"]]]
    irrelevant = derive_irrelevant()
    if irrelevant is None:
        return [], "no irrelevant control"
    faults.append(irrelevant)
    return faults, None


def weapon_faults(
    template: Template, data: dict[str, Any], clean: dict[str, Any]
) -> tuple[list[dict[str, Any]], str | None]:
    weapons = data["stateful_world"]["weapons"]
    monsters = data["stateful_world"]["monsters"]

    def stage_of(weapon: dict[str, Any]) -> str | None:
        cell = weapon["cell"]
        return first_transmission_stage(
            clean,
            weapon["sensed_by"],
            f"weapon:{weapon['weapon_type']}:{cell[0]}:{cell[1]}",
        )

    def base(weapon: dict[str, Any], stage_id: str) -> dict[str, Any]:
        return {
            "phase": stage_id,
            "source_agent": weapon["sensed_by"],
            "transmission_occurrence": 1,
            "fact_type": "weapon",
            "object_type": weapon["weapon_type"],
            "true_cell": list(weapon["cell"]),
        }

    def derive_relevant(cls: str, weapon_type: str) -> dict[str, Any] | None:
        recipient = PARTNER[WEAPON_SENSOR[weapon_type]]
        accepted = template.accidental_classes if cls == "accidental" else (
            "executed_with_harm",
        )
        candidates = []
        for weapon in [w for w in weapons if w["weapon_type"] == weapon_type]:
            stage_id = stage_of(weapon)
            if stage_id is None:
                continue
            true_cell = tuple(weapon["cell"])
            if cls == "accidental":
                rules = [
                    {"type": "fixed_offset", "delta": list(delta)}
                    for delta in WEAPON_OFFSETS
                    if chebyshev(true_cell, (true_cell[0] + delta[0], true_cell[1] + delta[1]))
                    >= 3
                ]
            else:
                rules = [
                    {"type": "route_decoy", "cell": monster["cell"]}
                    for monster in monsters
                    if monster["required_weapon"] == weapon_type
                ]
            for rule in rules:
                fault = {
                    "id": (
                        f"f-{weapon['id']}-offset"
                        if cls == "accidental"
                        else f"a-{weapon['id']}-monster-decoy"
                    ),
                    "class": cls,
                    **base(weapon, stage_id),
                    "kind": "coordinate_shift",
                    "selection_rule": rule,
                }
                check = classify(data, fault)
                if check is None or check["operational_classification"] not in accepted:
                    continue
                harmful = check["operational_classification"] == "executed_with_harm"
                if (
                    harmful
                    and template.cross_stage_faults
                    and harm_stage(data, fault) == stage_id
                ):
                    continue
                rank = accepted.index(check["operational_classification"])
                candidates.append((rank, len(candidates), fault))
        if not candidates:
            return None
        # Prefer the most harmful accepted class, then the first candidate.
        fault = min(candidates)[2]
        rule = fault["selection_rule"]
        fault["expected_effect"] = (
            f"{recipient} is told the {weapon_type} at {fault['true_cell']} is "
            + (
                f"offset by {rule['delta']}"
                if cls == "accidental"
                else f"on the monster cell {rule['cell']}"
            )
            + " and may lose HP crossing an unbeaten monster or waste steps."
        )
        return fault

    def derive_irrelevant() -> dict[str, Any] | None:
        for weapon in weapons:
            stage_id = stage_of(weapon)
            if stage_id is None:
                continue
            fault = {
                "id": f"f-{weapon['id']}-drop-once",
                "class": "accidental",
                **base(weapon, stage_id),
                "kind": "omission",
                "selection_rule": {"type": "drop_once"},
            }
            check = classify(data, fault)
            if check is None or check["operational_classification"] != "non_propagating":
                continue
            fault["expected_effect"] = (
                f"The first report of {weapon['id']} is dropped and retransmitted "
                "unchanged before it is needed; no HP should be lost."
            )
            return fault
        return None

    faults = []
    weapon_types = ["ice_sword", "fire_sword"]
    for cls in ("accidental", "adversarial"):
        fault = next(
            (item for item in (derive_relevant(cls, w) for w in weapon_types) if item), None
        )
        if fault is None:
            return [], f"no {cls} relevant fault"
        faults.append(fault)
        # Prefer the opposite message direction for the adversarial fault.
        used = fault["object_type"]
        weapon_types = [w for w in ("ice_sword", "fire_sword") if w != used] + [used]
    irrelevant = derive_irrelevant()
    if irrelevant is None:
        return [], "no irrelevant control"
    faults.append(irrelevant)
    return faults, None


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------


def structural_checks(template: Template, data: dict[str, Any]) -> str | None:
    rows = template.rows
    if len(rows[0]) > 11 or len(rows) > 9:
        return "map exceeds 11 x 9"
    segments = stage_segments(template)
    if template.suite == "basic":
        traps = {kind: {tuple(cell) for cell in cells} for kind, cells in data["hazards"].items()}
        all_traps = traps["fire"] | traps["water"]
        stage_needs: dict[int, bool] = {}
        for agent in AGENTS:
            partner_kind = SENSOR[PARTNER[agent]]
            needs_partner = False
            for index, segment in enumerate(segments):
                start, target = segment[agent]
                own = traps[SENSOR[agent]]
                informed = distance(rows, start, target, all_traps)
                uninformed = distance(rows, start, target, own)
                if informed is None or uninformed is None:
                    return f"{agent} has no route"
                if informed - uninformed > MAX_TRAP_DETOUR:
                    return f"{agent} safe detour exceeds {MAX_TRAP_DETOUR}"
                if any(
                    on_some_shortest_path(rows, start, target, cell, own)
                    for cell in traps[partner_kind]
                ):
                    needs_partner = True
                    stage_needs[index] = True
            if not needs_partner:
                return f"{agent} never needs a partner hazard report"
        for index in template.report_stages:
            if not stage_needs.get(index):
                return f"stage {index + 1} needs no partner hazard report"
        return None

    monsters = {tuple(m["cell"]): m for m in data["stateful_world"]["monsters"]}
    weapons = data["stateful_world"]["weapons"]
    for agent in AGENTS:
        carried = CARRIED_WEAPON[agent]
        guards = {cell for cell, m in monsters.items() if m["required_weapon"] == carried}
        start, goal = template.starts[agent], template.goals[agent]
        through = distance(rows, start, goal, set())
        around = distance(rows, start, goal, set(monsters))
        if template.avoidable_monster:
            if around is None or around - through < 4:
                return f"{agent} monster detour is too cheap to need a sword"
        elif around is not None:
            return f"{agent} can reach its goal without passing a monster"
        if not guards:
            return f"{agent} has no monster to defeat"
        # A sword that lies on a shortest route would be collected without any
        # report, making communication unnecessary.
        for weapon in weapons:
            if weapon["weapon_type"] != carried:
                continue
            for segment_start, target in (segment[agent] for segment in segments):
                if on_some_shortest_path(
                    rows, segment_start, target, tuple(weapon["cell"]), set()
                ):
                    return f"{weapon['id']} lies on a shortest route of {agent}"
    # Exercise re-planning when a chunk reaches a monster its fighter has not
    # yet seen.
    if template.hidden_monster and not any(
        chebyshev(cell, template.starts[agent]) > OBSERVATION_RADIUS
        for cell, monster in monsters.items()
        for agent in AGENTS
        if monster["required_weapon"] == CARRIED_WEAPON[agent]
    ):
        return "every monster is visible to its fighter at the start"
    return None


def clean_checks(template: Template, clean: dict[str, Any]) -> str | None:
    if not clean["final_success"]:
        return "scripted clean run fails"
    if clean["trap_entries"] or clean["damage_taken"]:
        return "scripted clean run takes damage"
    if template.suite == "extended":
        monster_count = len(template.monsters)
        if clean["monsters_defeated"] != monster_count:
            return "scripted clean run does not defeat every monster"
    return None


def restart_checks(data: dict[str, Any], faults: list[dict[str, Any]]) -> str | None:
    """Both forked restarts must complete for every relevant fault."""
    for fault in faults:
        if fault["kind"] == "omission":
            continue
        for trigger in ("delay", "after-fault-linked-harm"):
            result = run_experiment(
                data,
                ScriptedBackend(),
                mode="full-restart",
                fault_id=fault["id"],
                planning_granularity="chunk",
                chunk_size=3,
                detection_delay_ticks=0 if trigger == "delay" else None,
                detection_trigger=trigger,
                recovery_scope="both",
            )
            by_scope = result["paired_baselines"]["full_restart_by_scope"]
            reset = by_scope["environment-reset"]
            if not (reset["success"] and reset["full_health_success"]):
                return f"{fault['id']} environment reset does not recover ({trigger})"
            if data["schema_version"] >= 5 and not by_scope["agent-only"]["success"]:
                return f"{fault['id']} agent-only restart does not finish ({trigger})"
    return None


def attempt(template: Template, seed: int) -> tuple[dict[str, Any] | None, str]:
    try:
        return evaluate_seed(template, seed)
    except (MapValidationError, EpisodeError, ValueError) as exc:
        # A scripted episode that raises is a rejected candidate, not a bug.
        return None, f"{type(exc).__name__}: {str(exc).split(':')[0]}"


def evaluate_seed(template: Template, seed: int) -> tuple[dict[str, Any] | None, str]:
    rng = random.Random(f"{template.name}:{seed}")
    try:
        placement = sample_placement(template, rng)
        probe = build_map(template, placement)
        validate_map(probe)
    except (MapValidationError, ValueError) as exc:
        return None, f"invalid: {exc}"
    reason = structural_checks(template, probe)
    if reason:
        return None, reason
    clean = run_clean(probe)
    reason = clean_checks(template, clean)
    if reason:
        return None, reason
    steps = stage_steps(clean)
    budgets = {
        (index, agent): max(12, 2 * steps.get((index, agent), 0) + 6)
        for index in range(len(template.stages))
        for agent in AGENTS
    }
    data = build_map(template, placement, budgets=budgets)
    try:
        validate_map(data)
    except MapValidationError as exc:
        return None, f"invalid budget: {exc}"
    clean = run_clean(data)
    reason = clean_checks(template, clean)
    if reason:
        return None, f"with budgets: {reason}"
    derive = hazard_faults if template.suite == "basic" else weapon_faults
    faults, reason = derive(template, data, clean)
    if reason:
        return None, reason
    data["faults"] = faults
    validate_map(data)
    preflight = run_scheduler_preflight(data, planning_granularity="chunk", chunk_size=3)
    if not preflight["passed"]:
        return None, "final preflight fails"
    reason = restart_checks(data, faults)
    if reason:
        return None, reason
    data["generator"] = {
        "script": "scripts/generate_evaluation_maps.py",
        "version": GENERATOR_VERSION,
        "template": template.name,
        "suite": template.suite,
        "size_class": template.size_class,
        "seed": seed,
        "expected_classifications": {
            item["fault_id"]: item["operational_classification"]
            for item in preflight["fault_checks"]
        },
        "scripted_clean_actions": clean["executed_actions"],
    }
    return data, "accepted"


def generate(template: Template) -> tuple[dict[str, Any] | None, dict[str, int]]:
    """Return the first accepted seed's map and the rejection reasons before it."""
    rejections: dict[str, int] = {}
    for seed in range(MAX_SEEDS):
        data, reason = attempt(template, seed)
        if data is not None:
            data["generator"]["rejected_seeds"] = sum(rejections.values())
            return data, rejections
        rejections[reason] = rejections.get(reason, 0) + 1
    return None, rejections


def map_filename(template: Template) -> str:
    return f"{template.name.replace('-', '_')}.json"


def serialize(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2) + "\n"


def render(data: dict[str, Any]) -> str:
    grid = [list(row) for row in data["public_rows"]]

    def mark(cell: list[int], symbol: str) -> None:
        grid[cell[1]][cell[0]] = symbol

    for cell in data["hazards"]["fire"]:
        mark(cell, "F")
    for cell in data["hazards"]["water"]:
        mark(cell, "W")
    for weapon in data.get("stateful_world", {}).get("weapons", []):
        mark(weapon["cell"], "f" if weapon["weapon_type"] == "fire_sword" else "i")
    for monster in data.get("stateful_world", {}).get("monsters", []):
        mark(monster["cell"], "M" if monster["monster_type"] == "fire_monster" else "N")
    for agent in AGENTS:
        mark(data["agents"][agent]["goal"], "G")
    starts = data["agents"]
    if starts["P1"]["start"] == starts["P2"]["start"]:
        mark(starts["P1"]["start"], "S")
    else:
        mark(starts["P1"]["start"], "1")
        mark(starts["P2"]["start"], "2")
    return "\n".join("".join(row) for row in grid)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", nargs="*", help="template names to generate")
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate and compare with the files in --output instead of writing",
    )
    args = parser.parse_args()
    selected = [t for t in TEMPLATES if not args.only or t.name in args.only]
    if not args.check:
        args.output.mkdir(parents=True, exist_ok=True)
    failed = False
    for template in selected:
        data, rejections = generate(template)
        if data is None:
            failed = True
            print(f"FAILED {template.name}: {rejections}")
            continue
        path = args.output / map_filename(template)
        if args.check:
            matches = path.is_file() and path.read_text() == serialize(data)
            failed |= not matches
            print(f"{template.name}: {'matches' if matches else 'DIFFERS from'} {path}")
            continue
        path.write_text(serialize(data))
        print(f"{template.name}: seed {data['generator']['seed']} -> {path.name}")
        print(render(data))
        for fault in data["faults"]:
            print(
                f"  {fault['id']}: "
                f"{data['generator']['expected_classifications'][fault['id']]}"
            )
        if rejections:
            print(f"  rejected: {rejections}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
