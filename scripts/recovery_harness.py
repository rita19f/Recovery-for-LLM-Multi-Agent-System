#!/usr/bin/env python3
"""Minimal shared-world multi-agent recovery testbed.

The module provides a deterministic simulator, a scripted backend, an
OpenAI-compatible LLM backend, controlled message-fault injection, and
append-only artifacts/events suitable for clean, no-recovery, and full-restart
smoke experiments.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Iterable


AGENTS = ("P1", "P2")
PARTNER = {"P1": "P2", "P2": "P1"}
MOVES = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}
HTTP_USER_AGENT = "CSE291A-Grid-Recovery/0.1 (OpenAI-compatible client)"
MAX_PLAN_REPAIRS = 2
LOOP_REPETITIONS = 3
MAX_LOOP_PERIOD = 2


def elapsed_ms(start_ns: int) -> float:
    """Return monotonic elapsed time in milliseconds at logging precision."""
    return round((perf_counter_ns() - start_ns) / 1_000_000, 3)


BEARER_TOKEN_PATTERN = re.compile(r"Bearer\s+[^\s'\"\\]+")
OPENAI_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{8,}")


def redact_secrets(text: str, secrets: Iterable[str] = ()) -> str:
    """Remove API credentials from text before it is printed or persisted."""
    for secret in secrets:
        secret = (secret or "").strip()
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = BEARER_TOKEN_PATTERN.sub("Bearer [REDACTED]", text)
    return OPENAI_KEY_PATTERN.sub("[REDACTED]", text)


class MapValidationError(ValueError):
    """Raised when an environment map is invalid."""


class EpisodeError(RuntimeError):
    """Raised when an agent response cannot be safely interpreted."""


class LLMResponseParseError(EpisodeError):
    """Raised when an LLM response is text but cannot be parsed as JSON."""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise EpisodeError("LLM response must be a JSON object")
    return value


def normalize_chat_content(content: Any) -> str | None:
    """Normalize common OpenAI-compatible chat content representations.

    Most gateways return a string, while some return typed content blocks.  A
    reasoning-only response deliberately returns ``None`` here: reasoning text
    is not a substitute for the requested final JSON and must not be logged as
    model output.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts) if parts else None


def chat_response_shape(raw: Any) -> dict[str, Any]:
    """Return a diagnostic response summary without storing hidden reasoning."""
    if not isinstance(raw, dict):
        return {"top_level_type": type(raw).__name__}
    summary: dict[str, Any] = {
        "top_level_type": "object",
        "top_level_keys": sorted(raw),
    }
    choices = raw.get("choices")
    if not isinstance(choices, list):
        summary["choices_type"] = type(choices).__name__
        return summary
    summary["choice_count"] = len(choices)
    if not choices or not isinstance(choices[0], dict):
        return summary
    choice = choices[0]
    summary["choice_keys"] = sorted(choice)
    summary["finish_reason"] = choice.get("finish_reason")
    message = choice.get("message")
    if not isinstance(message, dict):
        summary["message_type"] = type(message).__name__
        return summary
    summary["message_keys"] = sorted(message)
    summary["content_type"] = type(message.get("content")).__name__
    reasoning = message.get("reasoning_content")
    summary["has_reasoning_content"] = reasoning is not None
    if isinstance(reasoning, str):
        summary["reasoning_content_chars"] = len(reasoning)
    return summary


def parse_sse_chat_response(body: bytes) -> dict[str, Any]:
    """Collapse an OpenAI-compatible SSE chat stream into one response object."""
    content_parts: list[str] = []
    finish_reason: str | None = None
    usage: dict[str, Any] = {}
    event_count = 0
    for raw_line in body.decode(errors="replace").splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError as exc:
            raise EpisodeError(f"invalid JSON in LLM SSE event: {data[:300]!r}") from exc
        event_count += 1
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        choices = event.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or choice.get("message") or {}
        part = delta.get("content")
        if isinstance(part, str):
            content_parts.append(part)
        if choice.get("finish_reason") is not None:
            finish_reason = choice["finish_reason"]
    if event_count == 0:
        raise EpisodeError("LLM streaming response contained no SSE data events")
    return {
        "choices": [
            {
                "message": {"content": "".join(content_parts)},
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }


def find_one(rows: list[str], symbol: str) -> tuple[int, int]:
    matches = [
        (x, y)
        for y, row in enumerate(rows)
        for x, value in enumerate(row)
        if value == symbol
    ]
    if len(matches) != 1:
        raise MapValidationError(
            f"expected exactly one {symbol!r}, found {len(matches)}"
        )
    return matches[0]


def as_cell(value: Iterable[int]) -> tuple[int, int]:
    items = tuple(value)
    if len(items) != 2 or not all(isinstance(item, int) for item in items):
        raise ValueError(f"invalid cell: {value!r}")
    return items[0], items[1]


def agent_cell(data: dict[str, Any], agent: str, key: str) -> tuple[int, int]:
    """Return a configured agent start or goal cell."""
    configured = data["agents"][agent].get(key)
    if configured is None:
        raise MapValidationError(f"{agent} must define {key}")
    return as_cell(configured)


def agent_actions(data: dict[str, Any], key: str, agent: str) -> list[str]:
    """Read a per-agent deterministic action sequence."""
    configured = data.get(key, {})
    if not isinstance(configured, dict):
        raise MapValidationError(f"{key} must define per-agent action lists")
    actions = configured.get(agent, [])
    if not isinstance(actions, list):
        raise MapValidationError(f"{key} for {agent} must be a list of actions")
    return actions


def phase_cell(phase: dict[str, Any], key: str, agent: str) -> tuple[int, int]:
    """Read a phase cell shared by both agents or configured per agent."""
    configured = phase[key]
    if isinstance(configured, dict):
        if set(configured) != set(AGENTS):
            raise MapValidationError(
                f"phase {phase.get('id', '<unknown>')} {key} must define P1 and P2"
            )
        configured = configured[agent]
    return as_cell(configured)


def phase_routes(phase: dict[str, Any], agent: str) -> list[list[str]]:
    """Read shared or per-agent reference routes."""
    configured = phase.get("reference_routes")
    if isinstance(configured, dict):
        if set(configured) != set(AGENTS):
            raise MapValidationError(
                f"phase {phase.get('id', '<unknown>')} reference_routes must define P1 and P2"
            )
        configured = configured[agent]
    if not isinstance(configured, list):
        raise MapValidationError(
            f"phase {phase.get('id', '<unknown>')} reference_routes must be a list"
        )
    return configured


def phase_for_agent(phase: dict[str, Any], agent: str) -> dict[str, Any]:
    """Normalize one shared phase specification for an agent-facing planner."""
    normalized = dict(phase)
    normalized["start"] = list(phase_cell(phase, "start", agent))
    normalized["target"] = list(phase_cell(phase, "target", agent))
    normalized["reference_routes"] = phase_routes(phase, agent)
    max_actions = phase.get("max_actions")
    if isinstance(max_actions, dict):
        if set(max_actions) != set(AGENTS):
            raise MapValidationError(
                f"phase {phase.get('id', '<unknown>')} max_actions must define P1 and P2"
            )
        max_actions = max_actions[agent]
    if not isinstance(max_actions, int) or max_actions < 1:
        raise MapValidationError(
            f"phase {phase.get('id', '<unknown>')} max_actions must be a positive integer"
        )
    normalized["max_actions"] = max_actions
    return normalized


def detect_navigation_loop(
    cells: list[list[int]],
    *,
    repetitions: int = LOOP_REPETITIONS,
    max_period: int = MAX_LOOP_PERIOD,
) -> dict[str, Any] | None:
    """Return a compact description of a repeated terminal position cycle.

    This is a failure detector, not a navigation policy: it never rejects or
    replaces an action. Periods one and two cover repeated invalid stays and
    the common A-B-A-B oscillation while leaving ordinary backtracking legal.
    """
    normalized = [as_cell(cell) for cell in cells]
    for period in range(1, max_period + 1):
        required = period * repetitions
        if len(normalized) < required:
            continue
        tail = normalized[-required:]
        pattern = tail[:period]
        if tail == pattern * repetitions:
            return {
                "period": period,
                "repetitions": repetitions,
                "cycle_cells": [list(cell) for cell in pattern],
            }
    return None


def trap_cells(data: dict[str, Any]) -> dict[str, set[tuple[int, int]]]:
    return {
        hazard: {as_cell(cell) for cell in cells}
        for hazard, cells in data["hazards"].items()
    }


def tile_at(data: dict[str, Any], cell: tuple[int, int]) -> str:
    x, y = cell
    return data["public_rows"][y][x]


def in_bounds(data: dict[str, Any], cell: tuple[int, int]) -> bool:
    x, y = cell
    return 0 <= x < data["dimensions"]["width"] and 0 <= y < data["dimensions"]["height"]


def validate_map(data: dict[str, Any]) -> dict[str, Any]:
    schema_version = data.get("schema_version")
    if schema_version not in {3, 4}:
        raise MapValidationError("environment requires schema_version 3 or 4")
    if data.get("world_model") != "shared":
        raise MapValidationError("shared-world schemas require world_model='shared'")
    width = data["dimensions"]["width"]
    height = data["dimensions"]["height"]
    rows = data["public_rows"]
    if len(rows) != height or any(len(row) != width for row in rows):
        raise MapValidationError("public_rows do not match declared dimensions")
    unknown = set("".join(rows)) - {"#", ".", "S", "G", "D", "T"}
    if unknown:
        raise MapValidationError(f"unknown public tiles: {sorted(unknown)}")
    if set(data["agents"]) != set(AGENTS):
        raise MapValidationError("agents must be exactly P1 and P2")
    if data["agents"]["P1"]["sensor"] != "fire":
        raise MapValidationError("P1 must use the fire sensor in the environment")
    if data["agents"]["P2"]["sensor"] != "water":
        raise MapValidationError("P2 must use the water sensor in the environment")

    starts = {agent: agent_cell(data, agent, "start") for agent in AGENTS}
    goals = {agent: agent_cell(data, agent, "goal") for agent in AGENTS}
    for agent in AGENTS:
        for label, cell in (("start", starts[agent]), ("goal", goals[agent])):
            if not in_bounds(data, cell) or tile_at(data, cell) == "#":
                raise MapValidationError(f"{agent} {label} {cell} is not on open terrain")

    traps = trap_cells(data)
    if set(traps) != {"fire", "water"}:
        raise MapValidationError("hazards must define exactly fire and water")
    flat_traps = set().union(*traps.values())
    if len(flat_traps) != sum(len(cells) for cells in traps.values()):
        raise MapValidationError("fire and water traps may not overlap")
    for hazard, cells in traps.items():
        for cell in cells:
            if not in_bounds(data, cell) or tile_at(data, cell) == "#":
                raise MapValidationError(f"{hazard} trap {cell} is not on open terrain")
            if cell in set(starts.values()) | set(goals.values()):
                raise MapValidationError(f"{hazard} trap may not occupy an agent start/goal")

    phases = data["phases"]
    if not phases:
        raise MapValidationError("at least one phase is required")
    cursors: dict[str, tuple[int, int]] = {}
    for agent in AGENTS:
        prelude = replay_public_actions(
            data, starts[agent], agent_actions(data, "prelude_actions", agent)
        )
        if not prelude["valid"]:
            raise MapValidationError(
                f"invalid prelude_actions for {agent}: {prelude['error']}"
            )
        cursors[agent] = prelude["position"]
    phase_summaries = []
    for phase in phases:
        agent_summaries: dict[str, dict[str, Any]] = {}
        for agent in AGENTS:
            normalized_phase = phase_for_agent(phase, agent)
            phase_start = as_cell(normalized_phase["start"])
            phase_target = as_cell(normalized_phase["target"])
            if cursors[agent] != phase_start:
                raise MapValidationError(
                    f"phase {phase['id']} for {agent} starts at {phase_start}, but it "
                    f"arrives at {cursors[agent]}"
                )
            reference_routes = normalized_phase["reference_routes"]
            if not reference_routes:
                raise MapValidationError(
                    f"phase {phase['id']} for {agent} must define at least one reference route"
                )
            route_summaries = []
            safe_route_indexes = []
            for route_index, actions in enumerate(reference_routes):
                if not isinstance(actions, list) or not actions:
                    raise MapValidationError(
                        f"phase {phase['id']} {agent} reference route {route_index} "
                        "must contain actions"
                    )
                if len(actions) > normalized_phase["max_actions"]:
                    raise MapValidationError(
                        f"phase {phase['id']} {agent} reference route {route_index} "
                        "exceeds max_actions"
                    )
                replay = replay_public_actions(data, phase_start, actions)
                if not replay["valid"] or replay["position"] != phase_target:
                    raise MapValidationError(
                        f"phase {phase['id']} {agent} reference route {route_index} does "
                        f"not reach target: {replay['error'] or replay['position']}"
                    )
                safe = not any(as_cell(cell) in flat_traps for cell in replay["trace"])
                if safe:
                    safe_route_indexes.append(route_index)
                route_summaries.append(
                    {
                        "route_index": route_index,
                        "actions": actions,
                        "cells": replay["trace"],
                        "safe_under_ground_truth": safe,
                    }
                )
            if not safe_route_indexes:
                raise MapValidationError(
                    f"phase {phase['id']} for {agent} has no trap-safe reference route"
                )
            oracle_route_index = safe_route_indexes[0]
            oracle = reference_routes[oracle_route_index]
            agent_summaries[agent] = {
                "start": list(phase_start),
                "target": list(phase_target),
                "oracle_route_index": oracle_route_index,
                "oracle_actions": oracle,
                "steps": len(oracle),
                "reference_routes": route_summaries,
            }
            cursors[agent] = phase_target
        per_agent_phase = any(
            isinstance(phase.get(key), dict)
            for key in ("start", "target", "reference_routes", "max_actions")
        )
        if per_agent_phase:
            phase_summaries.append(
                {
                    "id": phase["id"],
                    "steps": {
                        agent: agent_summaries[agent]["steps"] for agent in AGENTS
                    },
                    "agents": agent_summaries,
                }
            )
        else:
            phase_summaries.append({"id": phase["id"], **agent_summaries["P1"]})
    for agent in AGENTS:
        finale = replay_public_actions(
            data, cursors[agent], agent_actions(data, "finale_actions", agent)
        )
        if not finale["valid"] or finale["position"] != goals[agent]:
            raise MapValidationError(
                f"finale_actions for {agent} do not reach goal {goals[agent]}"
            )

    fault_ids: set[str] = set()
    phase_ids = {phase["id"] for phase in phases}
    for fault in data.get("faults", []):
        if fault["id"] in fault_ids:
            raise MapValidationError(f"duplicate fault id: {fault['id']}")
        fault_ids.add(fault["id"])
        if fault["phase"] not in phase_ids:
            raise MapValidationError(f"fault {fault['id']} names unknown phase")
        if fault["source_agent"] not in AGENTS:
            raise MapValidationError(f"fault {fault['id']} names unknown source agent")
        occurrence = fault.get("transmission_occurrence", 1)
        if not isinstance(occurrence, int) or occurrence < 1:
            raise MapValidationError(
                f"fault {fault['id']} transmission_occurrence must be positive"
            )
        true_cell = as_cell(fault["true_cell"])
        if true_cell not in traps[fault["hazard"]]:
            raise MapValidationError(
                f"fault {fault['id']} true_cell is not a {fault['hazard']} trap"
            )
        if fault["kind"] == "coordinate_shift":
            faulty_cell = resolve_faulty_cell(fault)
            if not in_bounds(data, faulty_cell):
                raise MapValidationError(f"fault {fault['id']} faulty_cell out of bounds")
            if faulty_cell == true_cell:
                raise MapValidationError(
                    f"fault {fault['id']} coordinate shift must change the cell"
                )
            if data.get("schema_version", 0) >= 4 and tile_at(data, faulty_cell) == "#":
                raise MapValidationError(
                    f"fault {fault['id']} selection rule resolves to a public wall"
                )
            rule = fault.get("selection_rule")
            if data.get("schema_version", 0) >= 4 and not isinstance(rule, dict):
                raise MapValidationError(
                    f"fault {fault['id']} requires a selection_rule in schema v4"
                )
            if isinstance(rule, dict):
                rule_type = rule.get("type")
                expected_rule = (
                    "fixed_offset"
                    if fault.get("class") == "accidental"
                    else "route_decoy"
                    if fault.get("class") == "adversarial"
                    else None
                )
                if expected_rule is not None and rule_type != expected_rule:
                    raise MapValidationError(
                        f"fault {fault['id']} class {fault.get('class')} requires "
                        f"selection_rule.type={expected_rule}"
                    )
        elif fault["kind"] != "omission":
            raise MapValidationError(f"unsupported environment fault kind: {fault['kind']}")
        elif data.get("schema_version", 0) >= 4:
            rule = fault.get("selection_rule")
            if not isinstance(rule, dict) or rule.get("type") != "drop_once":
                raise MapValidationError(
                    f"fault {fault['id']} omission requires selection_rule.type=drop_once"
                )

    return {
        "map_id": data["id"],
        "world_model": data["world_model"],
        "starts": {agent: list(cell) for agent, cell in starts.items()},
        "goals": {agent: list(cell) for agent, cell in goals.items()},
        "fire_traps": [list(cell) for cell in sorted(traps["fire"])],
        "water_traps": [list(cell) for cell in sorted(traps["water"])],
        "phases": phase_summaries,
        "fault_ids": sorted(fault_ids),
        "passed": True,
    }


def replay_public_actions(
    data: dict[str, Any], start: tuple[int, int], actions: list[str]
) -> dict[str, Any]:
    position = start
    trace = []
    for index, action in enumerate(actions):
        if action not in MOVES:
            return {
                "valid": False,
                "position": position,
                "error": f"unknown {action}",
                "trace": trace,
            }
        dx, dy = MOVES[action]
        target = (position[0] + dx, position[1] + dy)
        if not in_bounds(data, target) or tile_at(data, target) == "#":
            return {
                "valid": False,
                "position": position,
                "error": f"blocked public move {action} at step {index}",
                "trace": trace,
            }
        position = target
        trace.append(list(position))
    return {"valid": True, "position": position, "error": None, "trace": trace}


def shortest_path(
    data: dict[str, Any],
    start: tuple[int, int],
    target: tuple[int, int],
    *,
    blocked: set[tuple[int, int]],
) -> list[str] | None:
    queue = deque([start])
    parents: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {start: None}
    for_current = (("UP", (0, -1)), ("LEFT", (-1, 0)), ("RIGHT", (1, 0)), ("DOWN", (0, 1)))
    while queue:
        current = queue.popleft()
        if current == target:
            actions: list[str] = []
            cursor = current
            while parents[cursor] is not None:
                previous, action = parents[cursor]  # type: ignore[misc]
                actions.append(action)
                cursor = previous
            return list(reversed(actions))
        for action, (dx, dy) in for_current:
            nxt = current[0] + dx, current[1] + dy
            if nxt in parents or nxt in blocked or not in_bounds(data, nxt):
                continue
            if tile_at(data, nxt) == "#":
                continue
            parents[nxt] = (current, action)
            queue.append(nxt)
    return None


def visible_public_cells(
    data: dict[str, Any], position: tuple[int, int]
) -> list[dict[str, Any]]:
    radius = data["observation_radius"]
    cells = []
    for y in range(position[1] - radius, position[1] + radius + 1):
        for x in range(position[0] - radius, position[0] + radius + 1):
            cell = (x, y)
            if in_bounds(data, cell):
                cells.append({"cell": [x, y], "tile": tile_at(data, cell)})
    return cells


def compact_public_view(public_cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Render a rectangular local observation with explicit x/y labels."""
    tiles = {as_cell(item["cell"]): item["tile"] for item in public_cells}
    xs = sorted({cell[0] for cell in tiles})
    ys = sorted({cell[1] for cell in tiles})
    return {
        "columns_x": xs,
        "rows": [
            {
                "y": y,
                "tiles": "".join(tiles.get((x, y), "?") for x in xs),
            }
            for y in ys
        ],
        "unknown_symbol": "?",
    }


def annotated_public_view(
    *,
    public_cells: list[dict[str, Any]],
    position: tuple[int, int],
    target: tuple[int, int],
    available_facts: list[dict[str, Any]],
    visited_cells: list[list[int]],
) -> dict[str, Any]:
    """Overlay navigation state on the public grid without hidden information."""
    tiles = {as_cell(item["cell"]): item["tile"] for item in public_cells}
    xs = sorted({cell[0] for cell in tiles})
    ys = sorted({cell[1] for cell in tiles})
    visited = {as_cell(cell) for cell in visited_cells}
    hazards = {
        as_cell(fact["cell"]): "F" if fact["hazard"] == "fire" else "W"
        for fact in available_facts
        if as_cell(fact["cell"]) in tiles
    }

    def symbol(cell: tuple[int, int]) -> str:
        if cell == position:
            return "@"
        if cell == target:
            return "T"
        if cell in hazards:
            return hazards[cell]
        if cell in visited and tiles.get(cell) != "#":
            return "v"
        return tiles.get(cell, "?")

    return {
        "columns_x": xs,
        "rows": [
            {"y": y, "cells": "".join(symbol((x, y)) for x in xs)}
            for y in ys
        ],
        "legend": {
            "@": "current agent cell",
            "T": "current segment target",
            "F": "known fire hazard",
            "W": "known water hazard",
            "v": "previously visited traversable cell",
            "#": "wall",
            ".": "unvisited traversable floor",
            "S": "global start",
            "G": "global goal",
            "?": "unobserved cell",
        },
    }


def public_transition_graph(public_cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """List legal public moves within the observed cells, without using hazards."""
    tiles = {as_cell(item["cell"]): item["tile"] for item in public_cells}
    transitions = []
    for cell in sorted(tiles, key=lambda item: (item[1], item[0])):
        if tiles[cell] == "#":
            continue
        moves = {}
        for action, (dx, dy) in MOVES.items():
            target = cell[0] + dx, cell[1] + dy
            if target in tiles and tiles[target] != "#":
                moves[action] = list(target)
        transitions.append({"from": list(cell), "moves": moves})
    return transitions


def shortest_progress_moves(
    *,
    position: tuple[int, int],
    target: tuple[int, int],
    transitions: dict[tuple[int, int], dict[str, list[int]]],
) -> dict[str, list[int]]:
    """Return legal next moves on a shortest public-topology path to target."""
    legal = transitions.get(position, {})
    if not legal:
        return {}
    distances: dict[tuple[int, int], int] = {target: 0}
    queue = deque([target])
    reverse_edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for source, moves in transitions.items():
        for raw_destination in moves.values():
            reverse_edges.setdefault(as_cell(raw_destination), []).append(source)
    while queue:
        current = queue.popleft()
        for previous in reverse_edges.get(current, []):
            if previous not in distances:
                distances[previous] = distances[current] + 1
                queue.append(previous)
    scored = {
        action: distances[as_cell(destination)]
        for action, destination in legal.items()
        if as_cell(destination) in distances
    }
    if not scored:
        return {}
    best = min(scored.values())
    return {
        action: destination
        for action, destination in legal.items()
        if scored.get(action) == best
    }


def candidate_continuations(
    *,
    candidate_moves: dict[str, list[int]],
    target: tuple[int, int],
    transitions: dict[tuple[int, int], dict[str, list[int]]],
) -> dict[str, dict[str, Any]]:
    """Compile each immediate choice through forced public-topology steps."""
    continuations: dict[str, dict[str, Any]] = {}
    for first_action, raw_destination in candidate_moves.items():
        position = as_cell(raw_destination)
        actions = [first_action]
        cells = [list(position)]
        while position != target:
            following = shortest_progress_moves(
                position=position,
                target=target,
                transitions=transitions,
            )
            if len(following) != 1:
                break
            action, raw_next = next(iter(following.items()))
            next_position = as_cell(raw_next)
            if list(next_position) in cells:
                break
            actions.append(action)
            cells.append(list(next_position))
            position = next_position
        continuations[first_action] = {
            "actions": actions,
            "cells": cells,
            "stops_at": list(position),
            "reaches_segment_target": position == target,
        }
    return continuations


def shortest_transition_route(
    *,
    start: tuple[int, int],
    target: tuple[int, int],
    transitions: dict[tuple[int, int], dict[str, list[int]]],
    blocked: set[tuple[int, int]],
) -> dict[str, list[Any]] | None:
    """Find a shortest route using only the supplied public transition graph."""
    if start in blocked or target in blocked:
        return None
    queue = deque([start])
    parents: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {
        start: None
    }
    while queue:
        current = queue.popleft()
        if current == target:
            actions: list[str] = []
            cells: list[list[int]] = []
            cursor = current
            while parents[cursor] is not None:
                previous, action = parents[cursor]  # type: ignore[misc]
                actions.append(action)
                cells.append(list(cursor))
                cursor = previous
            actions.reverse()
            cells.reverse()
            return {"actions": actions, "cells": cells}
        for action, raw_destination in transitions.get(current, {}).items():
            destination = as_cell(raw_destination)
            if destination in parents or destination in blocked:
                continue
            parents[destination] = (current, action)
            queue.append(destination)
    return None


def visible_progress_route(
    *,
    start: tuple[int, int],
    target: tuple[int, int],
    transitions: dict[tuple[int, int], dict[str, list[int]]],
    blocked: set[tuple[int, int]],
    visited: set[tuple[int, int]],
) -> dict[str, list[Any]] | None:
    """Route to a safe visible frontier when the segment target is still unseen.

    The choice uses only cells in the supplied observation graph. Manhattan
    distance is used as a target-direction heuristic, not as evidence that an
    unobserved cell is traversable.
    """
    if start in blocked:
        return None
    queue = deque([start])
    parents: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {
        start: None
    }
    depths = {start: 0}
    while queue:
        current = queue.popleft()
        for action, raw_destination in transitions.get(current, {}).items():
            destination = as_cell(raw_destination)
            if destination in parents or destination in blocked:
                continue
            parents[destination] = (current, action)
            depths[destination] = depths[current] + 1
            queue.append(destination)
    candidates = [cell for cell in parents if cell != start]
    if not candidates:
        return None
    destination = min(
        candidates,
        key=lambda cell: (
            abs(cell[0] - target[0]) + abs(cell[1] - target[1]),
            cell in visited,
            depths[cell],
            cell[1],
            cell[0],
        ),
    )
    actions: list[str] = []
    cells: list[list[int]] = []
    cursor = destination
    while parents[cursor] is not None:
        previous, action = parents[cursor]  # type: ignore[misc]
        actions.append(action)
        cells.append(list(cursor))
        cursor = previous
    actions.reverse()
    cells.reverse()
    return {"actions": actions, "cells": cells}


def decision_error(
    *,
    next_action: Any,
    candidate_moves: dict[str, list[int]],
    continuations: dict[str, dict[str, Any]] | None,
    available_facts: list[dict[str, Any]],
    visited_cells: list[list[int]],
) -> str | None:
    """Check one decision using public topology and currently known hazards."""
    if not isinstance(next_action, str):
        return "next_action must be a string"
    if next_action not in candidate_moves:
        return (
            f"next_action {next_action!r} is not a candidate; choose one of "
            f"{sorted(candidate_moves)}"
        )
    known_hazards = {as_cell(fact["cell"]) for fact in available_facts}
    destination = as_cell(candidate_moves[next_action])
    visited = {as_cell(cell) for cell in visited_cells}
    unexplored_safe_actions = [
        action
        for action, raw_destination in candidate_moves.items()
        if as_cell(raw_destination) not in visited
        and as_cell(raw_destination) not in known_hazards
    ]
    if destination in visited and unexplored_safe_actions:
        return (
            f"next_action {next_action} revisits {list(destination)} while unexplored "
            f"immediately safe actions remain: {sorted(unexplored_safe_actions)}"
        )
    checked_cells = (
        continuations[next_action]["cells"]
        if continuations is not None
        else [candidate_moves[next_action]]
    )
    for raw_cell in checked_cells:
        cell = as_cell(raw_cell)
        if cell in known_hazards:
            scope = (
                "compiled public continuation"
                if continuations is not None
                else "immediate move"
            )
            return (
                f"next_action {next_action}: {scope} crosses known hazard cell "
                f"{list(cell)}"
            )
    return None


def make_sensor_facts(
    data: dict[str, Any], agent: str, position: tuple[int, int], attempt: str, phase: str
) -> list[dict[str, Any]]:
    sensor = data["agents"][agent]["sensor"]
    radius = data["observation_radius"]
    visible = []
    for cell in sorted(trap_cells(data)[sensor]):
        if max(abs(cell[0] - position[0]), abs(cell[1] - position[1])) <= radius:
            visible.append(
                {
                    # A physical fact keeps one identity for the entire attempt.  The
                    # phase/tick at which it was observed belongs to observation
                    # events, not to the fact identity itself.
                    "artifact_id": f"{attempt}:fact:{agent}:{sensor}:{cell[0]}:{cell[1]}",
                    "hazard": sensor,
                    "cell": list(cell),
                    "source_agent": agent,
                    "first_observed_phase": phase,
                }
            )
    return visible


def communication_messages(
    *, agent: str, phase: str, sensor_facts: list[dict[str, Any]], limit: int
) -> list[dict[str, str]]:
    payload = {
        "agent": agent,
        "partner": PARTNER[agent],
        "phase": phase,
        "sensor_facts": sensor_facts,
        "message_fact_limit": limit,
    }
    return [
        {
            "role": "system",
            "content": (
                "You are one robot in a shared-world cooperative navigation task. You and your "
                "partner move independently in the same coordinate-aligned physical map, but "
                "you have complementary private hazard sensors. The supplied sensor_facts are "
                "pending facts that have not yet been delivered to your partner. Select at least "
                "one and at most message_fact_limit facts. Do not invent, repeat, or alter "
                "coordinates. Return JSON "
                "only: {\"message_facts\":[{\"source_fact_id\":str,\"hazard\":\"fire|water\","
                "\"cell\":[x,y]}],\"note\":str}."
            ),
        },
        {"role": "user", "content": json.dumps(payload, sort_keys=True)},
    ]


def planning_messages(
    *,
    data: dict[str, Any],
    agent: str,
    phase: dict[str, Any],
    position: tuple[int, int],
    public_cells: list[dict[str, Any]],
    available_facts: list[dict[str, Any]],
    candidate_moves: dict[str, list[int]],
    visited_cells: list[list[int]],
    remaining_actions: int,
    planning_granularity: str,
    chunk_size: int = 3,
    last_action_feedback: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    payload = {
        "agent": agent,
        "world_model": data["world_model"],
        "agent_start": list(agent_cell(data, agent, "start")),
        "agent_goal": list(agent_cell(data, agent, "goal")),
        "phase": phase["id"],
        "coordinate_system": {
            "origin": "top-left",
            "cell_format": "[x, y]",
            "action_deltas": {action: list(delta) for action, delta in MOVES.items()},
            "has_heading": False,
            "movement_frame": "absolute_grid",
            "rotation_effect": "none",
        },
        "tile_legend": {
            "#": "blocked wall",
            ".": "traversable floor",
            "S": "traversable start marker",
            "G": "traversable goal marker",
        },
        "grid_dimensions": data["dimensions"],
        "current_cell": list(position),
        "segment_target": phase["target"],
        "remaining_actions": remaining_actions,
        "annotated_local_map": annotated_public_view(
            public_cells=public_cells,
            position=position,
            target=as_cell(phase["target"]),
            available_facts=available_facts,
            visited_cells=visited_cells,
        ),
        "available_hazard_facts": available_facts,
    }
    if planning_granularity == "route":
        payload["known_forbidden_cells"] = sorted(
            {
                tuple(fact["cell"])
                for fact in available_facts
            },
            key=lambda cell: (cell[1], cell[0]),
        )
        payload["known_forbidden_cells"] = [
            list(cell) for cell in payload["known_forbidden_cells"]
        ]
        payload["public_adjacency_graph"] = [
            {
                "cell": item["from"],
                "neighbors": list(item["moves"].values()),
            }
            for item in public_transition_graph(public_cells)
        ]
        task_instruction = (
            "Plan a concrete coordinate path from current_cell to segment_target. Choose the "
            "waypoint yourself and list every traversed cell in order in path_cells. Include "
            "current_cell as the first cell and segment_target as the final cell. The "
            "controller will translate consecutive coordinate differences into primitive grid "
            "actions and execute them. Do not output UP, DOWN, LEFT, or RIGHT."
        )
        output_instruction = (
            "Return JSON only: {\"waypoint\":[x,y],\"path_cells\":[[x,y],...],"
            "\"subgoal\":str,\"declared_dependency_ids\":[...],\"reason\":str}. "
            "waypoint must differ from current_cell, occur in path_cells, and identify the "
            "meaningful intermediate milestone or branch chosen. Before answering, verify each "
            "next cell is in the current cell's neighbors list and no path cell appears in "
            "known_forbidden_cells. "
            "Dependency IDs must be copied from dependency_id fields of hazard facts that "
            "influenced the route choice. Keep subgoal and reason concise."
        )
    elif planning_granularity == "chunk":
        chunk_action_limit = min(chunk_size, remaining_actions)
        payload["chunk_action_limit"] = chunk_action_limit
        payload["known_forbidden_cells"] = sorted(
            {tuple(fact["cell"]) for fact in available_facts},
            key=lambda cell: (cell[1], cell[0]),
        )
        payload["known_forbidden_cells"] = [
            list(cell) for cell in payload["known_forbidden_cells"]
        ]
        payload["public_transition_graph"] = public_transition_graph(public_cells)
        payload["recent_trajectory"] = visited_cells[-8:]
        payload["last_action_feedback"] = last_action_feedback
        task_instruction = (
            "Plan the next short open-loop navigation chunk. Choose the primitive actions "
            "yourself and output them directly in execution order. The simulator will execute "
            "exactly those actions, then return a fresh observation before the next chunk. "
            "Choose between one and chunk_action_limit actions. Stop early if the actions reach "
            "segment_target. waypoint must be the exact cell reached after all actions."
        )
        output_instruction = (
            "Return JSON only: {\"actions\":[\"UP|DOWN|LEFT|RIGHT\",...],"
            "\"waypoint\":[x,y],\"declared_dependency_ids\":[...],\"reason\":str}. "
            "Dependency IDs must be copied from dependency_id fields of hazard facts that "
            "influenced the action chunk. Keep reason concise and under 160 characters."
        )
    elif planning_granularity == "junction":
        payload["candidate_moves"] = candidate_moves
        payload["visited_cells"] = visited_cells
        payload["public_transition_graph"] = public_transition_graph(public_cells)
        transitions = {
            as_cell(item["from"]): item["moves"]
            for item in public_transition_graph(public_cells)
        }
        payload["candidate_continuations"] = candidate_continuations(
            candidate_moves=candidate_moves,
            target=as_cell(phase["target"]),
            transitions=transitions,
        )
        task_instruction = (
            "Choose exactly one primitive movement at the current grid junction. Use "
            "candidate_continuations to check the downstream forced corridor, but output only "
            "the immediate next_action."
        )
        output_instruction = (
            "Return JSON only: {\"next_action\":\"UP|DOWN|LEFT|RIGHT\","
            "\"declared_dependency_ids\":[...],\"reason\":str}. Dependency IDs must be "
            "copied from dependency_id fields of hazard facts that influenced the plan. "
            "Keep reason concise and under 160 characters."
        )
    elif planning_granularity == "step":
        payload["public_next_cells"] = sorted(
            candidate_moves.values(), key=lambda cell: (cell[1], cell[0])
        )
        payload["known_forbidden_cells"] = sorted(
            {tuple(fact["cell"]) for fact in available_facts},
            key=lambda cell: (cell[1], cell[0]),
        )
        payload["known_forbidden_cells"] = [
            list(cell) for cell in payload["known_forbidden_cells"]
        ]
        payload["recent_trajectory"] = visited_cells[-5:]
        payload["last_action_feedback"] = last_action_feedback
        task_instruction = (
            "This is one turn of closed-loop navigation. Choose exactly one adjacent coordinate "
            "from public_next_cells as next_cell. The simulator will execute only that one grid "
            "edge and then provide an updated position and last_action_feedback. You are choosing "
            "the path one concrete cell at a time; do not output a full route."
        )
        output_instruction = (
            "Return JSON only: {\"next_cell\":[x,y],"
            "\"declared_dependency_ids\":[...],\"reason\":str}. Dependency IDs must be "
            "copied from dependency_id fields of hazard facts that influenced the plan. "
            "Keep reason concise, state-grounded, and under 160 characters."
        )
    else:
        raise ValueError(f"unsupported planning granularity: {planning_granularity}")
    if planning_granularity == "route":
        decision_constraints = (
            "Use only annotated_local_map, public_adjacency_graph, and available_hazard_facts. "
            "Every consecutive pair, beginning with current_cell, must be connected by exactly "
            "one edge in public_adjacency_graph. Do not enter a cell listed in "
            "known_forbidden_cells. Unknown hidden hazards are unavailable and must not be "
            "assumed. The number of edges in path_cells must not exceed remaining_actions."
        )
    elif planning_granularity == "chunk":
        decision_constraints = (
            "Use only annotated_local_map, public_transition_graph, available_hazard_facts, "
            "recent_trajectory, and last_action_feedback. Simulate every proposed action from "
            "current_cell using the action deltas and the transition graph. Every action must "
            "be a public legal edge, and the resulting cells must avoid known_forbidden_cells. "
            "Do not assume unobserved cells are traversable. Unknown hidden hazards are "
            "unavailable and must not be assumed."
        )
    elif planning_granularity == "junction":
        decision_constraints = (
            "Use only candidate_moves, the visible public map, public_transition_graph, and "
            "available hazard facts; do not assume an unknown cell is traversable. The "
            "transition graph is the authoritative list of publicly legal moves. Select "
            "next_action only from candidate_moves while avoiding available hazards and "
            "unnecessary revisits and making progress toward segment_target."
        )
    else:
        decision_constraints = (
            "Use only annotated_local_map, public_next_cells, available_hazard_facts, "
            "known_forbidden_cells, recent_trajectory, and last_action_feedback. Choose exactly "
            "one next_cell from public_next_cells and do not choose a known_forbidden_cells "
            "cell. Backtracking is allowed when useful; revisits are not rejected by the "
            "controller. Unknown hidden hazards are unavailable and must not be assumed."
        )
    return [
        {
            "role": "system",
            "content": (
                task_instruction
                + " Coordinates use a top-left origin: UP changes [x,y] to [x,y-1], "
                "DOWN to [x,y+1], "
                "LEFT to [x-1,y], and RIGHT to [x+1,y]. A # tile is blocked; ., S, and G "
                "are traversable. The agent has no heading or facing direction: LEFT and RIGHT "
                "are absolute x-axis movements, and no action rotates the agent. Fire and water "
                "traps are dangerous to both robots. "
                + decision_constraints
                + " "
                + output_instruction
            ),
        },
        {"role": "user", "content": json.dumps(payload, sort_keys=True)},
    ]


def response_schema(kind: str) -> tuple[str, dict[str, Any]]:
    cell_schema = {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 2,
        "maxItems": 2,
    }
    if kind == "communicate":
        return (
            "communication_output",
            {
                "type": "object",
                "properties": {
                    "message_facts": {
                        "type": "array",
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_fact_id": {"type": "string"},
                                "hazard": {
                                    "type": "string",
                                    "enum": ["fire", "water"],
                                },
                                "cell": cell_schema,
                            },
                            "required": ["source_fact_id", "hazard", "cell"],
                            "additionalProperties": False,
                        },
                    },
                    "note": {"type": "string", "maxLength": 160},
                },
                "required": ["message_facts", "note"],
                "additionalProperties": False,
            },
        )
    if kind == "step_plan":
        return (
            "step_plan_output",
            {
                "type": "object",
                "properties": {
                    "next_cell": cell_schema,
                    "declared_dependency_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "reason": {"type": "string", "maxLength": 160},
                },
                "required": [
                    "next_cell",
                    "declared_dependency_ids",
                    "reason",
                ],
                "additionalProperties": False,
            },
        )
    if kind == "chunk_plan":
        return (
            "chunk_plan_output",
            {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(MOVES)},
                        "minItems": 1,
                    },
                    "waypoint": cell_schema,
                    "declared_dependency_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "reason": {"type": "string", "maxLength": 160},
                },
                "required": [
                    "actions",
                    "waypoint",
                    "declared_dependency_ids",
                    "reason",
                ],
                "additionalProperties": False,
            },
        )
    if kind == "plan":
        return (
            "plan_output",
            {
                "type": "object",
                "properties": {
                    "next_action": {
                        "type": "string",
                        "enum": list(MOVES),
                    },
                    "declared_dependency_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "reason": {"type": "string", "maxLength": 160},
                },
                "required": [
                    "next_action",
                    "declared_dependency_ids",
                    "reason",
                ],
                "additionalProperties": False,
            },
        )
    if kind == "route_plan":
        return (
            "route_plan_output",
            {
                "type": "object",
                "properties": {
                    "waypoint": cell_schema,
                    "path_cells": {
                        "type": "array",
                        "items": cell_schema,
                        "minItems": 2,
                    },
                    "subgoal": {"type": "string", "maxLength": 120},
                    "declared_dependency_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "reason": {"type": "string", "maxLength": 160},
                },
                "required": [
                    "waypoint",
                    "path_cells",
                    "subgoal",
                    "declared_dependency_ids",
                    "reason",
                ],
                "additionalProperties": False,
            },
        )
    raise ValueError(f"no response schema for LLM call kind: {kind}")


def validate_message_output(
    output: dict[str, Any], source_facts: list[dict[str, Any]], limit: int
) -> dict[str, Any]:
    facts = output.get("message_facts")
    if not isinstance(facts, list) or len(facts) > limit:
        raise EpisodeError("message_facts must be a list within the fact limit")
    source_by_id = {fact["artifact_id"]: fact for fact in source_facts}
    normalized = []
    for item in facts:
        if not isinstance(item, dict):
            raise EpisodeError("each message fact must be an object")
        source_id = item.get("source_fact_id")
        if source_id not in source_by_id:
            raise EpisodeError(f"message cites unknown source fact: {source_id!r}")
        hazard = item.get("hazard")
        if hazard not in {"fire", "water"}:
            raise EpisodeError(f"invalid hazard type: {hazard!r}")
        cell = as_cell(item.get("cell", []))
        source = source_by_id[source_id]
        if hazard != source["hazard"] or cell != as_cell(source["cell"]):
            raise EpisodeError(
                "message fact must exactly copy the cited sensor fact; "
                f"source {source_id!r} is {source['hazard']} at {source['cell']}"
            )
        normalized.append(
            {
                "source_fact_id": source_id,
                "hazard": hazard,
                "cell": list(cell),
            }
        )
    return {"message_facts": normalized, "note": str(output.get("note", ""))}


def validate_plan_output(
    output: dict[str, Any],
    allowed_dependency_ids: set[str],
    allowed_actions: set[str],
) -> dict[str, Any]:
    next_action = output.get("next_action")
    dependencies = output.get("declared_dependency_ids", [])
    if next_action not in allowed_actions:
        raise EpisodeError(
            f"next_action must be one of {sorted(allowed_actions)}, got {next_action!r}"
        )
    if not isinstance(dependencies, list) or any(
        not isinstance(item, str) for item in dependencies
    ):
        raise EpisodeError("declared_dependency_ids must be a list of strings")
    if len(dependencies) != len(set(dependencies)):
        raise EpisodeError("declared_dependency_ids must not contain duplicates")
    unknown_dependencies = set(dependencies) - allowed_dependency_ids
    if unknown_dependencies:
        raise EpisodeError(
            "plan declares unknown dependency IDs: "
            + ", ".join(sorted(unknown_dependencies))
        )
    return {
        "next_action": next_action,
        "declared_dependency_ids": dependencies,
        "reason": str(output.get("reason", "")),
    }


def assess_step_plan_output(
    output: dict[str, Any],
    *,
    allowed_dependency_ids: set[str],
    current_cell: tuple[int, int],
    candidate_moves: dict[str, list[int]],
    available_facts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assess one closed-loop coordinate choice without aborting the episode.

    Publicly invalid choices become a recorded no-op. A physically legal move
    remains executable even when the agent was told that its destination is a
    hazard; the environment, rather than a safety shield, then determines the
    outcome. This keeps "ignored correct warning" distinct from "missing or
    faulty warning" in the trace.
    """
    issues: list[str] = []
    issue_code: str | None = None
    try:
        next_cell = as_cell(output.get("next_cell", []))
    except (TypeError, ValueError):
        next_cell = current_cell
        issue_code = "invalid_next_cell"
        issues.append(
            "next_cell must be an integer [x,y] cell, got "
            f"{output.get('next_cell')!r}"
        )

    raw_dependencies = output.get("declared_dependency_ids", [])
    if not isinstance(raw_dependencies, list) or any(
        not isinstance(item, str) for item in raw_dependencies
    ):
        dependencies: list[str] = []
        issue_code = issue_code or "invalid_dependencies"
        issues.append("declared_dependency_ids must be a list of strings")
    else:
        dependencies = list(raw_dependencies)
        if len(dependencies) != len(set(dependencies)):
            issue_code = issue_code or "duplicate_dependencies"
            issues.append("declared_dependency_ids must not contain duplicates")
        unknown_dependencies = set(dependencies) - allowed_dependency_ids
        if unknown_dependencies:
            issue_code = issue_code or "unknown_dependencies"
            issues.append(
                "step plan declares unknown dependency IDs: "
                + ", ".join(sorted(unknown_dependencies))
            )

    valid_dependencies = list(
        dict.fromkeys(
            dependency_id
            for dependency_id in dependencies
            if dependency_id in allowed_dependency_ids
        )
    )
    destination_actions = {
        as_cell(destination): action for action, destination in candidate_moves.items()
    }
    next_action = destination_actions.get(next_cell)
    known_hazards = {as_cell(fact["cell"]) for fact in available_facts}
    if next_cell == current_cell:
        issue_code = issue_code or (
            "format_fallback" if output.get("_format_fallback") else "stay_requested"
        )
        issues.append("next_cell equals current_cell, so the agent remains in place")
    elif next_action is None:
        issue_code = issue_code or "non_adjacent_or_blocked"
        issues.append(
            f"next_cell {list(next_cell)} is not one of the public adjacent cells "
            f"{sorted(destination_actions)}"
        )

    accepted = not issues
    return {
        "accepted": accepted,
        "next_cell": list(next_cell),
        "next_action": next_action if accepted else None,
        "declared_dependency_ids": dependencies,
        "valid_dependency_ids": valid_dependencies,
        "reason": str(output.get("reason", "")),
        "known_hazard_selected": next_cell in known_hazards,
        "issue_code": issue_code,
        "feedback": None if accepted else "; ".join(issues),
    }


def validate_route_plan_output(
    output: dict[str, Any],
    allowed_dependency_ids: set[str],
    *,
    current_cell: tuple[int, int],
    target: tuple[int, int],
    public_cells: list[dict[str, Any]],
    available_facts: list[dict[str, Any]],
    remaining_actions: int,
) -> dict[str, Any]:
    """Validate and compile an LLM-proposed coordinate path using public facts only."""
    try:
        waypoint = as_cell(output.get("waypoint", []))
    except (TypeError, ValueError) as exc:
        raise EpisodeError(f"invalid waypoint: {output.get('waypoint')!r}") from exc
    raw_path = output.get("path_cells")
    if not isinstance(raw_path, list) or len(raw_path) < 2:
        raise EpisodeError("path_cells must contain at least a start and target cell")
    try:
        path_cells = [as_cell(cell) for cell in raw_path]
    except (TypeError, ValueError) as exc:
        raise EpisodeError("every path_cells item must be an integer [x,y] cell") from exc
    visible_tiles = {
        as_cell(item["cell"]): item["tile"] for item in public_cells
    }
    known_hazards = {as_cell(fact["cell"]) for fact in available_facts}
    delta_to_action = {delta: action for action, delta in MOVES.items()}
    actions: list[str] = []
    issues: list[str] = []
    if path_cells[0] != current_cell:
        issues.append(
            f"path_cells starts at {list(path_cells[0])}, not current_cell "
            f"{list(current_cell)}"
        )
    if path_cells[-1] != target:
        issues.append(
            f"path_cells ends at {list(path_cells[-1])}, not segment_target {list(target)}"
        )
    edge_count = len(path_cells) - 1
    if edge_count > remaining_actions:
        issues.append(
            f"path has {edge_count} edges but only {remaining_actions} actions remain"
        )
    if waypoint == current_cell:
        issues.append("waypoint must differ from current_cell")
    elif waypoint not in path_cells:
        issues.append(f"waypoint {list(waypoint)} must appear in path_cells")

    for step_number, (previous, cell) in enumerate(
        zip(path_cells, path_cells[1:]), start=1
    ):
        if cell not in visible_tiles:
            issues.append(
                f"path step {step_number} cell {list(cell)} is outside the visible public map"
            )
        elif visible_tiles[cell] == "#":
            issues.append(
                f"path step {step_number} enters public wall {list(cell)}"
            )
        if cell in known_hazards:
            issues.append(
                f"path step {step_number} enters known hazard {list(cell)}"
            )
        delta = cell[0] - previous[0], cell[1] - previous[1]
        action = delta_to_action.get(delta)
        if action is None:
            issues.append(
                f"path step {step_number} jumps from {list(previous)} to {list(cell)}; "
                "consecutive cells must differ by one public grid edge"
            )
        else:
            actions.append(action)

    if issues:
        raise EpisodeError("; ".join(issues))

    dependencies = output.get("declared_dependency_ids", [])
    if not isinstance(dependencies, list) or any(
        not isinstance(item, str) for item in dependencies
    ):
        raise EpisodeError("declared_dependency_ids must be a list of strings")
    if len(dependencies) != len(set(dependencies)):
        raise EpisodeError("declared_dependency_ids must not contain duplicates")
    unknown_dependencies = set(dependencies) - allowed_dependency_ids
    if unknown_dependencies:
        raise EpisodeError(
            "route plan declares unknown dependency IDs: "
            + ", ".join(sorted(unknown_dependencies))
        )
    return {
        "waypoint": list(waypoint),
        "path_cells": [list(cell) for cell in path_cells],
        "execution_cells": [list(cell) for cell in path_cells[1:]],
        "compiled_actions": actions,
        "subgoal": str(output.get("subgoal", "")),
        "declared_dependency_ids": dependencies,
        "reason": str(output.get("reason", "")),
    }


def validate_chunk_plan_output(
    output: dict[str, Any],
    allowed_dependency_ids: set[str],
    *,
    current_cell: tuple[int, int],
    target: tuple[int, int],
    public_cells: list[dict[str, Any]],
    available_facts: list[dict[str, Any]],
    remaining_actions: int,
    chunk_size: int,
) -> dict[str, Any]:
    """Validate an LLM-selected primitive-action chunk using public state only."""
    raw_actions = output.get("actions")
    action_limit = min(chunk_size, remaining_actions)
    if not isinstance(raw_actions, list) or not raw_actions:
        raise EpisodeError("actions must be a non-empty list")
    if len(raw_actions) > action_limit:
        raise EpisodeError(
            f"chunk has {len(raw_actions)} actions but the current limit is {action_limit}"
        )
    if any(not isinstance(action, str) or action not in MOVES for action in raw_actions):
        raise EpisodeError(
            f"every chunk action must be one of {sorted(MOVES)}"
        )

    try:
        waypoint = as_cell(output.get("waypoint", []))
    except (TypeError, ValueError) as exc:
        raise EpisodeError(f"invalid waypoint: {output.get('waypoint')!r}") from exc

    visible_tiles = {as_cell(item["cell"]): item["tile"] for item in public_cells}
    known_hazards = {as_cell(fact["cell"]) for fact in available_facts}
    position = current_cell
    execution_cells: list[list[int]] = []
    for step_number, action in enumerate(raw_actions, start=1):
        dx, dy = MOVES[action]
        destination = position[0] + dx, position[1] + dy
        if destination not in visible_tiles:
            raise EpisodeError(
                f"chunk action {step_number} {action} leaves the observed public map at "
                f"{list(destination)}"
            )
        if visible_tiles[destination] == "#":
            raise EpisodeError(
                f"chunk action {step_number} {action} enters public wall "
                f"{list(destination)}"
            )
        if destination in known_hazards:
            raise EpisodeError(
                f"chunk action {step_number} {action} enters known hazard "
                f"{list(destination)}"
            )
        position = destination
        execution_cells.append(list(position))
        if position == target and step_number != len(raw_actions):
            raise EpisodeError("chunk reaches segment_target before its final action")

    if waypoint != position:
        raise EpisodeError(
            f"waypoint {list(waypoint)} does not match action endpoint {list(position)}"
        )
    if waypoint == current_cell:
        raise EpisodeError("chunk actions must make progress to a different waypoint")

    dependencies = output.get("declared_dependency_ids", [])
    if not isinstance(dependencies, list) or any(
        not isinstance(item, str) for item in dependencies
    ):
        raise EpisodeError("declared_dependency_ids must be a list of strings")
    if len(dependencies) != len(set(dependencies)):
        raise EpisodeError("declared_dependency_ids must not contain duplicates")
    unknown_dependencies = set(dependencies) - allowed_dependency_ids
    if unknown_dependencies:
        raise EpisodeError(
            "chunk plan declares unknown dependency IDs: "
            + ", ".join(sorted(unknown_dependencies))
        )
    return {
        "actions": list(raw_actions),
        "waypoint": list(waypoint),
        "execution_cells": execution_cells,
        "declared_dependency_ids": dependencies,
        "reason": str(output.get("reason", "")),
    }


def resolve_faulty_cell(fault: dict[str, Any]) -> tuple[int, int]:
    """Resolve a coordinate fault from its precommitted operational rule."""
    rule = fault.get("selection_rule")
    if isinstance(rule, dict):
        rule_type = rule.get("type")
        if rule_type == "fixed_offset":
            delta = rule.get("delta")
            if (
                not isinstance(delta, list)
                or len(delta) != 2
                or any(not isinstance(value, int) for value in delta)
            ):
                raise MapValidationError(
                    f"fault {fault.get('id')} fixed_offset requires integer delta [dx, dy]"
                )
            true_cell = as_cell(fault["true_cell"])
            return true_cell[0] + delta[0], true_cell[1] + delta[1]
        if rule_type == "route_decoy":
            return as_cell(rule.get("cell"))
        raise MapValidationError(
            f"fault {fault.get('id')} has unsupported selection rule: {rule_type}"
        )
    # Backward compatibility for development maps created before schema v4.
    return as_cell(fault["faulty_cell"])


def inject_fault(
    message: dict[str, Any], fault: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    updated = deepcopy(message)
    true_cell = list(as_cell(fault["true_cell"]))
    matched = False
    new_facts = []
    for item in updated["message_facts"]:
        if item["hazard"] == fault["hazard"] and item["cell"] == true_cell:
            matched = True
            if fault["kind"] == "omission":
                continue
            if fault["kind"] == "coordinate_shift":
                item = dict(item)
                item["cell"] = list(resolve_faulty_cell(fault))
        new_facts.append(item)
    updated["message_facts"] = new_facts
    if matched:
        updated["note"] = (
            f"[controlled fault {fault['id']}: {fault['kind']} applied to "
            "structured message_facts after generation]"
        )
    return updated, matched


@dataclass
class Backend:
    call_records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def llm_calls(self) -> int:
        return sum(1 for record in self.call_records if record.get("backend") == "llm")

    def communicate(
        self, *, agent: str, phase: str, sensor_facts: list[dict[str, Any]], limit: int
    ) -> dict[str, Any]:
        raise NotImplementedError

    def plan(
        self,
        *,
        data: dict[str, Any],
        agent: str,
        phase: dict[str, Any],
        position: tuple[int, int],
        public_cells: list[dict[str, Any]],
        available_facts: list[dict[str, Any]],
        candidate_moves: dict[str, list[int]],
        visited_cells: list[list[int]],
        remaining_actions: int,
        planning_granularity: str,
        chunk_size: int = 3,
        last_action_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError


class ScriptedBackend(Backend):
    """Deterministic backend used to validate the harness without API calls."""

    def communicate(
        self, *, agent: str, phase: str, sensor_facts: list[dict[str, Any]], limit: int
    ) -> dict[str, Any]:
        output = {
            "message_facts": [
                {
                    "source_fact_id": fact["artifact_id"],
                    "hazard": fact["hazard"],
                    "cell": fact["cell"],
                }
                for fact in sensor_facts[:limit]
            ],
            "note": f"scripted {agent} {phase} sensor report",
        }
        self.call_records.append(
            {"backend": "scripted", "kind": "communicate", "agent": agent, "phase": phase}
        )
        return output

    def plan(
        self,
        *,
        data: dict[str, Any],
        agent: str,
        phase: dict[str, Any],
        position: tuple[int, int],
        public_cells: list[dict[str, Any]],
        available_facts: list[dict[str, Any]],
        candidate_moves: dict[str, list[int]],
        visited_cells: list[list[int]],
        remaining_actions: int,
        planning_granularity: str,
        chunk_size: int = 3,
        last_action_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if planning_granularity in {"route", "chunk"}:
            transitions = {
                as_cell(item["from"]): item["moves"]
                for item in public_transition_graph(public_cells)
            }
            blocked = {as_cell(fact["cell"]) for fact in available_facts}
            selected = shortest_transition_route(
                start=position,
                target=as_cell(phase["target"]),
                transitions=transitions,
                blocked=blocked,
            )
            if selected is None and planning_granularity == "chunk":
                selected = visible_progress_route(
                    start=position,
                    target=as_cell(phase["target"]),
                    transitions=transitions,
                    blocked=blocked,
                    visited={as_cell(cell) for cell in visited_cells},
                )
            if selected is None or len(selected["actions"]) > remaining_actions:
                raise EpisodeError(
                    f"scripted coordinate planner found no public path for {agent} in phase "
                    f"{phase['id']}"
                )
            dependencies = sorted(
                {
                    fact["dependency_id"]
                    for fact in available_facts
                    if fact.get("dependency_id")
                }
            )
            call_kind = "route_plan" if planning_granularity == "route" else "chunk_plan"
            self.call_records.append(
                {
                    "backend": "scripted",
                    "kind": call_kind,
                    "agent": agent,
                    "phase": phase["id"],
                }
            )
            if planning_granularity == "chunk":
                actions = selected["actions"][: min(chunk_size, remaining_actions)]
                cells = selected["cells"][: len(actions)]
                return {
                    "actions": actions,
                    "waypoint": cells[-1],
                    "declared_dependency_ids": dependencies,
                    "reason": "scripted safe direct-action chunk toward the segment target",
                }
            path_cells = [list(position), *selected["cells"]]
            return {
                "waypoint": path_cells[len(path_cells) // 2],
                "path_cells": path_cells,
                "subgoal": "follow the safest compiled route to the segment target",
                "declared_dependency_ids": dependencies,
                "reason": "scripted shortest route under currently available hazard facts",
            }
        blocked = {as_cell(fact["cell"]) for fact in available_facts}
        choices: list[tuple[int, str]] = []
        for action, raw_destination in candidate_moves.items():
            destination = as_cell(raw_destination)
            if destination in blocked:
                continue
            tail = shortest_path(
                data,
                destination,
                as_cell(phase["target"]),
                blocked=blocked,
            )
            if tail is not None and len(tail) + 1 <= remaining_actions:
                choices.append((len(tail) + 1, action))
        if not choices:
            raise EpisodeError(
                f"scripted planner found no safe candidate for {agent} in phase "
                f"{phase['id']}"
            )
        _, next_action = min(choices, key=lambda item: item[0])
        dependencies = sorted(
            {
                fact["dependency_id"]
                for fact in available_facts
                if fact.get("dependency_id")
            }
        )
        self.call_records.append(
            {"backend": "scripted", "kind": "plan", "agent": agent, "phase": phase["id"]}
        )
        if planning_granularity == "step":
            return {
                "next_cell": candidate_moves[next_action],
                "declared_dependency_ids": dependencies,
                "reason": "scripted safest adjacent cell toward the segment target",
            }
        return {
            "next_action": next_action,
            "declared_dependency_ids": dependencies,
            "reason": "scripted safest next move toward the segment target",
        }


class DryRunBackend(ScriptedBackend):
    """Scripted execution that also records the exact prompts for inspection."""

    def communicate(
        self, *, agent: str, phase: str, sensor_facts: list[dict[str, Any]], limit: int
    ) -> dict[str, Any]:
        result = super().communicate(
            agent=agent, phase=phase, sensor_facts=sensor_facts, limit=limit
        )
        self.call_records[-1]["prompt"] = communication_messages(
            agent=agent, phase=phase, sensor_facts=sensor_facts, limit=limit
        )
        return result

    def plan(
        self,
        *,
        data: dict[str, Any],
        agent: str,
        phase: dict[str, Any],
        position: tuple[int, int],
        public_cells: list[dict[str, Any]],
        available_facts: list[dict[str, Any]],
        candidate_moves: dict[str, list[int]],
        visited_cells: list[list[int]],
        remaining_actions: int,
        planning_granularity: str,
        chunk_size: int = 3,
        last_action_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = super().plan(
            data=data,
            agent=agent,
            phase=phase,
            position=position,
            public_cells=public_cells,
            available_facts=available_facts,
            candidate_moves=candidate_moves,
            visited_cells=visited_cells,
            remaining_actions=remaining_actions,
            planning_granularity=planning_granularity,
            chunk_size=chunk_size,
            last_action_feedback=last_action_feedback,
        )
        self.call_records[-1]["prompt"] = planning_messages(
            data=data,
            agent=agent,
            phase=phase,
            position=position,
            public_cells=public_cells,
            available_facts=available_facts,
            candidate_moves=candidate_moves,
            visited_cells=visited_cells,
            remaining_actions=remaining_actions,
            planning_granularity=planning_granularity,
            chunk_size=chunk_size,
            last_action_feedback=last_action_feedback,
        )
        return result


class LLMBackend(Backend):
    def __init__(
        self,
        *,
        url: str,
        key: str,
        model: str,
        timeout: int = 90,
        reasoning_effort: str = "none",
    ) -> None:
        super().__init__()
        if reasoning_effort not in {"none", "low", "medium", "high", "xhigh"}:
            raise ValueError(
                "reasoning_effort must be one of none, low, medium, high, xhigh"
            )
        # A trailing newline from a pasted key makes urllib raise an invalid
        # header error whose message contains the full Authorization value.
        key = key.strip()
        if not key or any(char.isspace() or not char.isprintable() for char in key):
            raise ValueError(
                "LLM API key is empty or contains whitespace/control characters"
            )
        self.url = url
        self.key = key
        self.model = model
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.force_streaming = False

    def _chat(
        self,
        *,
        kind: str,
        agent: str,
        phase: str,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        hostname = (urllib.parse.urlparse(self.url).hostname or "").lower()
        if hostname in {"api.together.ai", "api.together.xyz"}:
            payload["temperature"] = 0
            schema_name, schema = response_schema(kind)
            request_messages = deepcopy(messages)
            request_messages[0]["content"] += (
                " Follow this JSON Schema exactly: "
                + json.dumps(schema, separators=(",", ":"))
            )
            payload["messages"] = request_messages
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema},
            }
            payload["reasoning"] = {"enabled": False}
        elif hostname == "api.openai.com":
            schema_name, schema = response_schema(kind)
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            }
            payload["reasoning_effort"] = self.reasoning_effort
            payload["store"] = False
        else:
            payload["temperature"] = 0

        is_tritonai = hostname == "tritonai-api.ucsd.edu"
        effective_reasoning_effort: str | None = None
        if is_tritonai and self.model == "api-glm-5.3":
            # TritonAI reports that this hosted-vLLM route supports both
            # reasoning_effort and response_format.  GLM-5.3 otherwise tends
            # to spend the whole completion budget in hidden reasoning and can
            # return content=null.  It is an always-reasoning model, so map our
            # provider-neutral "none" default to its cheapest useful effort.
            effective_reasoning_effort = (
                "low" if self.reasoning_effort == "none" else self.reasoning_effort
            )
            payload["reasoning_effort"] = effective_reasoning_effort
            payload["response_format"] = {"type": "json_object"}
        # TritonAI's full GLM-5.3 route can spend more than 2,400 completion
        # tokens entirely on hidden reasoning for a navigation plan, returning
        # content=null with finish_reason=length.  Communication is simple and
        # empirically completes well below the generic limit, so keep its
        # cheaper limits while giving planning calls enough room to emit JSON.
        token_limits = (
            (4096, 8192)
            if is_tritonai
            and self.model == "api-glm-5.3"
            and kind != "communicate"
            else (1200, 2400)
        )
        last_error: Exception | None = None
        for request_number, token_limit in enumerate(token_limits, start=1):
            request_started_ns = perf_counter_ns()
            transport_attempts = 0
            token_limit_parameter = (
                "max_completion_tokens"
                if hostname == "api.openai.com" or is_tritonai
                else "max_tokens"
            )
            if token_limit_parameter == "max_completion_tokens":
                payload["max_completion_tokens"] = token_limit
            else:
                payload["max_tokens"] = token_limit
            used_streaming_fallback = False
            for transport_attempt in range(2):
                transport_attempts += 1
                use_streaming = self.force_streaming
                if use_streaming:
                    payload["stream"] = True
                    payload["stream_options"] = {"include_usage": True}
                body = json.dumps(payload).encode()
                request = urllib.request.Request(
                    self.url,
                    data=body,
                    headers={
                        "Authorization": f"Bearer {self.key}",
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream" if use_streaming else "application/json",
                        # Together's Cloudflare policy rejects urllib's default
                        # Python-urllib signature with HTTP 403 / error code 1010.
                        "User-Agent": HTTP_USER_AGENT,
                    },
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout) as response:
                        response_body = response.read()
                    if use_streaming:
                        raw = parse_sse_chat_response(response_body)
                    else:
                        raw = json.loads(response_body)
                    break
                except urllib.error.HTTPError as exc:
                    detail = redact_secrets(
                        exc.read().decode(errors="replace")[:1000], [self.key]
                    )
                    exc.close()
                    if (
                        exc.code == 400
                        and "streaming_required" in detail
                        and not self.force_streaming
                        and transport_attempt == 0
                    ):
                        self.force_streaming = True
                        used_streaming_fallback = True
                        continue
                    hint = ""
                    if exc.code == 403 and "1010" in detail:
                        hint = (
                            " The API gateway rejected the HTTP client signature "
                            "(Cloudflare 1010)."
                        )
                    self.call_records.append(
                        {
                            "backend": "llm",
                            "kind": kind,
                            "agent": agent,
                            "phase": phase,
                            "model": self.model,
                            "request_number": request_number,
                            "max_tokens": token_limit,
                            "token_limit_parameter": token_limit_parameter,
                            "reasoning_effort": effective_reasoning_effort,
                            "prompt": payload["messages"],
                            "latency_ms": elapsed_ms(request_started_ns),
                            "transport_attempts": transport_attempts,
                            "stream": use_streaming,
                            "streaming_fallback_used": used_streaming_fallback,
                            "error_type": "http_error",
                            "error": f"HTTP {exc.code}: {detail}{hint}",
                        }
                    )
                    raise EpisodeError(
                        f"LLM endpoint returned HTTP {exc.code}: {detail}{hint}"
                    ) from exc
                except urllib.error.URLError as exc:
                    error_text = redact_secrets(str(exc), [self.key])
                    self.call_records.append(
                        {
                            "backend": "llm",
                            "kind": kind,
                            "agent": agent,
                            "phase": phase,
                            "model": self.model,
                            "request_number": request_number,
                            "max_tokens": token_limit,
                            "token_limit_parameter": token_limit_parameter,
                            "reasoning_effort": effective_reasoning_effort,
                            "prompt": payload["messages"],
                            "latency_ms": elapsed_ms(request_started_ns),
                            "transport_attempts": transport_attempts,
                            "stream": use_streaming,
                            "streaming_fallback_used": used_streaming_fallback,
                            "error_type": "transport_error",
                            "error": error_text,
                        }
                    )
                    raise EpisodeError(
                        f"LLM endpoint request failed: {error_text}"
                    ) from exc
            else:
                raise EpisodeError("LLM transport retry exhausted")
            response_shape = chat_response_shape(raw)
            try:
                choice = raw["choices"][0]
                message = choice["message"]
                finish_reason = choice.get("finish_reason")
                refusal = message.get("refusal")
                if refusal:
                    raise EpisodeError(f"LLM refused the request: {refusal}")
                content = normalize_chat_content(message.get("content"))
                if content is None or not content.strip():
                    raise TypeError("message has no final text content")
            except EpisodeError:
                raise
            except (KeyError, IndexError, TypeError) as exc:
                record = {
                    "backend": "llm",
                    "kind": kind,
                    "agent": agent,
                    "phase": phase,
                    "model": self.model,
                    "request_number": request_number,
                    "max_tokens": token_limit,
                    "token_limit_parameter": token_limit_parameter,
                    "reasoning_effort": effective_reasoning_effort,
                    "prompt": payload["messages"],
                    "finish_reason": response_shape.get("finish_reason"),
                    "usage": raw.get("usage", {}) if isinstance(raw, dict) else {},
                    "latency_ms": elapsed_ms(request_started_ns),
                    "transport_attempts": transport_attempts,
                    "stream": use_streaming,
                    "streaming_fallback_used": used_streaming_fallback,
                    "error_type": "missing_final_content",
                    "response_shape": response_shape,
                }
                self.call_records.append(record)
                last_error = exc
                continue
            record = {
                "backend": "llm",
                "kind": kind,
                "agent": agent,
                "phase": phase,
                "model": self.model,
                "request_number": request_number,
                "max_tokens": token_limit,
                "token_limit_parameter": token_limit_parameter,
                "reasoning_effort": effective_reasoning_effort,
                "prompt": payload["messages"],
                "response_text": content,
                "finish_reason": finish_reason,
                "usage": raw.get("usage", {}),
                "latency_ms": elapsed_ms(request_started_ns),
                "transport_attempts": transport_attempts,
                "stream": use_streaming,
                "streaming_fallback_used": used_streaming_fallback,
            }
            self.call_records.append(record)
            try:
                return extract_json(content)
            except json.JSONDecodeError as exc:
                record["parse_error"] = str(exc)
                last_error = exc

        assert last_error is not None
        final_record = self.call_records[-1]
        if final_record.get("error_type") == "missing_final_content":
            raise LLMResponseParseError(
                "LLM returned no final text content after one retry; "
                f"finish_reason={final_record.get('finish_reason')!r}; "
                f"response_shape={final_record.get('response_shape')!r}"
            ) from last_error
        raise LLMResponseParseError(
            "LLM returned invalid JSON after one retry; "
            f"finish_reason={final_record.get('finish_reason')!r}; "
            f"parse_error={last_error}; "
            f"response_tail={final_record.get('response_text', '')[-400:]!r}"
        ) from last_error

    def communicate(
        self, *, agent: str, phase: str, sensor_facts: list[dict[str, Any]], limit: int
    ) -> dict[str, Any]:
        messages = communication_messages(
            agent=agent, phase=phase, sensor_facts=sensor_facts, limit=limit
        )
        return self._chat(kind="communicate", agent=agent, phase=phase, messages=messages)

    def plan(
        self,
        *,
        data: dict[str, Any],
        agent: str,
        phase: dict[str, Any],
        position: tuple[int, int],
        public_cells: list[dict[str, Any]],
        available_facts: list[dict[str, Any]],
        candidate_moves: dict[str, list[int]],
        visited_cells: list[list[int]],
        remaining_actions: int,
        planning_granularity: str,
        chunk_size: int = 3,
        last_action_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if planning_granularity == "chunk":
            messages = planning_messages(
                data=data,
                agent=agent,
                phase=phase,
                position=position,
                public_cells=public_cells,
                available_facts=available_facts,
                candidate_moves=candidate_moves,
                visited_cells=visited_cells,
                remaining_actions=remaining_actions,
                planning_granularity=planning_granularity,
                chunk_size=chunk_size,
                last_action_feedback=last_action_feedback,
            )
            output: dict[str, Any] = {}
            last_issue: str | None = None
            allowed_dependency_ids = {
                fact["dependency_id"]
                for fact in available_facts
                if fact.get("dependency_id")
            }
            for repair_round in range(MAX_PLAN_REPAIRS + 1):
                output = self._chat(
                    kind="chunk_plan",
                    agent=agent,
                    phase=phase["id"],
                    messages=messages,
                )
                self.call_records[-1]["semantic_attempt"] = repair_round + 1
                try:
                    validate_chunk_plan_output(
                        output,
                        allowed_dependency_ids=allowed_dependency_ids,
                        current_cell=position,
                        target=as_cell(phase["target"]),
                        public_cells=public_cells,
                        available_facts=available_facts,
                        remaining_actions=remaining_actions,
                        chunk_size=chunk_size,
                    )
                    return output
                except (EpisodeError, ValueError) as exc:
                    last_issue = str(exc)
                    self.call_records[-1]["semantic_validation_error"] = last_issue
                    if repair_round < MAX_PLAN_REPAIRS:
                        messages = messages + [
                            {
                                "role": "assistant",
                                "content": json.dumps(output, sort_keys=True),
                            },
                            {
                                "role": "user",
                                "content": (
                                    "The public action-chunk checker rejected that output: "
                                    + last_issue
                                    + ". Return corrected primitive actions using only the "
                                    "original public_transition_graph. Keep the list within "
                                    "chunk_action_limit, avoid known_forbidden_cells, and set "
                                    "waypoint to the exact final cell reached."
                                ),
                            },
                        ]
            raise EpisodeError(
                f"LLM chunk plan failed public validation after {MAX_PLAN_REPAIRS} "
                f"repairs: {last_issue}"
            )
        if planning_granularity == "route":
            messages = planning_messages(
                data=data,
                agent=agent,
                phase=phase,
                position=position,
                public_cells=public_cells,
                available_facts=available_facts,
                candidate_moves=candidate_moves,
                visited_cells=visited_cells,
                remaining_actions=remaining_actions,
                planning_granularity=planning_granularity,
                chunk_size=chunk_size,
                last_action_feedback=last_action_feedback,
            )
            output: dict[str, Any] = {}
            last_issue: str | None = None
            allowed_dependency_ids = {
                fact["dependency_id"]
                for fact in available_facts
                if fact.get("dependency_id")
            }
            for repair_round in range(MAX_PLAN_REPAIRS + 1):
                output = self._chat(
                    kind="route_plan",
                    agent=agent,
                    phase=phase["id"],
                    messages=messages,
                )
                self.call_records[-1]["semantic_attempt"] = repair_round + 1
                try:
                    validate_route_plan_output(
                        output,
                        allowed_dependency_ids=allowed_dependency_ids,
                        current_cell=position,
                        target=as_cell(phase["target"]),
                        public_cells=public_cells,
                        available_facts=available_facts,
                        remaining_actions=remaining_actions,
                    )
                    return output
                except (EpisodeError, ValueError) as exc:
                    last_issue = str(exc)
                    self.call_records[-1]["semantic_validation_error"] = last_issue
                    if repair_round < MAX_PLAN_REPAIRS:
                        messages = messages + [
                            {
                                "role": "assistant",
                                "content": json.dumps(output, sort_keys=True),
                            },
                            {
                                "role": "user",
                                "content": (
                                    "The public coordinate-path checker rejected that output: "
                                    + last_issue
                                    + ". Return a corrected waypoint and path_cells using only "
                                    "the original public_adjacency_graph and known hazards. "
                                    "Begin with current_cell, list every adjacent cell, end with "
                                    "segment_target, and exclude every known_forbidden_cells cell."
                                ),
                            },
                        ]
            raise EpisodeError(
                f"LLM route plan failed public validation after {MAX_PLAN_REPAIRS} "
                f"repairs: {last_issue}"
            )
        if planning_granularity == "step":
            messages = planning_messages(
                data=data,
                agent=agent,
                phase=phase,
                position=position,
                public_cells=public_cells,
                available_facts=available_facts,
                candidate_moves=candidate_moves,
                visited_cells=visited_cells,
                remaining_actions=remaining_actions,
                planning_granularity=planning_granularity,
                chunk_size=chunk_size,
                last_action_feedback=last_action_feedback,
            )
            try:
                return self._chat(
                    kind="step_plan",
                    agent=agent,
                    phase=phase["id"],
                    messages=messages,
                )
            except LLMResponseParseError as exc:
                return {
                    "next_cell": list(position),
                    "declared_dependency_ids": [],
                    "reason": "invalid JSON; deterministic STAY fallback",
                    "_format_fallback": True,
                    "_format_error": str(exc),
                }

        continuations = None
        if planning_granularity == "junction":
            transitions = {
                as_cell(item["from"]): item["moves"]
                for item in public_transition_graph(public_cells)
            }
            continuations = candidate_continuations(
                candidate_moves=candidate_moves,
                target=as_cell(phase["target"]),
                transitions=transitions,
            )
        messages = planning_messages(
            data=data,
            agent=agent,
            phase=phase,
            position=position,
            public_cells=public_cells,
            available_facts=available_facts,
            candidate_moves=candidate_moves,
            visited_cells=visited_cells,
            remaining_actions=remaining_actions,
            planning_granularity=planning_granularity,
            chunk_size=chunk_size,
            last_action_feedback=last_action_feedback,
        )
        output: dict[str, Any] = {}
        last_issue: str | None = None
        for repair_round in range(MAX_PLAN_REPAIRS + 1):
            output = self._chat(
                kind="plan", agent=agent, phase=phase["id"], messages=messages
            )
            self.call_records[-1]["semantic_attempt"] = repair_round + 1
            issue = decision_error(
                next_action=output.get("next_action"),
                candidate_moves=candidate_moves,
                continuations=continuations,
                available_facts=available_facts,
                visited_cells=visited_cells,
            )
            if issue is None:
                return output
            last_issue = issue
            self.call_records[-1]["semantic_validation_error"] = issue
            if repair_round < MAX_PLAN_REPAIRS:
                messages = messages + [
                    {
                        "role": "assistant",
                        "content": json.dumps(output, sort_keys=True),
                    },
                    {
                        "role": "user",
                        "content": (
                            "The public decision checker rejected that action: "
                            + issue
                            + ". Select exactly one corrected next_action from the original "
                            "candidate_moves and avoid every available_hazard_facts cell."
                        ),
                    },
                ]
        raise EpisodeError(
            f"LLM decision failed public validation after {MAX_PLAN_REPAIRS} repairs: "
            + str(last_issue)
        )


@dataclass
class AttemptTrace:
    attempt: str
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    counter: int = 0
    tracking_latency_ns: int = 0

    def artifact(self, artifact_type: str, **payload: Any) -> str:
        started_ns = perf_counter_ns()
        try:
            self.counter += 1
            artifact_id = f"{self.attempt}:artifact:{self.counter:04d}"
            self.artifacts.append(
                {"artifact_id": artifact_id, "artifact_type": artifact_type, **payload}
            )
            return artifact_id
        finally:
            self.tracking_latency_ns += perf_counter_ns() - started_ns

    def event(self, event_type: str, **payload: Any) -> None:
        started_ns = perf_counter_ns()
        try:
            self.events.append(
                {"event_index": len(self.events), "event_type": event_type, **payload}
            )
        finally:
            self.tracking_latency_ns += perf_counter_ns() - started_ns

    def add_tracking_time(self, started_ns: int) -> None:
        """Include dependency bookkeeping performed outside trace append calls."""
        self.tracking_latency_ns += perf_counter_ns() - started_ns

    @property
    def tracking_latency_ms(self) -> float:
        return round(self.tracking_latency_ns / 1_000_000, 3)


def execute_segment(
    *,
    data: dict[str, Any],
    agent: str,
    position: tuple[int, int],
    actions: list[str],
    target: tuple[int, int],
    plan_id: str,
    trace: AttemptTrace,
    step_state: dict[str, int],
) -> dict[str, Any]:
    traps = trap_cells(data)
    trap_lookup = {cell: hazard for hazard, cells in traps.items() for cell in cells}
    action_trace = []
    for action in actions:
        if step_state[agent] >= data["step_limit"]:
            return {
                "success": False,
                "position": position,
                "error": "step limit exceeded",
                "trace": action_trace,
            }
        dx, dy = MOVES[action]
        destination = position[0] + dx, position[1] + dy
        action_id = trace.artifact(
            "action",
            agent=agent,
            action=action,
            from_cell=list(position),
            to_cell=list(destination),
            source_ids=[plan_id],
        )
        step_state[agent] += 1
        if not in_bounds(data, destination) or tile_at(data, destination) == "#":
            trace.event(
                "action_failed",
                agent=agent,
                action_artifact_id=action_id,
                reason="public_obstacle",
            )
            return {
                "success": False,
                "position": position,
                "error": f"blocked public move into {destination}",
                "trace": action_trace,
            }
        position = destination
        item = {"action": action, "cell": list(position), "artifact_id": action_id}
        action_trace.append(item)
        trace.event("action_executed", agent=agent, **item)
        if position in trap_lookup:
            trace.event(
                "trap_triggered",
                agent=agent,
                action_artifact_id=action_id,
                hazard=trap_lookup[position],
                cell=list(position),
            )
            return {
                "success": False,
                "position": position,
                "error": f"triggered {trap_lookup[position]} trap at {position}",
                "trace": action_trace,
            }
    if position != target:
        return {
            "success": False,
            "position": position,
            "error": f"plan ended at {position}, expected {target}",
            "trace": action_trace,
        }
    return {"success": True, "position": position, "error": None, "trace": action_trace}


def execute_feedback_noop(
    *,
    data: dict[str, Any],
    agent: str,
    position: tuple[int, int],
    requested_cell: list[int],
    plan_id: str,
    reason: str,
    issue_code: str,
    trace: AttemptTrace,
    step_state: dict[str, int],
) -> dict[str, Any]:
    """Consume one environment turn while keeping the agent in place."""
    if step_state[agent] >= data["step_limit"]:
        return {
            "success": False,
            "position": position,
            "error": "step limit exceeded",
            "trace": [],
        }
    action_id = trace.artifact(
        "action",
        agent=agent,
        action="STAY",
        requested_cell=requested_cell,
        from_cell=list(position),
        to_cell=list(position),
        source_ids=[plan_id],
        outcome="no_op",
        issue_code=issue_code,
        reason=reason,
    )
    step_state[agent] += 1
    item = {
        "action": "STAY",
        "requested_cell": requested_cell,
        "cell": list(position),
        "artifact_id": action_id,
        "outcome": "no_op",
        "issue_code": issue_code,
        "reason": reason,
    }
    trace.event("action_noop", agent=agent, **item)
    return {"success": True, "position": position, "error": None, "trace": [item]}


def run_attempt(
    data: dict[str, Any],
    backend: Backend,
    *,
    attempt: str,
    fault: dict[str, Any] | None,
    planning_granularity: str,
    chunk_size: int = 3,
    detection_delay_ticks: int | None = None,
    verification_policy: str = "none",
) -> dict[str, Any]:
    attempt_started_ns = perf_counter_ns()
    before_calls = len(backend.call_records)
    validate_map(data)
    if planning_granularity not in {"route", "chunk", "step", "junction"}:
        raise ValueError(f"unsupported planning granularity: {planning_granularity}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if detection_delay_ticks is not None and detection_delay_ticks < 0:
        raise ValueError("detection_delay_ticks must be non-negative or None")
    if verification_policy not in {"none", "trusted-source"}:
        raise ValueError("verification_policy must be none or trusted-source")

    trace = AttemptTrace(attempt)
    starts = {agent: agent_cell(data, agent, "start") for agent in AGENTS}
    goals = {agent: agent_cell(data, agent, "goal") for agent in AGENTS}
    positions = dict(starts)
    position_history = {agent: [list(starts[agent])] for agent in AGENTS}
    step_state = {agent: 0 for agent in AGENTS}

    # Facts are stable semantic artifacts.  Observations are repeated temporal
    # evidence about those facts, and messages are per-delivery artifacts.
    fact_registry: dict[str, dict[str, dict[str, Any]]] = {
        agent: {} for agent in AGENTS
    }
    pending_fact_ids: dict[str, list[str]] = {agent: [] for agent in AGENTS}
    delivered_fact_ids: dict[str, set[str]] = {agent: set() for agent in AGENTS}
    received_memory: dict[str, dict[str, dict[str, Any]]] = {
        agent: {} for agent in AGENTS
    }
    eligible_transmissions: dict[tuple[str, str], int] = {}

    fault_injected = False
    fault_message_id: str | None = None
    fault_injected_tick: int | None = None
    fault_injected_phase: str | None = None
    fault_delivered_to_recipient = False
    fault_exposed_to_planner = False
    fault_consumed = False
    fault_explicitly_consumed = False
    fault_structurally_consumed = False
    fault_plan_relevant = False
    fault_executed_relevant_action = False
    fault_caused_failure = False
    plan_dependency_evaluations: list[dict[str, Any]] = []
    verification_checks = 0
    verification_rejections = 0
    detection: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    global_tick = 0

    trace.event(
        "attempt_started",
        map_id=data["id"],
        world_model=data["world_model"],
        starts={agent: list(cell) for agent, cell in starts.items()},
        goals={agent: list(cell) for agent, cell in goals.items()},
        fault_id=None if fault is None else fault["id"],
        scheduler="lockstep_primitive_tick",
        detection_delay_ticks=detection_delay_ticks,
        verification_policy=verification_policy,
    )

    def record_detection(*, phase_id: str, tick: int, reason: str) -> None:
        nonlocal detection
        if detection is not None or not fault_injected:
            return
        detection = {
            "fault_id": None if fault is None else fault["id"],
            "phase": phase_id,
            "tick": tick,
            "reason": reason,
            "injection_phase": fault_injected_phase,
            "injection_tick": fault_injected_tick,
            "observed_delay_ticks": (
                None
                if fault_injected_tick is None
                else max(0, tick - fault_injected_tick)
            ),
            "backend_call_count": len(backend.call_records) - before_calls,
            "artifact_count": len(trace.artifacts),
            "event_count": len(trace.events),
            "executed_action_count": sum(
                event["event_type"] == "action_executed" for event in trace.events
            ),
            "safety_violations": sum(
                event["event_type"] == "trap_triggered" for event in trace.events
            ),
            "elapsed_ms": elapsed_ms(attempt_started_ns),
        }
        trace.event("fault_detected", **detection)

    def observe(phase_id: str, tick: int) -> None:
        for agent in AGENTS:
            public_cells = visible_public_cells(data, positions[agent])
            visible_facts = make_sensor_facts(
                data, agent, positions[agent], attempt, phase_id
            )
            observation_id = trace.artifact(
                "observation",
                agent=agent,
                phase=phase_id,
                tick=tick,
                position=list(positions[agent]),
                public_cells=public_cells,
                sensor=data["agents"][agent]["sensor"],
                sensor_fact_ids=[fact["artifact_id"] for fact in visible_facts],
            )
            new_fact_ids = []
            repeated_fact_ids = []
            for fact in visible_facts:
                fact_id = fact["artifact_id"]
                if fact_id not in fact_registry[agent]:
                    canonical = {
                        **fact,
                        "artifact_type": "sensor_fact",
                        "first_observed_phase": phase_id,
                        "first_observed_tick": tick,
                        "source_ids": [observation_id],
                    }
                    fact_registry[agent][fact_id] = canonical
                    trace.artifacts.append(canonical)
                    pending_fact_ids[agent].append(fact_id)
                    new_fact_ids.append(fact_id)
                    trace.event(
                        "sensor_fact_discovered",
                        agent=agent,
                        phase=phase_id,
                        tick=tick,
                        sensor_fact_id=fact_id,
                        hazard=fact["hazard"],
                        cell=fact["cell"],
                        observation_artifact_id=observation_id,
                    )
                else:
                    repeated_fact_ids.append(fact_id)
                    trace.event(
                        "sensor_fact_reobserved",
                        agent=agent,
                        phase=phase_id,
                        tick=tick,
                        sensor_fact_id=fact_id,
                        observation_artifact_id=observation_id,
                    )
            trace.event(
                "observation_created",
                agent=agent,
                phase=phase_id,
                tick=tick,
                observation_artifact_id=observation_id,
                sensor_fact_ids=[fact["artifact_id"] for fact in visible_facts],
                new_sensor_fact_ids=new_fact_ids,
                repeated_sensor_fact_ids=repeated_fact_ids,
            )

    def communicate_pending(phase_id: str, tick: int) -> None:
        nonlocal fault_injected, fault_message_id, fault_injected_tick
        nonlocal fault_injected_phase
        nonlocal fault_delivered_to_recipient
        nonlocal verification_checks, verification_rejections
        for agent in AGENTS:
            pending = [
                fact_registry[agent][fact_id]
                for fact_id in pending_fact_ids[agent]
            ]
            if not pending:
                continue
            raw_message = backend.communicate(
                agent=agent,
                phase=phase_id,
                sensor_facts=pending,
                limit=data["message_fact_limit"],
            )
            normalized = validate_message_output(
                raw_message, pending, data["message_fact_limit"]
            )
            selected_source_ids = [
                item["source_fact_id"] for item in normalized["message_facts"]
            ]
            if not selected_source_ids:
                raise EpisodeError(
                    f"{agent} has pending facts but produced an empty communication"
                )

            delivered = normalized
            injected_here = False
            if (
                fault is not None
                and not fault_injected
                and fault["source_agent"] == agent
                and fault["phase"] == phase_id
            ):
                matching_items = [
                    item
                    for item in normalized["message_facts"]
                    if item["hazard"] == fault["hazard"]
                    and as_cell(item["cell"]) == as_cell(fault["true_cell"])
                ]
                if matching_items:
                    semantic_fact_id = matching_items[0]["source_fact_id"]
                    key = (agent, semantic_fact_id)
                    eligible_transmissions[key] = eligible_transmissions.get(key, 0) + 1
                    occurrence = int(fault.get("transmission_occurrence", 1))
                    if eligible_transmissions[key] == occurrence:
                        delivered, injected_here = inject_fault(normalized, fault)
                        fault_injected = injected_here

            message_id = trace.artifact(
                "message",
                agent=agent,
                recipient=PARTNER[agent],
                phase=phase_id,
                tick=tick,
                source_ids=selected_source_ids,
                content=delivered,
                controlled_fault_id=fault["id"] if injected_here and fault else None,
                lifecycle_status="faulty" if injected_here else "valid",
            )
            if injected_here:
                fault_message_id = message_id
                fault_injected_tick = tick
                fault_injected_phase = phase_id
                trace.event(
                    "fault_injected",
                    fault_id=fault["id"],
                    message_artifact_id=message_id,
                    declared_phase=fault.get("phase"),
                    actual_phase=phase_id,
                    tick=tick,
                    transmission_occurrence=fault.get("transmission_occurrence", 1),
                    selection_rule=fault.get("selection_rule"),
                )

            recipient = PARTNER[agent]
            effective_message_id = message_id
            effective_delivery = delivered
            if verification_policy == "trusted-source":
                verification_checks += 1
                expected_items = [
                    {
                        "source_fact_id": fact_id,
                        "hazard": fact_registry[agent][fact_id]["hazard"],
                        "cell": fact_registry[agent][fact_id]["cell"],
                    }
                    for fact_id in selected_source_ids
                ]
                actual_items = delivered["message_facts"]
                verified = actual_items == expected_items
                verification_id = trace.artifact(
                    "verification",
                    agent="trusted-source-verifier",
                    phase=phase_id,
                    tick=tick,
                    message_artifact_id=message_id,
                    source_fact_ids=selected_source_ids,
                    valid=verified,
                    source_ids=[message_id, *selected_source_ids],
                )
                trace.event(
                    "message_verified" if verified else "message_verification_failed",
                    verification_artifact_id=verification_id,
                    message_artifact_id=message_id,
                    phase=phase_id,
                    tick=tick,
                )
                if not verified:
                    verification_rejections += 1
                    effective_delivery = {
                        "message_facts": expected_items,
                        "note": "trusted-source verifier regenerated rejected message",
                    }
                    effective_message_id = trace.artifact(
                        "message",
                        agent=agent,
                        recipient=recipient,
                        phase=phase_id,
                        tick=tick,
                        source_ids=[verification_id, *selected_source_ids],
                        content=effective_delivery,
                        controlled_fault_id=None,
                        lifecycle_status="recomputed",
                        replaces_message_artifact_id=message_id,
                    )
                    trace.event(
                        "message_regenerated_after_verification",
                        rejected_message_artifact_id=message_id,
                        regenerated_message_artifact_id=effective_message_id,
                        verification_artifact_id=verification_id,
                        phase=phase_id,
                        tick=tick,
                    )

            effective_source_ids = {
                item["source_fact_id"]
                for item in effective_delivery["message_facts"]
            }
            pending_fact_ids[agent] = [
                fact_id
                for fact_id in pending_fact_ids[agent]
                if fact_id not in effective_source_ids
            ]
            delivered_fact_ids[agent].update(effective_source_ids)

            if injected_here and effective_message_id == message_id:
                fault_delivered_to_recipient = True
            delivered_memory_ids = []
            for item in effective_delivery["message_facts"]:
                memory_id = trace.artifact(
                    "memory_entry",
                    agent=recipient,
                    source_agent=agent,
                    semantic_fact_id=item["source_fact_id"],
                    hazard=item["hazard"],
                    cell=item["cell"],
                    phase=phase_id,
                    tick=tick,
                    source_ids=[effective_message_id],
                )
                received_memory[recipient][item["source_fact_id"]] = {
                    "artifact_id": memory_id,
                    "message_artifact_id": effective_message_id,
                    "source_agent": agent,
                    "fact": item,
                }
                delivered_memory_ids.append(memory_id)
            trace.event(
                "message_delivered",
                message_artifact_id=effective_message_id,
                original_message_artifact_id=message_id,
                sender=agent,
                recipient=recipient,
                phase=phase_id,
                tick=tick,
                source_fact_ids=sorted(effective_source_ids),
                memory_artifact_ids=delivered_memory_ids,
            )

    def available_facts_for(agent: str) -> list[dict[str, Any]]:
        started_ns = perf_counter_ns()
        available: list[dict[str, Any]] = []
        for fact in fact_registry[agent].values():
            available.append(
                {
                    "hazard": fact["hazard"],
                    "cell": fact["cell"],
                    "dependency_id": fact["artifact_id"],
                    "provenance": "own_sensor",
                }
            )
        for memory in received_memory[agent].values():
            item = memory["fact"]
            available.append(
                {
                    "hazard": item["hazard"],
                    "cell": item["cell"],
                    # The exact message, not the timeless physical fact, is the
                    # dependency that may later be declared faulty.
                    "dependency_id": memory["message_artifact_id"],
                    "memory_artifact_id": memory["artifact_id"],
                    "semantic_fact_id": item["source_fact_id"],
                    "provenance": f"message_from_{memory['source_agent']}",
                }
            )
        available.sort(
            key=lambda item: (
                item["hazard"], item["cell"][1], item["cell"][0], item["dependency_id"]
            )
        )
        trace.add_tracking_time(started_ns)
        return available

    def queued_cells(
        position: tuple[int, int], actions: list[str]
    ) -> list[tuple[int, int]]:
        cells = []
        cursor = position
        for action in actions:
            dx, dy = MOVES[action]
            cursor = cursor[0] + dx, cursor[1] + dy
            cells.append(cursor)
        return cells

    def fault_plan_labels(
        *,
        input_ids: list[str],
        declared_ids: list[str],
        compiler_ids: list[str],
        predicted_cells: list[list[int]],
    ) -> dict[str, Any]:
        context_exposed = bool(fault_message_id and fault_message_id in input_ids)
        declared = bool(fault_message_id and fault_message_id in declared_ids)
        structured = bool(fault_message_id and fault_message_id in compiler_ids)
        counterfactually_relevant = bool(
            context_exposed
            and fault is not None
            and list(as_cell(fault["true_cell"])) in predicted_cells
        )
        return {
            "fault_context_exposed": context_exposed,
            "fault_declared_dependency": declared,
            "fault_structured_dependency": structured,
            "fault_counterfactually_relevant_plan": counterfactually_relevant,
            "fault_addition_effect_unmeasured": bool(
                context_exposed
                and fault is not None
                and fault.get("kind") == "coordinate_shift"
            ),
        }

    def record_fault_plan(
        *,
        agent: str,
        phase_id: str,
        tick: int,
        plan_artifact_id: str,
        labels: dict[str, Any],
    ) -> None:
        nonlocal fault_consumed, fault_explicitly_consumed
        nonlocal fault_structurally_consumed, fault_exposed_to_planner
        nonlocal fault_plan_relevant
        if labels["fault_context_exposed"]:
            fault_exposed_to_planner = True
            evaluation = {
                "agent": agent,
                "phase": phase_id,
                "tick": tick,
                "plan_artifact_id": plan_artifact_id,
                "declared": labels["fault_declared_dependency"],
                "structured": labels["fault_structured_dependency"],
                "conservative": True,
                "counterfactually_relevant": labels[
                    "fault_counterfactually_relevant_plan"
                ],
            }
            plan_dependency_evaluations.append(evaluation)
            trace.event("fault_plan_dependency_evaluated", **evaluation)
        if labels["fault_declared_dependency"]:
            fault_explicitly_consumed = True
        if labels["fault_structured_dependency"]:
            fault_structurally_consumed = True
        if labels["fault_counterfactually_relevant_plan"]:
            fault_plan_relevant = True
        if labels["fault_declared_dependency"] or labels["fault_structured_dependency"]:
            fault_consumed = True
            trace.event(
                "fault_consumed",
                agent=agent,
                plan_artifact_id=plan_artifact_id,
                faulty_message_artifact_id=fault_message_id,
                declared=labels["fault_declared_dependency"],
                structured=labels["fault_structured_dependency"],
            )

    for agent in AGENTS:
        prelude = execute_segment(
            data=data,
            agent=agent,
            position=positions[agent],
            actions=agent_actions(data, "prelude_actions", agent),
            target=phase_cell(data["phases"][0], "start", agent),
            plan_id=f"{attempt}:deterministic:prelude",
            trace=trace,
            step_state=step_state,
        )
        positions[agent] = prelude["position"]
        position_history[agent].extend(item["cell"] for item in prelude["trace"])
        if not prelude["success"]:
            failure = {"agent": agent, "phase": "prelude", **prelude}
            break

    if failure is None:
        for phase_spec in data["phases"]:
            phase_id = phase_spec["id"]
            states: dict[str, dict[str, Any]] = {}
            for agent in AGENTS:
                phase = phase_for_agent(phase_spec, agent)
                states[agent] = {
                    "phase": phase,
                    "target": as_cell(phase["target"]),
                    "phase_steps": 0,
                    "action_trace": [],
                    "plan_ids": [],
                    "queue": [],
                    "queue_plan_id": None,
                    "pending_noop": None,
                    "last_action_feedback": None,
                    "phase_position_history": [list(positions[agent])],
                    "error": None,
                }

            phase_tick = 0
            while not all(
                positions[agent] == states[agent]["target"] for agent in AGENTS
            ):
                global_tick += 1
                phase_tick += 1
                trace.event(
                    "round_started",
                    phase=phase_id,
                    tick=global_tick,
                    phase_tick=phase_tick,
                    positions={agent: list(positions[agent]) for agent in AGENTS},
                )

                observe(phase_id, global_tick)
                communicate_pending(phase_id, global_tick)
                if (
                    detection_delay_ticks is not None
                    and fault_injected_tick is not None
                    and global_tick >= fault_injected_tick + detection_delay_ticks
                ):
                    record_detection(
                        phase_id=phase_id,
                        tick=global_tick,
                        reason=f"controlled_delay_{detection_delay_ticks}_ticks",
                    )

                available_by_agent = {
                    agent: available_facts_for(agent) for agent in AGENTS
                }

                # A multi-action plan is only a queued intention.  New sensor or
                # message evidence can interrupt it before the next primitive step.
                for agent in AGENTS:
                    state = states[agent]
                    if not state["queue"]:
                        continue
                    known_hazards = {
                        as_cell(item["cell"])
                        for item in available_by_agent[agent]
                    }
                    unsafe_cells = [
                        cell
                        for cell in queued_cells(positions[agent], state["queue"])
                        if cell in known_hazards
                    ]
                    if unsafe_cells:
                        trace.event(
                            "plan_interrupted",
                            agent=agent,
                            phase=phase_id,
                            tick=global_tick,
                            plan_artifact_id=state["queue_plan_id"],
                            reason="newly_known_hazard_on_remaining_path",
                            hazard_cells=[list(cell) for cell in unsafe_cells],
                            discarded_actions=list(state["queue"]),
                        )
                        state["queue"] = []
                        state["queue_plan_id"] = None

                # Both agents plan from the same round snapshot before either moves.
                for agent in AGENTS:
                    state = states[agent]
                    target = state["target"]
                    if positions[agent] == target or state["queue"] or state["pending_noop"]:
                        continue
                    phase = state["phase"]
                    if state["phase_steps"] >= phase["max_actions"]:
                        state["error"] = (
                            f"phase action budget exceeded before reaching {target}"
                        )
                        continue

                    available_facts = available_by_agent[agent]
                    input_ids = sorted(
                        {fact["dependency_id"] for fact in available_facts}
                    )
                    public_cells = visible_public_cells(data, positions[agent])
                    transitions = {
                        as_cell(item["from"]): item["moves"]
                        for item in public_transition_graph(public_cells)
                    }
                    if planning_granularity in {"step", "chunk", "route"}:
                        candidate_moves = transitions.get(positions[agent], {})
                    else:
                        candidate_moves = shortest_progress_moves(
                            position=positions[agent],
                            target=target,
                            transitions=transitions,
                        )
                    if planning_granularity != "route" and not candidate_moves:
                        state["error"] = (
                            f"no public route from {positions[agent]} to {target}"
                        )
                        continue

                    remaining_actions = phase["max_actions"] - state["phase_steps"]
                    needs_model_decision = (
                        planning_granularity in {"route", "chunk", "step"}
                        or len(candidate_moves) > 1
                    )
                    if not needs_model_decision:
                        next_action = next(iter(candidate_moves))
                        state["queue"] = [next_action]
                        state["queue_plan_id"] = (
                            state["queue_plan_id"]
                            or f"{attempt}:deterministic:{phase_id}:{agent}"
                        )
                        trace.event(
                            "corridor_action_selected",
                            agent=agent,
                            phase=phase_id,
                            tick=global_tick,
                            action=next_action,
                            position=list(positions[agent]),
                            source_plan_artifact_id=state["queue_plan_id"],
                        )
                        continue

                    raw_plan = backend.plan(
                        data=data,
                        agent=agent,
                        phase=phase,
                        position=positions[agent],
                        public_cells=public_cells,
                        available_facts=available_facts,
                        candidate_moves=(
                            {} if planning_granularity == "route" else candidate_moves
                        ),
                        visited_cells=position_history[agent],
                        remaining_actions=remaining_actions,
                        planning_granularity=planning_granularity,
                        chunk_size=chunk_size,
                        last_action_feedback=state["last_action_feedback"],
                    )

                    if planning_granularity == "route":
                        plan = validate_route_plan_output(
                            raw_plan,
                            allowed_dependency_ids=set(input_ids),
                            current_cell=positions[agent],
                            target=target,
                            public_cells=public_cells,
                            available_facts=available_facts,
                            remaining_actions=remaining_actions,
                        )
                        compiler_dependency_ids = sorted(
                            {fact["dependency_id"] for fact in available_facts}
                        )
                        fault_labels = fault_plan_labels(
                            input_ids=input_ids,
                            declared_ids=plan["declared_dependency_ids"],
                            compiler_ids=compiler_dependency_ids,
                            predicted_cells=plan["execution_cells"],
                        )
                        plan_id = trace.artifact(
                            "plan",
                            agent=agent,
                            phase=phase_id,
                            tick=global_tick,
                            planning_granularity=planning_granularity,
                            current_cell=list(positions[agent]),
                            waypoint=plan["waypoint"],
                            subgoal=plan["subgoal"],
                            proposed_path_cells=plan["path_cells"],
                            compiled_actions=plan["compiled_actions"],
                            compiled_cells=plan["execution_cells"],
                            declared_dependency_ids=plan["declared_dependency_ids"],
                            compiler_dependency_ids=compiler_dependency_ids,
                            input_artifact_ids=input_ids,
                            source_ids=plan["declared_dependency_ids"],
                            reason=plan["reason"],
                            **fault_labels,
                        )
                        state["queue"] = list(plan["compiled_actions"])
                        trace.event(
                            "coordinate_path_plan_created",
                            agent=agent,
                            phase=phase_id,
                            tick=global_tick,
                            plan_artifact_id=plan_id,
                            waypoint=plan["waypoint"],
                            compiled_steps=len(plan["compiled_actions"]),
                        )
                        record_fault_plan(
                            agent=agent,
                            phase_id=phase_id,
                            tick=global_tick,
                            plan_artifact_id=plan_id,
                            labels=fault_labels,
                        )
                    elif planning_granularity == "chunk":
                        plan = validate_chunk_plan_output(
                            raw_plan,
                            allowed_dependency_ids=set(input_ids),
                            current_cell=positions[agent],
                            target=target,
                            public_cells=public_cells,
                            available_facts=available_facts,
                            remaining_actions=remaining_actions,
                            chunk_size=chunk_size,
                        )
                        compiler_dependency_ids = sorted(
                            {fact["dependency_id"] for fact in available_facts}
                        )
                        fault_labels = fault_plan_labels(
                            input_ids=input_ids,
                            declared_ids=plan["declared_dependency_ids"],
                            compiler_ids=compiler_dependency_ids,
                            predicted_cells=plan["execution_cells"],
                        )
                        plan_id = trace.artifact(
                            "plan",
                            agent=agent,
                            phase=phase_id,
                            tick=global_tick,
                            planning_granularity=planning_granularity,
                            current_cell=list(positions[agent]),
                            waypoint=plan["waypoint"],
                            direct_actions=plan["actions"],
                            predicted_cells=plan["execution_cells"],
                            chunk_size_limit=min(chunk_size, remaining_actions),
                            declared_dependency_ids=plan["declared_dependency_ids"],
                            compiler_dependency_ids=compiler_dependency_ids,
                            input_artifact_ids=input_ids,
                            source_ids=plan["declared_dependency_ids"],
                            reason=plan["reason"],
                            **fault_labels,
                        )
                        state["queue"] = list(plan["actions"])
                        trace.event(
                            "action_chunk_plan_created",
                            agent=agent,
                            phase=phase_id,
                            tick=global_tick,
                            plan_artifact_id=plan_id,
                            waypoint=plan["waypoint"],
                            direct_actions=plan["actions"],
                        )
                        record_fault_plan(
                            agent=agent,
                            phase_id=phase_id,
                            tick=global_tick,
                            plan_artifact_id=plan_id,
                            labels=fault_labels,
                        )
                    elif planning_granularity == "step":
                        plan = assess_step_plan_output(
                            raw_plan,
                            allowed_dependency_ids=set(input_ids),
                            current_cell=positions[agent],
                            candidate_moves=candidate_moves,
                            available_facts=available_facts,
                        )
                        fault_labels = fault_plan_labels(
                            input_ids=input_ids,
                            declared_ids=plan["valid_dependency_ids"],
                            compiler_ids=[],
                            predicted_cells=(
                                [plan["next_cell"]] if plan["accepted"] else []
                            ),
                        )
                        plan_id = trace.artifact(
                            "plan",
                            agent=agent,
                            phase=phase_id,
                            tick=global_tick,
                            planning_granularity=planning_granularity,
                            current_cell=list(positions[agent]),
                            next_cell=plan["next_cell"],
                            public_next_cells=sorted(
                                candidate_moves.values(),
                                key=lambda cell: (cell[1], cell[0]),
                            ),
                            decision_status="accepted" if plan["accepted"] else "no_op",
                            decision_issue_code=plan["issue_code"],
                            decision_feedback=plan["feedback"],
                            known_hazard_selected=plan["known_hazard_selected"],
                            declared_dependency_ids=plan["declared_dependency_ids"],
                            input_artifact_ids=input_ids,
                            source_ids=plan["valid_dependency_ids"],
                            reason=plan["reason"],
                            **fault_labels,
                        )
                        trace.event(
                            "step_decision_created",
                            agent=agent,
                            phase=phase_id,
                            tick=global_tick,
                            plan_artifact_id=plan_id,
                            next_cell=plan["next_cell"],
                            public_next_cells=sorted(candidate_moves.values()),
                            accepted=plan["accepted"],
                            issue_code=plan["issue_code"],
                            known_hazard_selected=plan["known_hazard_selected"],
                        )
                        if plan["known_hazard_selected"]:
                            trace.event(
                                "known_hazard_selected",
                                agent=agent,
                                phase=phase_id,
                                tick=global_tick,
                                plan_artifact_id=plan_id,
                                cell=plan["next_cell"],
                            )
                        if plan["accepted"]:
                            state["queue"] = [plan["next_action"]]
                        else:
                            state["pending_noop"] = {
                                "requested_cell": plan["next_cell"],
                                "reason": plan["feedback"] or "invalid step decision",
                                "issue_code": plan["issue_code"] or "invalid_step",
                            }
                        record_fault_plan(
                            agent=agent,
                            phase_id=phase_id,
                            tick=global_tick,
                            plan_artifact_id=plan_id,
                            labels=fault_labels,
                        )
                    else:
                        plan = validate_plan_output(
                            raw_plan,
                            allowed_dependency_ids=set(input_ids),
                            allowed_actions=set(candidate_moves),
                        )
                        fault_labels = fault_plan_labels(
                            input_ids=input_ids,
                            declared_ids=plan["declared_dependency_ids"],
                            compiler_ids=[],
                            predicted_cells=[
                                list(candidate_moves[plan["next_action"]])
                            ],
                        )
                        plan_id = trace.artifact(
                            "plan",
                            agent=agent,
                            phase=phase_id,
                            tick=global_tick,
                            planning_granularity=planning_granularity,
                            current_cell=list(positions[agent]),
                            next_action=plan["next_action"],
                            candidate_moves=candidate_moves,
                            declared_dependency_ids=plan["declared_dependency_ids"],
                            input_artifact_ids=input_ids,
                            source_ids=plan["declared_dependency_ids"],
                            reason=plan["reason"],
                            **fault_labels,
                        )
                        state["queue"] = [plan["next_action"]]
                        trace.event(
                            "junction_decision_created",
                            agent=agent,
                            phase=phase_id,
                            tick=global_tick,
                            plan_artifact_id=plan_id,
                            next_action=plan["next_action"],
                            candidate_actions=sorted(candidate_moves),
                        )
                        record_fault_plan(
                            agent=agent,
                            phase_id=phase_id,
                            tick=global_tick,
                            plan_artifact_id=plan_id,
                            labels=fault_labels,
                        )

                    state["queue_plan_id"] = plan_id
                    state["plan_ids"].append(plan_id)

                planning_failure = next(
                    (
                        (agent, states[agent])
                        for agent in AGENTS
                        if states[agent]["error"] is not None
                    ),
                    None,
                )
                if planning_failure is not None:
                    failed_agent, failed_state = planning_failure
                    failure = {
                        "agent": failed_agent,
                        "phase": phase_id,
                        "success": False,
                        "position": positions[failed_agent],
                        "error": failed_state["error"],
                        "trace": failed_state["action_trace"],
                    }
                    break

                # Execute one primitive action (or one feedback no-op) per active
                # agent.  Both plans were produced from the same round snapshot.
                round_failures: list[tuple[str, str]] = []
                for agent in AGENTS:
                    state = states[agent]
                    if positions[agent] == state["target"]:
                        continue
                    plan_id = state["queue_plan_id"] or (
                        f"{attempt}:deterministic:{phase_id}:{agent}"
                    )
                    if state["pending_noop"] is not None:
                        pending_noop = state["pending_noop"]
                        result = execute_feedback_noop(
                            data=data,
                            agent=agent,
                            position=positions[agent],
                            requested_cell=pending_noop["requested_cell"],
                            plan_id=plan_id,
                            reason=pending_noop["reason"],
                            issue_code=pending_noop["issue_code"],
                            trace=trace,
                            step_state=step_state,
                        )
                        state["pending_noop"] = None
                        state["last_action_feedback"] = {
                            "success": False,
                            "requested_cell": pending_noop["requested_cell"],
                            "current_cell": list(positions[agent]),
                            "issue_code": pending_noop["issue_code"],
                            "reason": pending_noop["reason"],
                        }
                    else:
                        action = state["queue"].pop(0)
                        dx, dy = MOVES[action]
                        destination = positions[agent][0] + dx, positions[agent][1] + dy
                        is_fault_relevant_action = bool(
                            fault is not None
                            and fault_message_id is not None
                            and agent == PARTNER[fault["source_agent"]]
                            and destination == as_cell(fault["true_cell"])
                        )
                        result = execute_segment(
                            data=data,
                            agent=agent,
                            position=positions[agent],
                            actions=[action],
                            target=destination,
                            plan_id=plan_id,
                            trace=trace,
                            step_state=step_state,
                        )
                        positions[agent] = result["position"]
                        if is_fault_relevant_action:
                            fault_executed_relevant_action = True
                            trace.event(
                                "fault_counterfactual_action_executed",
                                agent=agent,
                                phase=phase_id,
                                tick=global_tick,
                                plan_artifact_id=plan_id,
                                destination=list(destination),
                                faulty_message_artifact_id=fault_message_id,
                            )
                            if not result["success"] and "trap" in str(
                                result["error"]
                            ).lower():
                                fault_caused_failure = True
                                trace.event(
                                    "fault_caused_failure",
                                    agent=agent,
                                    phase=phase_id,
                                    tick=global_tick,
                                    plan_artifact_id=plan_id,
                                    destination=list(destination),
                                    faulty_message_artifact_id=fault_message_id,
                                )
                        state["last_action_feedback"] = {
                            "success": result["success"],
                            "executed_actions": [action],
                            "requested_cell": list(destination),
                            "current_cell": list(positions[agent]),
                            "issue_code": None if result["success"] else "execution_failure",
                            "reason": result["error"],
                        }

                    state["action_trace"].extend(result["trace"])
                    position_history[agent].extend(
                        item["cell"] for item in result["trace"]
                    )
                    state["phase_position_history"].extend(
                        item["cell"] for item in result["trace"]
                    )
                    state["phase_steps"] += 1
                    if not result["success"]:
                        round_failures.append((agent, result["error"]))
                    elif positions[agent] != state["target"]:
                        loop = detect_navigation_loop(state["phase_position_history"])
                        if loop is not None:
                            trace.event(
                                "navigation_loop_detected",
                                agent=agent,
                                phase=phase_id,
                                tick=global_tick,
                                plan_artifact_id=plan_id,
                                **loop,
                            )
                            round_failures.append(
                                (
                                    agent,
                                    "navigation loop detected: "
                                    f"period {loop['period']} repeated "
                                    f"{loop['repetitions']} times",
                                )
                            )
                    if positions[agent] == state["target"] and state["queue"]:
                        trace.event(
                            "plan_queue_discarded_at_target",
                            agent=agent,
                            phase=phase_id,
                            tick=global_tick,
                            plan_artifact_id=plan_id,
                            discarded_actions=list(state["queue"]),
                        )
                        state["queue"] = []

                trace.event(
                    "round_finished",
                    phase=phase_id,
                    tick=global_tick,
                    phase_tick=phase_tick,
                    positions={agent: list(positions[agent]) for agent in AGENTS},
                )
                if round_failures:
                    failed_agent, error = round_failures[0]
                    failed_state = states[failed_agent]
                    failure = {
                        "agent": failed_agent,
                        "phase": phase_id,
                        "success": False,
                        "position": positions[failed_agent],
                        "error": error,
                        "trace": failed_state["action_trace"],
                    }
                    break

            if failure is not None:
                break

    if failure is None:
        for agent in AGENTS:
            finale = execute_segment(
                data=data,
                agent=agent,
                position=positions[agent],
                actions=agent_actions(data, "finale_actions", agent),
                target=goals[agent],
                plan_id=f"{attempt}:deterministic:finale",
                trace=trace,
                step_state=step_state,
            )
            positions[agent] = finale["position"]
            position_history[agent].extend(item["cell"] for item in finale["trace"])
            if not finale["success"] and failure is None:
                failure = {"agent": agent, "phase": "finale", **finale}

    if fault_injected and detection is None:
        record_detection(
            phase_id=(failure or {}).get("phase", "outcome"),
            tick=global_tick,
            reason="outcome",
        )

    success = failure is None and all(
        positions[agent] == goals[agent] for agent in AGENTS
    )
    trap_entries = sum(
        event["event_type"] == "trap_triggered" for event in trace.events
    )
    trace.event(
        "attempt_finished",
        success=success,
        final_positions={agent: list(cell) for agent, cell in positions.items()},
        failure=failure,
        ticks=global_tick,
        trap_entries=trap_entries,
        communication_state={
            agent: {
                "discovered": len(fact_registry[agent]),
                "delivered": len(delivered_fact_ids[agent]),
                "pending": len(pending_fact_ids[agent]),
            }
            for agent in AGENTS
        },
    )
    attempt_records = backend.call_records[before_calls:]
    attempt_llm_latency_ms = round(
        sum(
            float(record.get("latency_ms", 0.0))
            for record in attempt_records
            if record.get("backend") == "llm"
        ),
        3,
    )

    def dependency_confusion(method: str) -> dict[str, Any]:
        counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for evaluation in plan_dependency_evaluations:
            predicted = bool(evaluation[method])
            actual = bool(evaluation["counterfactually_relevant"])
            if predicted and actual:
                counts["tp"] += 1
            elif predicted:
                counts["fp"] += 1
            elif actual:
                counts["fn"] += 1
            else:
                counts["tn"] += 1
        positive_predictions = counts["tp"] + counts["fp"]
        actual_positives = counts["tp"] + counts["fn"]
        return {
            **counts,
            "precision": (
                None
                if positive_predictions == 0
                else round(counts["tp"] / positive_predictions, 4)
            ),
            "recall": (
                None
                if actual_positives == 0
                else round(counts["tp"] / actual_positives, 4)
            ),
        }

    return {
        "attempt": attempt,
        "success": success,
        "final_positions": {agent: list(cell) for agent, cell in positions.items()},
        "steps": step_state,
        "ticks": global_tick,
        "trap_entries": trap_entries,
        "safety_violations": trap_entries,
        "communication_state": {
            agent: {
                "discovered": len(fact_registry[agent]),
                "delivered": len(delivered_fact_ids[agent]),
                "pending": len(pending_fact_ids[agent]),
            }
            for agent in AGENTS
        },
        "fault_id": None if fault is None else fault["id"],
        "fault_class": None if fault is None else fault.get("class", "unspecified"),
        "fault_selection_rule": None if fault is None else fault.get("selection_rule"),
        "fault_declared_phase": None if fault is None else fault.get("phase"),
        "fault_injection_phase": fault_injected_phase,
        "fault_injection_tick": fault_injected_tick,
        "fault_injected": fault_injected,
        "fault_delivered_to_recipient": fault_delivered_to_recipient,
        "fault_exposed_to_planner": fault_exposed_to_planner,
        "fault_message_artifact_id": fault_message_id,
        "fault_consumed": fault_consumed,
        "fault_explicitly_consumed": fault_explicitly_consumed,
        "fault_structurally_consumed": fault_structurally_consumed,
        "fault_counterfactually_relevant_plan": fault_plan_relevant,
        "fault_counterfactually_relevant_executed": (
            fault_executed_relevant_action
        ),
        "fault_caused_failure": fault_caused_failure,
        "fault_affected_stages": sorted(
            {
                evaluation["phase"]
                for evaluation in plan_dependency_evaluations
                if evaluation["declared"]
                or evaluation["structured"]
                or evaluation["counterfactually_relevant"]
            }
        ),
        "affected_stage_count": len(
            {
                evaluation["phase"]
                for evaluation in plan_dependency_evaluations
                if evaluation["declared"]
                or evaluation["structured"]
                or evaluation["counterfactually_relevant"]
            }
        ),
        # The public action validator never executes a rejected move.  We still
        # report attempted known-hazard moves separately.
        "unauthorized_actions": 0,
        "blocked_unsafe_action_attempts": sum(
            event["event_type"] == "known_hazard_selected"
            for event in trace.events
        ),
        "fault_plan_evaluations": plan_dependency_evaluations,
        "dependency_quality": {
            method: dependency_confusion(method)
            for method in ("declared", "structured", "conservative")
        },
        "verification_policy": verification_policy,
        "verification_checks": verification_checks,
        "verification_rejections": verification_rejections,
        "detection": detection,
        "failure": failure,
        "latency_ms": {
            "end_to_end": elapsed_ms(attempt_started_ns),
            "llm": attempt_llm_latency_ms,
            "dependency_tracking": trace.tracking_latency_ms,
        },
        "artifacts": trace.artifacts,
        "events": trace.events,
    }


def sum_usage(records: list[dict[str, Any]]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for record in records:
        usage = record.get("usage", {})
        for key in totals:
            value = usage.get(key, 0)
            if isinstance(value, int):
                totals[key] += value
    return totals


def find_fault(data: dict[str, Any], fault_id: str | None) -> dict[str, Any] | None:
    if fault_id is None:
        return None
    for fault in data.get("faults", []):
        if fault["id"] == fault_id:
            return fault
    raise ValueError(f"unknown fault id: {fault_id}")


def run_experiment(
    data: dict[str, Any],
    backend: Backend,
    *,
    mode: str,
    fault_id: str | None,
    planning_granularity: str = "step",
    chunk_size: int = 3,
    detection_delay_ticks: int | None = None,
) -> dict[str, Any]:
    experiment_started_ns = perf_counter_ns()
    if mode not in {"clean", "no-recovery", "full-restart", "always-verify"}:
        raise ValueError(f"unsupported mode: {mode}")
    fault = None if mode == "clean" else find_fault(data, fault_id)
    if mode != "clean" and fault is None:
        raise ValueError(f"mode {mode} requires --fault")
    verification_policy = "trusted-source" if mode == "always-verify" else "none"

    before_calls = len(backend.call_records)
    first = run_attempt(
        data,
        backend,
        attempt="attempt-1",
        fault=fault,
        planning_granularity=planning_granularity,
        chunk_size=chunk_size,
        detection_delay_ticks=detection_delay_ticks,
        verification_policy=verification_policy,
    )
    first_record_count = len(backend.call_records) - before_calls
    attempts = [first]
    recovery_triggered = False
    recovery_latency_ms: float | None = None
    if mode == "full-restart" and first["fault_injected"]:
        recovery_triggered = True
        recovery_started_ns = perf_counter_ns()
        attempts.append(
            run_attempt(
                data,
                backend,
                attempt="attempt-2-restart",
                fault=None,
                planning_granularity=planning_granularity,
                chunk_size=chunk_size,
                detection_delay_ticks=None,
                verification_policy="none",
            )
        )
        recovery_latency_ms = elapsed_ms(recovery_started_ns)

    final_success = attempts[-1]["success"]
    trap_entries = sum(attempt["trap_entries"] for attempt in attempts)
    ever_failed = any(not attempt["success"] for attempt in attempts)
    recovered_success = recovery_triggered and final_success
    clean_episode_success = final_success and not ever_failed and trap_entries == 0
    records = backend.call_records[before_calls:]
    llm_latency_ms = round(
        sum(
            float(record.get("latency_ms", 0.0))
            for record in records
            if record.get("backend") == "llm"
        ),
        3,
    )
    dependency_tracking_latency_ms = round(
        sum(attempt["latency_ms"]["dependency_tracking"] for attempt in attempts),
        3,
    )
    paired_baselines: dict[str, Any] | None = None
    if mode == "full-restart" and recovery_triggered:
        first_records = records[:first_record_count]
        restart_records = records[first_record_count:]
        detection_snapshot = first.get("detection") or {}
        prefix_call_count = min(
            int(detection_snapshot.get("backend_call_count", len(first_records))),
            len(first_records),
        )
        prefix_records = first_records[:prefix_call_count]
        restart = attempts[-1]
        paired_baselines = {
            "shared_faulty_trajectory": first["attempt"],
            "detection": detection_snapshot,
            "no_recovery": {
                "success": first["success"],
                "trap_entries": first["trap_entries"],
                "backend_calls": len(first_records),
                "llm_calls": sum(
                    record.get("backend") == "llm" for record in first_records
                ),
                "usage": sum_usage(first_records),
                "latency_ms": first["latency_ms"]["end_to_end"],
                "accounting_note": "complete shared faulty trajectory",
            },
            "full_restart": {
                "success": restart["success"],
                "trap_entries": int(detection_snapshot.get("safety_violations", 0))
                + restart["trap_entries"],
                "backend_calls": len(prefix_records) + len(restart_records),
                "llm_calls": sum(
                    record.get("backend") == "llm"
                    for record in [*prefix_records, *restart_records]
                ),
                "usage": sum_usage([*prefix_records, *restart_records]),
                "latency_ms": round(
                    float(detection_snapshot.get("elapsed_ms", 0.0))
                    + restart["latency_ms"]["end_to_end"],
                    3,
                ),
                "diagnostic_suffix_excluded": (
                    len(first_records) - len(prefix_records)
                ),
                "wasted_executed_actions": int(
                    detection_snapshot.get("executed_action_count", 0)
                ),
                "wasted_backend_calls": len(prefix_records),
                "wasted_llm_calls": sum(
                    record.get("backend") == "llm" for record in prefix_records
                ),
                "wasted_usage": sum_usage(prefix_records),
                "accounting_note": (
                    "shared faulty prefix through detection plus clean restart"
                ),
            },
        }
    return {
        "map_id": data["id"],
        "mode": mode,
        "fault_id": fault_id,
        "fault_class": first["fault_class"],
        "fault_selection_rule": first["fault_selection_rule"],
        "fault_declared_phase": first["fault_declared_phase"],
        "fault_injection_phase": first["fault_injection_phase"],
        "fault_injection_tick": first["fault_injection_tick"],
        "planning_granularity": planning_granularity,
        "chunk_size": chunk_size,
        "detection_delay_ticks": detection_delay_ticks,
        "verification_policy": verification_policy,
        "recovery_triggered": recovery_triggered,
        "final_success": final_success,
        "success": final_success,
        "initial_success": first["success"],
        "recovered_success": recovered_success,
        "clean_episode_success": clean_episode_success,
        "ever_failed": ever_failed,
        "attempt_count": len(attempts),
        "trap_entries": trap_entries,
        "safety_violations": trap_entries,
        "fault_metrics": {
            key: first[key]
            for key in (
                "fault_injected",
                "fault_delivered_to_recipient",
                "fault_exposed_to_planner",
                "fault_explicitly_consumed",
                "fault_structurally_consumed",
                "fault_counterfactually_relevant_plan",
                "fault_counterfactually_relevant_executed",
                "fault_caused_failure",
                "affected_stage_count",
                "unauthorized_actions",
                "blocked_unsafe_action_attempts",
            )
        },
        "fault_affected_stages": first["fault_affected_stages"],
        "dependency_quality": first["dependency_quality"],
        "verification_checks": sum(
            attempt["verification_checks"] for attempt in attempts
        ),
        "verification_rejections": sum(
            attempt["verification_rejections"] for attempt in attempts
        ),
        "detection": first["detection"],
        "paired_baselines": paired_baselines,
        "attempts": attempts,
        "backend_calls": len(records),
        "llm_calls": sum(1 for record in records if record.get("backend") == "llm"),
        "usage": sum_usage(records),
        "latency_ms": {
            "end_to_end": elapsed_ms(experiment_started_ns),
            "initial_execution": first["latency_ms"]["end_to_end"],
            "recovery": recovery_latency_ms,
            "llm": llm_latency_ms,
            "dependency_tracking": dependency_tracking_latency_ms,
        },
        "call_records": records,
    }


def run_scheduler_preflight(
    data: dict[str, Any],
    *,
    planning_granularity: str = "chunk",
    chunk_size: int = 3,
) -> dict[str, Any]:
    """Exercise clean and fault cases with the production scheduler.

    This is a deterministic development gate, not evidence about LLM quality.
    Its purpose is to catch timing mistakes such as facts that are discovered
    after the intended consumer has already committed.
    """
    clean = run_experiment(
        data,
        ScriptedBackend(),
        mode="clean",
        fault_id=None,
        planning_granularity=planning_granularity,
        chunk_size=chunk_size,
    )
    fault_checks = []
    for fault in data.get("faults", []):
        result = run_experiment(
            data,
            ScriptedBackend(),
            mode="no-recovery",
            fault_id=fault["id"],
            planning_granularity=planning_granularity,
            chunk_size=chunk_size,
        )
        attempt = result["attempts"][0]
        fault_checks.append(
            {
                "fault_id": fault["id"],
                "fault_class": fault.get("class", "unspecified"),
                "declared_phase": fault.get("phase"),
                "actual_phase": attempt["fault_injection_phase"],
                "actual_tick": attempt["fault_injection_tick"],
                "phase_matches": (
                    attempt["fault_injection_phase"] == fault.get("phase")
                ),
                "fault_injected": attempt["fault_injected"],
                "fault_consumed": attempt["fault_consumed"],
                "fault_delivered_to_recipient": attempt[
                    "fault_delivered_to_recipient"
                ],
                "fault_exposed_to_planner": attempt["fault_exposed_to_planner"],
                "fault_explicitly_consumed": attempt[
                    "fault_explicitly_consumed"
                ],
                "fault_structurally_consumed": attempt[
                    "fault_structurally_consumed"
                ],
                "fault_counterfactually_relevant_plan": attempt[
                    "fault_counterfactually_relevant_plan"
                ],
                "fault_counterfactually_relevant_executed": attempt[
                    "fault_counterfactually_relevant_executed"
                ],
                "fault_caused_failure": attempt["fault_caused_failure"],
                "final_success": result["final_success"],
                "trap_entries": result["trap_entries"],
                "operational_classification": (
                    "outcome_relevant"
                    if attempt["fault_caused_failure"]
                    else "executed_without_task_failure"
                    if attempt["fault_counterfactually_relevant_executed"]
                    else "plan_relevant_without_execution"
                    if attempt["fault_counterfactually_relevant_plan"]
                    else "exposed_without_relevant_plan"
                    if attempt["fault_exposed_to_planner"]
                    else "non_propagating"
                ),
            }
        )
    return {
        "scheduler": "lockstep_primitive_tick",
        "planning_granularity": planning_granularity,
        "chunk_size": chunk_size,
        "clean_success": clean["final_success"],
        "clean_trap_entries": clean["trap_entries"],
        "fault_checks": fault_checks,
        "passed": clean["final_success"]
        and all(
            item["fault_injected"] and item["phase_matches"]
            for item in fault_checks
        ),
    }


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in items))


def write_run(output_dir: Path, config: dict[str, Any], result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    artifacts = [artifact for attempt in result["attempts"] for artifact in attempt["artifacts"]]
    events = [event for attempt in result["attempts"] for event in attempt["events"]]
    write_jsonl(output_dir / "artifacts.jsonl", artifacts)
    write_jsonl(output_dir / "events.jsonl", events)
    summary = {key: value for key, value in result.items() if key not in {"attempts", "call_records"}}
    summary["attempt_summaries"] = [
        {key: value for key, value in attempt.items() if key not in {"artifacts", "events"}}
        for attempt in result["attempts"]
    ]
    (output_dir / "episode-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "calls.json").write_text(json.dumps(result["call_records"], indent=2) + "\n")
