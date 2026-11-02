# Experiment Protocol

## Evidence split

`maps/development/` contains maps already used to design prompts, scheduler
semantics, faults, and metrics. Their results are development evidence only.

`maps/evaluation/` is reserved for held-out maps. Before final evaluation:

1. create and validate the maps without an LLM;
2. precommit fault IDs, recovery conditions, detection delays, repetitions, and
   analysis metrics in a new batch manifest;
3. record file hashes and the code revision;
4. freeze prompts and recovery logic;
5. run all declared conditions, retaining failures and timeouts.

Do not move the current development maps into the evaluation split merely by
renaming them.

The current evaluation suite contains three independently constructed maps:

- `coincident_detours.json`: 7 x 7 coincident-start/coincident-goal base case;
- `three_portal_exchange.json`: 9 x 9 medium exchange topology;
- `staged_crossing.json`: 11 x 9 staged topology with delayed observations.

Their geometry and fault scheduling are checked deterministically. The
precommitted evaluation matrix is `configs/evaluation_batch.json`; do not
revise it after inspecting held-out LLM outcomes.

## Evaluation matrix

The development batch currently covers:

- clean execution on the small and asymmetric maps;
- relevant accidental coordinate corruption;
- irrelevant accidental omission;
- relevant adversarial targeted coordinate corruption;
- paired no-recovery/full-restart accounting;
- trusted-source oracle verification;
- provenance-assisted LLM verification on the coincident base map;
- three repetitions per condition by default.

The evaluation matrix includes both immediate detection and one three-tick
delayed-detection condition. Detection delay is a controlled experimental
factor, not an automated detector. It contains 14 conditions and 42 planned
runs at the default three repetitions.

## Required measurements

- clean and recovered joint success;
- trap entries and other unsafe actions;
- fault injection, delivery, planner exposure, structured route relevance,
  executed harm, and caused failure;
- declared versus actual injection phase/tick and observed detection delay;
- affected workflow stages, blocked unsafe-action attempts, and executed
  unauthorized actions;
- plan-level dependency TP/FP/FN/TN;
- LLM calls and prompt/completion tokens;
- verifier TP/FP/FN/TN, calls, tokens, and latency;
- end-to-end, LLM, recovery, verification, and dependency-tracking latency;
- executed primitive actions and structured message count;
- wasted prefix work for full restart, plus retained and recomputed work once
  selective recovery is implemented;
- all failed, malformed, null, skipped, and timed-out runs.

`scripts/aggregate_results.py` converts batch summaries into the canonical
run-level, condition-level, paired-run, paired-condition, failure, and compact
overview files. API/program success and task success are separate metrics;
errors and timeouts remain in the primary all-runs denominator. Fault funnel
rates carry explicit `_completed` and `_all_runs` suffixes.

## Baseline interpretation

Full restart and no recovery must share the same first faulty trajectory.
Trusted-source always verify is an oracle prevention upper bound. `llm-verify`
receives the same trusted source facts, but its accept/reject decision is made
by an additional model call; false positives and false negatives are retained
as outcomes. It is therefore a provenance-assisted verifier rather than an
unassisted fault detector. The final recovery comparison must also include
checkpoint restart so that dependency tracking is credited only for savings
beyond ordinary checkpointing.

`fault_structured_route_relevant_plan` and
`fault_structured_route_relevant_executed` are deterministic operational labels
computed from the frozen route representation. They must not be described as
causal-dependency ground truth.
