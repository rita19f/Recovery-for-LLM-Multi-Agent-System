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
precommitted official matrix is `configs/evaluation_batch.json`; do not revise
it after inspecting official LLM outcomes.

## Phase 1 matrix

The development batch currently covers:

- clean execution on the small and asymmetric maps;
- relevant accidental coordinate corruption;
- irrelevant accidental omission;
- relevant adversarial targeted coordinate corruption;
- paired no-recovery/full-restart accounting;
- trusted-source always verification;
- three repetitions per condition by default.

At least one delayed-detection batch should be added after immediate-detection
runs are stable. Detection delay is a controlled experimental factor, not an
automated detector.

## Required measurements

- clean and recovered joint success;
- trap entries and other unsafe actions;
- fault injection, delivery, planner exposure, plan relevance, executed harm,
  and caused failure;
- plan-level dependency TP/FP/FN/TN;
- LLM calls and prompt/completion tokens;
- end-to-end, LLM, recovery, and dependency-tracking latency;
- retained work and recomputed work once selective recovery is implemented;
- all failed, malformed, null, skipped, and timed-out runs.

`scripts/aggregate_results.py` converts batch summaries into the canonical
run-level, condition-level, paired-recovery, failure, and Phase 1 overview
files. API/program success and task success are separate metrics; errors and
timeouts remain in the primary all-runs denominator.

## Baseline interpretation

Full restart and no recovery must share the same first faulty trajectory.
Trusted-source always verify is an oracle prevention upper bound. The final
recovery comparison must also include checkpoint restart so that dependency
tracking is credited only for savings beyond ordinary checkpointing.
