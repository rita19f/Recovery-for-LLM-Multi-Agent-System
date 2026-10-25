# Safe Recovery for LLM Multi-Agent Workflows

This repository studies how a multi-agent workflow should recover after a
known message artifact is found to be faulty. Two LLM-controlled agents move in
one partially observable grid: P1 detects fire, P2 detects water, and both must
communicate private hazard observations to reach their goals safely.

The main systems question is whether artifact dependencies can support safe
reuse while avoiding the cost of restarting the complete workflow.

## Current scope

The Phase 1 implementation provides:

- clean and fault-injected shared-world execution;
- accidental and adversarial structured message faults;
- no-recovery, full-restart, and trusted-source always-verify baselines;
- controlled detection timing;
- append-only observations, messages, plans, actions, and dependency metadata;
- paired no-recovery/full-restart accounting from the same faulty trajectory;
- calls, tokens, latency, safety, propagation, and dependency-quality metrics;
- deterministic preflight checks, unit tests, and an interactive trace viewer.

Checkpoint restart and selective recovery are not yet implemented. They are the
next recovery mechanisms to compare against the existing baselines.

## Repository layout

```text
configs/             reproducible batch manifests
docs/                experiment and reproducibility notes
maps/development/    maps used while designing prompts and the harness
maps/evaluation/     LLM-held-out maps frozen before official evaluation
results/summaries/   reviewed aggregate results suitable for version control
scripts/             experiment runner, batch runner, harness, and viewer server
tests/               deterministic unit and integration tests
viewer/              browser-based replay interface
```

Raw runs are written to `results/runs/` and batch executions to
`results/batches/`. Both directories are ignored by Git because they can be
large and may contain complete prompts. Reviewed aggregate tables belong in
`results/summaries/`.

## Requirements

- Python 3.11 (the pinned evaluation runtime)
- no third-party Python packages for the core harness
- TritonAI access for official LLM experiments

The course evaluation uses TritonAI. Other OpenAI-compatible endpoints may be
useful for local debugging, but they should not be mixed into the official
TritonAI result tables.

Configure the endpoint in the current shell:

```bash
export LLM_CHAT_COMPLETIONS_URL='https://tritonai-api.ucsd.edu/v1/chat/completions'
export LLM_API_KEY='YOUR_TRITONAI_KEY'
export LLM_MODEL='api-glm-5.3'
export LLM_REASONING_EFFORT='low'
```

Never place the real key in a tracked file. `.env` is ignored; `.env.example`
contains only placeholders.

## Validate the repository

Run the complete test suite:

```bash
python3 -m unittest discover -s tests -v
```

Run deterministic map and scheduler preflight checks:

```bash
python3 scripts/run_experiment.py \
  maps/development/shared_static_traps_small.json \
  --validate-only \
  --planning-granularity chunk \
  --chunk-size 3

python3 scripts/run_experiment.py \
  maps/development/shared_asymmetric_crossing.json \
  --validate-only \
  --planning-granularity chunk \
  --chunk-size 3
```

The test suite also validates all three evaluation maps and their precommitted
fault classifications without calling an LLM.

Preflight is a harness check, not evidence about LLM quality.

## Run one TritonAI experiment

Clean execution:

```bash
python3 scripts/run_experiment.py \
  maps/development/shared_asymmetric_crossing.json \
  --backend llm \
  --mode clean \
  --planning-granularity chunk \
  --chunk-size 3
```

Paired fault propagation and full restart:

```bash
python3 scripts/run_experiment.py \
  maps/development/shared_asymmetric_crossing.json \
  --backend llm \
  --mode full-restart \
  --fault f-water-lower-coordinate-shift \
  --planning-granularity chunk \
  --chunk-size 3 \
  --detection-delay-ticks 0
```

Trusted-source verification upper bound:

```bash
python3 scripts/run_experiment.py \
  maps/development/shared_asymmetric_crossing.json \
  --backend llm \
  --mode always-verify \
  --fault f-water-lower-coordinate-shift \
  --planning-granularity chunk \
  --chunk-size 3 \
  --detection-delay-ticks 0
```

`always-verify` compares messages with immutable source sensor facts. It is an
oracle prevention baseline and does not represent a deployable LLM verifier.

## Run a reproducible batch

Preview the development commands without spending API quota:

```bash
python3 scripts/run_batch.py configs/development_batch.json
```

Execute the batch after reviewing the printed plan:

```bash
python3 scripts/run_batch.py configs/development_batch.json \
  --backend llm \
  --execute
```

The batch runner:

- repeats every declared condition three times by default;
- enforces the manifest's TritonAI endpoint host and model identifier;
- creates one isolated directory per run;
- continues after failures and timeouts;
- records every outcome in `batch-summary.json`;
- records manifest/map hashes, Python version, Git revision when available,
  model, and endpoint host, but never writes the API key.

Use `--repetitions 1` for an inexpensive integration check. Do not use the
development batch as final held-out evidence.

The official held-out manifest is `configs/evaluation_batch.json`. Inspect its
36 planned runs before spending quota:

```bash
python3 scripts/run_batch.py configs/evaluation_batch.json
```

After the code, prompts, model, maps, and manifest are frozen, first run a
single-repetition integration batch, inspect every outcome, and only then run
the declared three repetitions:

```bash
python3 scripts/run_batch.py configs/evaluation_batch.json \
  --backend llm \
  --repetitions 1 \
  --execute

python3 scripts/run_batch.py configs/evaluation_batch.json \
  --backend llm \
  --execute
```

Aggregate a completed batch (replace `BATCH_NAME` with the generated directory):

```bash
python3 scripts/aggregate_results.py \
  results/batches/BATCH_NAME/batch-summary.json \
  --output results/summaries
```

The aggregation pipeline retains errors and timeouts in the primary success
denominator and emits run-level, condition-level, paired-recovery, failure, and
machine-readable summary files.

## Output and metric semantics

Each run writes:

```text
config.json
episode-summary.json
events.jsonl
artifacts.jsonl
calls.json
```

Important summary fields:

- `fault_metrics`: injection, delivery, planner exposure, declared/structured
  dependency, plan relevance, executed harm, and caused failure;
- `dependency_quality`: plan-level TP/FP/FN/TN, precision, and recall;
- `detection`: the chargeable prefix boundary;
- `paired_baselines.no_recovery`: complete first faulty attempt;
- `paired_baselines.full_restart`: shared prefix through detection plus restart;
- top-level `backend_calls`, `usage`, and `latency_ms`: actual harness work,
  including any post-detection diagnostic suffix.

Omission faults are reported separately when the missing fact has no positive
artifact edge. A delivered fault is not automatically counted as causal.

## Replay traces

Start the local viewer:

```bash
python3 scripts/serve_viewer.py --port 8765
```

Then open `http://127.0.0.1:8765`. The viewer exposes the ground-truth map,
agent belief views, paths, messages, plans, calls, and event timeline.

## Evaluation discipline

- Development maps may be used to revise prompts and code.
- Evaluation maps must be frozen before their LLM outcomes are inspected.
- Keep accidental and adversarial faults separate.
- Pin model, endpoint, Python version, maps, prompts, and code revision.
- Retain failed, malformed, and timed-out runs.
- Repeat stochastic runs and report variability rather than only successful
  examples.
- Distinguish navigation failure from recovery or verification success.
- Report pre-detection trap entries even if a later restart succeeds.

See [docs/experiment_protocol.md](docs/experiment_protocol.md) for the current
run matrix and held-out-map protocol.
