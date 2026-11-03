# Safe Recovery for LLM Multi-Agent Workflows

This repository studies what happens when a message passed between LLM agents
is wrong, and how expensive and how safe it is to recover from it.

Two LLM agents move through a shared, partially observable grid. They have
complementary private sensors (P1 senses fire, P2 senses water), so each agent
needs facts reported by its partner to reach its goal safely. The harness
injects faults into these structured messages, measures whether they propagate
into plans, unsafe actions, and failures, and compares recovery baselines.

## Workloads

- **Basic maps**: fire and water traps are terminal, so reaching the goal and
  reaching it safely are the same.
- **Extended maps**: agents have 3 HP. Monsters cost one HP per unarmed entry,
  and single-use swords defeat them. An agent can carry only the sword type its
  partner senses, so every sword must be learned from a partner report. Damage,
  pickups, and defeated monsters are persistent world state.

Each agent observes a local area, sends newly sensed facts to its partner,
plans a short chunk of moves, and executes one move per tick.

## Faults and baselines

Each evaluation map has three precommitted faults: an accidental coordinate
offset, an adversarial route decoy, and an irrelevant one-time omission.

| Baseline | Meaning |
| --- | --- |
| Clean | no fault |
| No recovery | the faulty run continues to the end |
| Full restart | after detection, all agent memory, messages, and plans are cleared; the physical world (positions, HP, items) is kept |
| Always verify | every message is checked against the trusted source facts (oracle upper bound) |
| LLM verify | an extra LLM call checks each message against the trusted facts |

No recovery and full restart share the same faulty prefix, so they are compared
pairwise. Detection is either immediate or after the fault first causes harm.
A full environment reset is available as a development-only oracle
(`--recovery-scope environment-reset`) and is not part of the formal matrix.

## Repository layout

```text
configs/    batch manifests (configs/evaluation_batch.json is the formal matrix)
docs/       experiment protocol
maps/       development maps and the ten generated evaluation maps
scripts/    harness, runners, map generator, aggregation, viewer server
tests/      deterministic tests (no LLM calls)
viewer/     browser replay of runs
results/    raw runs and batches (ignored by Git) and reviewed summaries
```

## Setup

The core harness uses only the Python standard library. The formal runs use a
self-contained conda environment pinned to CPython 3.14.2, so system or
Homebrew upgrades cannot change the interpreter during an experiment. Do not
install or update packages in it while a batch (or its resume) is pending:

```bash
conda create -n safe-mas-py314 --override-channels -c conda-forge "python=3.14.2=*_cp314"
conda activate safe-mas-py314
cp .env.example .env        # then fill in the TritonAI values
set -a; source .env; set +a
```

Required variables: `LLM_CHAT_COMPLETIONS_URL`, `LLM_API_KEY`, `LLM_MODEL`
(`api-glm-5.3`), and `LLM_REASONING_EFFORT` (`low`). Never commit the real key.

## Usage

Check the code and maps without calling an LLM:

```bash
python -m unittest discover -s tests
python scripts/generate_evaluation_maps.py --output maps/evaluation --check
python scripts/run_batch.py configs/evaluation_batch.json \
  --backend dry-run --repetitions 1 --execute --output-root /tmp/rehearsal
```

Run one experiment:

```bash
python scripts/run_experiment.py maps/evaluation/e1_pocket_crossing.json \
  --backend llm --mode full-restart --fault a-fire-sword-1-monster-decoy \
  --planning-granularity chunk --chunk-size 3 \
  --detection-trigger after-fault-caused-harm --recovery-scope agent-only
```

Run the formal batch (one repetition first, then all three), resume it if it is
interrupted, and aggregate the results:

```bash
python scripts/run_batch.py configs/evaluation_batch.json --backend llm --repetitions 1 --execute
python scripts/run_batch.py configs/evaluation_batch.json --backend llm --execute
python scripts/run_batch.py configs/evaluation_batch.json --backend llm --execute \
  --resume results/batches/BATCH_NAME
python scripts/aggregate_results.py results/batches/BATCH_NAME/batch-summary.json \
  --output results/summaries
```

Replay a run in the browser:

```bash
python scripts/serve_viewer.py --port 8765   # open http://127.0.0.1:8765
```

## Main metrics

- goal success and safe (full-health) success;
- the fault propagation funnel: delivered, used in a plan, executed, harm,
  failure;
- LLM calls, tokens, latency, executed actions, and messages;
- wasted work discarded by a restart and damage that a restart cannot undo;
- verifier accuracy and cost.

Errors and timeouts stay in every success denominator. See
[docs/experiment_protocol.md](docs/experiment_protocol.md) for the full
matrix, detection rules, and evaluation discipline.
