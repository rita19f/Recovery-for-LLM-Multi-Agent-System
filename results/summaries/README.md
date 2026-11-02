# Reviewed Summaries

Place aggregate CSV/JSON tables and plotting inputs here after checking that
they contain no API credentials or unnecessary full prompts.

Generate the standard evaluation tables from one or more completed batches:

```bash
python3 scripts/aggregate_results.py \
  results/batches/BATCH_NAME/batch-summary.json \
  --output results/summaries
```

The command writes:

- `run_level.csv`: one row for every planned run, including errors/timeouts;
- `condition_summary.csv`: success rates, fault funnel rates, pooled dependency
  and verifier confusion metrics, and mean/standard deviation of workload,
  cost, and latency;
- `paired_recovery.csv`: no-recovery versus full-restart values from the same
  faulty trajectory;
- `paired_condition_summary.csv`: paired success, cost, waste, and deltas
  aggregated across repetitions of each condition;
- `failure_summary.csv`: failures, timeouts, traps, and successful runs;
- `evaluation_summary.json`: compact machine-readable overview.

The six canonical aggregate files are allowed by `.gitignore`; raw batch
directories remain ignored.

## Current result snapshot

The frozen evaluation contains 42 planned runs: 40 completed with structured
outputs, one returned a malformed route, and one reached the API read timeout.
Clean execution succeeds in 7/9 trials. In the 14 completed paired runs with a
route-relevant coordinate fault, no recovery succeeds in 0/14 trials and enters
14 hazards, while full restart succeeds in 10/14 trials with no paired-path
hazard entries. Full restart adds 18,866 tokens and 24.9 seconds on average.

Dependency precision and recall use `structured route relevance` as an
operational reference, not causal ground truth. Across 45 plan-level
evaluations, LLM-declared IDs obtain 66.7% precision and 42.9% recall;
structured and conservative tracking both obtain 31.1% precision and 100%
recall. The current structured policy therefore does not yet improve
selectivity over the conservative policy.

Generate the reviewed dependency plot with the optional `matplotlib`
dependency:

```bash
python3 scripts/plot_dependency_quality.py \
  --input results/summaries/run_level.csv \
  --output-dir results/summaries
```

The command writes vector PDF and 300-DPI PNG versions of
`dependency_precision_recall`.
