# Reviewed Summaries

Place aggregate CSV/JSON tables and plotting inputs here after checking that
they contain no API credentials or unnecessary full prompts.

Generate the standard Phase 1 tables from one or more completed batches:

```bash
python3 scripts/aggregate_results.py \
  results/batches/BATCH_NAME/batch-summary.json \
  --output results/summaries
```

The command writes:

- `run_level.csv`: one row for every planned run, including errors/timeouts;
- `condition_summary.csv`: success rates, fault funnel rates, pooled dependency
  confusion metrics, and mean/standard deviation of cost and latency;
- `paired_recovery.csv`: no-recovery versus full-restart values from the same
  faulty trajectory;
- `paired_condition_summary.csv`: paired success, cost, waste, and deltas
  aggregated across repetitions of each condition;
- `failure_summary.csv`: failures, timeouts, traps, and successful runs;
- `phase1_summary.json`: compact machine-readable overview.

The six canonical aggregate files are allowed by `.gitignore`; raw batch
directories remain ignored.
