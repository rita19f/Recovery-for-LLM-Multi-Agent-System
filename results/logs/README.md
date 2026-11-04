# Formal evaluation logs

`formal-agent-only-v1.tar.gz` contains the complete raw records for the frozen
six-map evaluation: per-run configurations, model-call records, artifacts,
events, episode summaries, the single run error, and the batch state and
summary.

The archive is a mechanically sanitized copy of the original output. Local
paths below the repository root were replaced with `${REPO_ROOT}`, and any
remaining user-home prefix was replaced with `${USER_HOME}`. Experimental
events, model inputs and outputs, metrics, and usage values were not changed.
The archive was scanned for common API-key and authorization-header patterns
before publication.

Verify and extract it with:

```bash
cd results/logs
shasum -a 256 -c formal-agent-only-v1.sha256
mkdir -p /tmp/safe-mas-formal-logs
tar -xzf formal-agent-only-v1.tar.gz -C /tmp/safe-mas-formal-logs
```

The frozen run metadata and map hashes are recorded in
`formal-agent-only-v1.provenance.json`. Reviewed aggregate tables are generated
separately so that changes to analysis code never modify this raw archive.
