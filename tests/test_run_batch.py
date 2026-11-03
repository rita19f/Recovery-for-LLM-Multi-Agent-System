from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_BATCH = PROJECT_ROOT / "scripts" / "run_batch.py"


class RunBatchTest(unittest.TestCase):
    def test_completed_batch_can_be_resumed_without_rerunning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            output = root / "batch"
            manifest.write_text(
                json.dumps(
                    {
                        "name": "resume-test",
                        "repetitions": 1,
                        "defaults": {
                            "planning_granularity": "chunk",
                            "chunk_size": 3,
                        },
                        "runs": [
                            {
                                "id": "clean",
                                "map": "maps/development/shared_static_traps_small.json",
                                "mode": "clean",
                            }
                        ],
                    }
                )
            )
            first = subprocess.run(
                [
                    sys.executable,
                    str(RUN_BATCH),
                    str(manifest),
                    "--backend",
                    "dry-run",
                    "--output-root",
                    str(output),
                    "--execute",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stderr or first.stdout)
            state = json.loads((output / "batch-state.json").read_text())
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["completed_runs"], 1)
            self.assertEqual(state["status_counts"]["ok"], 1)

            resumed = subprocess.run(
                [
                    sys.executable,
                    str(RUN_BATCH),
                    str(manifest),
                    "--backend",
                    "dry-run",
                    "--resume",
                    str(output),
                    "--execute",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                resumed.returncode, 0, resumed.stderr or resumed.stdout
            )
            self.assertIn("Resuming 1/1 completed runs", resumed.stdout)
            summary = json.loads((output / "batch-summary.json").read_text())
            self.assertEqual(summary["completed_runs"], 1)
            self.assertEqual(len(summary["outcomes"]), 1)


if __name__ == "__main__":
    unittest.main()
