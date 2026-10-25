# Held-Out Evaluation Maps

These maps were constructed and checked with the deterministic scripted
preflight without consulting LLM outcomes. They are not renamed development
maps. Once the first official LLM evaluation begins, do not tune prompts,
recovery logic, faults, or topology against results from this directory.

| Map | Size | Starts/goals | Intended role |
| --- | --- | --- | --- |
| `coincident_detours.json` | 7 x 7 | same / same | simple base case with two short branch decisions |
| `three_portal_exchange.json` | 9 x 9 | separate / separate | side exchange, transient accidental fault, relevant adversarial fault |
| `staged_crossing.json` | 11 x 9 | separate / separate | two stages, delayed observations, relevant and non-propagating controls |

All maps satisfy the maximum 11 x 9 size of the largest development map. Each
registered fault has a precommitted deterministic classification tested in
`tests/test_evaluation_maps.py`. Preflight establishes reachability and fault
scheduling only; it is not an LLM performance result.
