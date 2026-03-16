# B6. Iteration loop: run → report → tighten graders

Deep eval quality comes from iteration, not one-off authoring.

## Loop

1. Run suite/scenario with broad thresholds.
2. Inspect failures and near-misses.
3. Tighten deterministic gates first.
4. Refine LLM rubrics second.
5. Re-run and compare.

## Commands

- `ash-hawk run <suite.yaml> --agent <name>`
- `ash-hawk scenario run <scenario-or-dir> --sut <adapter>`
- `ash-hawk report --run <run-id>`

## What to tune first

- Missing fixture references
- Overly permissive tool policies
- Grader weights that let weak outputs pass
- Budget thresholds that are too loose
