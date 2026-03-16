# C5. Deterministic trace graders

Trace graders score behavior from observable event streams, not guesswork.

## Event schema

Trace events are defined in `ash_hawk/scenario/trace.py`:
- `ModelMessageEvent`
- `ToolCallEvent`
- `ToolResultEvent`
- `VerificationEvent`
- `TodoEvent`
- `DiffEvent`
- `ArtifactEvent`
- `PolicyDecisionEvent`
- `RejectionEvent`
- `BudgetEvent`

## High-value graders

Registered in `ash_hawk/graders/registry.py` (trace assertions):
- `trace_schema`
- `verify_before_done`
- `evidence_required`
- `ordering`
- `budget` (budget compliance)
- `diff_constraints`

## Authoring guidance

- Start every new scenario with `trace_schema`.
- Add one protocol grader at a time (`verify_before_done`, then `ordering`, etc.).
- Keep each grader focused so failure reasons stay clear.
