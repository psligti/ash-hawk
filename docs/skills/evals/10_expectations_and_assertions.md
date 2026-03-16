# C4. Expectations: event ordering/diff/output assertions

Expectations describe what must happen (and must not happen) in a scenario run.

## Expectation fields

In `ExpectationConfig` (`ash_hawk/scenario/models.py`):
- `must_events`
- `must_not_events`
- `ordering_rules`
- `diff_assertions`
- `output_assertions`

## Design guidance

- Keep assertions atomic (one invariant per assertion).
- Prefer deterministic checks over broad semantic checks when possible.
- Use ordering for protocol requirements (e.g., TODO before verification before done).

## Skill links

- Trace event model: [C5](12_trace_graders.md)
- Compliance/policy packs: [C6](13_scenario_packs_for_skills_tools_mcp.md)
