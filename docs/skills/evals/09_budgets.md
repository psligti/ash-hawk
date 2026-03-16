# C3. Budgets: steps/tool calls/tokens/time

Budgets enforce bounded behavior and help prevent flaky/expensive runs.

## Scenario budget fields

`BudgetConfig` (`ash_hawk/scenario/models.py`) includes:
- `max_steps`
- `max_tool_calls`
- `max_tokens`
- `max_time_seconds`

## Authoring strategy

- Start lenient for discovery.
- Tighten budgets after observing stable behavior.
- Pair with budget graders to make limits enforceable in scoring.

## Skill links

- Budget grader details: [C5](12_trace_graders.md)
- Iteration loop to tighten thresholds: [B6](11_iteration_loop.md)
