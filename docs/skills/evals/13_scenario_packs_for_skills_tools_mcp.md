# C6. Scenario packs for skills/tools/MCP compliance

Use scenario packs as reusable protocol tests.

## Existing packs in this repo

- Policy pack: `examples/scenarios/policy/`
- Skills/tools/MCP compliance pack: `examples/scenarios/compliance/`

The compliance pack checks patterns such as:
- required trace events,
- MCP prefix/tool usage constraints,
- forbidden tool usage,
- required skill markers in output.

## Why this is a “deep vein”

These packs become reusable contract tests for agent behavior across models/versions.
Instead of ad-hoc checks, you keep an evolving library of scenario contracts.

## Skill links

- Trace graders: [C5](12_trace_graders.md)
- Policies and budgets: [B5](07_tool_policies.md), [C3](09_budgets.md)
