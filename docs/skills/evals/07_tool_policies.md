# B5. Policies: tool surface boundaries

Every serious eval should constrain the tool surface.

## Core model

`ToolSurfacePolicy` in `ash_hawk/types.py` controls:
- allowed/denied tools
- filesystem/network boundaries
- tool-call and token/cost budgets

## Authoring guideline

- Default-deny where possible.
- Explicitly allow only tools needed by the task.
- Set `max_tool_calls`/timeouts for deterministic behavior.

## Why this matters

Policy is both a safety mechanism and a grading signal. Many scenario packs test policy invariants directly.

## Skill links

- Scenario compliance pack examples: [C6](13_scenario_packs_for_skills_tools_mcp.md)
- Budgets: [C3](09_budgets.md)
