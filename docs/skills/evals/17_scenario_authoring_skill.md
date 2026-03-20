# C7. Scenario authoring skill: write high-signal scenarios

Use this when you need to create or expand `*.scenario.yaml` files for agent behavior evaluation.

## Goal

Produce scenarios that are:
- deterministic (stable pass/fail behavior),
- diagnostic (clear reason when they fail),
- diverse (cover different failure modes and behaviors).

## Workflow

1. Pick one behavior per scenario.
2. Write a prompt that forces that behavior to appear in trace/events.
3. Keep tools minimal (`allowed_tools`) and mock only what the behavior needs.
4. Set realistic budgets (`max_steps`, `max_tool_calls`, `max_time_seconds`) that catch runaway behavior.
5. Add deterministic graders first (`trace_schema`, ordering/content checks), then optional quality grader.
6. Validate and run the scenario before adding more complexity.

## Authoring template

```yaml
schema_version: "v1"
id: "my_behavior_case"
description: "One-line behavior target"

sut:
  type: "agentic_sdk"
  adapter: "bolt_merlin"
  config:
    agent: "execution"

inputs:
  prompt: |
    <single behavior-focused prompt>

tools:
  allowed_tools: ["read", "write", "edit", "bash"]
  mocks:
    read:
      input: {}
      result:
        content: "# fixture content\n"
    write:
      input: {}
      result:
        success: true
  fault_injection: {}

budgets:
  max_steps: 30
  max_tool_calls: 50
  max_tokens: 6000
  max_time_seconds: 180.0

expectations:
  must_events: ["ModelMessageEvent", "ToolCallEvent"]
  must_not_events: []
  ordering_rules:
    - before: "ModelMessageEvent"
      after: "ToolCallEvent"
  diff_assertions: []
  output_assertions: []

graders:
  - grader_type: "trace_schema"
    required: true
    config: {}
  - grader_type: "trace_content"
    required: true
    config:
      required_event_types: ["ToolCallEvent"]
  - grader_type: "trace_quality"
    required: false
    weight: 1.5
    config:
      target_tool_calls: 6
      tool_call_penalty: 0.08
      rejection_penalty: 0.12
      max_penalty: 0.6
      pass_threshold: 0.7
```

## Variation strategy (avoid copy-paste scenarios)

- Vary the core behavior axis, not just wording:
  - policy following,
  - todo/progress tracking,
  - tool selection discipline,
  - verify-before-done evidence,
  - error recovery.
- Vary stressors per scenario:
  - conflicting instructions,
  - tighter budgets,
  - partial tool failures,
  - ambiguous file layout.
- Keep each scenario single-purpose so failure attribution is obvious.

## Validation commands

```bash
uv run ash-hawk scenario validate examples/scenarios/bolt-merlin/
uv run ash-hawk scenario run examples/scenarios/bolt-merlin/
```

## Quality checklist

- Scenario id is unique and descriptive.
- Prompt targets exactly one primary behavior.
- Tool mocks are minimal and deterministic.
- Budgets can fail runaway behavior.
- At least one required deterministic grader is present.
- The scenario fails for the right reason when behavior is violated.

## Related skills

- `02_scenarios_101.md`
- `08_tooling_mocks_record_replay.md`
- `09_budgets.md`
- `10_expectations_and_assertions.md`
- `12_trace_graders.md`
- `13_scenario_packs_for_skills_tools_mcp.md`
