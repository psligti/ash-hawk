# C1. Scenarios 101: minimal v1 YAML

Use scenarios when you need deterministic checks on behavior/process via trace events.

## Minimal shape

```yaml
schema_version: "v1"
id: "hello_world"
sut:
  type: "agentic_sdk"
  adapter: "mock_adapter"
  config: {}
inputs:
  prompt: "Say hello world"
tools:
  allowed_tools: ["bash"]
  mocks: {}
  fault_injection: {}
budgets:
  max_steps: 3
  max_tool_calls: 5
  max_tokens: 100
  max_time_seconds: 10.0
expectations:
  must_events: []
  must_not_events: []
  ordering_rules: []
  diff_assertions: []
  output_assertions: []
graders:
  - grader_type: "trace_schema"
    config: {}
```

## Runtime mapping

- YAML loaded by `ash_hawk/scenario/loader.py:load_scenario`
- Validated into `ScenarioV1` (`ash_hawk/scenario/models.py`)
- Scenario converted to `EvalTask` by `ScenarioRunner._scenario_to_task` (`ash_hawk/scenario/runner.py`)
- Trace artifacts emitted in JSONL (`ash_hawk/scenario/trace.py`)

## Skill links

- Tool mocks + record/replay → [C2](08_tooling_mocks_record_replay.md)
- Budgets and enforcement → [C3](09_budgets.md)
- Ordering/diff/output assertions → [C4](10_expectations_and_assertions.md)
- Deterministic trace graders → [C5](12_trace_graders.md)

## Good starter examples

- `examples/scenarios/hello_world.scenario.yaml`
- `examples/scenarios/coding_agent_smoke.scenario.yaml`
