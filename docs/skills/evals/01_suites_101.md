# B1. Suites 101: minimal YAML

Use suites when you want task-centric evals (`tasks:`) with grader specs.

## Minimal shape

```yaml
id: my-suite-v1
name: My Suite
tasks:
  - id: t1
    description: Basic check
    input: "What is 2 + 2?"
    expected_output: "4"
    grader_specs:
      - grader_type: string_match
        config:
          expected: "4"
          mode: contains
```

## Runtime mapping

- YAML → `EvalSuite` and `EvalTask` models (`ash_hawk/types.py`)
- Each `grader_specs[]` item → `GraderSpec`
- Execution path: runner + trial executor

## Skill links

- Need fixtures? → [B2. Fixtures](04_fixtures_and_injection.md)
- Need better scoring? → [B3. Graders](05_graders_and_weights.md)
- Prefer scaffolded authoring? → [B4. Templates](06_templates_and_task_builders.md)

## Good starter example

- `examples/complete-eval/suites/complete-example.yaml`
