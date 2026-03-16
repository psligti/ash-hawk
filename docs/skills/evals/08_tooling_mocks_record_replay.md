# C2. Tooling: allowlist + mocks + record/replay

Scenarios model tool behavior explicitly.

## YAML fields

- `tools.allowed_tools`
- `tools.mocks`
- `tools.fault_injection`

These are part of `ToolingConfig` (`ash_hawk/scenario/models.py`).

## Practical pattern

- Start with `mock_adapter` + mocks for deterministic baseline.
- Move to real adapters once baseline is stable.
- Use record/replay flows when validating tool interactions over time.

## Skill links

- Scenarios basics: [C1](02_scenarios_101.md)
- Budgets and expectations: [C3](09_budgets.md), [C4](10_expectations_and_assertions.md)
