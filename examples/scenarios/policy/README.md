# Policy Eval Pack

This directory contains ash-hawk scenario evals for policy-agent behavior.

By default, scenarios use `mock_adapter` so validation and harness execution stay inside ash-hawk.
For real policy-agent integration, run the same scenarios with `--sut sdk_dawn_kestrel` and `--policy-mode ...`.

## Coverage

- Policy modes: `fsm`, `rules`, `react`, `plan_execute`, `router`
- Invariants: verification-before-done, evidence requirements, budget compliance
- Guardrails: tool restrictions, rejection telemetry, router fallback surface

## Scenarios

- `policy_fsm_bridge_baseline.scenario.yaml`
- `policy_rules_deterministic_baseline.scenario.yaml`
- `policy_react_repair_path.scenario.yaml`
- `policy_plan_execute_progression.scenario.yaml`
- `policy_router_mode_selection.scenario.yaml`
- `policy_rejection_disallowed_tool.scenario.yaml`
- `policy_rejection_disallowed_edit.scenario.yaml`
- `policy_approval_gate_high_risk.scenario.yaml`
- `policy_budget_soft_pressure.scenario.yaml`
- `policy_budget_hard_stop.scenario.yaml`
- `policy_done_gate_verification_required.scenario.yaml`
- `policy_todo_evidence_required.scenario.yaml`
- `policy_ordering_todo_verify_done.scenario.yaml`
- `policy_router_fallback_resilience.scenario.yaml`

## Validate

```bash
uv run ash-hawk scenario validate examples/scenarios/policy/
```

## Run one scenario

```bash
uv run ash-hawk scenario run examples/scenarios/policy/policy_rules_deterministic_baseline.scenario.yaml --sut mock_adapter
```

## Run one integration scenario against dawn-kestrel policy agents

```bash
uv run ash-hawk scenario run examples/scenarios/policy/policy_rules_deterministic_baseline.scenario.yaml --sut sdk_dawn_kestrel --policy-mode rules
```

## Run matrix across policies/models

```bash
uv run ash-hawk scenario matrix examples/scenarios/policy/ --sut sdk_dawn_kestrel --policies react,rules --models m1,m2
```
