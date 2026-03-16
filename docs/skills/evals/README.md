# Writing evals in ash-hawk (skill graph)

This is a graph (not a single linear doc) of skills for authoring evals in this repo.

## Entry points (pick one)

- **Eval suites (task-based)** → start here: [Suites 101](01_suites_101.md)
- **Scenarios (pytest-like, trace-based)** → start here: [Scenarios 101](02_scenarios_101.md)

## Graph

### A. Foundations

- [A1. Mental model: suite vs scenario](00_mental_model_suite_vs_scenario.md)
- [A2. File layout + discoverability](03_repo_layout_and_discovery.md)

### B. Suites (task-based)

- [B1. Suites 101: minimal YAML](01_suites_101.md)
- [B2. Fixtures: resolution + $injection](04_fixtures_and_injection.md)
- [B3. Graders: deterministic → LLM → composite](05_graders_and_weights.md)
- [B4. Templates: coding/conversational/research/custom](06_templates_and_task_builders.md)
- [B5. Policies: tool surface boundaries](07_tool_policies.md)
- [B6. Iteration loop: run → report → tighten graders](11_iteration_loop.md)

### C. Scenarios (trace-based)

- [C1. Scenarios 101: minimal v1 YAML](02_scenarios_101.md)
- [C2. Tooling: allowlist + mocks + record/replay](08_tooling_mocks_record_replay.md)
- [C3. Budgets: steps/tool calls/tokens/time](09_budgets.md)
- [C4. Expectations: event ordering/diff/output assertions](10_expectations_and_assertions.md)
- [C5. Deterministic trace graders: schema, verify-before-done, evidence-required](12_trace_graders.md)
- [C6. Scenario packs as skill/protocol compliance tests](13_scenario_packs_for_skills_tools_mcp.md)

### D. Quality & coverage

- [D1. Coverage mindset: deterministic first, then LLM](14_quality_and_coverage.md)
- [D2. Eval-enhancements: calibration + gap scorecard](15_eval_enhancements_calibration_and_gap_scorecard.md)
- [D3. How to create fast evals](16_fast_evals_for_other_projects.md)

## Working examples (read alongside)

- Suite example: `examples/complete-eval/suites/complete-example.yaml`
- Custom suite example: `examples/custom/mixed-eval-suite.yaml`
- Scenario example: `examples/scenarios/hello_world.scenario.yaml`
- Scenario packs: `examples/scenarios/policy/`, `examples/scenarios/compliance/`
