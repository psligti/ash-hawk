# B3. Graders: deterministic → LLM → composite

The strongest suites start deterministic, then add LLM judgment where deterministic checks stop.

## Built-in landscape

See built-ins in `ash_hawk/graders/registry.py` and interfaces in `ash_hawk/graders/base.py`.

Common types:
- Deterministic: `string_match`, `test_runner`, `static_analysis`, trace assertions
- LLM-based: `llm_judge`
- Composition: `composite`

## Weighting pattern

Good default for coding tasks:
- 0.6 test runner
- 0.2 static analysis
- 0.2 or 0.4 llm_judge (depending on strictness)

## Required graders

Use `required: true` for hard gates (e.g., tests must pass).

## Skill links

- From fixtures: [B2](04_fixtures_and_injection.md)
- Into templates: [B4](06_templates_and_task_builders.md)
- Iteration loop: [B6](11_iteration_loop.md)
