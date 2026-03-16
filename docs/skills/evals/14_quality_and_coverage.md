# D1. Quality & coverage mindset

For robust evals, think in layers:

1. **Deterministic correctness** (tests/schema/string/diff)
2. **Process correctness** (trace/order/budget/policy)
3. **Quality judgment** (LLM rubric)

## Practical rules

- Deterministic checks should dominate pass/fail.
- LLM graders should refine quality, not replace hard correctness gates.
- Keep grader outcomes explainable and independently debuggable.

## Skill links

- Grader weighting: [B3](05_graders_and_weights.md)
- Trace protocol checks: [C5](12_trace_graders.md)
