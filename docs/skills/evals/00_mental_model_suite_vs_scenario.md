# A1. Mental model: suite vs scenario

Ash-hawk supports two evaluation surfaces that look similar (both are YAML), but they produce different artifacts and are graded differently.

## Eval suite (task-based)

**What it is**: A list of `tasks:`. Each task is a prompt/input with optional fixtures and a list of `grader_specs`.

**What you get**:
- Trial transcripts (messages + tool calls)
- Per-grader results (deterministic + LLM-based)
- Suite summaries and reports

**When to use**:
- You can grade via tests, static analysis, string matching, schema checks, or LLM judge.
- You want a simple harness: “prompt in, score out”.

**Core references**:
- Authoring guide: `SKILL.md`
- Example: `examples/complete-eval/suites/complete-example.yaml`

## Scenario (pytest-like, trace-based)

**What it is**: A single `ScenarioV1` YAML contract (`schema_version: "v1"`) that defines the SUT adapter, tooling surface (mocks/record/replay), budgets, expectations, and deterministic graders over a normalized trace.

**What you get**:
- A normalized Trace v1 JSONL artifact with typed events (`ToolCallEvent`, `TodoEvent`, `VerificationEvent`, ...)
- Deterministic grading over observable artifacts (trace/diff/budget/order)

**When to use**:
- You care about protocol compliance and behavior over time (ordering, budgets, tool use).
- You want deterministic checks on “agent process”, not only final answer.

**Core references**:
- Scenario spec: `docs/specs/agent-evals-pytest-like-framework.md`
- Examples: `examples/scenarios/hello_world.scenario.yaml`, `examples/scenarios/policy/`

## Key decision: start from grading

- If you can express correctness as **tests/static rules** → prefer suites.
- If you need to enforce **process invariants** (verify-before-done, evidence-required, ordering, budgets) → prefer scenarios.
