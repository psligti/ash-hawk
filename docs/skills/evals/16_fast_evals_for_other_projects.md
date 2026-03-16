# D3. How to create fast evals

Use fast evals for quick, behavior-focused regression checks.

## Goal

Create evals that are:
- small input
- narrow objective
- deterministic when possible
- low cost
- easy to batch

## Fast eval shape

Use `evals:` (not `tasks:`):

```yaml
id: core-fast-v1
name: Core Fast Regression
parallelism: 4

defaults:
  grader_type: string_match
  pass_threshold: 0.7
  case_insensitive: true
  mode: contains

evals:
  - id: polite-greeting
    description: Greeting includes polite salutation
    input: Say hello in one sentence.
    expected: [hello, hi, hey]
    grader: string_match
    tags: [regression, deterministic]

  - id: user-json-shape
    description: Response matches required JSON shape
    input: Return JSON with fields name (string) and age (integer).
    grader: json_schema
    schema:
      type: object
      required: [name, age]
      properties:
        name: {type: string}
        age: {type: integer}
    tags: [regression, deterministic, format]

  - id: explanation-quality
    description: Quality check for subjective output
    input: Explain recursion in two sentences.
    grader:
      grader_type: llm_rubric
      pass_threshold: 0.7
      case_insensitive: false
      mode: exact
    rubric: |
      Score 1.0 if it clearly defines recursion and mentions a base case.
      Score 0.7 if mostly correct but missing one key detail.
      Score 0.0 if incorrect or off-topic.
    tags: [regression, llm]
```

## Authoring rules

- One behavior per eval.
- Prefer deterministic graders first: `string_match`, `regex`, `json_schema`.
- Use `llm_rubric` only when deterministic checks cannot express quality.
- Keep prompts short (1-4 lines target).
- Use stable expected outputs and explicit thresholds.

## Run

```bash
ash-hawk run evals/fast/core-fast-v1.yaml
```

## Maintenance loop

1. Add one fast eval for each bug fix.
2. Keep failing fast evals blocking in CI.
3. Convert flaky rubric checks to deterministic checks when possible.
