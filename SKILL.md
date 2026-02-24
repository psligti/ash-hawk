# ash-hawk

Python evaluation harness for AI agents. Use this to evaluate coding agents on structured tasks with automated grading.

## Installation

```bash
# From source (editable)
git clone <repo-url>
cd ash-hawk
uv sync

# As dependency in your project
[project.optional-dependencies]
eval = ["ash-hawk>=0.1.0"]
```

## Quick Start

```bash
# Run an evaluation suite (agent is required)
ash-hawk run path/to/suite.yaml --agent build

# List available suites and runs
ash-hawk list

# Generate report
ash-hawk report --run <run-id>
```

## Directory Structure

```
your-coding-agent/
├── evals/
│   ├── suites/
│   │   ├── code-generation.yaml
│   │   └── bug-fixing.yaml
│   └── fixtures/
│       ├── projects/
│       ├── test-files/
│       └── expected-outputs/
├── pyproject.toml
└── README.md
```

## Suite YAML Format

```yaml
id: my-suite-v1
name: My Evaluation Suite
description: Tests agent capabilities
version: 1.0.0
tags: [coding, python]

agent:
  name: build
  # optional overrides
  provider: zai-coding-plan
  model: glm-4.7
  # optional custom runner loading
  # class: custom.runner:MyRunner
  # location: ./evals/custom_runner.py
  # kwargs:
  #   temperature: 0.2

tasks:
  - id: task-001
    description: Implement binary search
    input:
      prompt: "Write a binary search function in Python"
      language: python
    expected_output: "A working binary search implementation"
    fixtures:
      test_file: ./fixtures/tests/test_binary_search.py
    grader_specs:
      - grader_type: test_runner
        config:
          test_file: $test_file
        weight: 0.6
        required: true
      - grader_type: llm_judge
        config:
          rubric: code_quality
          pass_threshold: 0.7
        weight: 0.4
    tags: [algorithms]
    timeout_seconds: 300
```

## Graders

### First Layer: Deterministic

| Grader | Type ID | Purpose |
|--------|---------|---------|
| String Match | `string_match` | Exact/contains/regex matching |
| Test Runner | `test_runner` | Execute pytest/unittest |
| Static Analysis | `static_analysis` | Run linters (ruff, mypy) |

### Second Layer: LLM-Based

| Grader | Type ID | Purpose |
|--------|---------|---------|
| LLM Judge | `llm_judge` | Quality assessment via LLM |
| Code Review | `code_review` | Code quality evaluation |

### Third Layer: Composite

| Grader | Type ID | Purpose |
|--------|---------|---------|
| Composite | `composite` | Combine multiple graders with weights |
| Aggregation | `aggregation` | Aggregate scores across trials |

## Grader Configurations

### string_match

```yaml
grader_type: string_match
config:
  expected: "exact text"
  mode: exact  # exact | contains | regex
  case_sensitive: false
```

### test_runner

```yaml
grader_type: test_runner
config:
  test_file: ./tests/test_my_func.py
  test_function: test_specific  # optional, run specific test
  pass_threshold: 1.0
  timeout_seconds: 60
```

### static_analysis

```yaml
grader_type: static_analysis
config:
  tools: [ruff, mypy]
  max_issues: 0
  fail_on_error: true
```

### llm_judge

```yaml
grader_type: llm_judge
config:
  rubric: code_quality
  criteria:
    - correctness
    - readability
    - efficiency
  pass_threshold: 0.7
  model: claude-3-5-sonnet
```

### composite

```yaml
grader_type: composite
config:
  graders:
    - grader_type: test_runner
      config:
        test_file: ./tests/test_func.py
      weight: 0.5
    - grader_type: llm_judge
      config:
        rubric: code_quality
      weight: 0.5
  aggregation: weighted_average
```

## Fixtures

Fixtures are files/directories referenced by tasks. Paths are resolved relative to the suite file.

### fixture resolution

```yaml
tasks:
  - id: task-1
    input:
      prompt: "Fix the bug in $source_file"
      working_dir: $project_dir
    fixtures:
      source_file: ./fixtures/buggy_code.py
      project_dir: ./fixtures/project/
      expected_output: ./fixtures/expected/fix.py
```

### fixture injection

Use `$fixture_name` in task input to inject resolved paths:

```yaml
input:
  prompt: "Read $input_file and write to $output_dir"
fixtures:
  input_file: ./data/input.json
  output_dir: ./output/
```

Becomes at runtime:

```python
{
  "prompt": "Read /abs/path/to/data/input.json and write to /abs/path/to/output/",
  ...
}
```

## CLI Commands

```bash
# Run suite (uses suite.agent by default)
ash-hawk run <suite.yaml> [--agent <agent-name>] [--model MODEL] [--provider PROVIDER]

# Override with explicit runner class/location
ash-hawk run <suite.yaml> [--agent-class module:Class] [--agent-location ./path/to/runner.py]

# List suites/runs
ash-hawk list [--suite SUITE_ID] [--run RUN_ID]

# Generate report
ash-hawk report --run <run-id> [--format json|html] [--output PATH]

# Validate suite
ash-hawk validate <suite.yaml>
```

Provider/model resolution order when running:
1. Explicit CLI overrides (`--provider`, `--model`)
2. Suite YAML `agent.provider` / `agent.model`
3. Agent-specific model config in dawn-kestrel agent registry
4. dawn-kestrel default account/settings (`provider_default`, `model_default`)

Agent selection order:
1. CLI `--agent`, `--agent-class`, `--agent-location`
2. Suite YAML `agent` block
3. No fallback (run fails if nothing is configured)

## Programmatic Usage

```python
from ash_hawk.execution import EvalRunner, FixtureResolver, TrialExecutor
from ash_hawk.storage import FileStorage
from ash_hawk.types import EvalSuite, EvalTask, ToolSurfacePolicy

# Load suite
suite = EvalSuite.from_yaml("evals/suites/code-gen.yaml")

# Setup
storage = FileStorage(base_path=".ash-hawk-results")
policy = ToolSurfacePolicy(allowed_tools=["read", "write", "bash"])
resolver = FixtureResolver("evals/suites/code-gen.yaml", suite)
executor = TrialExecutor(storage, policy, fixture_resolver=resolver)

# Run
runner = EvalRunner(config, storage, executor)
summary = await runner.run_suite(suite, agent_config, run_envelope)

# Results
print(f"Pass rate: {summary.metrics.pass_rate:.1%}")
print(f"Total cost: ${summary.metrics.total_cost_usd:.2f}")
```

## Configuration

Environment variables (prefix: `ASH_HAWK_`):

```bash
ASH_HAWK_PARALLELISM=4
ASH_HAWK_DEFAULT_TIMEOUT_SECONDS=300
ASH_HAWK_STORAGE_BACKEND=file
ASH_HAWK_STORAGE_PATH=.ash-hawk-results
ASH_HAWK_LOG_LEVEL=INFO
```

## Output Structure

```
.ash-hawk-results/
├── <suite-id>/
│   ├── suite.json
│   └── runs/
│       └── <run-id>/
│           ├── envelope.json
│           ├── summary.json
│           └── trials/
│               ├── <trial-id>.json
│               └── ...
```

## Related

- **dawn-kestrel**: Agent runtime SDK (ash-hawk dependency)
- **Phase 2 Roadmap**: `docs/PHASE2.md`
