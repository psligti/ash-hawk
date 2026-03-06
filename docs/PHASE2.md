# ash-hawk Phase 2 Roadmap

pytest-like developer experience improvements.

## Overview

Phase 1 implemented fixture resolution. Phase 2 adds:
1. `conftest.yaml` loading with inheritance
2. `pyproject.toml [tool.ash-hawk]` configuration
3. Suite discovery and filtering

### Agent Evals Framework

For detailed specifications on the pytest-like scenario evaluation framework, see:

- **[Agent Evals (Pytest-like Framework)](specs/agent-evals-pytest-like-framework.md)** - Complete technical specification for scenario-based evaluations


---

## 1. conftest.yaml Loading

### Goal

Allow shared configuration at directory level, inherited by child directories and suites.

### Directory Structure

```
evals/
├── conftest.yaml           # Root config
├── code-gen/
│   ├── conftest.yaml       # Inherits from parent
│   ├── basic-suite.yaml
│   └── advanced-suite.yaml
└── bug-fixing/
    └── regression-suite.yaml
```

### conftest.yaml Format

```yaml
# evals/conftest.yaml
name: my-agent-evals
version: 1.0.0

# Default policy for all suites in this directory
policy:
  allowed_tools: [read, write, bash, grep, glob, edit]
  network_allowed: false
  timeout_seconds: 300

# Shared fixtures available to all tasks
fixtures:
  sample_project: ./fixtures/sample-project/
  test_data: ./fixtures/test-data.json

# Default grader when task has none
default_grader:
  grader_type: llm_judge
  config:
    rubric: correctness
    pass_threshold: 0.7

# Agent configuration
agent:
  provider: zai
  model: zai-coding-plan/glm-5
```

### Inheritance Rules

1. Walk up directories from suite file to evals root
2. Collect all `conftest.yaml` files
3. Merge configs (child overrides parent)
4. Suite YAML has highest priority

### Implementation Tasks

```
[ ] Create ConftestLoader class
    [ ] Parse conftest.yaml format
    [ ] Walk directory tree upward
    [ ] Merge configs with correct precedence

[ ] Wire into CLI
    [ ] Load conftest before suite
    [ ] Apply merged config to suite/tasks

[ ] Add tests
    [ ] Single conftest
    [ ] Nested conftest inheritance
    [ ] Suite overrides conftest
```

---

## 2. pyproject.toml Configuration

### Goal

Project-level defaults in `[tool.ash-hawk]` section.

### Format

```toml
[tool.ash-hawk]
# Discovery
suite_patterns = ["*-suite.yaml", "*.suite.yaml"]
search_paths = ["evals", "tests/evals"]

# Defaults
parallelism = 4
default_timeout = 300
storage_backend = "file"
storage_path = ".ash-hawk-results"

# Agent defaults
provider = "zai"
model = "zai-coding-plan/glm-5"

# Logging
log_level = "INFO"
```

### Implementation Tasks

```
[ ] Extend EvalConfig
    [ ] Add pyproject.toml loading with tomli
    [ ] Merge with env vars (env has priority)

[ ] Update CLI
    [ ] Use config for discovery patterns
    [ ] Use config for default agent/model

[ ] Add tests
    [ ] Config loading
    [ ] Precedence (env > pyproject > defaults)
```

---

## 3. Suite Discovery

### Goal

pytest-like suite discovery and filtering.

### CLI Commands

```bash
# Auto-discover and run all suites
ash-hawk

# Run suites matching pattern
ash-hawk run -k "code-gen"

# Collect only (don't run)
ash-hawk collect

# Run specific directory
ash-hawk run evals/code-gen/

# Run with markers/tags
ash-hawk run -m "algorithms"
```

### Discovery Algorithm

1. Start from `search_paths` in config (default: `evals/`, `tests/evals/`)
2. Match files against `suite_patterns` (default: `*-suite.yaml`)
3. Filter by `-k` pattern if provided
4. Filter by `-m` markers/tags if provided

### Implementation Tasks

```
[ ] Create SuiteDiscoverer class
    [ ] Glob-based file discovery
    [ ] Pattern matching (-k)
    [ ] Tag filtering (-m)

[ ] Add CLI commands
    [ ] Default run (no args = discover all)
    [ ] collect subcommand
    [ ] -k filtering
    [ ] -m marker filtering

[ ] Add tests
    [ ] Discovery patterns
    [ ] Filtering logic
    [ ] Edge cases (empty, no match)
```

---

## Priority Order

| Priority | Feature | Effort | Impact |
|----------|---------|--------|--------|
| 1 | conftest.yaml loading | Medium | High - enables shared config |
| 2 | pyproject.toml config | Small | Medium - project-level defaults |
| 3 | Suite discovery | Medium | High - pytest-like DX |

---

## Future Considerations

### Phase 3 Potential

- Parallel execution across machines
- Result comparison between runs
- Regression detection
- Integration with CI/CD
- Web dashboard for results
- Custom grader plugins via entry points

### API Extensions

```python
# Custom grader registration
@ash_hawk.grader("my_custom_grader")
def my_grader(task, transcript, config):
    ...

# Hook system
@ash_hawk.hook("pre_trial")
def setup_fixtures(task):
    ...
```
