# AGENTS.md - Ash Hawk Codebase Guide

## Project Overview

Ash Hawk is a Python evaluation harness for AI agents. It provides structured task execution, automated grading (deterministic, LLM-based, composite), and comprehensive result tracking.

---

## Build/Lint/Test Commands

### Installation
```bash
uv sync                    # Install all dependencies
uv sync --group dev        # Install with dev dependencies
```

### Linting & Type Checking
```bash
uv run ruff check .        # Run ruff linter
uv run ruff check . --fix  # Auto-fix lint issues
uv run mypy ash_hawk       # Run mypy type checker (strict mode)
```

### Testing
```bash
uv run pytest              # Run all tests with coverage
uv run pytest tests/       # Run tests in specific directory
uv run pytest tests/graders/test_composite.py  # Run single test file
uv run pytest tests/graders/test_composite.py::TestWeightedMode  # Run single test class
uv run pytest tests/graders/test_composite.py::TestWeightedMode::test_weighted_equal_weights  # Run single test
uv run pytest -x           # Stop on first failure
uv run pytest -v           # Verbose output
uv run pytest --no-cov     # Run without coverage
```

### Building
```bash
uv build                   # Build package
```

### CLI Usage
```bash
ash-hawk run <suite.yaml> --agent <name>  # Run evaluation suite
ash-hawk list                             # List runs
ash-hawk report --run <run-id>            # Generate report
ash-hawk validate <suite.yaml>            # Validate suite YAML
```

### Utility Scripts
```bash
uv run python scripts/analyze_latest_transcript.py            # Diagnose latest thin_runtime tool gap
uv run python scripts/analyze_latest_transcript.py --run-id <run-id>  # Diagnose specific run
uv run python scripts/check_type_hygiene.py [files...]        # Type hygiene checker
```

### Skill Systems
- **Canonical skills** now live in `skills/<name>/SKILL.md`
- Thin runtime internal skills are DK-discovered `SKILL.md` files with `metadata.catalog_source: thin_runtime`
- Evaluation or utility skills may also live in `skills/`, but should use a different `metadata.catalog_source`
- Do not add new skill content to `ash_hawk/thin_runtime/catalog/skills/`, `.dawn-kestrel/skills/`, or `.opencode/skills/`

---

## Code Style Guidelines

### Imports
```python
# Standard library first (alphabetically)
from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, Literal

# Third-party second
import pydantic as pd
import pytest

# Local imports last
from ash_hawk.types import EvalTranscript, EvalTrial, GraderResult
```

### Naming Conventions
- **Files**: `snake_case.py` (e.g., `llm_judge.py`, `test_runner.py`)
- **Classes**: `PascalCase` (e.g., `CompositeGrader`, `EvalSuite`)
- **Functions/Methods**: `snake_case` (e.g., `grade_trial`, `is_tool_allowed`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_TIMEOUT_SECONDS`)
- **Private attributes**: `_leading_underscore` (e.g., `_name`, `_weights`)
- **Enums**: `PascalCase` for class, `UPPER_CASE` for values

### Type Hints (MANDATORY)
```python
# All functions must have complete type hints
async def grade(
    self,
    trial: EvalTrial,
    transcript: EvalTranscript,
    spec: GraderSpec,
) -> GraderResult:
    ...

# Use modern union syntax
def __init__(self, name: str | None = None): ...

# Use list/dict generics directly
items: list[str] = []
config: dict[str, Any] = {}
```

### Pydantic Models
```python
# All models use strict validation with extra="forbid"
class MyModel(pd.BaseModel):
    field: str = pd.Field(description="Field description")
    optional: int | None = pd.Field(default=None, description="Optional field")

    model_config = pd.ConfigDict(extra="forbid")

# Use computed_field for derived properties
@pd.computed_field(return_type=int)
def total(self) -> int:
    return self.input + self.output
```

### Docstrings
```python
def method(self, arg: str) -> bool:
    """Brief one-line description.

    Longer description if needed.

    Args:
        arg: Description of arg.

    Returns:
        Description of return value.
    """
```

### Error Handling
- Raise specific exceptions with descriptive messages
- Use `ValueError` for validation errors
- Include context in error messages

### Formatting (via ruff)
- Line length: 100 characters
- Target Python: 3.11+
- Ruff rules enabled: E, F, I, W, UP (pyupgrade)

### Async Patterns
```python
# All I/O operations are async
async def grade(self, trial: EvalTrial, ...) -> GraderResult:
    result = await some_async_operation()
    return result

# Use pytest.mark.asyncio for async tests
@pytest.mark.asyncio
async def test_something():
    result = await my_async_func()
    assert result is True
```

### Test Patterns
```python
# Fixtures at module level
@pytest.fixture
def trial():
    return EvalTrial(id="trial-1", task_id="task-1")

# Class-based test organization
class TestCompositeGrader:
    """Test CompositeGrader."""

    @pytest.mark.asyncio
    async def test_weighted_scoring(self, trial, transcript, spec):
        # Arrange
        grader = CompositeGrader([...])
        # Act
        result = await grader.grade(trial, transcript, spec)
        # Assert
        assert result.score == 0.75
```

---

## Project Structure

```
ash-hawk/
├── ash_hawk/           # Main package
│   ├── agents/         # Agent adapters (dawn-kestrel, etc.)
│   ├── auto_research/  # Auto-research cycle runner
│   ├── cli/            # CLI commands (run, thin, improve)
│   ├── graders/        # Graders (base, llm_judge, composite, etc.)
│   ├── improve/        # Improve loop, diagnosis, patching
│   ├── improvement/    # Guardrails, fixture splitting
│   ├── policy/         # Tool policy enforcement
│   ├── prompts/        # LLM prompt templates
│   ├── scenario/       # Scenario running, adapters, trials
│   ├── storage/        # Backends (file, sqlite)
│   ├── tracing.py      # Trace/event utilities
│   ├── types.py        # Core type definitions
│   ├── context.py      # Eval context management
│   └── config.py       # Configuration management
├── tests/              # Test suite (mirrors ash_hawk structure)
├── evals/              # Evaluation suites and fixtures
├── examples/           # Example suites and scenarios
├── docs/               # Documentation
├── pyproject.toml      # Project config
└── ruff.toml           # Linter config
```

---

## Note-Lark Knowledge Management

Use note-lark MCP tools for persistent knowledge capture.

### Critical call shape

All note-lark tool calls must wrap arguments under `payload={...}`.

### Quick capture (append-only)

```python
note_lark_memory_append(payload={
    "scope": "project",
    "project": "ash-hawk",
    "source": "session",
    "raw_text": "Composite grader weighting needs per-domain calibration.",
    "tags": ["ash-hawk", "graders", "calibration"],
})
```

### Structured learning note

```python
note_lark_memory_structured(payload={
    "title": "Composite grader calibration pattern",
    "memory_type": "procedural",
    "scope": "project",
    "project": "ash-hawk",
    "status": "structured",
    "confidence": 0.9,
    "evidence_count": 2,
    "tags": ["ash-hawk", "evals", "graders"],
    "body": "# Learning\n\nCalibrate grader weights per suite domain.",
})
```

### Create typed notes (docs/specs/skills)

```python
note_lark_notes_create(payload={
    "title": "Ash Hawk grader authoring guide",
    "type": "doc",
    "scope": "project",
    "project": "ash-hawk",
    "frontmatter": {
        "doc_status": "draft",
    },
    "body": "## Overview\n\nGuidance for creating deterministic and LLM graders.",
})
```

### Workflow

1. Session start: `note_lark_memory_search` / `note_lark_notes_search`
2. During work: `note_lark_memory_append` for raw learnings
3. Consolidation: `note_lark_memory_structured` for validated patterns
4. Durable docs/specs: `note_lark_notes_create` with required `frontmatter`

---

## Key Dependencies

- **pydantic** (v2+): Data validation, strict mode
- **click**: CLI framework
- **pytest**: Testing with pytest-asyncio
- **ruff**: Fast linting and formatting
- **mypy**: Strict type checking
- **rich**: CLI output formatting
- **aiofiles/aiohttp/aiosqlite/asyncpg**: Async I/O

---

## Environment Variables

```bash
ASH_HAWK_PARALLELISM=4
ASH_HAWK_LLM_MAX_WORKERS=4
ASH_HAWK_LLM_TIMEOUT_SECONDS=300
ASH_HAWK_TRIAL_MAX_WORKERS=4
ASH_HAWK_DEFAULT_TIMEOUT_SECONDS=300
ASH_HAWK_STORAGE_BACKEND=file
ASH_HAWK_STORAGE_PATH=.ash-hawk
ASH_HAWK_LOG_LEVEL=INFO
```

---

## Request Queue Integration

Ash Hawk uses a two-level request queue for throttling to avoid agent timeouts.

### Queue Architecture

1. **LLM Request Queue** (`LLMRequestQueue`): Throttles concurrent LLM API calls across all trials
   - Uses local asyncio.Semaphore for concurrency control
   - Tracks wait times and token usage
   - Configured via `ASH_HAWK_LLM_MAX_WORKERS` (default: 4)

2. **Trial Execution Queue** (`TrialExecutionQueue`): Throttles concurrent trial execution
   - Uses local asyncio.Semaphore for concurrency control
   - Configured via `ASH_HAWK_TRIAL_MAX_WORKERS` (default: 4)

### Configuration

```bash
ASH_HAWK_LLM_MAX_WORKERS=4        # Max concurrent LLM requests
ASH_HAWK_LLM_TIMEOUT_SECONDS=300  # LLM request timeout in seconds
ASH_HAWK_TRIAL_MAX_WORKERS=4       # Max concurrent trials
```

### Usage

```python
from ash_hawk.scenario.runner import EvalRunner
from ash_hawk.scenario.trial import TrialExecutor

async def run_with_throttling():
    config = get_config()
    storage = FileStorage(base_path="./results")
    policy = ToolSurfacePolicy()

    trial_executor = TrialExecutor(storage, policy, agent_runner)
    runner = EvalRunner(config, storage, trial_executor)

    summary = await runner.run_suite(suite, agent_config, run_envelope)

    queue_stats = await runner.llm_queue.get_stats()
    print(f"LLM queue: {queue_stats}")
```

### Key Integration Points

- **DawnKestrelAgentRunner**: Automatically uses the global LLM queue when registered by EvalRunner
- **EvalRunner**: Creates and registers both queues during initialization
- **ResourceTracker**: Tracks queue wait times and peak depth

### Monitoring

```python
queue_stats = await runner.llm_queue.get_stats()
print(f"Total requests: {queue_stats['total_requests']}")
print(f"Avg wait time: {queue_stats['total_wait_time'] / max(1, queue_stats['total_requests']):.2f}s")
```

---

## Important Notes

- **Never** suppress type errors (`# type: ignore`, `Any` abuse)
- **Never** skip hooks or tests
- **Always** run `ruff check` and `mypy` before committing
- **Always** use async for I/O operations
- **Always** add type hints to all public functions
- **Always** use `extra="forbid"` in Pydantic models

---

## Git Practices

### Atomic Commits

Each commit should represent a single, logical change:

- **One concern per commit**: A commit should do one thing (fix a bug, add a feature, refactor)
- **Self-contained**: Each commit should leave the codebase in a working state
- **Revertable**: Any commit should be revertable without breaking the build
- **Descriptive messages**: Explain *why*, not *what*

**GOOD:**
```bash
git commit -m "fix: handle empty fixture list in registry lookup"
git commit -m "feat: add json_schema grader for structure validation"
git commit -m "refactor: extract common validation logic into base class"
```

**BAD:**
```bash
git commit -m "fixes"                    # Too vague
git commit -m "WIP"                      # Not atomic
git commit -m "fix bug and add tests"    # Two concerns
```

### Commit Message Format

Use conventional commits:

```
<type>: <description>

[optional body]
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `refactor:` - Code refactoring
- `test:` - Adding/updating tests
- `docs:` - Documentation only
- `chore:` - Maintenance tasks

### Commit Frequently

- **Commit after each logical unit** of work is complete
- **Commit periodically during implementation** - don't wait until the end
- **Don't batch multiple features** in one commit
- **Commit before switching tasks** or ending a session
- **Commit working code only** - tests should pass

**CRITICAL**: As an AI agent, you MUST commit changes periodically throughout a session. Do NOT wait for the user to ask. After completing each todo item or logical unit, commit immediately.

**Workflow:**
```bash
# After completing a feature or fix
git add ash_hawk/eval_packs/
git commit -m "feat: add evaluator packs with registry"
git push origin main

# After completing another unit
git add ash_hawk/integration/
git commit -m "feat: add post-run review hooks"
git push origin main
```

**Workflow:**
```bash
# Complete a feature
git add -p                    # Stage hunks selectively
git commit -m "feat: add support for custom fixtures"
git push origin main

# Fix a bug
git add -p
git commit -m "fix: handle None in grader registry"
git push origin main
```

### Pre-Commit Checklist

- [ ] `ruff check .` passes
- [ ] `mypy ash_hawk` passes
- [ ] `uv run pytest` passes (or affected tests)
- [ ] Commit message is descriptive
- [ ] Commit is atomic (single concern)
