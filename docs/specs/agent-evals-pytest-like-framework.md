# Agent Evals (Pytest-like Scenarios) + Agentic SDK (inside ash-hawk)

## TL;DR

> **Summary**: Add a scenario-based, pytest-like evaluation surface to `ash-hawk` with a stable Scenario v1 YAML schema, normalized Trace v1 JSONL events, adapters for SDK + coding-agent SUTs, and deterministic graders.

> **Deliverables**: Scenario/Trace schemas + loader/validator, scenario runner + artifact/trace persistence, deterministic trace/diff/budget graders, SDK adapter (dawn-kestrel) + coding-agent subprocess adapter, CLI (`ash-hawk scenario ...`), report w/ grader diffs + trace excerpts, matrix runner, record/replay for tool mocks, scenario template generator, smoke tests + examples.

> **Effort**: XL

> **Parallel**: YES - 4 waves

> **Critical Path**: Scenario+Trace schemas → runner+storage+trace persistence → adapters → deterministic graders → CLI+report+matrix/record → tests/examples

## Context

### Original Request

- Add the provided spec to the repo, plan it, then implement it.

### Interview Summary

- Integration decision: extend existing `ash-hawk` CLI/package (no separate `agent-evals` CLI).

### Metis Review (gaps addressed)

- Define Scenario v1 + Trace v1 as stable contracts (versioned, additive evolution).
- Keep persisted trace schema independent of the internal event bus (`ash_hawk/events.py`).
- Avoid scope creep into full pytest plugin ecosystem; start with CLI discovery/run/report.

### Alignment With `dawn-kestrel-policy-engine`

- Reuse enforcement semantics from `ash_hawk/types.py:ToolSurfacePolicy` + `ash_hawk/policy/enforcer.py:PolicyEnforcer` and avoid a parallel "second policy" in ToolingHarness.
- Ensure trace/grader semantics match dawn-kestrel harness invariants (verify-before-done, evidence-required, approval-before-risk).
- Do not depend on dawn-kestrel policy-engine Task 4 (proposal validator) telemetry being present or stable. Always emit Trace v1 policy decisions/rejections at the ash-hawk boundary from `PolicyEnforcer` outcomes + observed tool execution, and optionally enrich with policy-engine telemetry when available.

## Work Objectives

### Core Objective

- Provide a pytest-like developer experience for authoring scenario-based evals, runnable deterministically and gradable via deterministic graders over normalized trace artifacts.

### Deliverables

- Spec document added to repo: `docs/specs/agent-evals-pytest-like-framework.md` (verbatim or lightly adapted).
- Scenario v1 YAML schema + loader/validator.
- Trace v1 normalized event models + JSONL writer/reader.
- Scenario runner with:
  - SUT adapter interface + registry
  - tool mock layer + record/replay
  - artifact storage integration
- Deterministic graders:
  - verify-before-done
  - evidence-required (TODO events)
  - ordering rules
  - diff constraints
  - schema validation (structured outputs)
  - budget compliance
- CLI:
  - `ash-hawk scenario validate`
  - `ash-hawk scenario run`
  - `ash-hawk scenario report`
  - `ash-hawk scenario matrix`
  - `ash-hawk scenario new`
  - `ash-hawk scenario record`
  - `ash-hawk scenario replay`
- Tests + examples proving:
  - add scenario file → run locally in 1 command
  - same scenario runs on both SUT types (SDK + coding_agent adapter)

### Definition of Done (verifiable)

- `uv run ruff check .`
- `uv run mypy ash_hawk`
- `uv run pytest`
- `uv run ash-hawk scenario validate examples/scenarios/hello_world.scenario.yaml`
- `uv run pytest -q tests/scenario/test_sdk_adapter_trace.py`
- `uv run ash-hawk scenario run examples/scenarios/hello_world.scenario.yaml --sut coding_agent_subprocess`
- `uv run ash-hawk scenario matrix examples/scenarios/ --sut mock_adapter --policies react,rules --models m1,m2` outputs a comparison table
- `uv run pytest -q tests/cli/test_scenario_record_replay.py`

### Must Have

- Stable, versioned Scenario v1 + Trace v1 contracts.
- Deterministic-first grading over observable artifacts.
- Storage integration that persists normalized trace JSONL as first-class artifacts.

### Must NOT Have (guardrails)

- No pytest plugin (for now).
- No distributed execution beyond local parallelism.
- No coupling persisted trace format to the internal event bus.
- No embedding full agent implementations; adapters may provide a tiny test double only for smoke tests.

## Verification Strategy

> ZERO HUMAN INTERVENTION — all verification is agent-executed.

- Test decision: tests-after (pytest already configured) with focused unit tests + 1-2 smoke CLI tests.
- Evidence: each major CLI path writes artifacts under `.ash-hawk/...` and tests assert on artifact existence + schema.

## Execution Strategy

### Parallel Execution Waves

- **Wave 1**: Contracts + registries (1-4)
- **Wave 2**: Runner + storage + tooling/trace persistence (5-8)
- **Wave 3**: Adapters + deterministic graders (9-12)
- **Wave 4**: CLI + matrix/record + examples + e2e tests (13-16)

### Dependency Matrix (full, all tasks)

```
1 → 13,15,16
2 → 5,6,15
3 → 5,6,8,12
4 → 6
5 → 6,10
6 → 7,9,10,11,12,13
7 → 8,11,13
8 → 13
9 → 10
10 → 13
11 → 12,15
12 → 13,15
13 → 14,15
14 → 16
15 → 16
16 → Final verification wave
```

### Agent Dispatch Summary

- Wave 1: 4 tasks (writing + deep + quick)
- Wave 2: 4 tasks (deep + quick)
- Wave 3: 4 tasks (quick + deep)
- Wave 4: 4 tasks (deep + quick)

## Technical Specifications

### Scenario v1 Schema

The Scenario v1 YAML schema defines the structure for scenario-based evaluations:

```yaml
schema_version: 1
id: string
description: string
sut:
  type: 'coding_agent' | 'agentic_sdk'
  adapter: string
  config: dict[str, Any]
inputs:
  user_message: string
  repo_fixture: path | null
  initial_todos: list | null
  memory_seed: dict | null
tools:
  allowed_tools: list[str]
  mocks: dict
  fault_injection: dict
budgets:
  max_steps: int
  max_tool_calls: int
  max_tokens: int | null
  max_time_seconds: int
expectations:
  must_events: list[str]
  must_not_events: list[str]
  ordering_rules: list
  diff_assertions: dict
  output_assertions: dict
graders: list[GraderSpec]
```

### Trace v1 Schema

Trace v1 defines normalized event models for deterministic grading:

- **Base Event**: `schema_version`, `event_type`, `ts` (ISO timestamp), `data`
- **Event Types**:
  - `ModelMessageEvent` - agent messages
  - `ToolCallEvent` - tool invocations
  - `ToolResultEvent` - tool outputs
  - `PolicyDecisionEvent` - policy enforcement outcomes
  - `RejectionEvent` - policy rejections
  - `BudgetEvent` - budget consumption
  - `VerificationEvent` - verification results
  - `DiffEvent` - git diff information
  - `ArtifactEvent` - artifact paths
  - `TodoEvent` - TODO item state changes

### Storage Layout

```
.ash-hawk/
├── {suite_id}/
│   └── runs/
│       └── {run_id}/
│           ├── envelope.json
│           ├── trials/
│           │   ├── {trial_id}.json
│           │   └── {trial_id}.trace.jsonl
│           └── artifacts/
│               └── {trial_id}/
│                   ├── stdout.txt
│                   ├── stderr.txt
│                   └── diff.patch
└── tool_mocks/
    └── {scenario_id}/
        └── trace.jsonl
```

### Adapter Interface

```python
class ScenarioAdapter(Protocol):
    name: str
    
    def run_scenario(
        self,
        scenario: ScenarioV1,
        workdir: Path,
        tooling_harness: ToolingHarness,
        budgets: BudgetConfig
    ) -> tuple[str, list[dict], dict[str, Path]]:
        """
        Returns:
            - final_output: string result from agent
            - trace_events: list of Trace v1 event dicts
            - artifacts: dict mapping artifact names to paths
        """
        ...
```

### Grader Specifications

Deterministic graders operate on normalized trace artifacts:

1. **TraceSchemaGrader** (`trace_schema`): Validates all trace events conform to schema
2. **VerifyBeforeDoneGrader** (`verify_before_done`): Ensures verification before completion
3. **EvidenceRequiredGrader** (`evidence_required`): Validates TODO events have evidence
4. **OrderingGrader** (`ordering`): Enforces event ordering constraints
5. **BudgetComplianceGrader** (`budget`): Validates budget limits
6. **DiffConstraintsGrader** (`diff_constraints`): Validates diff patterns and constraints

### CLI Commands

```bash
# Validate scenario files
ash-hawk scenario validate PATH

# Run scenarios
ash-hawk scenario run PATH --sut <adapter> [--record | --replay]

# Generate report
ash-hawk scenario report --run <run-id>

# Matrix comparison
ash-hawk scenario matrix PATH --sut <adapter> --policies <csv> --models <csv>

# Generate new scenario
ash-hawk scenario new --type <type> --name <slug>

# Record tool mocks
ash-hawk scenario record PATH --sut <adapter>

# Replay with mocks
ash-hawk scenario replay --run <run-id>
```

## Success Criteria

- All Definition of Done commands pass.
- New scenario surface is discoverable, deterministic under replay, and produces actionable failure output.

## Commit Strategy

- Atomic commits by capability slice (contracts, runner/storage, adapters, graders, CLI/report, examples/tests).
