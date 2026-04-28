from __future__ import annotations

import importlib
from pathlib import Path

from ash_hawk.thin_runtime.catalog_loader import load_agent_specs, load_skill_specs
from ash_hawk.thin_runtime.models import (
    ContextCategory,
    ContextFieldSpec,
    HookSpec,
    HookStage,
    MemoryScopeKind,
    MemoryScopeSpec,
    ThinRuntimeCatalog,
    ToolSpec,
)
from ash_hawk.thin_runtime.tool_types import (
    ToolDependencies,
    ToolExample,
    ToolExampleField,
    ToolObservabilityConfig,
    ToolPermissions,
    ToolRetryGuidance,
)


def build_default_catalog() -> ThinRuntimeCatalog:
    catalog_root = Path(__file__).with_name("catalog")
    return ThinRuntimeCatalog(
        agents=load_agent_specs(catalog_root),
        skills=load_skill_specs(catalog_root),
        tools=_build_tools(),
        hooks=_build_hooks(),
        memory_scopes=_build_memory_scopes(),
        context_fields=_build_context_fields(),
    )


def _build_tools() -> list[ToolSpec]:
    tool_data = {
        "run_eval": "Execute one evaluation run.",
        "run_eval_repeated": "Execute repeated evaluation runs.",
        "aggregate_scores": "Aggregate repeated evaluation scores.",
        "run_baseline_eval": "Run the baseline evaluation path.",
        "run_targeted_validation": "Run targeted validation on affected scenarios.",
        "run_integrity_validation": "Run full integrity validation.",
        "verify_outcome": "Verify an outcome against configured graders or checks.",
        "audit_claims": "Compare claims to transcript and tool evidence.",
        "detect_regressions": "Detect regressions outside the target scope.",
        "call_llm_structured": "Call an LLM expecting structured output.",
        "read": "Read file contents with line numbers.",
        "write": "Write content to a file.",
        "edit": "Replace an exact string in a file.",
        "glob": "Find files matching a glob pattern.",
        "grep": "Search file contents using regex.",
        "bash": "Run shell commands and capture output.",
        "todoread": "Read the current todo list.",
        "todowrite": "Write or update the todo list.",
        "test": "Run project tests.",
        "load_workspace_state": "Load workspace information and current file state.",
        "scope_workspace": "Constrain the active workspace scope.",
        "prepare_isolated_workspace": "Create an isolated workspace for safe execution.",
        "sync_workspace_changes": "Sync accepted changes back to the primary workspace.",
        "commit_workspace_changes": "Commit accepted workspace changes.",
        "diff_workspace_changes": "Compute workspace diffs and changed paths.",
        "mutate_agent_files": "Apply deterministic mutations to agent files.",
        "detect_agent_config": "Locate active agent configuration.",
        "search_knowledge": "Search lessons, rules, and historical knowledge.",
        "delegate_task": "Delegate a bounded sub-task to another agent.",
    }
    return [_build_tool_spec(name, description) for name, description in tool_data.items()]


def _build_tool_spec(name: str, description: str) -> ToolSpec:
    command_module = importlib.import_module(f"ash_hawk.thin_runtime.tool_impl.{name}")
    command = getattr(command_module, "COMMAND")
    return ToolSpec(
        id=f"tool.{name}",
        name=name,
        kind="tool",
        version="1.0.0",
        status="active",
        summary=command.summary,
        goal=f"Produce the concrete outcome for tool '{name}'.",
        description=description,
        python_file=f"ash_hawk/thin_runtime/tool_impl/{name}.py",
        entrypoint=f"ash_hawk.thin_runtime.tool_impl.{name}",
        callable="run",
        capabilities=[
            command.summary,
            "Returns validated output",
            "Supports deterministic execution where possible",
        ],
        when_to_use=command.when_to_use,
        when_not_to_use=command.when_not_to_use,
        inputs=command.dk_inputs,
        outputs=command.outputs,
        side_effects=command.side_effects,
        permissions=_tool_permissions(name),
        risk_level=command.risk_level,
        timeout_seconds=command.timeout_seconds,
        idempotent=_tool_idempotent(name),
        supports_dry_run=_tool_supports_dry_run(name),
        failure_modes=[
            "Invalid input shape",
            "Timeout during execution",
            "Dependency unavailable",
            "Permission boundary violation",
        ],
        retry_guidance=ToolRetryGuidance(
            retryable_errors=["timeout", "transient_dependency_failure"],
            non_retryable_errors=["invalid_input", "authorization_failure"],
        ),
        observability=ToolObservabilityConfig(
            emit_logs=True,
            emit_metrics=True,
            emit_trace_events=True,
            log_fields=["tool_name", "execution_id", "latency_ms", "success"],
        ),
        dependencies=ToolDependencies(
            internal=[f"ash_hawk.thin_runtime.tool_impl.{name}"],
            external=[],
        ),
        examples=[
            ToolExample(
                description="Basic invocation",
                input_fields=[ToolExampleField(name="goal_id", string_value="example-goal")],
                expected_output_fields=[ToolExampleField(name="success", boolean_value=True)],
            )
        ],
        completion_criteria=command.completion_criteria,
        escalation_rules=command.escalation_rules,
        deterministic=True,
        required_contexts=_tool_contexts(name),
        produces_contexts=_tool_output_contexts(name),
    )


def _tool_permissions(name: str) -> ToolPermissions:
    return ToolPermissions(
        filesystem=name
        in {
            "read",
            "write",
            "edit",
            "glob",
            "grep",
            "bash",
            "todoread",
            "todowrite",
            "test",
            "load_workspace_state",
            "scope_workspace",
            "prepare_isolated_workspace",
            "sync_workspace_changes",
            "commit_workspace_changes",
            "diff_workspace_changes",
            "mutate_agent_files",
            "detect_agent_config",
        },
        network=name in {"call_llm_structured"},
        subprocess=name in {"commit_workspace_changes", "bash", "test"},
        external_services=["thin_runner"]
        if name
        in {
            "run_eval",
            "run_eval_repeated",
            "run_baseline_eval",
            "run_targeted_validation",
            "run_integrity_validation",
        }
        else [],
    )


def _tool_idempotent(name: str) -> bool:
    del name
    return True


def _tool_supports_dry_run(name: str) -> bool:
    return name in {"commit_workspace_changes", "sync_workspace_changes", "mutate_agent_files"}


def _tool_contexts(name: str) -> list[str]:
    context_map = {
        "read": ["workspace_context"],
        "write": ["workspace_context"],
        "edit": ["workspace_context"],
        "glob": ["workspace_context"],
        "grep": ["workspace_context"],
        "bash": ["workspace_context"],
        "todoread": ["workspace_context"],
        "todowrite": ["workspace_context"],
        "test": ["workspace_context"],
        "run_baseline_eval": ["goal_context", "workspace_context"],
        "run_targeted_validation": ["evaluation_context", "workspace_context"],
        "run_integrity_validation": ["evaluation_context", "workspace_context"],
        "delegate_task": ["goal_context", "runtime_context"],
    }
    return context_map.get(name, [])


def _tool_output_contexts(name: str) -> list[str]:
    output_map = {
        "run_eval": ["evaluation_context"],
        "run_eval_repeated": ["evaluation_context"],
        "aggregate_scores": ["evaluation_context"],
        "run_baseline_eval": ["evaluation_context"],
        "run_targeted_validation": ["evaluation_context"],
        "run_integrity_validation": ["evaluation_context"],
        "verify_outcome": ["evaluation_context"],
        "audit_claims": ["evaluation_context", "audit_context"],
        "detect_regressions": ["evaluation_context"],
        "call_llm_structured": ["audit_context"],
        "read": ["audit_context"],
        "write": ["workspace_context", "audit_context"],
        "edit": ["workspace_context", "audit_context"],
        "glob": ["audit_context"],
        "grep": ["audit_context"],
        "bash": ["audit_context"],
        "todoread": ["audit_context"],
        "todowrite": ["audit_context"],
        "test": ["audit_context"],
        "load_workspace_state": ["workspace_context"],
        "scope_workspace": ["workspace_context"],
        "prepare_isolated_workspace": ["workspace_context"],
        "sync_workspace_changes": ["workspace_context", "audit_context"],
        "commit_workspace_changes": ["workspace_context", "audit_context"],
        "diff_workspace_changes": ["workspace_context"],
        "mutate_agent_files": ["workspace_context"],
        "detect_agent_config": ["workspace_context"],
        "search_knowledge": ["memory_context"],
        "delegate_task": ["runtime_context", "audit_context"],
    }
    return output_map.get(name, [])


def _build_hooks() -> list[HookSpec]:
    hook_data = [
        ("before_run", HookStage.BEFORE_RUN, "Before a runtime execution starts."),
        ("after_run", HookStage.AFTER_RUN, "After a runtime execution completes."),
        ("before_agent", HookStage.BEFORE_AGENT, "Before an agent begins work."),
        ("after_agent", HookStage.AFTER_AGENT, "After an agent completes work."),
        ("before_skill", HookStage.BEFORE_SKILL, "Before a skill becomes active."),
        ("after_skill", HookStage.AFTER_SKILL, "After a skill completes its contribution."),
        ("before_tool", HookStage.BEFORE_TOOL, "Before a tool executes."),
        ("after_tool", HookStage.AFTER_TOOL, "After a tool executes."),
        (
            "before_delegation",
            HookStage.BEFORE_DELEGATION,
            "Before a delegated run is started.",
        ),
        (
            "after_delegation",
            HookStage.AFTER_DELEGATION,
            "After a delegated run completes.",
        ),
        ("before_memory_read", HookStage.BEFORE_MEMORY_READ, "Before memory is read."),
        ("after_memory_read", HookStage.AFTER_MEMORY_READ, "After memory is read."),
        ("before_memory_write", HookStage.BEFORE_MEMORY_WRITE, "Before memory is written."),
        ("after_memory_write", HookStage.AFTER_MEMORY_WRITE, "After memory is written."),
        (
            "on_policy_decision",
            HookStage.ON_POLICY_DECISION,
            "When the runtime records a tool-selection decision.",
        ),
        ("on_stop_condition", HookStage.ON_STOP_CONDITION, "When a stop condition fires."),
        (
            "on_observed_event",
            HookStage.ON_OBSERVED_EVENT,
            "When a nested tool/process event is observed during execution.",
        ),
        (
            "after_dream_state",
            HookStage.AFTER_DREAM_STATE,
            "After dream-state consolidation finishes.",
        ),
    ]
    return [
        HookSpec(name=name, stage=stage, description=description)
        for name, stage, description in hook_data
    ]


def _build_memory_scopes() -> list[MemoryScopeSpec]:
    return [
        MemoryScopeSpec(
            name="working_memory",
            kind=MemoryScopeKind.WORKING,
            description="Current run state, active hypotheses, and scratchpad data.",
            writable_by=["coordinator", "improver", "memory_manager"],
        ),
        MemoryScopeSpec(
            name="session_memory",
            kind=MemoryScopeKind.SESSION,
            description="Session-local decisions, retries, validation history, and temporary artifacts.",
            writable_by=[
                "coordinator",
                "improver",
                "executor",
                "verifier",
                "memory_manager",
                "researcher",
            ],
        ),
        MemoryScopeSpec(
            name="episodic_memory",
            kind=MemoryScopeKind.EPISODIC,
            description="Prior attempts and outcomes across runs.",
            writable_by=["memory_manager"],
        ),
        MemoryScopeSpec(
            name="semantic_memory",
            kind=MemoryScopeKind.SEMANTIC,
            description="Durable learned rules, boosts, penalties, and known patterns.",
            writable_by=["reviewer", "memory_manager"],
        ),
        MemoryScopeSpec(
            name="personal_memory",
            kind=MemoryScopeKind.PERSONAL,
            description="Operator defaults, preferences, and repo-specific heuristics.",
            writable_by=["memory_manager"],
        ),
        MemoryScopeSpec(
            name="artifact_memory",
            kind=MemoryScopeKind.ARTIFACT,
            description="Artifact-backed memory for runs, transcripts, events, and summaries.",
            writable_by=[
                "coordinator",
                "improver",
                "executor",
                "verifier",
                "reviewer",
                "memory_manager",
            ],
        ),
    ]


def _build_context_fields() -> list[ContextFieldSpec]:
    return [
        ContextFieldSpec(
            name="goal_context",
            category=ContextCategory.GOAL,
            description="Top-level run goal, sub-goals, and targets.",
            required=True,
        ),
        ContextFieldSpec(
            name="runtime_context",
            category=ContextCategory.RUNTIME,
            description="Run id, agent, budgets, and active runtime state.",
            required=True,
        ),
        ContextFieldSpec(
            name="workspace_context",
            category=ContextCategory.WORKSPACE,
            description="Repo root, agent root, allowed files, and isolated workspace paths.",
            required=True,
        ),
        ContextFieldSpec(
            name="evaluation_context",
            category=ContextCategory.EVALUATION,
            description="Baseline, targeted, integrity, acceptance, and regression evidence.",
            required=True,
        ),
        ContextFieldSpec(
            name="failure_context",
            category=ContextCategory.FAILURE,
            description="Failures, buckets, suspicious reviews, clusters, and hypotheses.",
            required=True,
        ),
        ContextFieldSpec(
            name="memory_context",
            category=ContextCategory.MEMORY,
            description="Relevant working, episodic, semantic, and preference memory.",
            required=True,
        ),
        ContextFieldSpec(
            name="tool_context",
            category=ContextCategory.TOOL,
            description="Allowed tools, policy decisions, registrations, and tool budgets.",
            required=True,
        ),
        ContextFieldSpec(
            name="audit_context",
            category=ContextCategory.AUDIT,
            description="Transcripts, traces, artifact paths, and event records.",
            required=True,
        ),
    ]
