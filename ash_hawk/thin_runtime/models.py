# type-hygiene: skip-file
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import pydantic as pd

from ash_hawk.thin_runtime.tool_types import (
    ToolCallContext,
    ToolDependencies,
    ToolExample,
    ToolExecutionPayload,
    ToolObservabilityConfig,
    ToolPermissions,
    ToolRetryGuidance,
    ToolSchemaSpec,
)


def _empty_examples() -> list[dict[str, Any]]:
    return []


class HookStage(StrEnum):
    BEFORE_RUN = "before_run"
    AFTER_RUN = "after_run"
    BEFORE_AGENT = "before_agent"
    AFTER_AGENT = "after_agent"
    BEFORE_SKILL = "before_skill"
    AFTER_SKILL = "after_skill"
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"
    BEFORE_DELEGATION = "before_delegation"
    AFTER_DELEGATION = "after_delegation"
    BEFORE_BASELINE_EVAL = "before_baseline_eval"
    AFTER_BASELINE_EVAL = "after_baseline_eval"
    BEFORE_TARGETED_VALIDATION = "before_targeted_validation"
    AFTER_TARGETED_VALIDATION = "after_targeted_validation"
    BEFORE_INTEGRITY_VALIDATION = "before_integrity_validation"
    AFTER_INTEGRITY_VALIDATION = "after_integrity_validation"
    BEFORE_ACCEPTANCE = "before_acceptance"
    AFTER_ACCEPTANCE = "after_acceptance"
    BEFORE_MEMORY_READ = "before_memory_read"
    AFTER_MEMORY_READ = "after_memory_read"
    BEFORE_MEMORY_WRITE = "before_memory_write"
    AFTER_MEMORY_WRITE = "after_memory_write"
    BEFORE_MEMORY_CONSOLIDATION = "before_memory_consolidation"
    AFTER_MEMORY_CONSOLIDATION = "after_memory_consolidation"
    BEFORE_WORKSPACE_PREPARE = "before_workspace_prepare"
    AFTER_WORKSPACE_PREPARE = "after_workspace_prepare"
    BEFORE_SYNC_BACK = "before_sync_back"
    AFTER_SYNC_BACK = "after_sync_back"
    BEFORE_COMMIT = "before_commit"
    AFTER_COMMIT = "after_commit"
    ON_POLICY_DECISION = "on_policy_decision"
    ON_RETRY = "on_retry"
    ON_STOP_CONDITION = "on_stop_condition"
    ON_SUSPICIOUS_RUN = "on_suspicious_run"
    ON_FAILURE_BUCKETED = "on_failure_bucketed"
    ON_ARTIFACT_WRITTEN = "on_artifact_written"
    ON_OBSERVED_EVENT = "on_observed_event"
    AFTER_DREAM_STATE = "after_dream_state"


class MemoryScopeKind(StrEnum):
    WORKING = "working"
    SESSION = "session"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PERSONAL = "personal"
    ARTIFACT = "artifact"


class ContextCategory(StrEnum):
    GOAL = "goal"
    RUNTIME = "runtime"
    WORKSPACE = "workspace"
    EVALUATION = "evaluation"
    FAILURE = "failure"
    MEMORY = "memory"
    TOOL = "tool"
    AUDIT = "audit"


class RuntimeGoal(pd.BaseModel):
    goal_id: str = pd.Field(description="Stable goal identifier")
    description: str = pd.Field(description="Human-readable goal")
    target_score: float | None = pd.Field(default=None, ge=0.0, le=1.0)
    max_iterations: int = pd.Field(default=10, ge=1)

    model_config = pd.ConfigDict(extra="forbid")


class AgentSpec(pd.BaseModel):
    id: str = pd.Field(default="", description="Stable agent identifier")
    name: str = pd.Field(description="Unique agent name")
    kind: str = pd.Field(default="agent", description="Catalog kind")
    version: str = pd.Field(default="1.0.0", description="Version")
    status: str = pd.Field(default="active", description="Status")
    summary: str = pd.Field(default="", description="Summary")
    role: str = pd.Field(default="", description="Role")
    mission: str = pd.Field(default="", description="Mission")
    goal: str = pd.Field(default="", description="Primary goal the agent owns")
    default_goal_description: str = pd.Field(
        default="",
        description="Default runtime goal description when no explicit goal is provided",
    )
    description: str = pd.Field(default="", description="Agent summary")
    instructions_markdown: str = pd.Field(default="", description="Agent instructions markdown")
    file: str = pd.Field(default="", description="Catalog file")
    authority_level: str = pd.Field(default="bounded", description="Authority level")
    scope: str = pd.Field(default="narrow", description="Scope")
    iteration_budget_mode: str = pd.Field(
        default="tool",
        description="How max_iterations is counted for this agent",
    )
    iteration_completion_tools: list[str] = pd.Field(
        default_factory=list,
        description="Tools that mark a completed iteration when iteration_budget_mode is not tool",
    )
    primary_objectives: list[str] = pd.Field(default_factory=list)
    non_goals: list[str] = pd.Field(default_factory=list)
    success_definition: list[str] = pd.Field(default_factory=list)
    when_to_activate: list[str] = pd.Field(default_factory=list)
    when_not_to_activate: list[str] = pd.Field(default_factory=list)
    available_tools: list[str] = pd.Field(default_factory=list)
    available_skills: list[str] = pd.Field(default_factory=list)
    decision_policy: dict[str, Any] = pd.Field(default_factory=dict)
    tool_selection_policy: list[str] = pd.Field(default_factory=list)
    skill_selection_policy: list[str] = pd.Field(default_factory=list)
    delegation_policy: dict[str, Any] = pd.Field(default_factory=dict)
    memory_policy: dict[str, Any] = pd.Field(default_factory=dict)
    verification_policy: dict[str, Any] = pd.Field(default_factory=dict)
    budgets: dict[str, Any] = pd.Field(default_factory=dict)
    risk_controls: dict[str, Any] = pd.Field(default_factory=dict)
    reporting_contract: dict[str, Any] = pd.Field(default_factory=dict)
    completion_criteria: list[str] = pd.Field(default_factory=list)
    escalation_rules: list[str] = pd.Field(default_factory=list)
    dependencies: dict[str, Any] = pd.Field(default_factory=dict)
    examples: list[dict[str, Any]] = pd.Field(default_factory=_empty_examples)
    skill_names: list[str] = pd.Field(default_factory=list)
    hook_names: list[str] = pd.Field(default_factory=list)
    memory_read_scopes: list[str] = pd.Field(default_factory=list)
    memory_write_scopes: list[str] = pd.Field(default_factory=list)

    model_config = pd.ConfigDict(extra="forbid")


class SkillSpec(pd.BaseModel):
    name: str = pd.Field(description="Unique skill name")
    description: str = pd.Field(default="", description="Skill summary")
    instructions_markdown: str = pd.Field(default="", description="Skill instructions markdown")
    id: str = pd.Field(default="", description="Stable skill identifier")
    kind: str = pd.Field(default="skill", description="Catalog kind")
    version: str = pd.Field(default="1.0.0", description="Version")
    status: str = pd.Field(default="active", description="Status")
    summary: str = pd.Field(default="", description="Summary")
    goal: str = pd.Field(default="", description="Skill goal")
    file: str = pd.Field(default="", description="Catalog file")
    category: str = pd.Field(default="reasoning", description="Category")
    scope: str = pd.Field(default="narrow", description="Scope")
    when_to_use: list[str] = pd.Field(default_factory=list)
    when_not_to_use: list[str] = pd.Field(default_factory=list)
    triggers: list[str] = pd.Field(default_factory=list)
    anti_triggers: list[str] = pd.Field(default_factory=list)
    prerequisites: list[str] = pd.Field(default_factory=list)
    inputs_expected: list[str] = pd.Field(default_factory=list)
    procedure: list[str] = pd.Field(default_factory=list)
    decision_points: list[str] = pd.Field(default_factory=list)
    fallback_strategy: list[str] = pd.Field(default_factory=list)
    outputs: dict[str, Any] = pd.Field(default_factory=dict)
    completion_criteria: list[str] = pd.Field(default_factory=list)
    escalation_rules: list[str] = pd.Field(default_factory=list)
    dependencies: dict[str, Any] = pd.Field(default_factory=dict)
    examples: list[dict[str, Any]] = pd.Field(default_factory=_empty_examples)
    tool_names: list[str] = pd.Field(default_factory=list)
    input_contexts: list[str] = pd.Field(default_factory=list)
    output_contexts: list[str] = pd.Field(default_factory=list)
    memory_read_scopes: list[str] = pd.Field(default_factory=list)
    memory_write_scopes: list[str] = pd.Field(default_factory=list)

    model_config = pd.ConfigDict(extra="forbid")


class ToolSpec(pd.BaseModel):
    name: str = pd.Field(description="Unique tool name")
    description: str = pd.Field(description="Deterministic tool summary")
    id: str = pd.Field(default="", description="Stable tool identifier")
    kind: str = pd.Field(default="tool", description="Catalog kind")
    version: str = pd.Field(default="1.0.0", description="Version")
    status: str = pd.Field(default="active", description="Status")
    summary: str = pd.Field(default="", description="Summary")
    goal: str = pd.Field(default="", description="Tool goal")
    python_file: str = pd.Field(default="", description="Python file path")
    entrypoint: str = pd.Field(default="", description="Entrypoint name")
    callable: str = pd.Field(default="run", description="Callable name")
    capabilities: list[str] = pd.Field(default_factory=list)
    when_to_use: list[str] = pd.Field(default_factory=list)
    when_not_to_use: list[str] = pd.Field(default_factory=list)
    inputs: ToolSchemaSpec = pd.Field(default_factory=ToolSchemaSpec)
    outputs: ToolSchemaSpec = pd.Field(default_factory=ToolSchemaSpec)
    side_effects: list[str] = pd.Field(default_factory=list)
    permissions: ToolPermissions = pd.Field(default_factory=ToolPermissions)
    risk_level: str = pd.Field(default="low", description="Risk level")
    timeout_seconds: int = pd.Field(default=30, ge=1)
    idempotent: bool = pd.Field(default=True)
    supports_dry_run: bool = pd.Field(default=False)
    failure_modes: list[str] = pd.Field(default_factory=list)
    retry_guidance: ToolRetryGuidance = pd.Field(default_factory=ToolRetryGuidance)
    observability: ToolObservabilityConfig = pd.Field(default_factory=ToolObservabilityConfig)
    dependencies: ToolDependencies = pd.Field(default_factory=ToolDependencies)
    examples: list[ToolExample] = pd.Field(default_factory=list)
    completion_criteria: list[str] = pd.Field(default_factory=list)
    escalation_rules: list[str] = pd.Field(default_factory=list)
    deterministic: bool = pd.Field(default=True)
    required_contexts: list[str] = pd.Field(default_factory=list)
    produces_contexts: list[str] = pd.Field(default_factory=list)

    model_config = pd.ConfigDict(extra="forbid")


class HookSpec(pd.BaseModel):
    name: str = pd.Field(description="Unique hook name")
    stage: HookStage = pd.Field(description="Hook execution stage")
    description: str = pd.Field(description="Hook summary")

    model_config = pd.ConfigDict(extra="forbid")


class MemoryScopeSpec(pd.BaseModel):
    name: str = pd.Field(description="Memory scope name")
    kind: MemoryScopeKind = pd.Field(description="Memory scope kind")
    description: str = pd.Field(description="What this memory stores")
    writable_by: list[str] = pd.Field(default_factory=list)

    model_config = pd.ConfigDict(extra="forbid")


class ContextFieldSpec(pd.BaseModel):
    name: str = pd.Field(description="Context field name")
    category: ContextCategory = pd.Field(description="Context category")
    description: str = pd.Field(description="Field summary")
    required: bool = pd.Field(default=False)

    model_config = pd.ConfigDict(extra="forbid")


class ToolCall(pd.BaseModel):
    tool_name: str = pd.Field(description="Tool name")
    goal_id: str = pd.Field(description="Goal id")
    caller_agent: str | None = pd.Field(default=None)
    caller_skill: str | None = pd.Field(default=None)
    agent_text: str | None = pd.Field(default=None)
    skills: list[str] = pd.Field(default_factory=list)
    remaining_tools: list[str] = pd.Field(default_factory=list)
    available_contexts: list[str] = pd.Field(default_factory=list)
    iterations: int | None = pd.Field(default=None, description="Completed loop iterations")
    tool_call_count: int | None = pd.Field(default=None, description="Executed tool calls")
    max_iterations: int | None = pd.Field(default=None)
    retry_count: int | None = pd.Field(default=None)
    context: ToolCallContext = pd.Field(default_factory=ToolCallContext)

    model_config = pd.ConfigDict(extra="forbid")


class ToolResult(pd.BaseModel):
    tool_name: str = pd.Field(description="Tool name")
    success: bool = pd.Field(description="Whether tool succeeded")
    payload: ToolExecutionPayload = pd.Field(default_factory=ToolExecutionPayload)
    error: str | None = pd.Field(default=None)

    model_config = pd.ConfigDict(extra="forbid")


class HookEvent(pd.BaseModel):
    hook_name: str = pd.Field(description="Hook name")
    stage: HookStage = pd.Field(description="Stage")
    payload: dict[str, Any] = pd.Field(default_factory=dict)

    model_config = pd.ConfigDict(extra="forbid")


class DelegationRecord(pd.BaseModel):
    agent_name: str = pd.Field(description="Delegated agent name")
    goal_id: str = pd.Field(description="Delegated goal id")
    selected_tool_names: list[str] = pd.Field(default_factory=list)
    success: bool = pd.Field(description="Whether delegated execution succeeded")
    error: str | None = pd.Field(default=None)

    model_config = pd.ConfigDict(extra="forbid")


class ContextSnapshot(pd.BaseModel):
    goal: dict[str, Any] = pd.Field(default_factory=dict)
    runtime: dict[str, Any] = pd.Field(default_factory=dict)
    workspace: dict[str, Any] = pd.Field(default_factory=dict)
    evaluation: dict[str, Any] = pd.Field(default_factory=dict)
    failure: dict[str, Any] = pd.Field(default_factory=dict)
    memory: dict[str, Any] = pd.Field(default_factory=dict)
    tool: dict[str, Any] = pd.Field(default_factory=dict)
    audit: dict[str, Any] = pd.Field(default_factory=dict)

    model_config = pd.ConfigDict(extra="forbid")


def _empty_agent_specs() -> list[AgentSpec]:
    return []


def _empty_skill_specs() -> list[SkillSpec]:
    return []


def _empty_tool_specs() -> list[ToolSpec]:
    return []


def _empty_hook_specs() -> list[HookSpec]:
    return []


def _empty_memory_specs() -> list[MemoryScopeSpec]:
    return []


def _empty_context_specs() -> list[ContextFieldSpec]:
    return []


def _empty_tool_results() -> list[ToolResult]:
    return []


def _empty_hook_events() -> list[HookEvent]:
    return []


def _empty_strings() -> list[str]:
    return []


def _empty_delegations() -> list[DelegationRecord]:
    return []


class ThinRuntimeCatalog(pd.BaseModel):
    agents: list[AgentSpec] = pd.Field(default_factory=_empty_agent_specs)
    skills: list[SkillSpec] = pd.Field(default_factory=_empty_skill_specs)
    tools: list[ToolSpec] = pd.Field(default_factory=_empty_tool_specs)
    hooks: list[HookSpec] = pd.Field(default_factory=_empty_hook_specs)
    memory_scopes: list[MemoryScopeSpec] = pd.Field(default_factory=_empty_memory_specs)
    context_fields: list[ContextFieldSpec] = pd.Field(default_factory=_empty_context_specs)

    model_config = pd.ConfigDict(extra="forbid")

    @pd.model_validator(mode="after")
    def validate_catalog_references(self) -> ThinRuntimeCatalog:
        skill_names = {skill.name for skill in self.skills}
        tool_names = {tool.name for tool in self.tools}
        hook_names = {hook.name for hook in self.hooks}
        memory_names = {scope.name: scope for scope in self.memory_scopes}
        context_names = {field.name for field in self.context_fields}

        for agent in self.agents:
            missing_skills = sorted(set(agent.skill_names) - skill_names)
            if missing_skills:
                raise ValueError(
                    f"Agent '{agent.name}' references unknown skills: {missing_skills}"
                )
            missing_hooks = sorted(set(agent.hook_names) - hook_names)
            if missing_hooks:
                raise ValueError(f"Agent '{agent.name}' references unknown hooks: {missing_hooks}")
            for scope_name in agent.memory_read_scopes + agent.memory_write_scopes:
                if scope_name not in memory_names:
                    raise ValueError(
                        f"Agent '{agent.name}' references unknown memory scope: {scope_name}"
                    )
            for scope_name in agent.memory_write_scopes:
                if agent.name not in memory_names[scope_name].writable_by:
                    raise ValueError(
                        f"Agent '{agent.name}' cannot write declared memory scope '{scope_name}'"
                    )

        for skill in self.skills:
            missing_tools = sorted(set(skill.tool_names) - tool_names)
            if missing_tools:
                raise ValueError(f"Skill '{skill.name}' references unknown tools: {missing_tools}")
            for context_name in skill.input_contexts + skill.output_contexts:
                if context_name not in context_names:
                    raise ValueError(
                        f"Skill '{skill.name}' references unknown context: {context_name}"
                    )
            for scope_name in skill.memory_read_scopes + skill.memory_write_scopes:
                if scope_name not in memory_names:
                    raise ValueError(
                        f"Skill '{skill.name}' references unknown memory scope: {scope_name}"
                    )

        return self


class ThinRuntimeExecutionResult(pd.BaseModel):
    run_id: str = pd.Field(description="Runtime run id")
    goal: RuntimeGoal = pd.Field(description="Runtime goal")
    agent: AgentSpec = pd.Field(description="Active agent")
    active_skills: list[SkillSpec] = pd.Field(default_factory=_empty_skill_specs)
    tools: list[ToolSpec] = pd.Field(default_factory=_empty_tool_specs)
    context: ContextSnapshot = pd.Field(default_factory=ContextSnapshot)
    available_contexts: list[str] = pd.Field(default_factory=_empty_strings)
    selected_tool_names: list[str] = pd.Field(default_factory=_empty_strings)
    tool_results: list[ToolResult] = pd.Field(default_factory=_empty_tool_results)
    emitted_hooks: list[HookEvent] = pd.Field(default_factory=_empty_hook_events)
    delegations: list[DelegationRecord] = pd.Field(default_factory=_empty_delegations)
    memory_snapshot: dict[str, dict[str, Any]] = pd.Field(default_factory=dict)
    artifact_dir: str = pd.Field(description="Persisted artifact directory")
    success: bool = pd.Field(description="Whether the agent run succeeded")
    error: str | None = pd.Field(default=None)

    model_config = pd.ConfigDict(extra="forbid")


@dataclass(frozen=True)
class RegistrySummary:
    agents: int
    skills: int
    tools: int
    hooks: int
    memory_scopes: int
    context_fields: int
