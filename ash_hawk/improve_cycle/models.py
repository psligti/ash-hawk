from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKED = "blocked"


class FailureCategory(StrEnum):
    POLICY_ORDERING = "policy_ordering"
    POLICY_GUARDRAIL = "policy_guardrail"
    SKILL_MISSING = "skill_missing"
    SKILL_WEAK = "skill_weak"
    TOOL_MISSING = "tool_missing"
    TOOL_INTERFACE_POOR = "tool_interface_poor"
    TOOL_OBSERVABILITY_POOR = "tool_observability_poor"
    HARNESS_LIMITATION = "harness_limitation"
    EVAL_GAP = "eval_gap"
    PLANNER_FAILURE = "planner_failure"
    MEMORY_CONTEXT_FAILURE = "memory_context_failure"
    RETRIEVAL_FAILURE = "retrieval_failure"
    NONDETERMINISM = "nondeterminism"
    ENVIRONMENTAL_FLAKE = "environmental_flake"
    MULTI_CAUSAL = "multi_causal"


class ProposalType(StrEnum):
    POLICY_PATCH = "policy_patch"
    POLICY_REORDER = "policy_reorder"
    PLAYBOOK_UPDATE = "playbook_update"
    SKILL_CREATE = "skill_create"
    SKILL_REVISE = "skill_revise"
    PROMPT_GUARDRAIL = "prompt_guardrail"
    BEHAVIORAL_CHECKLIST = "behavioral_checklist"
    TOOL_CREATE = "tool_create"
    TOOL_REVISE = "tool_revise"
    TOOL_WRAPPER_UPDATE = "tool_wrapper_update"
    HARNESS_PATCH = "harness_patch"
    EVAL_PATCH = "eval_patch"
    EVAL_EXPANSION = "eval_expansion"
    OBSERVABILITY_IMPROVEMENT = "observability_improvement"
    CONFIG_ADJUSTMENT = "config_adjustment"


class PromotionStatus(StrEnum):
    PROMOTE_GLOBAL = "promote_global"
    PROMOTE_AGENT_SPECIFIC = "promote_agent_specific"
    PROMOTE_PACK_SPECIFIC = "promote_pack_specific"
    HOLD_FOR_MORE_DATA = "hold_for_more_data"
    DEMOTE = "demote"
    RETIRE = "retire"
    ROLLBACK = "rollback"


class EvidenceRef(BaseModel):
    artifact_id: str
    kind: str
    pointer: str | None = None
    note: str | None = None

    model_config = ConfigDict(extra="forbid")


class MetricValue(BaseModel):
    name: str
    value: float
    unit: str | None = None
    baseline_value: float | None = None
    delta: float | None = None

    model_config = ConfigDict(extra="forbid")


class ReviewFinding(BaseModel):
    finding_id: str
    title: str
    summary: str
    severity: Severity
    category: FailureCategory | None = None
    evidence: list[EvidenceRef] = cast(list[EvidenceRef], [])
    metrics: list[MetricValue] = cast(list[MetricValue], [])
    tags: list[str] = []
    strategy: str | None = None
    sub_strategy: str | None = None

    model_config = ConfigDict(extra="forbid")


class RunArtifactBundle(BaseModel):
    run_id: str
    experiment_id: str
    agent_id: str
    eval_pack_id: str
    scenario_ids: list[str]
    timestamp: str
    transcripts: list[str] = []
    tool_traces: list[dict[str, Any]] = cast(list[dict[str, Any]], [])
    outputs: list[dict[str, Any]] = cast(list[dict[str, Any]], [])
    metrics: list[MetricValue] = cast(list[MetricValue], [])
    active_lessons: list[str] = []

    model_config = ConfigDict(extra="forbid")


class CompetitorOutput(BaseModel):
    baseline_run_id: str
    replay_run_id: str
    improved: bool
    summary: str
    metrics_before: list[MetricValue] = cast(list[MetricValue], [])
    metrics_after: list[MetricValue] = cast(list[MetricValue], [])
    evidence: list[EvidenceRef] = cast(list[EvidenceRef], [])

    model_config = ConfigDict(extra="forbid")


class TranslatorOutput(BaseModel):
    normalized_findings: list[ReviewFinding] = cast(list[ReviewFinding], [])
    schema_valid: bool = True
    mapping_notes: list[str] = Field(default_factory=list)
    rejected_inputs: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class AnalystOutput(BaseModel):
    findings: list[ReviewFinding] = cast(list[ReviewFinding], [])
    risk_areas: list[str] = []
    recurring_patterns: list[str] = []
    efficiency_metrics: list[MetricValue] = cast(list[MetricValue], [])
    summary: str

    model_config = ConfigDict(extra="forbid")


class FailureClassification(BaseModel):
    category: FailureCategory
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    evidence: list[EvidenceRef] = cast(list[EvidenceRef], [])

    model_config = ConfigDict(extra="forbid")


class TriageOutput(BaseModel):
    primary_cause: FailureClassification
    secondary_causes: list[FailureClassification] = cast(list[FailureClassification], [])
    primary_owner: Literal["coach", "architect", "both", "block"]
    recommended_actions: list[str] = []
    notes: str | None = None

    model_config = ConfigDict(extra="forbid")


class ImprovementProposal(BaseModel):
    proposal_id: str
    source_role: Literal["coach", "architect"]
    proposal_type: ProposalType
    title: str
    summary: str
    rationale: str
    target_surface: str
    confidence: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    evidence: list[EvidenceRef] = cast(list[EvidenceRef], [])
    expected_benefits: list[str] = []
    expected_tradeoffs: list[str] = []
    experiment_hints: list[str] = []
    rollback_notes: str | None = None

    model_config = ConfigDict(extra="forbid")


class CuratedLesson(BaseModel):
    lesson_id: str
    proposal_id: str
    proposal_type: ProposalType
    title: str
    summary: str
    target_surface: str
    approved: bool
    curation_notes: str
    confidence: float
    risk_level: RiskLevel
    lineage: list[str] = []

    model_config = ConfigDict(extra="forbid")


class ExperimentPlan(BaseModel):
    experiment_plan_id: str
    lesson_ids: list[str]
    mode: Literal["isolated", "bundled", "ab", "adversarial", "regression", "cross_pack"]
    scenario_ids: list[str] = []
    eval_pack_ids: list[str] = []
    repeat_count: int = 1
    acceptance_criteria: list[str] = []
    rejection_criteria: list[str] = []
    rollback_criteria: list[str] = []
    max_latency_delta_pct: float | None = None
    max_token_delta_pct: float | None = None
    notes: str | None = None

    model_config = ConfigDict(extra="forbid")


class AppliedChange(BaseModel):
    path: str
    surface: str
    change_kind: str
    description: str

    model_config = ConfigDict(extra="forbid")


class ChangeSet(BaseModel):
    change_set_id: str
    lesson_ids: list[str]
    applied_changes: list[AppliedChange] = cast(list[AppliedChange], [])
    rollback_plan: list[str] = []
    temp_only: bool = False

    model_config = ConfigDict(extra="forbid")


class VerificationCheck(BaseModel):
    name: str
    passed: bool
    summary: str
    metrics: list[MetricValue] = cast(list[MetricValue], [])

    model_config = ConfigDict(extra="forbid")


class VerificationReport(BaseModel):
    verification_id: str
    change_set_id: str
    passed: bool
    overall_summary: str
    score_delta: float | None = None
    variance: float | None = None
    regression_count: int = 0
    checks: list[VerificationCheck] = cast(list[VerificationCheck], [])
    recommendation: Literal["reject", "hold", "promote"]
    notes: list[str] = []

    model_config = ConfigDict(extra="forbid")


class PromotionDecision(BaseModel):
    decision_id: str
    lesson_id: str
    status: PromotionStatus
    scope: str
    reason: str
    effective_version: str | None = None
    rollback_trigger: str | None = None

    model_config = ConfigDict(extra="forbid")


class KnowledgeEntry(BaseModel):
    knowledge_id: str
    lesson_id: str
    kind: str
    title: str
    summary: str
    applicability_conditions: list[str] = []
    anti_patterns: list[str] = []
    references: list[EvidenceRef] = cast(list[EvidenceRef], [])

    model_config = ConfigDict(extra="forbid")


class ExperimentHistorySummary(BaseModel):
    agent_id: str
    experiment_count: int
    promoted_lessons: int
    retired_lessons: int
    common_failure_categories: list[str] = []
    recurring_regressions: list[str] = []
    trend_notes: list[str] = []

    model_config = ConfigDict(extra="forbid")


class AdversarialScenario(BaseModel):
    scenario_id: str
    title: str
    target_weakness: str
    description: str
    expected_failure_mode: str
    evaluation_hooks: list[str] = []

    model_config = ConfigDict(extra="forbid")


class RolePromptPack(BaseModel):
    system_prompt_path: str
    task_template_path: str
    rubric_path: str
    example_paths: list[str] = []

    model_config = ConfigDict(extra="forbid")


class RoleRuntimeConfig(BaseModel):
    model_name: str
    temperature: float = 0.0
    max_output_tokens: int | None = None
    structured_output_required: bool = True
    allowed_tools: list[str] = []

    model_config = ConfigDict(extra="forbid")


class RoleContract(BaseModel):
    role_name: str
    mission: str
    allowed_actions: list[str] = []
    forbidden_actions: list[str] = []
    decision_rules: list[str] = []
    quality_bar: list[str] = []
    failure_behavior: list[str] = []
    tool_access: list[str] = []
    prompt_pack: RolePromptPack
    runtime_config: RoleRuntimeConfig

    model_config = ConfigDict(extra="forbid")


class RoleLifecycleEvent(BaseModel):
    event_id: str
    event_type: Literal["role_started", "role_completed", "role_failed"]
    role: str
    run_id: str
    experiment_id: str
    duration_ms: int | None = None
    status: Literal["success", "failed", "started"]
    input_refs: list[str] = []
    output_refs: list[str] = []
    metrics: dict[str, float | int] = cast(dict[str, float | int], {})
    error_info: str | None = None
    timestamp: str

    model_config = ConfigDict(extra="forbid")


class ImproveCycleCheckpoint(BaseModel):
    checkpoint_id: str
    run_id: str
    experiment_id: str
    last_completed_role: Literal["analyst", "curator", "verifier", "promotion_manager", "complete"]
    status: Literal["in_progress", "completed"]
    state: dict[str, Any] = cast(dict[str, Any], {})
    saved_at: str

    model_config = ConfigDict(extra="forbid")
