# type-hygiene: skip-file
from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

import pydantic as pd


class SchemaFieldType(StrEnum):
    OBJECT = "object"
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"


class ToolFieldSpec(pd.BaseModel):
    name: str
    type: SchemaFieldType
    description: str
    required: bool = False
    item_type: SchemaFieldType | None = None
    default_string: str | None = None
    default_integer: int | None = None
    default_number: float | None = None
    default_boolean: bool | None = None

    model_config = pd.ConfigDict(extra="forbid")


class ToolSchemaSpec(pd.BaseModel):
    schema_type: str = pd.Field(default="object")
    properties: list[ToolFieldSpec] = pd.Field(default_factory=list)
    required: list[str] = pd.Field(default_factory=list)

    model_config = pd.ConfigDict(extra="forbid")

    def validate_payload(self, data: Mapping[str, object]) -> list[str]:
        errors: list[str] = []
        fields_by_name = {field.name: field for field in self.properties}
        for field_name in self.required:
            if field_name not in data or data[field_name] is None:
                errors.append(f"Missing required field: {field_name}")
        for field_name, field in fields_by_name.items():
            if field_name not in data:
                continue
            value = data[field_name]
            if value is None:
                if field.required:
                    errors.append(f"Field '{field_name}' cannot be null")
                continue
            if not _matches_schema_type(value, field.type, field.item_type):
                errors.append(
                    f"Field '{field_name}' expected {field.type.value}"
                    + (
                        f"[{field.item_type.value}]"
                        if field.type is SchemaFieldType.ARRAY and field.item_type is not None
                        else ""
                    )
                )
        return errors


def _matches_schema_type(
    value: object,
    field_type: SchemaFieldType,
    item_type: SchemaFieldType | None,
) -> bool:
    if field_type is SchemaFieldType.STRING:
        return isinstance(value, str)
    if field_type is SchemaFieldType.INTEGER:
        return isinstance(value, int) and not isinstance(value, bool)
    if field_type is SchemaFieldType.NUMBER:
        return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
    if field_type is SchemaFieldType.BOOLEAN:
        return isinstance(value, bool)
    if field_type is SchemaFieldType.OBJECT:
        return isinstance(value, pd.BaseModel)
    if field_type is SchemaFieldType.ARRAY:
        if not isinstance(value, list):
            return False
        if item_type is None:
            return True
        return all(_matches_schema_type(item, item_type, None) for item in value)
    return False


class ToolPermissions(pd.BaseModel):
    filesystem: bool = False
    network: bool = False
    subprocess: bool = False
    external_services: list[str] = pd.Field(default_factory=list)

    model_config = pd.ConfigDict(extra="forbid")


class ToolRetryGuidance(pd.BaseModel):
    retryable_errors: list[str] = pd.Field(default_factory=list)
    non_retryable_errors: list[str] = pd.Field(default_factory=list)

    model_config = pd.ConfigDict(extra="forbid")


class ToolObservabilityConfig(pd.BaseModel):
    emit_logs: bool = True
    emit_metrics: bool = True
    emit_trace_events: bool = True
    log_fields: list[str] = pd.Field(default_factory=list)

    model_config = pd.ConfigDict(extra="forbid")


class ToolExecutionObservability(pd.BaseModel):
    tool_name: str
    latency_ms: int
    success: bool

    model_config = pd.ConfigDict(extra="forbid")


class ToolDependencies(pd.BaseModel):
    internal: list[str] = pd.Field(default_factory=list)
    external: list[str] = pd.Field(default_factory=list)

    model_config = pd.ConfigDict(extra="forbid")


class ToolExampleField(pd.BaseModel):
    name: str
    string_value: str | None = None
    integer_value: int | None = None
    number_value: float | None = None
    boolean_value: bool | None = None
    string_list_value: list[str] = pd.Field(default_factory=list)

    model_config = pd.ConfigDict(extra="forbid")


class ToolExample(pd.BaseModel):
    description: str
    input_fields: list[ToolExampleField] = pd.Field(default_factory=list)
    expected_output_fields: list[ToolExampleField] = pd.Field(default_factory=list)

    model_config = pd.ConfigDict(extra="forbid")


class ToolContractView(pd.BaseModel):
    name: str
    summary: str
    when_to_use: list[str] = pd.Field(default_factory=list)
    when_not_to_use: list[str] = pd.Field(default_factory=list)
    inputs: ToolSchemaSpec = pd.Field(default_factory=ToolSchemaSpec)
    outputs: ToolSchemaSpec = pd.Field(default_factory=ToolSchemaSpec)
    side_effects: list[str] = pd.Field(default_factory=list)
    risk_level: str = "low"
    timeout_seconds: int = 30
    completion_criteria: list[str] = pd.Field(default_factory=list)
    escalation_rules: list[str] = pd.Field(default_factory=list)

    model_config = pd.ConfigDict(extra="forbid")


class ScoreSummary(pd.BaseModel):
    score: float | None = None
    status: str | None = None
    tool: str | None = None

    model_config = pd.ConfigDict(extra="forbid")


class AcceptanceStatus(pd.BaseModel):
    accepted: bool | None = None
    reason: str | None = None

    model_config = pd.ConfigDict(extra="forbid")


class VerificationStatus(pd.BaseModel):
    verified: bool | None = None
    evidence_count: int | None = None

    model_config = pd.ConfigDict(extra="forbid")


class ClaimAuditStatus(pd.BaseModel):
    aligned: bool | None = None

    model_config = pd.ConfigDict(extra="forbid")


class Phase2Metrics(pd.BaseModel):
    coverage: float | None = None

    model_config = pd.ConfigDict(extra="forbid")


class Phase2GateStatus(pd.BaseModel):
    passed: bool | None = None

    model_config = pd.ConfigDict(extra="forbid")


class FailureBucketCount(pd.BaseModel):
    bucket: str
    count: int

    model_config = pd.ConfigDict(extra="forbid")


class FailureCluster(pd.BaseModel):
    cluster: str
    count: int

    model_config = pd.ConfigDict(extra="forbid")


class RankedHypothesis(pd.BaseModel):
    name: str
    score: float
    rationale: str = ""
    target_files: list[str] = pd.Field(default_factory=list)
    ideal_outcome: str = ""

    model_config = pd.ConfigDict(extra="forbid")


class TraceRecord(pd.BaseModel):
    goal_id: str | None = None
    agent: str | None = None
    tool: str | None = None
    skill: str | None = None
    event_type: str | None = None
    phase: str | None = None
    rationale: str | None = None
    success: bool | None = None
    error: str | None = None
    preview: str | None = None

    model_config = pd.ConfigDict(extra="forbid")


class TranscriptRecord(pd.BaseModel):
    speaker: str | None = None
    type: str | None = None
    tool: str | None = None
    skill: str | None = None
    message: str | None = None
    success: bool | None = None

    model_config = pd.ConfigDict(extra="forbid")


class DreamQueueEntry(pd.BaseModel):
    scope: str
    key: str
    value: Any | None = None
    value_json: str | None = None

    model_config = pd.ConfigDict(extra="forbid")


class RuntimeToolContext(pd.BaseModel):
    lead_agent: str | None = None
    active_skills: list[str] = pd.Field(default_factory=list)
    max_iterations: int | None = None
    iteration_budget_mode: str | None = None
    completed_iterations: int | None = None
    remaining_iterations: int | None = None
    active_agent: str | None = None
    available_contexts: list[str] = pd.Field(default_factory=list)
    agent_text: str | None = None
    last_decision: str | None = None
    preferred_tool: str | None = None
    stop_checked: bool | None = None
    stop_reason: str | None = None
    resolved_agent_goal: str | None = None
    loaded_agent_definition: str | None = None
    parsed_response: bool | None = None
    retry_count: int | None = None
    executed_action_plan: bool | None = None
    delegated_to: str | None = None

    model_config = pd.ConfigDict(extra="forbid")


class WorkspaceToolContext(pd.BaseModel):
    workdir: str | None = None
    repo_root: str | None = None
    source_root: str | None = None
    package_name: str | None = None
    allowed_target_files: list[str] = pd.Field(default_factory=list)
    changed_files: list[str] = pd.Field(default_factory=list)
    isolated_workspace: bool | None = None
    isolated_workspace_path: str | None = None
    mutated_files: list[str] = pd.Field(default_factory=list)
    agent_config: str | None = None
    scenario_path: str | None = None
    scenario_targets: list[str] = pd.Field(default_factory=list)
    scenario_required_files: list[str] = pd.Field(default_factory=list)
    scenario_summary: str | None = None

    model_config = pd.ConfigDict(extra="forbid")


class EvaluationToolContext(pd.BaseModel):
    baseline_summary: ScoreSummary = pd.Field(default_factory=ScoreSummary)
    targeted_validation_summary: ScoreSummary = pd.Field(default_factory=ScoreSummary)
    integrity_summary: ScoreSummary = pd.Field(default_factory=ScoreSummary)
    last_eval_summary: ScoreSummary = pd.Field(default_factory=ScoreSummary)
    repeat_eval_summary: ScoreSummary = pd.Field(default_factory=ScoreSummary)
    regressions: list[str] = pd.Field(default_factory=list)
    aggregated_score: float | None = None
    acceptance: AcceptanceStatus = pd.Field(default_factory=AcceptanceStatus)
    verification: VerificationStatus = pd.Field(default_factory=VerificationStatus)
    claim_audit: ClaimAuditStatus = pd.Field(default_factory=ClaimAuditStatus)
    phase2_metrics: Phase2Metrics = pd.Field(default_factory=Phase2Metrics)
    phase2_gate: Phase2GateStatus = pd.Field(default_factory=Phase2GateStatus)

    model_config = pd.ConfigDict(extra="forbid")


class FailureToolContext(pd.BaseModel):
    suspicious_reviews: list[str] = pd.Field(default_factory=list)
    failure_buckets: list[FailureBucketCount] = pd.Field(default_factory=list)
    explanations: list[str] = pd.Field(default_factory=list)
    failure_family: str | None = None
    clustered_failures: list[FailureCluster] = pd.Field(default_factory=list)
    ranked_hypotheses: list[RankedHypothesis] = pd.Field(default_factory=list)
    concepts: list[str] = pd.Field(default_factory=list)
    diagnosis_mode: str | None = None

    model_config = pd.ConfigDict(extra="forbid")


class MemoryToolContext(pd.BaseModel):
    run_state: RuntimeToolContext = pd.Field(default_factory=RuntimeToolContext)
    snapshot_state: dict[str, bool] = pd.Field(default_factory=dict)
    resumed_state: dict[str, bool] = pd.Field(default_factory=dict)
    working_memory: dict[str, Any] = pd.Field(default_factory=dict)
    session_memory: dict[str, Any] = pd.Field(default_factory=dict)
    episodic_memory: dict[str, Any] = pd.Field(default_factory=dict)
    semantic_memory: dict[str, Any] = pd.Field(default_factory=dict)
    personal_memory: dict[str, Any] = pd.Field(default_factory=dict)
    working_snapshot_loaded: bool | None = None
    session_loaded: bool | None = None
    episodic_loaded: bool | None = None
    semantic_loaded: bool | None = None
    personal_loaded: bool | None = None
    search_results: list[str] = pd.Field(default_factory=list)
    last_memory_entry_written: bool | None = None
    episode_recorded: bool | None = None
    lesson_saved: bool | None = None
    lessons_consolidated: bool | None = None
    calibration_factor: float | None = None
    skip_hypothesis: bool | None = None
    backfilled: bool | None = None
    snapshot_saved: bool | None = None
    resumed: bool | None = None

    model_config = pd.ConfigDict(extra="forbid")


class ToolStateContext(pd.BaseModel):
    active_tools: list[str] = pd.Field(default_factory=list)
    tool_contracts: list[ToolContractView] = pd.Field(default_factory=list)
    policy_decisions: list[str] = pd.Field(default_factory=list)
    registered_mcp_tools: list[str] = pd.Field(default_factory=list)
    last_tool: str | None = None
    available_tools_snapshot: list[str] = pd.Field(default_factory=list)
    resolved_toolset: list[str] = pd.Field(default_factory=list)
    policy_checked: bool | None = None
    context_assembled: bool | None = None
    context_sections: list[str] = pd.Field(default_factory=list)

    model_config = pd.ConfigDict(extra="forbid")


class AuditRunResult(pd.BaseModel):
    run_id: str | None = None
    success: bool | None = None
    message: str | None = None
    error: str | None = None
    aggregate_score: float | None = None
    aggregate_passed: bool | None = None

    model_config = pd.ConfigDict(extra="forbid")


class AuditToolContext(pd.BaseModel):
    events: list[TraceRecord] = pd.Field(default_factory=list)
    artifacts: list[str] = pd.Field(default_factory=list)
    transcripts: list[TranscriptRecord] = pd.Field(default_factory=list)
    validation_tools: list[str] = pd.Field(default_factory=list)
    iteration_logs: list[str] = pd.Field(default_factory=list)
    decision_trace: list[str] = pd.Field(default_factory=list)
    sync_events: list[str] = pd.Field(default_factory=list)
    commit_events: list[str] = pd.Field(default_factory=list)
    agent_invocations: list[str] = pd.Field(default_factory=list)
    llm_calls: list[str] = pd.Field(default_factory=list)
    deterministic_tool_calls: list[str] = pd.Field(default_factory=list)
    tool_usage: list[str] = pd.Field(default_factory=list)
    delegation_requests: list[str] = pd.Field(default_factory=list)
    run_summary: dict[str, str] = pd.Field(default_factory=dict)
    diff_report: dict[str, int] = pd.Field(default_factory=dict)
    run_result: AuditRunResult = pd.Field(default_factory=AuditRunResult)

    model_config = pd.ConfigDict(extra="forbid")


class ToolCallContext(pd.BaseModel):
    runtime: RuntimeToolContext = pd.Field(default_factory=RuntimeToolContext)
    workspace: WorkspaceToolContext = pd.Field(default_factory=WorkspaceToolContext)
    evaluation: EvaluationToolContext = pd.Field(default_factory=EvaluationToolContext)
    failure: FailureToolContext = pd.Field(default_factory=FailureToolContext)
    memory: MemoryToolContext = pd.Field(default_factory=MemoryToolContext)
    tool: ToolStateContext = pd.Field(default_factory=ToolStateContext)
    audit: AuditToolContext = pd.Field(default_factory=AuditToolContext)

    model_config = pd.ConfigDict(extra="forbid")


class DelegationRequest(pd.BaseModel):
    agent_name: str
    requested_skills: list[str] = pd.Field(default_factory=list)
    goal_id: str
    description: str

    model_config = pd.ConfigDict(extra="forbid")


class ToolExecutionPayload(pd.BaseModel):
    runtime_updates: RuntimeToolContext = pd.Field(default_factory=RuntimeToolContext)
    workspace_updates: WorkspaceToolContext = pd.Field(default_factory=WorkspaceToolContext)
    evaluation_updates: EvaluationToolContext = pd.Field(default_factory=EvaluationToolContext)
    failure_updates: FailureToolContext = pd.Field(default_factory=FailureToolContext)
    memory_updates: MemoryToolContext = pd.Field(default_factory=MemoryToolContext)
    tool_updates: ToolStateContext = pd.Field(default_factory=ToolStateContext)
    audit_updates: AuditToolContext = pd.Field(default_factory=AuditToolContext)
    delegation: DelegationRequest | None = None
    stop: bool = False
    message: str = ""
    errors: list[str] = pd.Field(default_factory=list)
    observability: ToolExecutionObservability | None = None

    model_config = pd.ConfigDict(extra="forbid")
