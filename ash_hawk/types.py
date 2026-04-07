# type-hygiene: skip-file
"""Ash Hawk - Core type definitions for evaluation harness.

This module defines all core data models for the Ash-Hawk evaluation harness.
All models use Pydantic with strict validation (extra="forbid").

Key types:
- EvalStatus, FailureMode, ToolPermission: Enums
- TokenUsage, ToolSurfacePolicy: Resource tracking
- GraderSpec/GraderResult: Typed grader configuration and results
- EvalAgentConfig, EvalMcpServerConfig: Agent configuration
- EvalTranscript, EvalOutcome, TrialResult: Execution records
- EvalTrial: Single evaluation attempt (deprecated, kept for compatibility)
- RunEnvelope, TrialEnvelope: Reproducibility metadata
"""

from __future__ import annotations

import enum
import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import pydantic as pd

# =============================================================================
# ENUMS
# =============================================================================


class EvalStatus(enum.StrEnum):
    """Status of an evaluation task or trial."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class FailureMode(enum.StrEnum):
    """Failure modes for evaluation trials."""

    TIMEOUT = "timeout"
    TOOL_DENIED = "tool_denied"
    CRASH = "crash"
    JUDGE_ERROR = "judge_error"
    POLICY_VIOLATION = "policy_violation"
    AGENT_ERROR = "agent_error"
    VALIDATION_ERROR = "validation_error"
    RESOURCE_EXCEEDED = "resource_exceeded"


class ToolPermission(enum.StrEnum):
    """Permission levels for tool access."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


# =============================================================================
# TOKEN USAGE
# =============================================================================


class TokenUsage(pd.BaseModel):
    """Token usage tracking for a trial."""

    input: int = 0
    output: int = 0
    reasoning: int = 0
    cache_read: int = 0
    cache_write: int = 0

    @pd.computed_field(return_type=int)
    def total(self) -> int:
        """Total tokens used (input + output + reasoning)."""
        return self.input + self.output + self.reasoning

    model_config = pd.ConfigDict(extra="forbid")


# =============================================================================
# TOOL SURFACE POLICY
# =============================================================================


class ToolSurfacePolicy(pd.BaseModel):
    """Policy defining the tool surface and permission boundary for trials.

    This defines what tools an agent can use, what filesystem paths are
    accessible, network rules, and various resource limits.
    """

    allowed_tools: list[str] = pd.Field(
        default_factory=list,
        description="List of allowed tool names (glob patterns supported)",
    )
    denied_tools: list[str] = pd.Field(
        default_factory=list,
        description="List of denied tool names (takes precedence over allowed)",
    )
    default_permission: ToolPermission = pd.Field(
        default=ToolPermission.ASK,
        description="Default action when tool is not in allowlist/denylist",
    )
    allowed_roots: list[str] = pd.Field(
        default_factory=list,
        description="List of allowed filesystem root paths",
    )
    network_allowed: bool = pd.Field(
        default=False,
        description="Whether network access is allowed",
    )
    network_allowlist: list[str] = pd.Field(
        default_factory=list,
        description="List of allowed network hosts/domains if network_allowed=True",
    )
    max_tool_calls: int | None = pd.Field(
        default=None,
        description="Maximum number of tool calls allowed per trial",
    )
    timeout_seconds: float = pd.Field(
        default=300.0,
        description="Maximum trial execution time in seconds",
    )
    token_budget: int | None = pd.Field(
        default=None,
        description="Maximum total tokens allowed per trial",
    )
    cost_budget_usd: float | None = pd.Field(
        default=None,
        description="Maximum cost in USD allowed per trial",
    )
    max_file_size_bytes: int | None = pd.Field(
        default=None,
        description="Maximum file size that can be read/written",
    )
    env_vars_allowed: list[str] = pd.Field(
        default_factory=list,
        description="Environment variables the agent is allowed to access",
    )
    max_tool_depth: int = pd.Field(
        default=10,
        ge=1,
        description="Maximum nested tool call depth",
    )

    def is_tool_allowed(self, tool_name: str) -> ToolPermission:
        """Check if a tool is allowed based on the policy.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            ToolPermission indicating the action to take.
        """
        import fnmatch

        # Check denylist first (highest priority)
        for pattern in self.denied_tools:
            if fnmatch.fnmatch(tool_name, pattern):
                return ToolPermission.DENY

        # Check allowlist
        for pattern in self.allowed_tools:
            if fnmatch.fnmatch(tool_name, pattern):
                return ToolPermission.ALLOW

        # Fall back to default
        return self.default_permission

    model_config = pd.ConfigDict(extra="forbid")


# =============================================================================
# GRADER TYPES
# =============================================================================


class GraderSpec(pd.BaseModel):
    """Typed configuration for a grader."""

    grader_type: str = pd.Field(
        description="Type of grader (e.g., 'string_match', 'test_runner', 'llm_judge')",
    )
    config: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Grader-specific configuration",
    )
    weight: float = pd.Field(
        default=1.0,
        ge=0.0,
        description="Weight for this grader in aggregate scoring",
    )
    required: bool = pd.Field(
        default=False,
        description="Whether this grader must pass for overall success",
    )
    timeout_seconds: float | None = pd.Field(
        default=None,
        description="Timeout for grader execution",
    )

    model_config = pd.ConfigDict(extra="forbid")


class GraderResult(pd.BaseModel):
    """Result from a single grader evaluation."""

    grader_type: str = pd.Field(
        description="Type of grader that produced this result",
    )
    grader_id: str | None = pd.Field(
        default=None,
        description="Unique identifier for the grader instance",
    )
    score: float = pd.Field(
        ge=0.0,
        le=1.0,
        description="Score from 0.0 to 1.0",
    )
    passed: bool = pd.Field(
        description="Whether the trial passed this grader's evaluation",
    )
    details: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Detailed results specific to grader type",
    )
    error_message: str | None = pd.Field(
        default=None,
        description="Error message if grader execution failed",
    )
    execution_time_seconds: float | None = pd.Field(
        default=None,
        description="Time taken to execute the grader",
    )
    confidence: float | None = pd.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence level of the judgment (0.0-1.0)",
    )
    needs_review: bool = pd.Field(
        default=False,
        description="Whether this result needs human review",
    )
    review_reason: str | None = pd.Field(
        default=None,
        description="Reason for needing review if needs_review is True",
    )

    model_config = pd.ConfigDict(extra="forbid")


class EvalMcpServerConfig(pd.BaseModel):
    """Configuration for an MCP server attached to an agent."""

    name: str = pd.Field(description="Logical MCP server name")
    command: str = pd.Field(description="Executable command for the MCP server")
    args: list[str] = pd.Field(
        default_factory=list,
        description="Command-line arguments passed to the MCP server",
    )
    env: dict[str, str] = pd.Field(
        default_factory=dict,
        description="Environment variables for the MCP server process",
    )
    cwd: str | None = pd.Field(
        default=None,
        description="Optional working directory for the MCP server process",
    )

    model_config = pd.ConfigDict(extra="forbid")


class EvalAgentConfig(pd.BaseModel):
    """Configuration for an evaluation agent."""

    name: str | None = pd.Field(
        default=None,
        description="Agent name to resolve from dawn-kestrel registry",
    )
    provider: str | None = pd.Field(
        default=None,
        description="Provider override for this eval",
    )
    model: str | None = pd.Field(
        default=None,
        description="Model override for this eval",
    )
    class_name: str | None = pd.Field(
        default=None,
        alias="class",
        description="Runner class name or import path",
    )
    location: str | None = pd.Field(
        default=None,
        description="Optional file path for runner class loading",
    )
    kwargs: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Keyword arguments passed to the runner constructor",
    )
    mcp_servers: list[EvalMcpServerConfig] = pd.Field(
        default_factory=list,
        description="MCP servers to attach to the agent runtime",
    )

    model_config = pd.ConfigDict(extra="forbid", populate_by_name=True)


# =============================================================================
# TRANSCRIPT
# =============================================================================


class EvalTranscript(pd.BaseModel):
    """Complete record of a trial execution.

    Captures all messages, tool calls, timing, and resource usage
    for reproducibility and analysis.
    """

    messages: list[dict[str, Any]] = pd.Field(
        default_factory=list,
        description="List of messages exchanged during the trial",
    )
    tool_calls: list[dict[str, Any]] = pd.Field(
        default_factory=list,
        description="List of tool calls made during the trial",
    )
    trace_events: list[dict[str, Any]] = pd.Field(
        default_factory=list,
        description="Normalized trace events captured during the trial",
    )
    token_usage: TokenUsage = pd.Field(
        default_factory=TokenUsage,
        description="Token usage statistics",
    )
    cost_usd: float = pd.Field(
        default=0.0,
        ge=0.0,
        description="Total cost in USD for the trial",
    )
    duration_seconds: float = pd.Field(
        default=0.0,
        ge=0.0,
        description="Total execution time in seconds",
    )
    agent_response: str | dict[str, Any] | None = pd.Field(
        default=None,
        description="Final agent response/output",
    )
    error_trace: str | None = pd.Field(
        default=None,
        description="Stack trace if an error occurred",
    )

    model_config = pd.ConfigDict(extra="forbid")


# =============================================================================
# OUTCOME
# =============================================================================


class EvalOutcome(pd.BaseModel):
    """Final outcome state of a trial."""

    status: EvalStatus = pd.Field(
        description="Final status of the trial",
    )
    failure_mode: FailureMode | None = pd.Field(
        default=None,
        description="Failure mode if status is ERROR or CANCELLED",
    )
    error_message: str | None = pd.Field(
        default=None,
        description="Human-readable error message",
    )
    completed_at: str | None = pd.Field(
        default=None,
        description="ISO timestamp when the trial completed",
    )

    @classmethod
    def success(cls) -> EvalOutcome:
        """Create a successful outcome."""
        return cls(
            status=EvalStatus.COMPLETED,
            completed_at=datetime.now(UTC).isoformat(),
        )

    @classmethod
    def failure(
        cls,
        failure_mode: FailureMode,
        error_message: str | None = None,
    ) -> EvalOutcome:
        """Create a failure outcome."""
        return cls(
            status=EvalStatus.ERROR,
            failure_mode=failure_mode,
            error_message=error_message,
            completed_at=datetime.now(UTC).isoformat(),
        )

    model_config = pd.ConfigDict(extra="forbid")


# =============================================================================
# TRIAL RESULT
# =============================================================================


class TrialResult(pd.BaseModel):
    """Aggregated results from all graders for a single trial."""

    trial_id: str = pd.Field(
        description="ID of the trial this result belongs to",
    )
    outcome: EvalOutcome = pd.Field(
        description="Final outcome of the trial",
    )
    transcript: EvalTranscript = pd.Field(
        default_factory=EvalTranscript,
        description="Full transcript of the trial",
    )
    grader_results: list[GraderResult] = pd.Field(
        default_factory=list,
        description="Results from each grader",
    )
    aggregate_score: float = pd.Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weighted aggregate score from all graders",
    )
    aggregate_passed: bool = pd.Field(
        default=False,
        description="Whether the trial passed overall evaluation",
    )

    model_config = pd.ConfigDict(extra="forbid")


# =============================================================================
# EVALUATION TRIAL (deprecated, kept for auto_research compatibility)
# =============================================================================


class EvalTrial(pd.BaseModel):
    """Single attempt at evaluating an agent on a task.

    DEPRECATED: EvalTrial is deprecated and will be removed in a future version.
    Kept for auto_research compatibility.
    """

    id: str = pd.Field(
        description="Unique identifier for this trial",
    )
    task_id: str = pd.Field(
        description="ID of the task being evaluated",
    )
    status: EvalStatus = pd.Field(
        default=EvalStatus.PENDING,
        description="Current status of the trial",
    )
    attempt_number: int = pd.Field(
        default=1,
        ge=1,
        description="Attempt number for this task",
    )
    input_snapshot: str | dict[str, Any] | None = pd.Field(
        default=None,
        description="Snapshot of the input at trial start",
    )
    task_tags: list[str] = pd.Field(
        default_factory=list,
        description="Tags from the associated task for filtering",
    )
    result: TrialResult | None = pd.Field(
        default=None,
        description="Results from the trial (populated after completion)",
    )
    envelope: TrialEnvelope | None = pd.Field(
        default=None,
        description="Trial envelope with reproducibility metadata",
    )

    model_config = pd.ConfigDict(extra="forbid")


# =============================================================================
# ENVELOPES (Reproducibility Metadata)
# =============================================================================


class RunEnvelope(pd.BaseModel):
    """Reproducibility metadata for an entire evaluation run."""

    run_id: str = pd.Field(
        description="Unique identifier for this run",
    )
    suite_id: str = pd.Field(
        description="ID of the evaluation suite being run",
    )
    suite_hash: str = pd.Field(
        description="Hash of suite content for integrity verification",
    )
    harness_version: str = pd.Field(
        description="Version of the ash-hawk harness",
    )
    git_commit: str | None = pd.Field(
        default=None,
        description="Git commit hash of the harness code",
    )
    agent_name: str = pd.Field(
        description="Name/identifier of the agent being evaluated",
    )
    agent_version: str | None = pd.Field(
        default=None,
        description="Version of the agent",
    )
    provider: str = pd.Field(
        description="LLM provider (e.g., 'anthropic', 'openai', 'zai')",
    )
    model: str = pd.Field(
        description="Model identifier (e.g., 'claude-3-5-sonnet-20241022')",
    )
    model_params: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Model parameters (temperature, top_p, max_tokens, etc.)",
    )
    seed: int | None = pd.Field(
        default=None,
        description="Random seed if determinism is supported",
    )
    tool_policy_hash: str = pd.Field(
        description="Hash of the tool policy for integrity verification",
    )
    python_version: str = pd.Field(
        description="Python version used for the run",
    )
    os_info: str = pd.Field(
        description="Operating system information",
    )
    config_snapshot: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Sanitized snapshot of configuration",
    )
    created_at: str = pd.Field(
        description="ISO timestamp when the run was created",
    )

    model_config = pd.ConfigDict(extra="forbid")


class TrialEnvelope(pd.BaseModel):
    """Per-trial metadata referencing the parent RunEnvelope."""

    trial_id: str = pd.Field(
        description="Unique identifier for this trial",
    )
    run_id: str = pd.Field(
        description="ID of the parent run",
    )
    task_id: str = pd.Field(
        description="ID of the task being evaluated",
    )
    attempt_number: int = pd.Field(
        default=1,
        ge=1,
        description="Attempt number (for tasks with max_attempts > 1)",
    )
    policy_snapshot: ToolSurfacePolicy = pd.Field(
        description="Snapshot of the tool policy used for this trial",
    )
    created_at: str = pd.Field(
        description="ISO timestamp when the trial was created",
    )
    started_at: str | None = pd.Field(
        default=None,
        description="ISO timestamp when the trial started execution",
    )
    completed_at: str | None = pd.Field(
        default=None,
        description="ISO timestamp when the trial completed",
    )

    model_config = pd.ConfigDict(extra="forbid")


# =============================================================================
# EVALUATION TASK (kept for scenario/runner.py and execution/ compatibility)
# =============================================================================


class EvalTask(pd.BaseModel):
    """Single test case for evaluation."""

    id: str = pd.Field(description="Unique identifier for this task")
    description: str = pd.Field(default="", description="Human-readable description")
    input: str | dict[str, Any] = pd.Field(
        description="Task input - either a string prompt or structured payload",
    )
    expected_output: str | dict[str, Any] | None = pd.Field(
        default=None,
        description="Expected output for deterministic grading (optional)",
    )
    grader_specs: list[GraderSpec] = pd.Field(
        default_factory=list,
        description="List of grader configurations for this task",
    )
    tags: list[str] = pd.Field(default_factory=list, description="Tags for categorizing tasks")
    metadata: dict[str, Any] = pd.Field(
        default_factory=dict, description="Additional task metadata"
    )
    fixtures: dict[str, str] = pd.Field(
        default_factory=dict,
        description="Paths to fixture files/resources",
    )
    timeout_seconds: float | None = pd.Field(
        default=None,
        description="Task-specific timeout override",
    )
    max_attempts: int = pd.Field(default=1, ge=1, description="Maximum attempts allowed")

    model_config = pd.ConfigDict(extra="forbid")


# =============================================================================
# EVALUATION SUITE (kept for scenario/runner.py and execution/ compatibility)
# =============================================================================


class EvalSuite(pd.BaseModel):
    """Collection of related evaluation tasks."""

    id: str = pd.Field(description="Unique identifier for this suite")
    name: str = pd.Field(description="Human-readable name for the suite")
    description: str = pd.Field(default="", description="Detailed description")
    tasks: list[EvalTask] = pd.Field(default_factory=list, description="List of tasks")
    tags: list[str] = pd.Field(default_factory=list, description="Tags for categorizing")
    metadata: dict[str, Any] = pd.Field(default_factory=dict, description="Suite metadata")
    version: str = pd.Field(default="1.0.0", description="Suite schema version")
    agent: EvalAgentConfig | None = pd.Field(
        default=None,
        description="Default agent configuration for all tasks",
    )

    @pd.computed_field(return_type=int)
    def task_count(self) -> int:
        """Number of tasks in the suite."""
        return len(self.tasks)

    model_config = pd.ConfigDict(extra="forbid")


# =============================================================================
# SUITE METRICS (kept for execution/runner.py compatibility)
# =============================================================================


class SuiteMetrics(pd.BaseModel):
    """Aggregate metrics for an evaluation suite run."""

    suite_id: str = pd.Field(description="ID of the suite")
    run_id: str = pd.Field(description="ID of the run")
    total_tasks: int = pd.Field(description="Total number of tasks")
    completed_tasks: int = pd.Field(default=0, description="Completed tasks count")
    passed_tasks: int = pd.Field(default=0, description="Passed tasks count")
    failed_tasks: int = pd.Field(default=0, description="Failed tasks count")
    pass_rate: float = pd.Field(default=0.0, ge=0.0, le=1.0, description="Overall pass rate")
    mean_score: float = pd.Field(default=0.0, ge=0.0, le=1.0, description="Mean score")
    total_tokens: TokenUsage = pd.Field(
        default_factory=TokenUsage,
        description="Total token usage",
    )
    total_cost_usd: float = pd.Field(default=0.0, ge=0.0, description="Total cost in USD")
    total_duration_seconds: float = pd.Field(default=0.0, ge=0.0, description="Execution time")
    latency_p50_seconds: float | None = pd.Field(default=None, description="Median latency")
    latency_p95_seconds: float | None = pd.Field(default=None, description="P95 latency")
    latency_p99_seconds: float | None = pd.Field(default=None, description="P99 latency")
    pass_at_k: dict[int, float] = pd.Field(
        default_factory=dict,
        description="pass@k metrics",
    )
    created_at: str = pd.Field(description="ISO timestamp when metrics were computed")

    model_config = pd.ConfigDict(extra="forbid")


# =============================================================================
# RUN SUMMARY (kept for scenario/runner.py and execution/ compatibility)
# =============================================================================


class EvalRunSummary(pd.BaseModel):
    """Complete summary of an evaluation run."""

    envelope: RunEnvelope = pd.Field(description="Run envelope with reproducibility metadata")
    metrics: SuiteMetrics = pd.Field(description="Aggregate metrics for the run")
    trials: list[EvalTrial] = pd.Field(
        default_factory=list,
        description="All trials in the run",
    )

    model_config = pd.ConfigDict(extra="forbid")


# =============================================================================
# CONTRACT TYPES (absorbed from contracts/run_artifact.py)
# =============================================================================


class ToolCallRecord(pd.BaseModel):
    """Record of a single tool call within a run."""

    tool_name: str = pd.Field(description="Name of the tool that was called")
    outcome: str = pd.Field(
        default="success",
        description="Result of the tool call (success, failure, denied)",
    )
    duration_ms: int | None = pd.Field(
        default=None, description="Duration of the tool call in milliseconds"
    )
    error_message: str | None = pd.Field(
        default=None, description="Error message if the call failed"
    )
    input_args: dict[str, Any] = pd.Field(
        default_factory=dict, description="Input arguments passed to the tool"
    )
    output: str | dict[str, Any] | None = pd.Field(
        default=None, description="Output from the tool (truncated if large)"
    )
    timestamp: datetime | None = pd.Field(default=None, description="When the tool call was made")

    model_config = pd.ConfigDict(extra="allow")


class StepRecord(pd.BaseModel):
    """Record of a reasoning step within a run."""

    step_id: str = pd.Field(default="", description="Unique identifier for this step")
    step_type: str = pd.Field(
        default="action", description="Type of step (plan, action, reflection, etc.)"
    )
    content: str | dict[str, Any] | None = pd.Field(
        default=None, description="The content of the step"
    )
    outcome: str = pd.Field(default="pending", description="Result of this step")
    timestamp: datetime | None = pd.Field(default=None, description="When the step was recorded")
    status: str = pd.Field(
        default="pending",
        description="Status of this step (pending, running, completed, failed)",
    )
    started_at: datetime | None = pd.Field(default=None, description="When this step started")
    completed_at: datetime | None = pd.Field(default=None, description="When this step completed")
    tool_calls: list[ToolCallRecord] = pd.Field(
        default_factory=list, description="Tool calls made during this step"
    )

    model_config = pd.ConfigDict(extra="allow")


class RunArtifact(pd.BaseModel):
    """Complete artifact from a completed agent run."""

    run_id: str = pd.Field(description="Unique identifier for this run")
    suite_id: str | None = pd.Field(
        default=None, description="ID of the evaluation suite this run belongs to"
    )
    agent_id: str = pd.Field(
        default="unknown", description="Identifier of the agent that was evaluated"
    )
    agent_name: str = pd.Field(
        default="unknown", description="Name of the agent that was evaluated"
    )
    task_type: str | None = pd.Field(
        default=None,
        description="Type of task performed (e.g., 'pr_review', 'code_change')",
    )
    outcome: str = pd.Field(
        default="success",
        description="Overall outcome of the run (success, failure, error)",
    )
    tool_calls: list[ToolCallRecord] = pd.Field(
        default_factory=list, description="List of tool calls made during the run"
    )
    steps: list[StepRecord] = pd.Field(
        default_factory=list, description="List of reasoning steps recorded during the run"
    )
    messages: list[dict[str, Any]] = pd.Field(
        default_factory=list, description="List of messages exchanged during the run"
    )
    total_duration_ms: int | None = pd.Field(
        default=None, description="Total duration of the run in milliseconds"
    )
    token_usage: dict[str, int] = pd.Field(
        default_factory=dict, description="Token usage statistics (input, output, total)"
    )
    cost_usd: float | None = pd.Field(default=None, description="Total cost in USD for the run")
    error_message: str | None = pd.Field(
        default=None, description="Error message if the run failed"
    )
    metadata: dict[str, Any] = pd.Field(
        default_factory=dict, description="Additional metadata about the run"
    )
    created_at: datetime | None = pd.Field(default=None, description="When the run was created")
    completed_at: datetime | None = pd.Field(default=None, description="When the run completed")
    overall_score: float = pd.Field(
        default=0.0, ge=0.0, le=1.0, description="Overall score from evaluation (0.0 to 1.0)"
    )
    metrics: dict[str, float] = pd.Field(
        default_factory=dict,
        description="Computed metrics (efficiency_score, quality_score, safety_score, etc.)",
    )
    trial_ids: list[str] = pd.Field(
        default_factory=list, description="IDs of trials included in this run artifact"
    )

    model_config = pd.ConfigDict(extra="allow")

    def is_successful(self) -> bool:
        return self.outcome == "success"

    def get_tool_success_rate(self) -> float:
        if not self.tool_calls:
            return 0.0
        successful = sum(1 for tc in self.tool_calls if tc.outcome == "success")
        return successful / len(self.tool_calls)

    def get_total_tokens(self) -> int:
        return self.token_usage.get("total", 0)

    @classmethod
    def from_dawn_kestrel(cls, dk_artifact: Any) -> RunArtifact:
        tool_calls = []
        for tc in getattr(dk_artifact, "tool_calls", []):
            tool_calls.append(
                ToolCallRecord(
                    tool_name=getattr(tc, "tool_name", "unknown"),
                    outcome=getattr(tc, "outcome", "success"),
                    duration_ms=getattr(tc, "duration_ms"),
                    error_message=getattr(tc, "error_message"),
                    input_args=getattr(tc, "arguments", {}),
                    output=getattr(tc, "result_preview"),
                )
            )
        steps = []
        for step in getattr(dk_artifact, "steps", []):
            steps.append(
                StepRecord(
                    step_id=getattr(step, "step_id", ""),
                    step_type=getattr(step, "title", "action"),
                    content=getattr(step, "summary"),
                    outcome=getattr(step, "status", "pending"),
                    started_at=getattr(step, "started_at"),
                    completed_at=getattr(step, "finished_at"),
                )
            )
        telemetry = getattr(dk_artifact, "telemetry", {})
        token_usage = {
            "input": telemetry.get("input_tokens", 0),
            "output": telemetry.get("output_tokens", 0),
            "total": telemetry.get("total_tokens", 0),
        }
        return cls(
            run_id=getattr(dk_artifact, "run_id", ""),
            agent_id=getattr(dk_artifact, "agent_id", "unknown"),
            agent_name=getattr(dk_artifact, "agent_id", "unknown"),
            task_type=getattr(dk_artifact, "task_type"),
            outcome=getattr(dk_artifact, "outcome", "success"),
            tool_calls=tool_calls,
            steps=steps,
            messages=[],
            total_duration_ms=telemetry.get("duration_ms"),
            token_usage=token_usage,
            cost_usd=telemetry.get("cost_usd"),
            error_message=getattr(dk_artifact, "outcome_details"),
            metadata=getattr(dk_artifact, "metadata", {}),
            created_at=getattr(dk_artifact, "created_at"),
        )


# =============================================================================
# BRIDGE TYPES (absorbed from bridge/__init__.py)
# =============================================================================

_HASHABLE_EXTENSIONS: frozenset[str] = frozenset(
    {".md", ".py", ".yaml", ".yml", ".json", ".txt", ".toml", ".cfg", ".ini"}
)


class TelemetrySink(Protocol):
    """Protocol for receiving telemetry events from agent runs."""

    async def on_iteration_start(self, data: dict[str, Any]) -> None: ...
    async def on_iteration_end(self, data: dict[str, Any]) -> None: ...
    async def on_action_decision(self, data: dict[str, Any]) -> None: ...
    async def on_tool_result(self, data: dict[str, Any]) -> None: ...
    async def on_run_complete(self, data: dict[str, Any]) -> None: ...


@dataclass
class TranscriptData:
    """Captured transcript from agent run."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    trace_events: list[dict[str, Any]] = field(default_factory=list)
    token_usage: dict[str, int] = field(
        default_factory=lambda: {
            "input": 0,
            "output": 0,
            "reasoning": 0,
            "cache_read": 0,
            "cache_write": 0,
        }
    )
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    agent_response: str = ""
    error_trace: str | None = None

    def to_eval_transcript(self) -> Any:
        return EvalTranscript(
            messages=self.messages,
            tool_calls=self.tool_calls,
            trace_events=self.trace_events,
            token_usage=TokenUsage(
                input=self.token_usage.get("input", 0),
                output=self.token_usage.get("output", 0),
                reasoning=self.token_usage.get("reasoning", 0),
                cache_read=self.token_usage.get("cache_read", 0),
                cache_write=self.token_usage.get("cache_write", 0),
            ),
            cost_usd=self.cost_usd,
            duration_seconds=self.duration_seconds,
            agent_response=self.agent_response,
            error_trace=self.error_trace,
        )


@dataclass
class OutcomeData:
    """Captured outcome from agent run."""

    success: bool
    message: str = ""
    error: str | None = None

    def to_eval_outcome(self) -> Any:
        if self.success:
            return EvalOutcome.success()
        return EvalOutcome.failure(
            FailureMode.AGENT_ERROR,
            self.error or self.message,
        )


@dataclass
class RunResult:
    """Result from run_real_agent."""

    transcript: TranscriptData
    outcome: OutcomeData
    run_id: str = ""
    iterations: int = 0
    tools_used: list[str] = field(default_factory=list)
    manifest: RunManifest | None = None


class RunManifest(pd.BaseModel):
    """Provenance identity for a thin run."""

    run_id: str = pd.Field(description="Unique run identifier")
    scenario_path: str = pd.Field(description="Path to scenario YAML")
    scenario_hash: str = pd.Field(description="SHA-256 of scenario file content")
    agent_path: str = pd.Field(description="Path to agent directory")
    agent_hash: str = pd.Field(
        default="", description="SHA-256 of primary agent file (agent.md / AGENT.md)"
    )
    skill_hashes: dict[str, str] = pd.Field(
        default_factory=dict, description="{skill_name: SHA-256 of skill file}"
    )
    tool_hashes: dict[str, str] = pd.Field(
        default_factory=dict, description="{tool_name: SHA-256 of tool file}"
    )
    policy_hash: str = pd.Field(default="", description="SHA-256 of policy.md if present")
    model_name: str = pd.Field(default="", description="Model used for the run")
    variant: str = pd.Field(default="", description="Free-form variant tag (--variant flag)")
    seed: int | None = pd.Field(default=None, description="Deterministic seed if provided")
    grader_set: list[str] = pd.Field(default_factory=list, description="Grader type names used")
    timestamp: str = pd.Field(description="ISO 8601 UTC timestamp")
    ash_hawk_version: str = pd.Field(default="0.1.1", description="Ash-hawk harness version")

    model_config = pd.ConfigDict(extra="forbid")


class DiffFieldChange(pd.BaseModel):
    field: str = pd.Field(description="Manifest field name")
    baseline: str = pd.Field(default="", description="Value in baseline run")
    candidate: str = pd.Field(default="", description="Value in candidate run")


class DiffReport(pd.BaseModel):
    """Structured comparison between two thin runs."""

    baseline_run_id: str = pd.Field(description="Run ID of the baseline")
    candidate_run_id: str = pd.Field(description="Run ID of the candidate")
    baseline_score: float | None = pd.Field(
        default=None, description="Aggregate score of baseline run"
    )
    candidate_score: float | None = pd.Field(
        default=None, description="Aggregate score of candidate run"
    )
    score_delta: float | None = pd.Field(
        default=None, description="candidate_score - baseline_score"
    )
    field_changes: list[DiffFieldChange] = pd.Field(
        default_factory=list, description="Manifest fields that differ"
    )
    grader_deltas: dict[str, dict[str, Any]] = pd.Field(
        default_factory=dict,
        description="{grader_type: {baseline_score, candidate_score, delta, flipped}}",
    )
    recommendation: str = pd.Field(
        default="",
        description="keep / reject / inconclusive recommendation",
    )
    timestamp: str = pd.Field(description="ISO 8601 UTC timestamp")

    model_config = pd.ConfigDict(extra="forbid")


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file's contents."""
    try:
        content = path.read_bytes()
        return hashlib.sha256(content).hexdigest()
    except OSError:
        return ""


def compute_directory_hashes(
    directory: Path,
    extensions: frozenset[str] | None = None,
) -> dict[str, str]:
    """Compute SHA-256 hashes for all text files in a directory tree."""
    if extensions is None:
        extensions = _HASHABLE_EXTENSIONS

    if not directory.is_dir():
        return {}

    hashes: dict[str, str] = {}
    for child in sorted(directory.rglob("*")):
        if not child.is_file():
            continue
        if child.suffix.lower() not in extensions:
            continue
        rel = child.relative_to(directory).as_posix()
        digest = compute_file_hash(child)
        if digest:
            hashes[rel] = digest

    return hashes


def build_run_manifest(
    *,
    run_id: str | None,
    scenario_path: Path,
    agent_path: Path,
    model_name: str,
    variant: str = "",
    seed: int | None = None,
    grader_set: list[str] | None = None,
    ash_hawk_version: str = "0.1.1",
) -> RunManifest:
    """Build a RunManifest with provenance hashes for a thin run."""
    effective_run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"

    scenario_hash = compute_file_hash(scenario_path)

    agent_hash = ""
    for candidate in ("AGENT.md", "agent.md"):
        agent_file = agent_path / candidate
        if agent_file.is_file():
            agent_hash = compute_file_hash(agent_file)
            break

    skill_hashes: dict[str, str] = {}
    skills_dir = agent_path / "skills"
    if skills_dir.is_dir():
        skill_hashes = compute_directory_hashes(skills_dir)

    tool_hashes: dict[str, str] = {}
    tools_dir = agent_path / "tools"
    if tools_dir.is_dir():
        tool_hashes = compute_directory_hashes(tools_dir)

    policy_hash = ""
    for candidate in ("policy.md", "POLICY.md"):
        policy_file = agent_path / candidate
        if policy_file.is_file():
            policy_hash = compute_file_hash(policy_file)
            break

    return RunManifest(
        run_id=effective_run_id,
        scenario_path=str(scenario_path),
        scenario_hash=scenario_hash,
        agent_path=str(agent_path),
        agent_hash=agent_hash,
        skill_hashes=skill_hashes,
        tool_hashes=tool_hashes,
        policy_hash=policy_hash,
        model_name=model_name,
        variant=variant,
        seed=seed,
        grader_set=grader_set or [],
        timestamp=datetime.now(UTC).isoformat(),
        ash_hawk_version=ash_hawk_version,
    )


# =============================================================================
# TYPE ALIASES
# =============================================================================

StorageBackend = Literal["file", "sqlite"]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "EvalStatus",
    "FailureMode",
    "ToolPermission",
    # Core types
    "TokenUsage",
    "ToolSurfacePolicy",
    "GraderSpec",
    "GraderResult",
    "EvalMcpServerConfig",
    "EvalAgentConfig",
    # Transcript & Outcome
    "EvalTranscript",
    "EvalOutcome",
    "TrialResult",
    "EvalTrial",
    # Task & Suite
    "EvalTask",
    "EvalSuite",
    "SuiteMetrics",
    "EvalRunSummary",
    # Envelopes
    "RunEnvelope",
    "TrialEnvelope",
    # Contract types (absorbed)
    "ToolCallRecord",
    "StepRecord",
    "RunArtifact",
    # Bridge types (absorbed)
    "TelemetrySink",
    "TranscriptData",
    "OutcomeData",
    "RunResult",
    "RunManifest",
    "DiffFieldChange",
    "DiffReport",
    "compute_file_hash",
    "compute_directory_hashes",
    "build_run_manifest",
    # Type aliases
    "StorageBackend",
]
