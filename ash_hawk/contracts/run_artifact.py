"""Run artifact contract for cross-agent evaluation.

This contract defines the interface for run artifacts that are
analyzed by the improvement pipeline. It provides a compatibility layer
for dawn-kestrel runs while being self-contained in ash-hawk.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pydantic as pd


class ToolCallRecord(pd.BaseModel):
    """Record of a single tool call within a run.

    Attributes:
        tool_name: Name of the tool that was called.
        outcome: Result of the tool call (success, failure, denied).
        duration_ms: Duration of the tool call in milliseconds.
        error_message: Error message if the call failed.
        input_args: Input arguments passed to the tool.
        output: Output from the tool (truncated if large).
        timestamp: When the tool call was made.
    """

    tool_name: str = pd.Field(
        description="Name of the tool that was called",
    )
    outcome: str = pd.Field(
        default="success",
        description="Result of the tool call (success, failure, denied)",
    )
    duration_ms: int | None = pd.Field(
        default=None,
        description="Duration of the tool call in milliseconds",
    )
    error_message: str | None = pd.Field(
        default=None,
        description="Error message if the call failed",
    )
    input_args: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Input arguments passed to the tool",
    )
    output: str | dict[str, Any] | None = pd.Field(
        default=None,
        description="Output from the tool (truncated if large)",
    )
    timestamp: datetime | None = pd.Field(
        default=None,
        description="When the tool call was made",
    )

    model_config = pd.ConfigDict(extra="allow")


class StepRecord(pd.BaseModel):
    """Record of a reasoning step within a run.

    Attributes:
        step_id: Unique identifier for this step.
        step_type: Type of step (plan, action, reflection, etc.).
        content: The content of the step.
        outcome: Result of this step.
        timestamp: When the step was recorded.
        status: Status of this step (pending, running, completed, failed).
        started_at: When this step started.
        completed_at: When this step completed.
        tool_calls: Tool calls made during this step.
    """

    step_id: str = pd.Field(
        default="",
        description="Unique identifier for this step",
    )
    step_type: str = pd.Field(
        default="action",
        description="Type of step (plan, action, reflection, etc.)",
    )
    content: str | dict[str, Any] | None = pd.Field(
        default=None,
        description="The content of the step",
    )
    outcome: str = pd.Field(
        default="pending",
        description="Result of this step",
    )
    timestamp: datetime | None = pd.Field(
        default=None,
        description="When the step was recorded",
    )
    status: str = pd.Field(
        default="pending",
        description="Status of this step (pending, running, completed, failed)",
    )
    started_at: datetime | None = pd.Field(
        default=None,
        description="When this step started",
    )
    completed_at: datetime | None = pd.Field(
        default=None,
        description="When this step completed",
    )
    tool_calls: list[ToolCallRecord] = pd.Field(
        default_factory=list,
        description="Tool calls made during this step",
    )

    model_config = pd.ConfigDict(extra="allow")


class RunArtifact(pd.BaseModel):
    """Complete artifact from a completed agent run.

    This is the primary data structure analyzed by the improvement pipeline.
    It captures all relevant information from a run for analysis, including
    tool calls, reasoning steps, messages, and outcomes.

    Compatible with dawn-kestrel RunArtifact via from_dawn_kestrel() method.

    Attributes:
        run_id: Unique identifier for this run.
        suite_id: ID of the evaluation suite this run belongs to.
        agent_id: Identifier of the agent (dawn-kestrel compatible).
        agent_name: Name of the agent that was evaluated (alias for agent_id).
        task_type: Type of task performed (e.g., 'pr_review', 'code_change').
        outcome: Overall outcome of the run (success, failure, error).
        tool_calls: List of tool calls made during the run.
        steps: List of reasoning steps recorded during the run.
        messages: List of messages exchanged during the run.
        total_duration_ms: Total duration of the run in milliseconds.
        token_usage: Token usage statistics (input, output, total).
        cost_usd: Total cost in USD for the run.
        error_message: Error message if the run failed.
        metadata: Additional metadata about the run.
        created_at: When the run was created.
        completed_at: When the run completed.
    """

    run_id: str = pd.Field(
        description="Unique identifier for this run",
    )
    suite_id: str | None = pd.Field(
        default=None,
        description="ID of the evaluation suite this run belongs to",
    )
    agent_id: str = pd.Field(
        default="unknown",
        description="Identifier of the agent that was evaluated",
    )
    agent_name: str = pd.Field(
        default="unknown",
        description="Name of the agent that was evaluated (alias for agent_id)",
    )
    task_type: str | None = pd.Field(
        default=None,
        description="Type of task performed (e.g., 'pr_review', 'code_change', 'content_gen')",
    )
    outcome: str = pd.Field(
        default="success",
        description="Overall outcome of the run (success, failure, error)",
    )
    tool_calls: list[ToolCallRecord] = pd.Field(
        default_factory=list,
        description="List of tool calls made during the run",
    )
    steps: list[StepRecord] = pd.Field(
        default_factory=list,
        description="List of reasoning steps recorded during the run",
    )
    messages: list[dict[str, Any]] = pd.Field(
        default_factory=list,
        description="List of messages exchanged during the run",
    )
    total_duration_ms: int | None = pd.Field(
        default=None,
        description="Total duration of the run in milliseconds",
    )
    token_usage: dict[str, int] = pd.Field(
        default_factory=dict,
        description="Token usage statistics (input, output, total)",
    )
    cost_usd: float | None = pd.Field(
        default=None,
        description="Total cost in USD for the run",
    )
    error_message: str | None = pd.Field(
        default=None,
        description="Error message if the run failed",
    )
    metadata: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Additional metadata about the run",
    )
    created_at: datetime | None = pd.Field(
        default=None,
        description="When the run was created",
    )
    completed_at: datetime | None = pd.Field(
        default=None,
        description="When the run completed",
    )
    overall_score: float = pd.Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall score from evaluation (0.0 to 1.0)",
    )
    metrics: dict[str, float] = pd.Field(
        default_factory=dict,
        description="Computed metrics (efficiency_score, quality_score, safety_score, etc.)",
    )
    trial_ids: list[str] = pd.Field(
        default_factory=list,
        description="IDs of trials included in this run artifact",
    )

    model_config = pd.ConfigDict(extra="allow")

    def is_successful(self) -> bool:
        """Check if the run was successful."""
        return self.outcome == "success"

    def get_tool_success_rate(self) -> float:
        """Calculate the tool call success rate."""
        if not self.tool_calls:
            return 0.0
        successful = sum(1 for tc in self.tool_calls if tc.outcome == "success")
        return successful / len(self.tool_calls)

    def get_total_tokens(self) -> int:
        """Get total tokens used."""
        return self.token_usage.get("total", 0)

    @classmethod
    def from_dawn_kestrel(cls, dk_artifact: Any) -> RunArtifact:
        """Convert a dawn-kestrel RunArtifact to ash-hawk RunArtifact.

        Args:
            dk_artifact: RunArtifact from dawn_kestrel.contracts.

        Returns:
            Ash Hawk RunArtifact with field mappings applied.
        """
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
