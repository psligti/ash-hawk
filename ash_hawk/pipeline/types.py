"""Pipeline types for improvement workflow.

Defines the core types used across the improvement pipeline:
- PipelineContext: Execution context for a pipeline run
- PipelineRole: Available roles in the pipeline
- PipelineStepResult: Result of a single step execution
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

import pydantic as pd


class PipelineRole(StrEnum):
    """Roles in the improvement pipeline.

    Roles run in sequence to transform run artifacts into curated lessons:
    1. COMPETITOR: Re-attempts or replays the run (optional)
    2. TRANSLATOR: Converts raw competitor output into validated strategy
    3. ANALYST: Analyzes failures, generates findings
    4. COACH: Generates policy/playbook proposals
    5. ARCHITECT: Generates harness/tool proposals
    6. CURATOR: Approves/rejects proposals into lessons
    """

    COMPETITOR = "competitor"
    TRANSLATOR = "translator"
    ANALYST = "analyst"
    COACH = "coach"
    ARCHITECT = "architect"
    CURATOR = "curator"


class PipelineContext(pd.BaseModel):
    """Execution context for a pipeline run.

    Carries state through the pipeline roles and tracks progress.

    Attributes:
        run_artifact_id: ID of the run artifact being analyzed.
        review_request_id: ID of the review request that started this pipeline.
        role: Current active role in the pipeline.
        target_agent: Agent being evaluated.
        experiment_id: Optional experiment ID for parallel trial isolation.
        inputs: Input data for the current role.
        outputs: Accumulated outputs from completed roles.
        metadata: Additional context metadata.
    """

    run_artifact_id: str = pd.Field(
        description="ID of the run artifact being analyzed",
    )
    review_request_id: str = pd.Field(
        description="ID of the review request that started this pipeline",
    )
    role: PipelineRole = pd.Field(
        description="Current active role in the pipeline",
    )
    target_agent: str = pd.Field(
        description="Agent being evaluated",
    )
    experiment_id: str | None = pd.Field(
        default=None,
        description="Optional experiment ID for parallel trial isolation",
    )
    inputs: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Input data for the current role",
    )
    outputs: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Accumulated outputs from completed roles",
    )
    metadata: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Additional context metadata",
    )

    model_config = pd.ConfigDict(extra="forbid")

    def advance_to(self, next_role: PipelineRole) -> PipelineContext:
        """Create a new context for the next role, preserving outputs."""
        return PipelineContext(
            run_artifact_id=self.run_artifact_id,
            review_request_id=self.review_request_id,
            role=next_role,
            target_agent=self.target_agent,
            experiment_id=self.experiment_id,
            inputs={},
            outputs=self.outputs.copy(),
            metadata=self.metadata.copy(),
        )


class PipelineStepResult(pd.BaseModel):
    """Result of executing a single pipeline step.

    Tracks execution status, timing, and any outputs or errors.

    Attributes:
        step_id: Unique identifier for this step.
        role: Role that executed this step.
        status: Execution status (pending, running, completed, failed, skipped).
        started_at: When execution started.
        completed_at: When execution completed.
        error: Error message if status is failed.
        outputs: Outputs from this step.
    """

    step_id: str = pd.Field(
        description="Unique identifier for this step",
    )
    role: PipelineRole = pd.Field(
        description="Role that executed this step",
    )
    status: str = pd.Field(
        default="pending",
        description="Execution status",
    )
    started_at: datetime | None = pd.Field(
        default=None,
        description="When execution started",
    )
    completed_at: datetime | None = pd.Field(
        default=None,
        description="When execution completed",
    )
    error: str | None = pd.Field(
        default=None,
        description="Error message if status is failed",
    )
    outputs: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Outputs from this step",
    )

    model_config = pd.ConfigDict(extra="forbid")


class PipelineResult(pd.BaseModel):
    """Result of running the complete improvement pipeline.

    Captures all outputs from the pipeline execution for thread-safe
    access without relying on orchestrator instance state.

    Attributes:
        review_request_id: ID of the review request.
        run_artifact_id: ID of the run artifact analyzed.
        target_agent: Agent that was evaluated.
        experiment_id: Optional experiment ID for isolation.
        steps: Results from each pipeline role.
        proposals: Generated improvement proposals.
        lessons: Curated lessons from the pipeline.
        comparison: Optional comparison result if baseline was provided.
    """

    review_request_id: str = pd.Field(
        description="ID of the review request",
    )
    run_artifact_id: str = pd.Field(
        description="ID of the run artifact analyzed",
    )
    target_agent: str = pd.Field(
        description="Agent that was evaluated",
    )
    experiment_id: str | None = pd.Field(
        default=None,
        description="Optional experiment ID for isolation",
    )
    steps: dict[str, PipelineStepResult] = pd.Field(
        default_factory=dict,
        description="Results from each pipeline role keyed by role value",
    )
    proposals: list[Any] = pd.Field(
        default_factory=list,
        description="Generated improvement proposals",
    )
    lessons: list[Any] = pd.Field(
        default_factory=list,
        description="Curated lessons from the pipeline",
    )
    comparison: dict[str, Any] | None = pd.Field(
        default=None,
        description="Optional comparison result if baseline was provided",
    )

    model_config = pd.ConfigDict(extra="forbid")

    def get_step(self, role: PipelineRole) -> PipelineStepResult | None:
        return self.steps.get(role.value)
