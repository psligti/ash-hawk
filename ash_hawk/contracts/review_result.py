"""Review result contract for evaluation outcomes."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import pydantic as pd


class ReviewFinding(pd.BaseModel):
    """Single finding from a review evaluation.

    Attributes:
        finding_id: Unique identifier for this finding.
        category: Category of finding (e.g., 'efficiency', 'quality', 'safety').
        severity: Severity level (info, warning, critical).
        title: Short title describing the finding.
        description: Detailed description of the finding.
        evidence_refs: References to evidence in the run artifact.
        recommendation: Suggested improvement action.
    """

    finding_id: str = pd.Field(
        description="Unique identifier for this finding",
    )
    category: str = pd.Field(
        description="Category of finding (e.g., 'efficiency', 'quality', 'safety')",
    )
    severity: Literal["info", "warning", "critical"] = pd.Field(
        description="Severity level",
    )
    title: str = pd.Field(
        description="Short title describing the finding",
    )
    description: str = pd.Field(
        description="Detailed description of the finding",
    )
    evidence_refs: list[str] = pd.Field(
        default_factory=list,
        description="References to evidence in the run artifact",
    )
    recommendation: str | None = pd.Field(
        default=None,
        description="Suggested improvement action",
    )

    model_config = pd.ConfigDict(extra="forbid")


class ReviewMetrics(pd.BaseModel):
    """Quantitative metrics from the review.

    Attributes:
        score: Overall score (0.0 to 1.0).
        efficiency_score: Tool call efficiency score.
        quality_score: Output quality score.
        safety_score: Policy compliance score.
        custom_metrics: Additional agent-specific metrics.
    """

    score: float = pd.Field(
        ge=0.0,
        le=1.0,
        description="Overall score (0.0 to 1.0)",
    )
    efficiency_score: float | None = pd.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Tool call efficiency score",
    )
    quality_score: float | None = pd.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Output quality score",
    )
    safety_score: float | None = pd.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Policy compliance score",
    )
    custom_metrics: dict[str, float] = pd.Field(
        default_factory=dict,
        description="Additional agent-specific metrics",
    )

    model_config = pd.ConfigDict(extra="forbid")


class ReviewResult(pd.BaseModel):
    """Result of evaluating a completed run.

    Contains findings, metrics, and any generated improvement proposals.
    Produced by Ash Hawk's review service.

    Attributes:
        review_id: Unique identifier for this review.
        request_id: ID of the ReviewRequest that triggered this review.
        run_artifact_id: ID of the run artifact that was reviewed.
        target_agent: Agent that was evaluated.
        status: Review status (completed, partial, failed).
        findings: List of findings from the evaluation.
        metrics: Quantitative metrics from the review.
        proposal_ids: IDs of any generated ImprovementProposals.
        comparison: Before/after comparison if baseline was provided.
        created_at: Timestamp when the review was completed.
        error_message: Error details if status is failed.
    """

    review_id: str = pd.Field(
        description="Unique identifier for this review",
    )
    request_id: str = pd.Field(
        description="ID of the ReviewRequest that triggered this review",
    )
    run_artifact_id: str = pd.Field(
        description="ID of the run artifact that was reviewed",
    )
    target_agent: str = pd.Field(
        description="Agent that was evaluated",
    )
    status: Literal["completed", "partial", "failed"] = pd.Field(
        description="Review status",
    )
    findings: list[ReviewFinding] = pd.Field(
        default_factory=list,
        description="List of findings from the evaluation",
    )
    metrics: ReviewMetrics = pd.Field(
        description="Quantitative metrics from the review",
    )
    proposal_ids: list[str] = pd.Field(
        default_factory=list,
        description="IDs of any generated ImprovementProposals",
    )
    comparison: dict[str, Any] | None = pd.Field(
        default=None,
        description="Before/after comparison if baseline was provided",
    )
    created_at: datetime = pd.Field(
        description="Timestamp when the review was completed",
    )
    error_message: str | None = pd.Field(
        default=None,
        description="Error details if status is failed",
    )

    @pd.computed_field(return_type=int)
    def critical_count(self) -> int:
        """Count of critical severity findings."""
        return sum(1 for f in self.findings if f.severity == "critical")

    @pd.computed_field(return_type=int)
    def warning_count(self) -> int:
        """Count of warning severity findings."""
        return sum(1 for f in self.findings if f.severity == "warning")

    model_config = pd.ConfigDict(extra="forbid")
