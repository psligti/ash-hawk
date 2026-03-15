"""Fixtures for pipeline tests.

Provides mock classes for dawn_kestrel.contracts.run_artifact.RunArtifact
to avoid external dependencies during testing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pydantic as pd
import pytest

from ash_hawk.contracts import (
    CuratedLesson,
    ImprovementProposal,
    ReviewFinding,
    ReviewMetrics,
    ReviewRequest,
    ReviewResult,
)
from ash_hawk.pipeline.types import PipelineContext, PipelineRole


class MockToolCall(pd.BaseModel):
    """Mock tool call for testing."""

    tool_name: str = pd.Field(default="read")
    outcome: str = pd.Field(default="success")
    duration_ms: int | None = pd.Field(default=100)
    error_message: str | None = pd.Field(default=None)

    model_config = pd.ConfigDict(extra="allow")


class MockRunArtifact(pd.BaseModel):
    """Mock run artifact for testing.

    Mimics dawn_kestrel.contracts.run_artifact.RunArtifact structure.
    """

    run_id: str = pd.Field(default_factory=lambda: f"run-{uuid4().hex[:8]}")
    outcome: str = pd.Field(default="success")
    tool_calls: list[MockToolCall] = pd.Field(default_factory=list)
    steps: list[dict[str, Any]] = pd.Field(default_factory=list)
    messages: list[dict[str, Any]] = pd.Field(default_factory=list)

    model_config = pd.ConfigDict(extra="allow")


@pytest.fixture
def mock_run_artifact() -> MockRunArtifact:
    return MockRunArtifact(
        run_id="run-test-001",
        outcome="success",
        tool_calls=[
            MockToolCall(tool_name="read", outcome="success", duration_ms=50),
            MockToolCall(tool_name="write", outcome="success", duration_ms=100),
        ],
    )


@pytest.fixture
def mock_failed_run_artifact() -> MockRunArtifact:
    return MockRunArtifact(
        run_id="run-test-failed-001",
        outcome="failure",
        tool_calls=[
            MockToolCall(tool_name="read", outcome="success", duration_ms=50),
            MockToolCall(
                tool_name="bash",
                outcome="failure",
                duration_ms=5000,
                error_message="Command timed out after 5000ms",
            ),
        ],
    )


@pytest.fixture
def review_request() -> ReviewRequest:
    return ReviewRequest(
        run_artifact_id="run-test-001",
        target_agent="test-agent",
        eval_suite=["efficiency", "quality"],
        review_mode="standard",
        persistence_mode="curate",
    )


@pytest.fixture
def pipeline_context() -> PipelineContext:
    return PipelineContext(
        run_artifact_id="run-test-001",
        review_request_id="review-test-001",
        role=PipelineRole.ANALYST,
        target_agent="test-agent",
    )


@pytest.fixture
def improvement_proposal() -> ImprovementProposal:
    return ImprovementProposal(
        proposal_id="prop-test-001",
        origin_run_id="run-test-001",
        origin_review_id="review-test-001",
        target_agent="test-agent",
        proposal_type="policy",
        title="Add timeout handling",
        rationale="Prevent timeout failures in tool calls",
        expected_benefit="Improved reliability",
        risk_level="low",
        diff_payload={"timeout_seconds": 30},
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def curated_lesson() -> CuratedLesson:
    return CuratedLesson(
        lesson_id="lesson-test-001",
        source_proposal_id="prop-test-001",
        applies_to_agents=["test-agent"],
        lesson_type="policy",
        title="Add timeout handling",
        description="Prevent timeout failures in tool calls",
        lesson_payload={"timeout_seconds": 30},
        validation_status="approved",
        version=1,
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def review_finding() -> ReviewFinding:
    return ReviewFinding(
        finding_id="finding-test-001",
        category="reliability",
        severity="warning",
        title="Tool call timed out",
        description="Tool bash timed out after 5000ms",
        evidence_refs=["tool_calls.1"],
        recommendation="Add timeout handling",
    )


@pytest.fixture
def review_metrics() -> ReviewMetrics:
    return ReviewMetrics(
        score=0.75,
        efficiency_score=0.8,
        quality_score=0.9,
        safety_score=1.0,
    )


@pytest.fixture
def review_result(review_metrics: ReviewMetrics) -> ReviewResult:
    return ReviewResult(
        review_id="review-test-001",
        request_id="run-test-001",
        run_artifact_id="run-test-001",
        target_agent="test-agent",
        status="completed",
        findings=[],
        metrics=review_metrics,
        created_at=datetime.now(UTC),
    )
