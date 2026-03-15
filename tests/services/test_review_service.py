"""Tests for ReviewService."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ash_hawk.contracts import ReviewFinding, ReviewMetrics, ReviewRequest, ReviewResult
from ash_hawk.services.review_service import ReviewService

from tests.pipeline.conftest import MockRunArtifact


@pytest.fixture
def review_service() -> ReviewService:
    return ReviewService()


@pytest.fixture
def review_request() -> ReviewRequest:
    return ReviewRequest(
        run_artifact_id="run-review-001",
        target_agent="test-agent",
        eval_suite=["efficiency"],
        review_mode="standard",
        persistence_mode="curate",
    )


class TestReviewServiceInit:
    def test_init_creates_service(self):
        service = ReviewService()
        assert service is not None


class TestReviewServiceReview:
    def test_review_returns_review_result(
        self, review_service: ReviewService, review_request: ReviewRequest
    ):
        artifact = MockRunArtifact(run_id="run-review-001")
        result = review_service.review(review_request, artifact)
        assert isinstance(result, ReviewResult)

    def test_review_sets_review_id(
        self, review_service: ReviewService, review_request: ReviewRequest
    ):
        artifact = MockRunArtifact(run_id="run-review-001")
        result = review_service.review(review_request, artifact)
        assert result.review_id.startswith("review-")

    def test_review_sets_run_artifact_id(
        self, review_service: ReviewService, review_request: ReviewRequest
    ):
        artifact = MockRunArtifact(run_id="run-review-001")
        result = review_service.review(review_request, artifact)
        assert result.run_artifact_id == "run-review-001"

    def test_review_sets_target_agent(
        self, review_service: ReviewService, review_request: ReviewRequest
    ):
        artifact = MockRunArtifact(run_id="run-review-001")
        result = review_service.review(review_request, artifact)
        assert result.target_agent == "test-agent"

    def test_review_sets_status_completed_on_success(
        self, review_service: ReviewService, review_request: ReviewRequest
    ):
        artifact = MockRunArtifact(
            run_id="run-review-001",
            outcome="success",
            tool_calls=[],
        )
        result = review_service.review(review_request, artifact)
        assert result.status == "completed"

    def test_review_includes_findings(
        self, review_service: ReviewService, review_request: ReviewRequest
    ):
        artifact = MockRunArtifact(run_id="run-review-001")
        result = review_service.review(review_request, artifact)
        assert isinstance(result.findings, list)

    def test_review_includes_metrics(
        self, review_service: ReviewService, review_request: ReviewRequest
    ):
        artifact = MockRunArtifact(run_id="run-review-001")
        result = review_service.review(review_request, artifact)
        assert isinstance(result.metrics, ReviewMetrics)

    def test_review_sets_created_at(
        self, review_service: ReviewService, review_request: ReviewRequest
    ):
        artifact = MockRunArtifact(run_id="run-review-001")
        result = review_service.review(review_request, artifact)
        assert result.created_at is not None


class TestReviewServiceWithFailedRun:
    def test_review_processes_failure(
        self, review_service: ReviewService, review_request: ReviewRequest
    ):
        from tests.pipeline.conftest import MockToolCall

        artifact = MockRunArtifact(
            run_id="run-review-001",
            outcome="failure",
            tool_calls=[
                MockToolCall(
                    tool_name="bash",
                    outcome="failure",
                    error_message="Command failed",
                ),
            ],
        )
        result = review_service.review(review_request, artifact)
        assert result.status == "completed"
        assert len(result.findings) > 0

    def test_review_failure_generates_findings(
        self, review_service: ReviewService, review_request: ReviewRequest
    ):
        from tests.pipeline.conftest import MockToolCall

        artifact = MockRunArtifact(
            run_id="run-review-001",
            outcome="failure",
            tool_calls=[
                MockToolCall(
                    tool_name="bash",
                    outcome="failure",
                    error_message="Timeout exceeded",
                ),
            ],
        )
        result = review_service.review(review_request, artifact)
        critical_findings = [f for f in result.findings if f.severity == "critical"]
        assert len(critical_findings) > 0


class TestReviewServiceResultStructure:
    def test_review_result_has_proposal_ids(
        self, review_service: ReviewService, review_request: ReviewRequest
    ):
        artifact = MockRunArtifact(
            run_id="run-review-001",
            outcome="failure",
            tool_calls=[],
        )
        result = review_service.review(review_request, artifact)
        assert isinstance(result.proposal_ids, list)

    def test_review_result_critical_count(
        self, review_service: ReviewService, review_request: ReviewRequest
    ):
        artifact = MockRunArtifact(
            run_id="run-review-001",
            outcome="failure",
            tool_calls=[],
        )
        result = review_service.review(review_request, artifact)
        assert isinstance(result.critical_count, int)

    def test_review_result_warning_count(
        self, review_service: ReviewService, review_request: ReviewRequest
    ):
        artifact = MockRunArtifact(run_id="run-review-001")
        result = review_service.review(review_request, artifact)
        assert isinstance(result.warning_count, int)


class TestReviewServiceEdgeCases:
    def test_review_empty_tool_calls(
        self, review_service: ReviewService, review_request: ReviewRequest
    ):
        artifact = MockRunArtifact(
            run_id="run-review-001",
            outcome="success",
            tool_calls=[],
        )
        result = review_service.review(review_request, artifact)
        assert result.status == "completed"

    def test_review_all_successful_tool_calls(
        self, review_service: ReviewService, review_request: ReviewRequest
    ):
        from tests.pipeline.conftest import MockToolCall

        artifact = MockRunArtifact(
            run_id="run-review-001",
            outcome="success",
            tool_calls=[
                MockToolCall(tool_name="read", outcome="success"),
                MockToolCall(tool_name="write", outcome="success"),
                MockToolCall(tool_name="bash", outcome="success"),
            ],
        )
        result = review_service.review(review_request, artifact)
        assert result.metrics.score == 1.0

    def test_review_mixed_tool_call_outcomes(
        self, review_service: ReviewService, review_request: ReviewRequest
    ):
        from tests.pipeline.conftest import MockToolCall

        artifact = MockRunArtifact(
            run_id="run-review-001",
            outcome="success",
            tool_calls=[
                MockToolCall(tool_name="read", outcome="success"),
                MockToolCall(tool_name="bash", outcome="failure", error_message="Error"),
            ],
        )
        result = review_service.review(review_request, artifact)
        assert result.metrics.score == 0.5

    def test_review_with_different_review_modes(self, review_service: ReviewService):
        for mode in ["quick", "standard", "deep"]:
            request = ReviewRequest(
                run_artifact_id="run-review-001",
                target_agent="test-agent",
                review_mode=mode,
            )
            artifact = MockRunArtifact(run_id="run-review-001")
            result = review_service.review(request, artifact)
            assert result.status == "completed"

    def test_review_with_different_persistence_modes(self, review_service: ReviewService):
        for mode in ["none", "propose", "curate"]:
            request = ReviewRequest(
                run_artifact_id="run-review-001",
                target_agent="test-agent",
                persistence_mode=mode,
            )
            artifact = MockRunArtifact(run_id="run-review-001")
            result = review_service.review(request, artifact)
            assert result.status == "completed"
