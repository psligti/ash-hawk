"""Tests for ReviewService and LessonService."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ash_hawk.contracts import CuratedLesson, ImprovementProposal, ReviewRequest
from ash_hawk.services.lesson_service import LessonService
from ash_hawk.services.review_service import ReviewService


@pytest.fixture
def review_request() -> ReviewRequest:
    return ReviewRequest(
        run_artifact_id="run-test-001",
        target_agent="test-agent",
        eval_suite=["efficiency"],
        review_mode="standard",
        persistence_mode="curate",
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
        rationale="Prevent timeout failures",
        expected_benefit="Improved reliability",
        risk_level="low",
        diff_payload={"timeout_seconds": 30},
        created_at=datetime.now(UTC),
    )


class TestReviewService:
    """Tests for ReviewService."""

    def test_init(self):
        service = ReviewService()
        assert service is not None


class TestLessonService:
    """Tests for LessonService."""

    def test_init(self):
        service = LessonService()
        assert service is not None

    def test_approve_proposal(self, improvement_proposal: ImprovementProposal):
        service = LessonService()
        lesson = service.approve_proposal(improvement_proposal)

        assert lesson is not None
        assert lesson.lesson_id == "lesson-prop-test-001"
        assert lesson.source_proposal_id == "prop-test-001"
        assert lesson.validation_status == "approved"

    def test_approve_proposal_with_custom_agents(self, improvement_proposal: ImprovementProposal):
        service = LessonService()
        lesson = service.approve_proposal(
            improvement_proposal,
            applies_to_agents=["agent-a", "agent-b"],
        )

        assert "agent-a" in lesson.applies_to_agents
        assert "agent-b" in lesson.applies_to_agents

    def test_get_lesson(self, improvement_proposal: ImprovementProposal):
        service = LessonService()
        created = service.approve_proposal(improvement_proposal)

        retrieved = service.get_lesson(created.lesson_id)
        assert retrieved is not None
        assert retrieved.lesson_id == created.lesson_id

    def test_get_lesson_unknown_returns_none(self):
        service = LessonService()
        retrieved = service.get_lesson("unknown-id")
        assert retrieved is None

    def test_get_lessons_for_agent(self, improvement_proposal: ImprovementProposal):
        service = LessonService()
        service.approve_proposal(improvement_proposal)

        lessons = service.get_lessons_for_agent("test-agent")
        assert len(lessons) == 1

    def test_list_lessons(self, improvement_proposal: ImprovementProposal):
        service = LessonService()
        service.approve_proposal(improvement_proposal)

        lessons = service.list_lessons()
        assert len(lessons) == 1

    def test_list_lessons_filters_by_status(self, improvement_proposal: ImprovementProposal):
        service = LessonService()
        service.approve_proposal(improvement_proposal)

        approved = service.list_lessons(status="approved")
        assert len(approved) == 1

        deprecated = service.list_lessons(status="deprecated")
        assert len(deprecated) == 0

    def test_deactivate_lesson(self, improvement_proposal: ImprovementProposal):
        service = LessonService()
        created = service.approve_proposal(improvement_proposal)

        deactivated = service.deactivate_lesson(created.lesson_id)
        assert deactivated is not None
        assert deactivated.validation_status == "rolled_back"

    def test_get_provenance(self, improvement_proposal: ImprovementProposal):
        service = LessonService()
        created = service.approve_proposal(improvement_proposal)

        provenance = service.get_provenance(created.lesson_id)
        assert provenance is not None
        assert provenance.lesson_id == created.lesson_id
        assert provenance.source_proposal_id == "prop-test-001"

    def test_get_lessons_from_run(self, improvement_proposal: ImprovementProposal):
        service = LessonService()
        service.approve_proposal(improvement_proposal)

        lesson_ids = service.get_lessons_from_run("run-test-001")
        assert len(lesson_ids) == 1

    def test_delete_lesson(self, improvement_proposal: ImprovementProposal):
        service = LessonService()
        created = service.approve_proposal(improvement_proposal)

        result = service.delete_lesson(created.lesson_id)
        assert result is True

        retrieved = service.get_lesson(created.lesson_id)
        assert retrieved is None

    def test_delete_unknown_returns_false(self):
        service = LessonService()
        result = service.delete_lesson("unknown-id")
        assert result is False
