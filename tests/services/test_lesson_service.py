"""Tests for LessonService."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ash_hawk.contracts import CuratedLesson, ImprovementProposal
from ash_hawk.services.lesson_service import LessonService


@pytest.fixture
def lesson_service() -> LessonService:
    return LessonService()


@pytest.fixture
def proposal() -> ImprovementProposal:
    return ImprovementProposal(
        proposal_id="prop-lesson-001",
        origin_run_id="run-lesson-001",
        origin_review_id="review-lesson-001",
        target_agent="test-agent",
        proposal_type="policy",
        title="Test proposal",
        rationale="For lesson service testing",
        expected_benefit="Better behavior",
        risk_level="low",
        diff_payload={"timeout_seconds": 30},
        status="pending",
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def existing_lesson() -> CuratedLesson:
    return CuratedLesson(
        lesson_id="lesson-existing-001",
        source_proposal_id="prop-existing-001",
        applies_to_agents=["test-agent"],
        lesson_type="policy",
        title="Existing lesson",
        description="Already exists",
        lesson_payload={"key": "value"},
        validation_status="approved",
        version=1,
        created_at=datetime.now(UTC),
    )


class TestLessonServiceInit:
    def test_init_creates_service(self):
        service = LessonService()
        assert service is not None

    def test_init_with_storage_path(self, tmp_path):
        service = LessonService(storage_path=tmp_path / "lessons")
        assert service is not None


class TestLessonServiceApproveProposal:
    def test_approve_proposal_returns_lesson(
        self, lesson_service: LessonService, proposal: ImprovementProposal
    ):
        lesson = lesson_service.approve_proposal(proposal)
        assert isinstance(lesson, CuratedLesson)

    def test_approve_proposal_sets_lesson_id(
        self, lesson_service: LessonService, proposal: ImprovementProposal
    ):
        lesson = lesson_service.approve_proposal(proposal)
        assert lesson.lesson_id == f"lesson-{proposal.proposal_id}"

    def test_approve_proposal_sets_source(
        self, lesson_service: LessonService, proposal: ImprovementProposal
    ):
        lesson = lesson_service.approve_proposal(proposal)
        assert lesson.source_proposal_id == proposal.proposal_id

    def test_approve_proposal_uses_target_agent_by_default(
        self, lesson_service: LessonService, proposal: ImprovementProposal
    ):
        lesson = lesson_service.approve_proposal(proposal)
        assert proposal.target_agent in lesson.applies_to_agents

    def test_approve_proposal_uses_custom_agents(
        self, lesson_service: LessonService, proposal: ImprovementProposal
    ):
        custom_agents = ["agent-1", "agent-2"]
        lesson = lesson_service.approve_proposal(proposal, applies_to_agents=custom_agents)
        assert lesson.applies_to_agents == custom_agents

    def test_approve_proposal_copies_fields(
        self, lesson_service: LessonService, proposal: ImprovementProposal
    ):
        lesson = lesson_service.approve_proposal(proposal)
        assert lesson.lesson_type == proposal.proposal_type
        assert lesson.title == proposal.title
        assert lesson.description == proposal.rationale
        assert lesson.lesson_payload == proposal.diff_payload

    def test_approve_proposal_sets_approved_status(
        self, lesson_service: LessonService, proposal: ImprovementProposal
    ):
        lesson = lesson_service.approve_proposal(proposal)
        assert lesson.validation_status == "approved"

    def test_approve_proposal_sets_version(
        self, lesson_service: LessonService, proposal: ImprovementProposal
    ):
        lesson = lesson_service.approve_proposal(proposal)
        assert lesson.version == 1


class TestLessonServiceGetLesson:
    def test_get_lesson_returns_approved_lesson(
        self, lesson_service: LessonService, proposal: ImprovementProposal
    ):
        approved = lesson_service.approve_proposal(proposal)
        retrieved = lesson_service.get_lesson(approved.lesson_id)
        assert retrieved is not None
        assert retrieved.lesson_id == approved.lesson_id

    def test_get_lesson_returns_none_for_unknown(self, lesson_service: LessonService):
        retrieved = lesson_service.get_lesson("unknown-lesson")
        assert retrieved is None


class TestLessonServiceGetLessonsForAgent:
    def test_get_lessons_for_agent_returns_matching(
        self, lesson_service: LessonService, proposal: ImprovementProposal
    ):
        lesson_service.approve_proposal(proposal)
        lessons = lesson_service.get_lessons_for_agent(proposal.target_agent)
        assert len(lessons) == 1

    def test_get_lessons_for_agent_returns_empty_for_unknown(self, lesson_service: LessonService):
        lessons = lesson_service.get_lessons_for_agent("unknown-agent")
        assert lessons == []

    def test_get_lessons_for_agent_multiple_lessons(
        self, lesson_service: LessonService, proposal: ImprovementProposal
    ):
        proposal2 = ImprovementProposal(
            proposal_id="prop-lesson-002",
            origin_run_id="run-lesson-001",
            target_agent="test-agent",
            proposal_type="skill",
            title="Second proposal",
            rationale="Another lesson",
            expected_benefit="More improvements",
            risk_level="medium",
            status="pending",
            created_at=datetime.now(UTC),
        )
        lesson_service.approve_proposal(proposal)
        lesson_service.approve_proposal(proposal2)

        lessons = lesson_service.get_lessons_for_agent("test-agent")
        assert len(lessons) == 2


class TestLessonServiceListLessons:
    def test_list_lessons_returns_all(
        self, lesson_service: LessonService, proposal: ImprovementProposal
    ):
        lesson_service.approve_proposal(proposal)
        lessons = lesson_service.list_lessons()
        assert len(lessons) == 1

    def test_list_lessons_filters_by_status(
        self, lesson_service: LessonService, proposal: ImprovementProposal
    ):
        lesson_service.approve_proposal(proposal)
        approved = lesson_service.list_lessons(status="approved")
        assert len(approved) == 1

        deprecated = lesson_service.list_lessons(status="deprecated")
        assert len(deprecated) == 0

    def test_list_lessons_filters_by_type(
        self, lesson_service: LessonService, proposal: ImprovementProposal
    ):
        lesson_service.approve_proposal(proposal)
        policies = lesson_service.list_lessons(lesson_type="policy")
        assert len(policies) == 1

        skills = lesson_service.list_lessons(lesson_type="skill")
        assert len(skills) == 0

    def test_list_lessons_empty(self, lesson_service: LessonService):
        lessons = lesson_service.list_lessons()
        assert lessons == []


class TestLessonServiceRollbackLesson:
    def test_rollback_lesson_returns_rolled_back(
        self, lesson_service: LessonService, proposal: ImprovementProposal
    ):
        lesson = lesson_service.approve_proposal(proposal)
        lesson_payload_v2 = {"timeout_seconds": 60}
        lesson_v2 = CuratedLesson(
            lesson_id=lesson.lesson_id,
            source_proposal_id=lesson.source_proposal_id,
            applies_to_agents=lesson.applies_to_agents,
            lesson_type=lesson.lesson_type,
            title=lesson.title,
            description="Updated description",
            lesson_payload=lesson_payload_v2,
            validation_status="approved",
            version=2,
            created_at=lesson.created_at,
        )
        lesson_service._store.store(lesson_v2)
        lesson_service._rollback.snapshot(lesson)
        lesson_service._rollback.snapshot(lesson_v2)

        rolled_back = lesson_service.rollback_lesson(lesson.lesson_id)
        assert rolled_back is not None

    def test_rollback_lesson_returns_none_for_no_history(self, lesson_service: LessonService):
        rolled_back = lesson_service.rollback_lesson("unknown-lesson")
        assert rolled_back is None


class TestLessonServiceDeactivateLesson:
    def test_deactivate_lesson_changes_status(
        self, lesson_service: LessonService, proposal: ImprovementProposal
    ):
        lesson = lesson_service.approve_proposal(proposal)
        deactivated = lesson_service.deactivate_lesson(lesson.lesson_id)
        assert deactivated is not None
        assert deactivated.validation_status == "rolled_back"

    def test_deactivate_lesson_returns_none_for_unknown(self, lesson_service: LessonService):
        deactivated = lesson_service.deactivate_lesson("unknown-lesson")
        assert deactivated is None


class TestLessonServiceProvenance:
    def test_get_provenance_returns_record(
        self, lesson_service: LessonService, proposal: ImprovementProposal
    ):
        lesson = lesson_service.approve_proposal(proposal)
        provenance = lesson_service.get_provenance(lesson.lesson_id)
        assert provenance is not None
        assert provenance.lesson_id == lesson.lesson_id
        assert provenance.source_proposal_id == proposal.proposal_id

    def test_get_provenance_returns_none_for_unknown(self, lesson_service: LessonService):
        provenance = lesson_service.get_provenance("unknown-lesson")
        assert provenance is None

    def test_get_lessons_from_run_returns_ids(
        self, lesson_service: LessonService, proposal: ImprovementProposal
    ):
        lesson_service.approve_proposal(proposal)
        lesson_ids = lesson_service.get_lessons_from_run(proposal.origin_run_id)
        assert len(lesson_ids) == 1

    def test_get_lessons_from_run_returns_empty_for_unknown(self, lesson_service: LessonService):
        lesson_ids = lesson_service.get_lessons_from_run("unknown-run")
        assert lesson_ids == []


class TestLessonServiceDeleteLesson:
    def test_delete_lesson_removes_lesson(
        self, lesson_service: LessonService, proposal: ImprovementProposal
    ):
        lesson = lesson_service.approve_proposal(proposal)
        result = lesson_service.delete_lesson(lesson.lesson_id)
        assert result is True
        assert lesson_service.get_lesson(lesson.lesson_id) is None

    def test_delete_lesson_returns_false_for_unknown(self, lesson_service: LessonService):
        result = lesson_service.delete_lesson("unknown-lesson")
        assert result is False


class TestLessonServiceIntegration:
    def test_full_lifecycle_approve_deactivate(
        self, lesson_service: LessonService, proposal: ImprovementProposal
    ):
        lesson = lesson_service.approve_proposal(proposal)
        assert lesson.validation_status == "approved"

        retrieved = lesson_service.get_lesson(lesson.lesson_id)
        assert retrieved is not None

        agent_lessons = lesson_service.get_lessons_for_agent(proposal.target_agent)
        assert len(agent_lessons) == 1

        deactivated = lesson_service.deactivate_lesson(lesson.lesson_id)
        assert deactivated is not None
        assert deactivated.validation_status == "rolled_back"

    def test_multiple_proposals_same_agent(
        self, lesson_service: LessonService, proposal: ImprovementProposal
    ):
        proposal2 = ImprovementProposal(
            proposal_id="prop-multi-002",
            origin_run_id="run-multi-001",
            target_agent="test-agent",
            proposal_type="skill",
            title="Second skill",
            rationale="Another improvement",
            expected_benefit="More capabilities",
            risk_level="medium",
            status="pending",
            created_at=datetime.now(UTC),
        )
        lesson1 = lesson_service.approve_proposal(proposal)
        lesson2 = lesson_service.approve_proposal(proposal2)

        assert lesson1.lesson_id != lesson2.lesson_id

        agent_lessons = lesson_service.get_lessons_for_agent("test-agent")
        assert len(agent_lessons) == 2

    def test_proposal_with_multiple_agents(
        self, lesson_service: LessonService, proposal: ImprovementProposal
    ):
        custom_agents = ["agent-alpha", "agent-beta", "agent-gamma"]
        lesson = lesson_service.approve_proposal(proposal, applies_to_agents=custom_agents)

        for agent in custom_agents:
            agent_lessons = lesson_service.get_lessons_for_agent(agent)
            assert any(stored.lesson_id == lesson.lesson_id for stored in agent_lessons)
