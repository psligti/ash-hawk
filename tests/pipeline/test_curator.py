"""Tests for CuratorRole."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ash_hawk.contracts import CuratedLesson, ImprovementProposal
from ash_hawk.pipeline.curator import CuratorRole


@pytest.fixture
def curator() -> CuratorRole:
    return CuratorRole()


@pytest.fixture
def pending_proposal() -> ImprovementProposal:
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
        evidence_refs=["test"],
        status="pending",
        created_at=datetime.now(UTC),
        confidence=1.0,
    )


@pytest.fixture
def approved_proposal() -> ImprovementProposal:
    return ImprovementProposal(
        proposal_id="prop-approved-001",
        origin_run_id="run-test-001",
        origin_review_id="review-test-001",
        target_agent="test-agent",
        proposal_type="policy",
        title="Already approved",
        rationale="This was already approved",
        expected_benefit="None",
        risk_level="low",
        status="approved",
        created_at=datetime.now(UTC),
        confidence=1.0,
    )


@pytest.fixture
def rejected_proposal() -> ImprovementProposal:
    return ImprovementProposal(
        proposal_id="prop-rejected-001",
        origin_run_id="run-test-001",
        origin_review_id="review-test-001",
        target_agent="test-agent",
        proposal_type="policy",
        title="Already rejected",
        rationale="This was rejected",
        expected_benefit="None",
        risk_level="high",
        status="rejected",
        created_at=datetime.now(UTC),
        confidence=1.0,
    )


class TestCuratorRoleCurate:
    def test_curate_empty_proposals(self, curator: CuratorRole):
        lessons = curator.curate([])
        assert lessons == []

    def test_curate_auto_appro_pending_proposal(
        self, curator: CuratorRole, pending_proposal: ImprovementProposal
    ):
        lessons = curator.curate([pending_proposal], auto_appro=True)
        assert len(lessons) == 1
        assert lessons[0].source_proposal_id == pending_proposal.proposal_id

    def test_curate_no_auto_appro(
        self, curator: CuratorRole, pending_proposal: ImprovementProposal
    ):
        lessons = curator.curate([pending_proposal], auto_appro=False)
        assert lessons == []

    def test_curate_skips_non_pending_proposals(
        self,
        curator: CuratorRole,
        approved_proposal: ImprovementProposal,
        rejected_proposal: ImprovementProposal,
    ):
        lessons = curator.curate([approved_proposal, rejected_proposal], auto_appro=True)
        assert lessons == []

    def test_curate_creates_lesson_with_correct_fields(
        self, curator: CuratorRole, pending_proposal: ImprovementProposal
    ):
        lessons = curator.curate([pending_proposal], auto_appro=True)
        lesson = lessons[0]

        assert lesson.lesson_type == pending_proposal.proposal_type
        assert lesson.title == pending_proposal.title
        assert lesson.description == pending_proposal.rationale
        assert lesson.lesson_payload == pending_proposal.diff_payload
        assert lesson.validation_status == "approved"
        assert lesson.version == 1
        assert pending_proposal.target_agent in lesson.applies_to_agents

    def test_curate_multiple_proposals(
        self, curator: CuratorRole, pending_proposal: ImprovementProposal
    ):
        proposal2 = ImprovementProposal(
            proposal_id="prop-test-002",
            origin_run_id="run-test-001",
            origin_review_id="review-test-001",
            target_agent="test-agent",
            proposal_type="skill",
            title="Add new skill",
            rationale="Improve capability",
            expected_benefit="Better results",
            risk_level="medium",
            status="pending",
            created_at=datetime.now(UTC),
            evidence_refs=["test"],
            confidence=1.0,
        )
        lessons = curator.curate([pending_proposal, proposal2], auto_appro=True)
        assert len(lessons) == 2


class TestCuratorRoleReject:
    def test_reject_sets_status(self, curator: CuratorRole, pending_proposal: ImprovementProposal):
        rejected = curator.reject(pending_proposal, "Too risky")
        assert rejected.status == "rejected"
        assert rejected.rejection_reason == "Too risky"

    def test_reject_sets_reviewed_at(
        self, curator: CuratorRole, pending_proposal: ImprovementProposal
    ):
        rejected = curator.reject(pending_proposal, "Not needed")
        assert rejected.reviewed_at is not None


class TestCuratorRoleDefer:
    def test_defer_keeps_pending_status(
        self, curator: CuratorRole, pending_proposal: ImprovementProposal
    ):
        deferred = curator.defer(pending_proposal)
        assert deferred.status == "pending"

    def test_defer_sets_reviewed_at(
        self, curator: CuratorRole, pending_proposal: ImprovementProposal
    ):
        deferred = curator.defer(pending_proposal)
        assert deferred.reviewed_at is not None


class TestCuratorRoleLessonTypes:
    def test_curate_policy_proposal(self, curator: CuratorRole):
        proposal = ImprovementProposal(
            proposal_id="prop-policy-001",
            origin_run_id="run-test-001",
            target_agent="test-agent",
            proposal_type="policy",
            title="Policy update",
            rationale="Improve behavior",
            expected_benefit="Better compliance",
            risk_level="low",
            status="pending",
            created_at=datetime.now(UTC),
            evidence_refs=["test"],
            confidence=1.0,
        )
        lessons = curator.curate([proposal], auto_appro=True)
        assert lessons[0].lesson_type == "policy"

    def test_curate_skill_proposal(self, curator: CuratorRole):
        proposal = ImprovementProposal(
            proposal_id="prop-skill-001",
            origin_run_id="run-test-001",
            target_agent="test-agent",
            proposal_type="skill",
            title="New skill",
            rationale="Add capability",
            expected_benefit="New functionality",
            risk_level="medium",
            status="pending",
            created_at=datetime.now(UTC),
            evidence_refs=["test"],
            confidence=1.0,
        )
        lessons = curator.curate([proposal], auto_appro=True)
        assert lessons[0].lesson_type == "skill"

    def test_curate_tool_proposal(self, curator: CuratorRole):
        proposal = ImprovementProposal(
            proposal_id="prop-tool-001",
            origin_run_id="run-test-001",
            target_agent="test-agent",
            proposal_type="tool",
            title="Tool improvement",
            rationale="Better tools",
            expected_benefit="Efficiency",
            risk_level="medium",
            status="pending",
            created_at=datetime.now(UTC),
            evidence_refs=["test"],
            confidence=1.0,
        )
        lessons = curator.curate([proposal], auto_appro=True)
        assert lessons[0].lesson_type == "tool"

    def test_curate_harness_proposal(self, curator: CuratorRole):
        proposal = ImprovementProposal(
            proposal_id="prop-harness-001",
            origin_run_id="run-test-001",
            target_agent="test-agent",
            proposal_type="harness",
            title="Harness update",
            rationale="Improve harness",
            expected_benefit="Better testing",
            risk_level="low",
            status="pending",
            created_at=datetime.now(UTC),
            evidence_refs=["test"],
            confidence=1.0,
        )
        lessons = curator.curate([proposal], auto_appro=True)
        assert lessons[0].lesson_type == "harness"

    def test_curate_eval_proposal(self, curator: CuratorRole):
        proposal = ImprovementProposal(
            proposal_id="prop-eval-001",
            origin_run_id="run-test-001",
            target_agent="test-agent",
            proposal_type="eval",
            title="Eval improvement",
            rationale="Better evaluation",
            expected_benefit="More accurate",
            risk_level="low",
            status="pending",
            created_at=datetime.now(UTC),
            evidence_refs=["test"],
            confidence=1.0,
        )
        lessons = curator.curate([proposal], auto_appro=True)
        assert lessons[0].lesson_type == "eval"
