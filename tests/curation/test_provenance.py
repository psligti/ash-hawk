"""Tests for ProvenanceTracker."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ash_hawk.contracts import CuratedLesson, ImprovementProposal
from ash_hawk.curation.provenance import ProvenanceRecord, ProvenanceTracker


@pytest.fixture
def tracker() -> ProvenanceTracker:
    return ProvenanceTracker()


@pytest.fixture
def proposal() -> ImprovementProposal:
    return ImprovementProposal(
        proposal_id="prop-prov-001",
        origin_run_id="run-prov-001",
        origin_review_id="review-prov-001",
        target_agent="test-agent",
        proposal_type="policy",
        title="Test proposal",
        rationale="For provenance testing",
        expected_benefit="Test tracking",
        risk_level="low",
        status="pending",
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def lesson() -> CuratedLesson:
    return CuratedLesson(
        lesson_id="lesson-prov-001",
        source_proposal_id="prop-prov-001",
        applies_to_agents=["test-agent"],
        lesson_type="policy",
        title="Test lesson",
        description="For provenance testing",
        lesson_payload={},
        validation_status="approved",
        version=1,
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def lesson2() -> CuratedLesson:
    return CuratedLesson(
        lesson_id="lesson-prov-002",
        source_proposal_id="prop-prov-002",
        applies_to_agents=["test-agent"],
        lesson_type="skill",
        title="Second lesson",
        description="Another lesson",
        lesson_payload={},
        validation_status="approved",
        version=1,
        created_at=datetime.now(UTC),
    )


class TestProvenanceRecord:
    def test_init_sets_fields(self):
        record = ProvenanceRecord(
            lesson_id="lesson-001",
            source_proposal_id="prop-001",
            origin_run_id="run-001",
            origin_review_id="review-001",
        )
        assert record.lesson_id == "lesson-001"
        assert record.source_proposal_id == "prop-001"
        assert record.origin_run_id == "run-001"
        assert record.origin_review_id == "review-001"
        assert record.created_at is not None
        assert record.metadata == {}

    def test_init_without_review_id(self):
        record = ProvenanceRecord(
            lesson_id="lesson-002",
            source_proposal_id="prop-002",
            origin_run_id="run-002",
        )
        assert record.origin_review_id is None


class TestProvenanceTrackerTrack:
    def test_track_returns_record(
        self, tracker: ProvenanceTracker, lesson: CuratedLesson, proposal: ImprovementProposal
    ):
        record = tracker.track(lesson, proposal)
        assert isinstance(record, ProvenanceRecord)
        assert record.lesson_id == lesson.lesson_id
        assert record.source_proposal_id == proposal.proposal_id

    def test_track_stores_record(
        self, tracker: ProvenanceTracker, lesson: CuratedLesson, proposal: ImprovementProposal
    ):
        tracker.track(lesson, proposal)
        retrieved = tracker.get_lesson_provenance(lesson.lesson_id)
        assert retrieved is not None
        assert retrieved.lesson_id == lesson.lesson_id

    def test_track_indexes_by_run(
        self, tracker: ProvenanceTracker, lesson: CuratedLesson, proposal: ImprovementProposal
    ):
        tracker.track(lesson, proposal)
        run_lessons = tracker.get_lessons_from_run(proposal.origin_run_id)
        assert lesson.lesson_id in run_lessons

    def test_track_indexes_by_review(
        self, tracker: ProvenanceTracker, lesson: CuratedLesson, proposal: ImprovementProposal
    ):
        tracker.track(lesson, proposal)
        review_lessons = tracker.get_lessons_from_review(proposal.origin_review_id)
        assert lesson.lesson_id in review_lessons

    def test_track_multiple_lessons(
        self,
        tracker: ProvenanceTracker,
        lesson: CuratedLesson,
        lesson2: CuratedLesson,
        proposal: ImprovementProposal,
    ):
        tracker.track(lesson, proposal)
        proposal2 = ImprovementProposal(
            proposal_id="prop-prov-002",
            origin_run_id="run-prov-001",
            origin_review_id="review-prov-001",
            target_agent="test-agent",
            proposal_type="skill",
            title="Second proposal",
            rationale="Another",
            expected_benefit="Test",
            risk_level="low",
            status="pending",
            created_at=datetime.now(UTC),
        )
        tracker.track(lesson2, proposal2)

        run_lessons = tracker.get_lessons_from_run("run-prov-001")
        assert len(run_lessons) == 2


class TestProvenanceTrackerGetLessonProvenance:
    def test_get_lesson_provenance_returns_record(
        self, tracker: ProvenanceTracker, lesson: CuratedLesson, proposal: ImprovementProposal
    ):
        tracker.track(lesson, proposal)
        record = tracker.get_lesson_provenance(lesson.lesson_id)
        assert record is not None
        assert record.lesson_id == lesson.lesson_id

    def test_get_lesson_provenance_returns_none_for_unknown(self, tracker: ProvenanceTracker):
        record = tracker.get_lesson_provenance("unknown-lesson")
        assert record is None


class TestProvenanceTrackerGetLessonsFromRun:
    def test_get_lessons_from_run_returns_ids(
        self, tracker: ProvenanceTracker, lesson: CuratedLesson, proposal: ImprovementProposal
    ):
        tracker.track(lesson, proposal)
        lesson_ids = tracker.get_lessons_from_run(proposal.origin_run_id)
        assert lesson.lesson_id in lesson_ids

    def test_get_lessons_from_run_returns_empty_for_unknown(self, tracker: ProvenanceTracker):
        lesson_ids = tracker.get_lessons_from_run("unknown-run")
        assert lesson_ids == []

    def test_get_lessons_from_run_returns_copy(
        self, tracker: ProvenanceTracker, lesson: CuratedLesson, proposal: ImprovementProposal
    ):
        tracker.track(lesson, proposal)
        ids1 = tracker.get_lessons_from_run(proposal.origin_run_id)
        ids2 = tracker.get_lessons_from_run(proposal.origin_run_id)
        assert ids1 is not ids2


class TestProvenanceTrackerGetLessonsFromReview:
    def test_get_lessons_from_review_returns_ids(
        self, tracker: ProvenanceTracker, lesson: CuratedLesson, proposal: ImprovementProposal
    ):
        tracker.track(lesson, proposal)
        lesson_ids = tracker.get_lessons_from_review(proposal.origin_review_id)
        assert lesson.lesson_id in lesson_ids

    def test_get_lessons_from_review_returns_empty_for_none(
        self, tracker: ProvenanceTracker, lesson: CuratedLesson
    ):
        proposal_no_review = ImprovementProposal(
            proposal_id="prop-no-review",
            origin_run_id="run-001",
            origin_review_id=None,
            target_agent="test-agent",
            proposal_type="policy",
            title="No review",
            rationale="Test",
            expected_benefit="Test",
            risk_level="low",
            status="pending",
            created_at=datetime.now(UTC),
        )
        tracker.track(lesson, proposal_no_review)
        lesson_ids = tracker.get_lessons_from_review("unknown-review")
        assert lesson_ids == []


class TestProvenanceTrackerGetLineage:
    def test_get_lineage_returns_dict(
        self, tracker: ProvenanceTracker, lesson: CuratedLesson, proposal: ImprovementProposal
    ):
        tracker.track(lesson, proposal)
        lineage = tracker.get_lineage(lesson.lesson_id)
        assert isinstance(lineage, dict)
        assert lineage["lesson_id"] == lesson.lesson_id
        assert lineage["source_proposal_id"] == proposal.proposal_id
        assert lineage["origin_run_id"] == proposal.origin_run_id
        assert lineage["origin_review_id"] == proposal.origin_review_id
        assert "created_at" in lineage

    def test_get_lineage_returns_empty_for_unknown(self, tracker: ProvenanceTracker):
        lineage = tracker.get_lineage("unknown-lesson")
        assert lineage == {}


class TestProvenanceTrackerExportAuditTrail:
    def test_export_audit_trail_returns_list(
        self, tracker: ProvenanceTracker, lesson: CuratedLesson, proposal: ImprovementProposal
    ):
        tracker.track(lesson, proposal)
        trail = tracker.export_audit_trail()
        assert isinstance(trail, list)
        assert len(trail) == 1
        assert trail[0]["lesson_id"] == lesson.lesson_id

    def test_export_audit_trail_empty(self, tracker: ProvenanceTracker):
        trail = tracker.export_audit_trail()
        assert trail == []

    def test_export_audit_trail_multiple(
        self,
        tracker: ProvenanceTracker,
        lesson: CuratedLesson,
        lesson2: CuratedLesson,
        proposal: ImprovementProposal,
    ):
        tracker.track(lesson, proposal)
        proposal2 = ImprovementProposal(
            proposal_id="prop-prov-002",
            origin_run_id="run-prov-002",
            origin_review_id="review-prov-002",
            target_agent="test-agent",
            proposal_type="skill",
            title="Second",
            rationale="Test",
            expected_benefit="Test",
            risk_level="low",
            status="pending",
            created_at=datetime.now(UTC),
        )
        tracker.track(lesson2, proposal2)

        trail = tracker.export_audit_trail()
        assert len(trail) == 2
        lesson_ids = {t["lesson_id"] for t in trail}
        assert lesson.lesson_id in lesson_ids
        assert lesson2.lesson_id in lesson_ids
