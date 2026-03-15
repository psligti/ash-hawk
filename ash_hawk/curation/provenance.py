"""Provenance tracker for tracing run to lesson lineage."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ash_hawk.contracts import CuratedLesson, ImprovementProposal


class ProvenanceRecord:
    """Tracks the provenance of a curated lesson."""

    def __init__(
        self,
        lesson_id: str,
        source_proposal_id: str,
        origin_run_id: str,
        origin_review_id: str | None = None,
    ) -> None:
        self.lesson_id = lesson_id
        self.source_proposal_id = source_proposal_id
        self.origin_run_id = origin_run_id
        self.origin_review_id = origin_review_id
        self.created_at = datetime.now(UTC)
        self.metadata: dict[str, Any] = {}


class ProvenanceTracker:
    """Tracks the full lineage from run artifact to curated lesson.

    Enables:
    - Tracing why a lesson exists
    - Finding all lessons from a specific run
    - Audit trail for behavioral changes
    """

    def __init__(self) -> None:
        self._records: dict[str, ProvenanceRecord] = {}
        self._by_run: dict[str, list[str]] = {}
        self._by_review: dict[str, list[str]] = {}

    def track(
        self,
        lesson: CuratedLesson,
        proposal: ImprovementProposal,
    ) -> ProvenanceRecord:
        record = ProvenanceRecord(
            lesson_id=lesson.lesson_id,
            source_proposal_id=proposal.proposal_id,
            origin_run_id=proposal.origin_run_id,
            origin_review_id=proposal.origin_review_id,
        )

        self._records[lesson.lesson_id] = record

        if proposal.origin_run_id not in self._by_run:
            self._by_run[proposal.origin_run_id] = []
        self._by_run[proposal.origin_run_id].append(lesson.lesson_id)

        if proposal.origin_review_id:
            if proposal.origin_review_id not in self._by_review:
                self._by_review[proposal.origin_review_id] = []
            self._by_review[proposal.origin_review_id].append(lesson.lesson_id)

        return record

    def get_lesson_provenance(self, lesson_id: str) -> ProvenanceRecord | None:
        return self._records.get(lesson_id)

    def get_lessons_from_run(self, run_id: str) -> list[str]:
        return self._by_run.get(run_id, []).copy()

    def get_lessons_from_review(self, review_id: str) -> list[str]:
        return self._by_review.get(review_id, []).copy()

    def get_lineage(self, lesson_id: str) -> dict[str, Any]:
        record = self._records.get(lesson_id)
        if not record:
            return {}

        return {
            "lesson_id": record.lesson_id,
            "source_proposal_id": record.source_proposal_id,
            "origin_run_id": record.origin_run_id,
            "origin_review_id": record.origin_review_id,
            "created_at": record.created_at.isoformat(),
        }

    def export_audit_trail(self) -> list[dict[str, Any]]:
        return [self.get_lineage(lid) for lid in self._records]
