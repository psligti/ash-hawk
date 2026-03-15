"""Lesson service for managing curated lessons."""

from __future__ import annotations

from pathlib import Path

from ash_hawk.contracts import CuratedLesson, ImprovementProposal
from ash_hawk.curation.provenance import ProvenanceRecord
from ash_hawk.curation.rollback import RollbackManager
from ash_hawk.curation.store import LessonStore
from ash_hawk.curation.provenance import ProvenanceTracker


class LessonService:
    """Service for managing the lifecycle of curated lessons.

    Provides a unified interface for storing, retrieving, and
    managing lessons with provenance tracking and rollback support.
    """

    def __init__(self, storage_path: Path | None = None) -> None:
        self._store = LessonStore(storage_path)
        self._provenance = ProvenanceTracker()
        self._rollback = RollbackManager()

    def approve_proposal(
        self,
        proposal: ImprovementProposal,
        applies_to_agents: list[str] | None = None,
    ) -> CuratedLesson:
        lesson = CuratedLesson(
            lesson_id=f"lesson-{proposal.proposal_id}",
            source_proposal_id=proposal.proposal_id,
            applies_to_agents=applies_to_agents or [proposal.target_agent],
            lesson_type=proposal.proposal_type,
            title=proposal.title,
            description=proposal.rationale,
            lesson_payload=proposal.diff_payload,
            validation_status="approved",
            version=1,
            created_at=proposal.created_at,
        )

        self._store.store(lesson)
        self._provenance.track(lesson, proposal)
        self._rollback.snapshot(lesson)

        return lesson

    def get_lesson(self, lesson_id: str) -> CuratedLesson | None:
        return self._store.get(lesson_id)

    def get_lessons_for_agent(self, agent_id: str) -> list[CuratedLesson]:
        return self._store.get_for_agent(agent_id)

    def list_lessons(
        self,
        status: str | None = None,
        lesson_type: str | None = None,
    ) -> list[CuratedLesson]:
        return self._store.list_all(status=status, lesson_type=lesson_type)

    def rollback_lesson(
        self,
        lesson_id: str,
        target_version: int | None = None,
    ) -> CuratedLesson | None:
        rolled_back = self._rollback.rollback(lesson_id, target_version)
        if rolled_back:
            self._store.store(rolled_back)
        return rolled_back

    def deactivate_lesson(self, lesson_id: str) -> CuratedLesson | None:
        return self._store.update_status(lesson_id, "rolled_back")

    def get_provenance(self, lesson_id: str) -> ProvenanceRecord | None:
        return self._provenance.get_lesson_provenance(lesson_id)

    def get_lessons_from_run(self, run_id: str) -> list[str]:
        return self._provenance.get_lessons_from_run(run_id)

    def delete_lesson(self, lesson_id: str) -> bool:
        return self._store.delete(lesson_id)
