"""Async lesson service with persistent storage."""

from __future__ import annotations

from pathlib import Path

from ash_hawk.contracts import CuratedLesson, ImprovementProposal
from ash_hawk.curation.persistent_store import PersistentLessonStore
from ash_hawk.curation.provenance import ProvenanceRecord, ProvenanceTracker


class AsyncLessonService:
    """Async service for managing curated lessons with persistent storage.

    Provides the same interface as LessonService but with async operations
    backed by SQLite for durability and concurrency safety.
    """

    def __init__(self, storage_path: Path | str | None = None) -> None:
        self._store = PersistentLessonStore(storage_path)
        self._provenance = ProvenanceTracker()

    async def approve_proposal(
        self,
        proposal: ImprovementProposal,
        applies_to_agents: list[str] | None = None,
        experiment_id: str | None = None,
        strategy: str | None = None,
        sub_strategies: list[str] | None = None,
    ) -> CuratedLesson:
        lesson_payload = proposal.diff_payload.copy()
        if experiment_id:
            lesson_payload["experiment_id"] = experiment_id
        if strategy:
            lesson_payload["strategy"] = strategy
        if sub_strategies:
            lesson_payload["sub_strategies"] = sub_strategies

        lesson = CuratedLesson(
            lesson_id=f"lesson-{proposal.proposal_id}",
            source_proposal_id=proposal.proposal_id,
            applies_to_agents=applies_to_agents or [proposal.target_agent],
            lesson_type=proposal.proposal_type,
            title=proposal.title,
            description=proposal.rationale,
            lesson_payload=lesson_payload,
            validation_status="approved",
            version=1,
            created_at=proposal.created_at,
        )

        await self._store.store(lesson)
        self._provenance.track(lesson, proposal)

        return lesson

    async def get_lesson(self, lesson_id: str) -> CuratedLesson | None:
        return await self._store.get(lesson_id)

    async def get_lessons_for_agent(
        self,
        agent_id: str,
        experiment_id: str | None = None,
    ) -> list[CuratedLesson]:
        return await self._store.get_for_agent(agent_id, experiment_id)

    async def list_lessons(
        self,
        status: str | None = None,
        lesson_type: str | None = None,
        experiment_id: str | None = None,
        strategy: str | None = None,
    ) -> list[CuratedLesson]:
        return await self._store.list_all(
            status=status,
            lesson_type=lesson_type,
            experiment_id=experiment_id,
            strategy=strategy,
        )

    async def rollback_lesson(
        self,
        lesson_id: str,
        target_version: int | None = None,
    ) -> CuratedLesson | None:
        return await self._store.rollback(lesson_id, target_version)

    async def deactivate_lesson(self, lesson_id: str) -> CuratedLesson | None:
        return await self._store.update_status(lesson_id, "rolled_back")

    async def deprecate_lesson(self, lesson_id: str) -> CuratedLesson | None:
        return await self._store.update_status(lesson_id, "deprecated")

    def get_provenance(self, lesson_id: str) -> ProvenanceRecord | None:
        return self._provenance.get_lesson_provenance(lesson_id)

    def get_lessons_from_run(self, run_id: str) -> list[str]:
        return self._provenance.get_lessons_from_run(run_id)

    async def delete_lesson(self, lesson_id: str) -> bool:
        return await self._store.delete(lesson_id)

    async def get_history(self, lesson_id: str) -> list[CuratedLesson]:
        return await self._store.get_history(lesson_id)

    async def close(self) -> None:
        await self._store.close()
