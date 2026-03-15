"""Lesson store for persisting curated lessons."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from ash_hawk.contracts import CuratedLesson


class LessonStore:
    """Persists and retrieves curated lessons.

    Lessons are stored with versioning and metadata for provenance tracking.
    """

    def __init__(self, storage_path: Path | None = None) -> None:
        self._storage_path = storage_path or Path(".ash-hawk/lessons")
        self._lessons: dict[str, CuratedLesson] = {}
        self._by_agent: dict[str, list[str]] = {}
        self._by_proposal: dict[str, str] = {}

    def store(self, lesson: CuratedLesson) -> str:
        lesson_id = lesson.lesson_id
        self._lessons[lesson_id] = lesson
        self._by_proposal[lesson.source_proposal_id] = lesson_id

        for agent in lesson.applies_to_agents:
            if agent not in self._by_agent:
                self._by_agent[agent] = []
            self._by_agent[agent].append(lesson_id)

        return lesson_id

    def get(self, lesson_id: str) -> CuratedLesson | None:
        return self._lessons.get(lesson_id)

    def get_by_proposal(self, proposal_id: str) -> CuratedLesson | None:
        lesson_id = self._by_proposal.get(proposal_id)
        if lesson_id:
            return self._lessons.get(lesson_id)
        return None

    def get_for_agent(self, agent_id: str) -> list[CuratedLesson]:
        lesson_ids = self._by_agent.get(agent_id, [])
        lessons = []
        for lid in lesson_ids:
            lesson = self._lessons.get(lid)
            if lesson and lesson.validation_status == "approved":
                lessons.append(lesson)
        return lessons

    def list_all(
        self,
        status: str | None = None,
        lesson_type: str | None = None,
    ) -> list[CuratedLesson]:
        lessons = list(self._lessons.values())
        if status:
            lessons = [lesson for lesson in lessons if lesson.validation_status == status]
        if lesson_type:
            lessons = [lesson for lesson in lessons if lesson.lesson_type == lesson_type]
        return lessons

    def update_status(
        self,
        lesson_id: str,
        new_status: Literal["approved", "deprecated", "rolled_back"],
    ) -> CuratedLesson | None:
        lesson = self._lessons.get(lesson_id)
        if not lesson:
            return None

        updated = CuratedLesson(
            lesson_id=lesson.lesson_id,
            source_proposal_id=lesson.source_proposal_id,
            applies_to_agents=lesson.applies_to_agents,
            lesson_type=lesson.lesson_type,
            title=lesson.title,
            description=lesson.description,
            lesson_payload=lesson.lesson_payload,
            validation_status=new_status,
            version=lesson.version,
            created_at=lesson.created_at,
            updated_at=datetime.now(UTC),
            rollback_of=lesson.rollback_of,
        )
        self._lessons[lesson_id] = updated
        return updated

    def delete(self, lesson_id: str) -> bool:
        if lesson_id not in self._lessons:
            return False

        lesson = self._lessons[lesson_id]
        del self._lessons[lesson_id]

        if lesson.source_proposal_id in self._by_proposal:
            del self._by_proposal[lesson.source_proposal_id]

        for agent in lesson.applies_to_agents:
            if agent in self._by_agent:
                self._by_agent[agent] = [lid for lid in self._by_agent[agent] if lid != lesson_id]

        return True
