"""Lesson store for persisting curated lessons.

DEPRECATED: The lesson store is deprecated and will be removed in a future version.
Use the new improvement module (ash_hawk.improvement) which applies unified diffs
directly to agent source files instead of storing intermediate lesson JSON.

Migration:
- Replace LessonStore with DiffApplier from ash_hawk.improvement
- Lessons are now applied as unified diffs to .dawn-kestrel/agents/{name}/*.md
- Use ImproverAgent to generate diffs from failed grades
"""

from __future__ import annotations

import fcntl
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ash_hawk.contracts import CuratedLesson

warnings.warn(
    "LessonStore is deprecated. Use ash_hawk.improvement.DiffApplier instead. "
    "Improvements now apply directly as unified diffs to agent source files.",
    DeprecationWarning,
    stacklevel=2,
)


class LessonStore:
    """Persists and retrieves curated lessons.

    Lessons are stored with versioning and metadata for provenance tracking.
    Supports both in-memory and file-backed persistence.
    """

    def __init__(self, storage_path: Path | None = None) -> None:
        self._storage_path = storage_path or Path(".ash-hawk/lessons")
        self._lessons: dict[str, CuratedLesson] = {}
        self._by_agent: dict[str, list[str]] = {}
        self._by_proposal: dict[str, str] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Load lessons from disk if not already loaded."""
        if self._loaded:
            return

        if self._storage_path and self._storage_path.exists():
            lessons_file = self._storage_path / "lessons.json"
            if lessons_file.exists():
                try:
                    with open(lessons_file) as f:
                        data = json.load(f)
                    for lesson_data in data.get("lessons", []):
                        lesson = CuratedLesson(**lesson_data)
                        self._lessons[lesson.lesson_id] = lesson
                        self._by_proposal[lesson.source_proposal_id] = lesson.lesson_id
                        for agent in lesson.applies_to_agents:
                            if agent not in self._by_agent:
                                self._by_agent[agent] = []
                            self._by_agent[agent].append(lesson.lesson_id)
                except (json.JSONDecodeError, KeyError):
                    pass
        self._loaded = True

    def _persist(self) -> None:
        if not self._storage_path:
            return

        self._storage_path.mkdir(parents=True, exist_ok=True)
        lessons_file = self._storage_path / "lessons.json"

        data: dict[str, Any] = {"lessons": []}
        for lesson in self._lessons.values():
            lesson_dict = lesson.model_dump()
            lesson_dict["created_at"] = lesson.created_at.isoformat() if lesson.created_at else None
            lesson_dict["updated_at"] = lesson.updated_at.isoformat() if lesson.updated_at else None
            data["lessons"].append(lesson_dict)

        with open(lessons_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            json.dump(data, f, indent=2, default=str)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def store(self, lesson: CuratedLesson) -> str:
        self._ensure_loaded()
        lesson_id = lesson.lesson_id
        self._lessons[lesson_id] = lesson
        self._by_proposal[lesson.source_proposal_id] = lesson_id

        for agent in lesson.applies_to_agents:
            if agent not in self._by_agent:
                self._by_agent[agent] = []
            if lesson_id not in self._by_agent[agent]:
                self._by_agent[agent].append(lesson_id)

        self._persist()
        return lesson_id

    def get(self, lesson_id: str) -> CuratedLesson | None:
        self._ensure_loaded()
        return self._lessons.get(lesson_id)

    def get_by_proposal(self, proposal_id: str) -> CuratedLesson | None:
        self._ensure_loaded()
        lesson_id = self._by_proposal.get(proposal_id)
        if lesson_id:
            return self._lessons.get(lesson_id)
        return None

    def get_for_agent(
        self,
        agent_id: str,
        experiment_id: str | None = None,
    ) -> list[CuratedLesson]:
        self._ensure_loaded()
        lesson_ids = self._by_agent.get(agent_id, [])
        lessons: list[CuratedLesson] = []
        for lid in lesson_ids:
            lesson = self._lessons.get(lid)
            if lesson and lesson.validation_status == "approved":
                if experiment_id is None or lesson.experiment_id == experiment_id:
                    lessons.append(lesson)
        return lessons

    def list_all(
        self,
        status: str | None = None,
        lesson_type: str | None = None,
        experiment_id: str | None = None,
    ) -> list[CuratedLesson]:
        self._ensure_loaded()
        lessons = list(self._lessons.values())
        if status:
            lessons = [lesson for lesson in lessons if lesson.validation_status == status]
        if lesson_type:
            lessons = [lesson for lesson in lessons if lesson.lesson_type == lesson_type]
        if experiment_id:
            lessons = [lesson for lesson in lessons if lesson.experiment_id == experiment_id]
        return lessons

    def update_status(
        self,
        lesson_id: str,
        new_status: Literal["approved", "deprecated", "rolled_back"],
    ) -> CuratedLesson | None:
        self._ensure_loaded()
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
        self._persist()
        return updated

    def delete(self, lesson_id: str) -> bool:
        self._ensure_loaded()
        if lesson_id not in self._lessons:
            return False

        lesson = self._lessons[lesson_id]
        del self._lessons[lesson_id]

        if lesson.source_proposal_id in self._by_proposal:
            del self._by_proposal[lesson.source_proposal_id]

        for agent in lesson.applies_to_agents:
            if agent in self._by_agent:
                self._by_agent[agent] = [lid for lid in self._by_agent[agent] if lid != lesson_id]

        self._persist()
        return True
