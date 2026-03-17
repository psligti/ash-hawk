"""Experiment-scoped lesson storage with namespacing."""

from __future__ import annotations

import fcntl
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ash_hawk.contracts import CuratedLesson


class ExperimentStore:
    """Experiment-scoped lesson storage with namespacing.

    Wraps lesson storage to provide experiment-isolated persistence.
    Lessons are stored in .ash-hawk/experiments/{experiment_id}/lessons.json
    with thread-safe writes using file locking.
    """

    def __init__(self, base_path: Path | None = None) -> None:
        self._base_path = base_path or Path(".ash-hawk")
        self._experiments_path = self._base_path / "experiments"
        self._global_path = self._base_path / "lessons"
        # In-memory caches per experiment
        self._lessons_by_experiment: dict[str, dict[str, CuratedLesson]] = {}
        self._by_agent_by_experiment: dict[str, dict[str, list[str]]] = {}
        self._loaded_experiments: set[str] = set()

    def _get_experiment_path(self, experiment_id: str) -> Path:
        """Get the storage path for an experiment."""
        return self._experiments_path / experiment_id / "lessons"

    def _ensure_loaded(self, experiment_id: str) -> None:
        """Load lessons from disk for an experiment if not already loaded."""
        if experiment_id in self._loaded_experiments:
            return

        exp_path = self._get_experiment_path(experiment_id)
        lessons_file = exp_path / "lessons.json"

        if experiment_id not in self._lessons_by_experiment:
            self._lessons_by_experiment[experiment_id] = {}
        if experiment_id not in self._by_agent_by_experiment:
            self._by_agent_by_experiment[experiment_id] = {}

        if lessons_file.exists():
            try:
                with open(lessons_file) as f:
                    data = json.load(f)
                for lesson_data in data.get("lessons", []):
                    lesson = CuratedLesson(**lesson_data)
                    self._lessons_by_experiment[experiment_id][lesson.lesson_id] = lesson
                    for agent in lesson.applies_to_agents:
                        if agent not in self._by_agent_by_experiment[experiment_id]:
                            self._by_agent_by_experiment[experiment_id][agent] = []
                        self._by_agent_by_experiment[experiment_id][agent].append(lesson.lesson_id)
            except (json.JSONDecodeError, KeyError):
                pass

        self._loaded_experiments.add(experiment_id)

    def _persist(self, experiment_id: str) -> None:
        """Persist lessons to disk with file locking."""
        exp_path = self._get_experiment_path(experiment_id)
        exp_path.mkdir(parents=True, exist_ok=True)
        lessons_file = exp_path / "lessons.json"

        lessons = self._lessons_by_experiment.get(experiment_id, {})
        data: dict[str, Any] = {"lessons": []}
        for lesson in lessons.values():
            lesson_dict = lesson.model_dump()
            lesson_dict["created_at"] = lesson.created_at.isoformat() if lesson.created_at else None
            lesson_dict["updated_at"] = lesson.updated_at.isoformat() if lesson.updated_at else None
            data["lessons"].append(lesson_dict)

        with open(lessons_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            json.dump(data, f, indent=2, default=str)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def store(self, lesson: CuratedLesson, experiment_id: str) -> str:
        """Store a lesson in the specified experiment."""
        self._ensure_loaded(experiment_id)

        # Ensure lesson has experiment_id set
        if lesson.experiment_id != experiment_id:
            lesson = CuratedLesson(
                **lesson.model_dump(exclude={"experiment_id"}),
                experiment_id=experiment_id,
            )

        lesson_id = lesson.lesson_id
        self._lessons_by_experiment[experiment_id][lesson_id] = lesson

        if experiment_id not in self._by_agent_by_experiment:
            self._by_agent_by_experiment[experiment_id] = {}

        for agent in lesson.applies_to_agents:
            if agent not in self._by_agent_by_experiment[experiment_id]:
                self._by_agent_by_experiment[experiment_id][agent] = []
            if lesson_id not in self._by_agent_by_experiment[experiment_id][agent]:
                self._by_agent_by_experiment[experiment_id][agent].append(lesson_id)

        self._persist(experiment_id)
        return lesson_id

    def get(self, lesson_id: str, experiment_id: str) -> CuratedLesson | None:
        """Get a lesson by ID from an experiment."""
        self._ensure_loaded(experiment_id)
        return self._lessons_by_experiment.get(experiment_id, {}).get(lesson_id)

    def get_for_agent(
        self,
        agent_id: str,
        experiment_id: str,
    ) -> list[CuratedLesson]:
        """Get lessons for an agent within an experiment."""
        self._ensure_loaded(experiment_id)
        lesson_ids = self._by_agent_by_experiment.get(experiment_id, {}).get(agent_id, [])
        lessons: list[CuratedLesson] = []
        for lid in lesson_ids:
            lesson = self._lessons_by_experiment.get(experiment_id, {}).get(lid)
            if lesson and lesson.validation_status == "approved":
                lessons.append(lesson)
        return lessons

    def list_all(
        self,
        experiment_id: str,
        status: str | None = None,
        lesson_type: str | None = None,
    ) -> list[CuratedLesson]:
        """List all lessons in an experiment with optional filters."""
        self._ensure_loaded(experiment_id)
        lessons = list(self._lessons_by_experiment.get(experiment_id, {}).values())
        if status:
            lessons = [lesson for lesson in lessons if lesson.validation_status == status]
        if lesson_type:
            lessons = [lesson for lesson in lessons if lesson.lesson_type == lesson_type]
        return lessons

    def update_status(
        self,
        lesson_id: str,
        experiment_id: str,
        new_status: Literal["approved", "deprecated", "rolled_back"],
    ) -> CuratedLesson | None:
        """Update lesson status within an experiment."""
        self._ensure_loaded(experiment_id)
        lesson = self._lessons_by_experiment.get(experiment_id, {}).get(lesson_id)
        if not lesson:
            return None

        updated = CuratedLesson(
            **lesson.model_dump(exclude={"validation_status", "updated_at"}),
            validation_status=new_status,
            updated_at=datetime.now(UTC),
        )
        self._lessons_by_experiment[experiment_id][lesson_id] = updated
        self._persist(experiment_id)
        return updated

    def delete(self, lesson_id: str, experiment_id: str) -> bool:
        """Delete a lesson from an experiment."""
        self._ensure_loaded(experiment_id)
        lessons = self._lessons_by_experiment.get(experiment_id, {})
        if lesson_id not in lessons:
            return False

        lesson = lessons[lesson_id]
        del lessons[lesson_id]

        # Clean up agent index
        by_agent = self._by_agent_by_experiment.get(experiment_id, {})
        for agent in lesson.applies_to_agents:
            if agent in by_agent:
                by_agent[agent] = [lid for lid in by_agent[agent] if lid != lesson_id]

        self._persist(experiment_id)
        return True

    def list_experiments(self) -> list[str]:
        """List all experiment IDs with stored lessons."""
        if not self._experiments_path.exists():
            return []
        return [d.name for d in self._experiments_path.iterdir() if d.is_dir()]

    def promote_to_global(self, experiment_id: str, lesson_ids: list[str]) -> list[str]:
        """Promote specified lessons from experiment to global store.

        Returns list of promoted lesson IDs.
        """
        self._ensure_loaded(experiment_id)
        promoted: list[str] = []

        # Load global store
        global_store_path = self._global_path
        global_store_path.mkdir(parents=True, exist_ok=True)
        global_lessons_file = global_store_path / "lessons.json"

        global_lessons: dict[str, Any] = {"lessons": []}
        if global_lessons_file.exists():
            try:
                with open(global_lessons_file) as f:
                    global_lessons = json.load(f)
            except (json.JSONDecodeError, KeyError):
                pass

        for lesson_id in lesson_ids:
            lesson = self._lessons_by_experiment.get(experiment_id, {}).get(lesson_id)
            if lesson:
                # Remove experiment_id for global promotion
                promoted_lesson = CuratedLesson(
                    **lesson.model_dump(exclude={"experiment_id"}),
                    experiment_id=None,
                )
                lesson_dict = promoted_lesson.model_dump()
                lesson_dict["created_at"] = (
                    promoted_lesson.created_at.isoformat() if promoted_lesson.created_at else None
                )
                lesson_dict["updated_at"] = (
                    promoted_lesson.updated_at.isoformat() if promoted_lesson.updated_at else None
                )
                lesson_dict["promoted_from"] = experiment_id
                global_lessons["lessons"].append(lesson_dict)
                promoted.append(lesson_id)

        # Write global store with locking
        with open(global_lessons_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            json.dump(global_lessons, f, indent=2, default=str)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return promoted
