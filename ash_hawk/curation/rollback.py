"""Rollback manager for lesson version control."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ash_hawk.contracts import CuratedLesson, ImprovementProposal


class RollbackManager:
    """Manages lesson rollback and version history.

    Tracks lesson versions and supports rollback to previous states.
    """

    def __init__(self) -> None:
        self._history: dict[str, list[CuratedLesson]] = {}
        self._rolled_back: dict[str, str] = {}

    def snapshot(self, lesson: CuratedLesson) -> None:
        lesson_id = lesson.lesson_id
        if lesson_id not in self._history:
            self._history[lesson_id] = []

        snapshot_lesson = CuratedLesson(
            lesson_id=lesson.lesson_id,
            source_proposal_id=lesson.source_proposal_id,
            applies_to_agents=lesson.applies_to_agents.copy(),
            lesson_type=lesson.lesson_type,
            title=lesson.title,
            description=lesson.description,
            lesson_payload=lesson.lesson_payload.copy(),
            validation_status=lesson.validation_status,
            version=lesson.version,
            created_at=lesson.created_at,
            updated_at=lesson.updated_at,
            rollback_of=lesson.rollback_of,
        )
        self._history[lesson_id].append(snapshot_lesson)

    def rollback(
        self,
        lesson_id: str,
        target_version: int | None = None,
    ) -> CuratedLesson | None:
        history = self._history.get(lesson_id, [])
        if not history:
            return None

        if target_version is None:
            if len(history) < 2:
                return None
            target = history[-2]
        else:
            matching = [h for h in history if h.version == target_version]
            if not matching:
                return None
            target = matching[-1]

        rolled_back = CuratedLesson(
            lesson_id=target.lesson_id,
            source_proposal_id=target.source_proposal_id,
            applies_to_agents=target.applies_to_agents,
            lesson_type=target.lesson_type,
            title=target.title,
            description=target.description,
            lesson_payload=target.lesson_payload,
            validation_status="rolled_back",
            version=target.version + 1,
            created_at=target.created_at,
            updated_at=datetime.now(UTC),
            rollback_of=lesson_id,
        )

        self._rolled_back[rolled_back.lesson_id] = lesson_id
        self._history[lesson_id].append(rolled_back)

        return rolled_back

    def get_history(self, lesson_id: str) -> list[CuratedLesson]:
        return self._history.get(lesson_id, []).copy()

    def get_rollback_chain(self, lesson_id: str) -> list[str]:
        chain: list[str] = []
        current = lesson_id
        while current in self._rolled_back:
            chain.append(current)
            current = self._rolled_back[current]
        return chain

    def can_rollback(self, lesson_id: str) -> bool:
        history = self._history.get(lesson_id, [])
        return len(history) >= 2
