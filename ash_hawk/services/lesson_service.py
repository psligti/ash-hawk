"""Lesson service for managing curated lessons."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ash_hawk.contracts import CuratedLesson, ImprovementProposal
from ash_hawk.curation.provenance import ProvenanceRecord, ProvenanceTracker
from ash_hawk.curation.rollback import RollbackManager
from ash_hawk.curation.store import LessonStore

if TYPE_CHECKING:
    from ash_hawk.curation.persistent_store import PersistentLessonStore


class LessonServiceError(Exception):
    """Base error for lesson service operations."""

    pass


class ExperimentIdRequiredError(LessonServiceError):
    """Raised when experiment_id is required but not provided."""

    pass


class LessonService:
    """Service for managing the lifecycle of curated lessons.

    IMPORTANT: For production use with parallel trials and persistent
    provenance tracking, use AsyncLessonService instead. This sync
    service uses in-memory provenance tracking which is lost on restart.

    Enforces experiment_id by default for parallel trial isolation.
    """

    def __init__(self, storage_path: Path | None = None) -> None:
        self._store = LessonStore(storage_path)
        self._provenance = ProvenanceTracker()
        self._rollback = RollbackManager()

    def approve_proposal(
        self,
        proposal: ImprovementProposal,
        applies_to_agents: list[str] | None = None,
        experiment_id: str | None = None,
        require_experiment_id: bool = True,
    ) -> CuratedLesson:
        """Approve a proposal and create a curated lesson.

        Args:
            proposal: The improvement proposal to approve.
            applies_to_agents: Agents this lesson applies to.
            experiment_id: REQUIRED for parallel trial isolation.
            require_experiment_id: If True, raises error when experiment_id is None.

        Returns:
            The created CuratedLesson.

        Raises:
            ExperimentIdRequiredError: If require_experiment_id=True and no experiment_id.
        """
        if experiment_id is None:
            if require_experiment_id:
                raise ExperimentIdRequiredError(
                    "experiment_id is REQUIRED for lesson curation in production. "
                    "Lessons without experiment_id pollute the global namespace and "
                    "can cause cross-trial contamination. Set require_experiment_id=False "
                    "only for local development or single-user scenarios."
                )
            import warnings

            warnings.warn(
                "experiment_id not provided - lesson will be stored in global namespace. "
                "Pass experiment_id for parallel trial isolation. "
                "Set require_experiment_id=True to enforce this requirement.",
                UserWarning,
                stacklevel=2,
            )

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
            experiment_id=experiment_id,
        )

        self._store.store(lesson)
        self._provenance.track(lesson, proposal)
        self._rollback.snapshot(lesson)

        return lesson

    def get_lesson(self, lesson_id: str) -> CuratedLesson | None:
        return self._store.get(lesson_id)

    def get_lessons_for_agent(
        self,
        agent_id: str,
        experiment_id: str | None = None,
    ) -> list[CuratedLesson]:
        return self._store.get_for_agent(agent_id, experiment_id=experiment_id)

    def list_lessons(
        self,
        status: str | None = None,
        lesson_type: str | None = None,
        experiment_id: str | None = None,
    ) -> list[CuratedLesson]:
        return self._store.list_all(
            status=status,
            lesson_type=lesson_type,
            experiment_id=experiment_id,
        )

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
