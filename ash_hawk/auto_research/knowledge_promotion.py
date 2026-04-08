from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ash_hawk.improve.lesson_store import Lesson

logger = logging.getLogger(__name__)


@dataclass
class PromotionCriteria:
    """Criteria for promoting an improvement to persistent knowledge."""

    min_improvement: float = 0.05  # 5% improvement required
    min_consecutive_successes: int = 3  # Must succeed N times in a row
    max_regression: float = 0.02  # No more than 2% regression on other metrics
    require_stability: bool = True  # Must not regress on re-run


@dataclass
class PromotedLesson:
    """A lesson that has been promoted to persistent knowledge."""

    lesson_id: str
    source_iteration: int
    hypothesis_summary: str
    root_cause: str
    target_files: list[str]
    score_delta: float
    promoted_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    promotion_confidence: float = 0.0
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> PromotedLesson:
        """Deserialize from dictionary."""
        return cls(**data)  # type: ignore[arg-type]


class KnowledgePromoter:
    """Evaluate improvements and promote validated lessons.

    Promotion means writing to local .ash-hawk/promoted/ and optionally
    to note-lark via MCP tools.

    Note: note-lark integration is optional. If note-lark tools are not available,
    promotion still works with local storage only.
    """

    def __init__(
        self,
        criteria: PromotionCriteria | None = None,
        storage_dir: Path | None = None,
        note_lark_enabled: bool = False,
    ) -> None:
        """Initialize promoter.

        Args:
            criteria: Promotion criteria (uses defaults if None).
            storage_dir: Directory for promoted lessons (default: .ash-hawk/promoted/).
            note_lark_enabled: Whether to attempt note-lark MCP integration.
        """
        self._criteria = criteria or PromotionCriteria()
        self._storage_dir = storage_dir or Path(".ash-hawk/promoted")
        self._note_lark_enabled = note_lark_enabled
        self._consecutive_successes: dict[str, int] = {}  # target_file -> count

    def should_promote(
        self,
        lesson: Lesson,
    ) -> tuple[bool, str]:
        """Determine if a lesson should be promoted.

        Checks:
        1. Outcome must be "kept" (not reverted)
        2. score_delta >= min_improvement
        3. Track consecutive successes per target file
        4. consecutive_successes >= min_consecutive_successes

        Returns:
            Tuple of (should_promote, reason).
        """
        # Check 1: outcome must be "kept"
        if lesson.outcome != "kept":
            logger.info(
                "Promotion check: lesson=%s delta=%.4f outcome=%s",
                lesson.lesson_id,
                lesson.score_delta,
                lesson.outcome,
            )
            logger.info("Promotion denied: outcome is %s, not 'kept'", lesson.outcome)
            return (False, f"outcome is '{lesson.outcome}', not 'kept'")

        # Check 2: score delta must meet minimum
        if lesson.score_delta < self._criteria.min_improvement:
            logger.info(
                "Promotion check: lesson=%s delta=%.4f outcome=%s",
                lesson.lesson_id,
                lesson.score_delta,
                lesson.outcome,
            )
            logger.info(
                "Promotion denied: delta %.4f below threshold %.4f",
                lesson.score_delta,
                self._criteria.min_improvement,
            )
            return (
                False,
                f"delta {lesson.score_delta:.4f} below threshold "
                f"{self._criteria.min_improvement:.4f}",
            )

        # Check 3 & 4: track consecutive successes per target file
        primary_target = lesson.target_files[0] if lesson.target_files else "__none__"
        self._consecutive_successes[primary_target] = (
            self._consecutive_successes.get(primary_target, 0) + 1
        )
        consecutive = self._consecutive_successes[primary_target]

        logger.info(
            "Promotion check: lesson=%s delta=%.4f outcome=%s",
            lesson.lesson_id,
            lesson.score_delta,
            lesson.outcome,
        )

        if consecutive < self._criteria.min_consecutive_successes:
            logger.info(
                "Promotion denied: %d/%d consecutive successes for %s",
                consecutive,
                self._criteria.min_consecutive_successes,
                primary_target,
            )
            return (
                False,
                f"{consecutive}/{self._criteria.min_consecutive_successes} "
                f"consecutive successes for {primary_target}",
            )

        logger.info(
            "Promotion approved: delta=%.4f consecutive=%d target=%s",
            lesson.score_delta,
            consecutive,
            primary_target,
        )
        return (True, "all criteria met")

    def promote(
        self,
        lesson: Lesson,
        tags: list[str] | None = None,
    ) -> PromotedLesson | None:
        """Promote a lesson to persistent storage.

        Steps:
        1. Create PromotedLesson from the input lesson
        2. Save to local JSON at .ash-hawk/promoted/{id}.json
        3. If note_lark_enabled, attempt note-lark promotion

        Args:
            lesson: The lesson to promote.
            tags: optional additional tags.

        Returns:
            PromotedLesson if successful, None on failure.
        """
        promoted_id = str(uuid.uuid4())
        merged_tags = list(set((lesson.tags or []) + (tags or [])))

        promoted = PromotedLesson(
            lesson_id=promoted_id,
            source_iteration=lesson.iteration,
            hypothesis_summary=lesson.hypothesis_summary,
            root_cause=lesson.root_cause,
            target_files=list(lesson.target_files),
            score_delta=lesson.score_delta,
            promotion_confidence=min(1.0, lesson.score_delta),
            tags=merged_tags,
        )

        logger.info(
            "Promoting lesson: %s (delta=%.4f)",
            promoted_id,
            promoted.score_delta,
        )

        try:
            path = self._save_local(promoted)
            logger.info("Lesson promoted to: %s", path)
        except OSError:
            logger.exception("Failed to save promoted lesson %s", promoted_id)
            return None

        if self._note_lark_enabled:
            self._promote_to_note_lark(promoted)

        return promoted

    def _save_local(self, promoted: PromotedLesson) -> Path:
        """Save promoted lesson to local JSON file.

        Uses atomic write (write to .tmp, rename).

        Args:
            promoted: The promoted lesson to save.

        Returns:
            Path to the saved file.
        """
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        path = self._storage_dir / f"{promoted.lesson_id}.json"
        tmp_path = path.with_suffix(".json.tmp")
        try:
            tmp_path.write_text(
                json.dumps(promoted.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            tmp_path.replace(path)
        except BaseException:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        return path

    def _promote_to_note_lark(self, promoted: PromotedLesson) -> bool:
        """Attempt to promote lesson to note-lark MCP.

        This is a best-effort operation. Actual note-lark MCP calls
        should be done from the CLI layer, not here. This method
        prepares the data and logs intent.

        Args:
            promoted: The promoted lesson to send to note-lark.

        Returns:
            True to indicate data was prepared.
        """
        logger.info(
            "Note-lark promotion prepared: lesson=%s tags=%s",
            promoted.lesson_id,
            promoted.tags,
        )
        logger.info(
            "Note-lark payload: hypothesis=%s root_cause=%s",
            promoted.hypothesis_summary[:80],
            promoted.root_cause[:80],
        )
        return True

    def load_promoted(self) -> list[PromotedLesson]:
        """Load all promoted lessons from storage.

        Returns:
            List of all promoted lessons.
        """
        if not self._storage_dir.is_dir():
            return []

        results: list[PromotedLesson] = []
        for json_file in sorted(self._storage_dir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                results.append(PromotedLesson.from_dict(data))
            except (json.JSONDecodeError, TypeError, ValueError):
                logger.warning("Failed to load promoted lesson from %s", json_file, exc_info=True)

        logger.info("Loaded %d promoted lessons from %s", len(results), self._storage_dir)
        return results

    def promotion_count(self) -> int:
        """Return number of promoted lessons.

        Returns:
            Count of promoted lesson files on disk.
        """
        if not self._storage_dir.is_dir():
            return 0
        return len(list(self._storage_dir.glob("*.json")))

    def reset_consecutive(self) -> None:
        """Reset consecutive success tracking."""
        self._consecutive_successes.clear()
