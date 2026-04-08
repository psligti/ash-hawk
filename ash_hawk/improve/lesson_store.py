from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Lesson:
    """A lesson learned during the improvement loop."""

    lesson_id: str
    trial_id: str
    hypothesis_summary: str
    root_cause: str
    target_files: list[str]
    outcome: str
    score_before: float
    score_after: float
    score_delta: float
    iteration: int
    agent_path: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize the lesson to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Lesson:
        """Deserialize a lesson from a dictionary."""
        return cls(**data)  # type: ignore[arg-type]


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity on lowercased word sets."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


class LessonStore:
    """Persistent JSON-file-backed lesson store.

    Stores lessons at ``.ash-hawk/lessons/{lesson_id}.json``.
    Provides search, dedup, and consultation APIs.
    """

    def __init__(self, lessons_dir: Path | None = None) -> None:
        """Initialize with storage directory. Defaults to .ash-hawk/lessons/"""
        self._lessons_dir = lessons_dir or Path(".ash-hawk/lessons")

    def save(self, lesson: Lesson) -> Path:
        """Save a lesson to disk.

        Creates the lessons directory if needed and writes the lesson
        as JSON using an atomic write strategy.

        Args:
            lesson: The lesson to persist.

        Returns:
            Path to the saved lesson file.
        """
        self._lessons_dir.mkdir(parents=True, exist_ok=True)
        path = self._lessons_dir / f"{lesson.lesson_id}.json"
        tmp_path = path.with_suffix(".json.tmp")
        try:
            tmp_path.write_text(
                json.dumps(lesson.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            tmp_path.replace(path)
        except BaseException:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

        logger.info(
            "Lesson saved: id=%s outcome=%s delta=%.4f trial=%s",
            lesson.lesson_id,
            lesson.outcome,
            lesson.score_delta,
            lesson.trial_id,
        )
        return path

    def load_all(self) -> list[Lesson]:
        """Load all lessons from disk.

        Returns:
            List of all stored lessons.
        """
        if not self._lessons_dir.is_dir():
            return []

        lessons: list[Lesson] = []
        for json_file in sorted(self._lessons_dir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                lessons.append(Lesson.from_dict(data))
            except (json.JSONDecodeError, TypeError, ValueError):
                logger.warning("Failed to load lesson from %s", json_file, exc_info=True)
        logger.info("Loaded %d lessons from %s", len(lessons), self._lessons_dir)
        return lessons

    def load_for_target(self, target_files: list[str]) -> list[Lesson]:
        """Load lessons that touched any of the given target files.

        Args:
            target_files: Files to filter by.

        Returns:
            Lessons whose ``target_files`` overlap with the query.
        """
        target_set = set(target_files)
        all_lessons = self.load_all()
        matched = [lesson for lesson in all_lessons if set(lesson.target_files) & target_set]
        logger.debug("Found %d lessons for targets %s", len(matched), target_files)
        return matched

    def find_similar(
        self,
        hypothesis_summary: str,
        root_cause: str,
        threshold: float = 0.6,
    ) -> list[Lesson]:
        """Find lessons with similar hypothesis or root cause.

        Uses simple word-overlap similarity (Jaccard on lowercased word
        sets). Returns lessons above the similarity threshold.

        Args:
            hypothesis_summary: The hypothesis text to compare.
            root_cause: The root cause text to compare.
            threshold: Minimum Jaccard similarity (0.0-1.0).

        Returns:
            Lessons with similarity above the threshold.
        """
        all_lessons = self.load_all()
        matched: list[Lesson] = []
        for lesson in all_lessons:
            hyp_sim = _jaccard_similarity(hypothesis_summary, lesson.hypothesis_summary)
            cause_sim = _jaccard_similarity(root_cause, lesson.root_cause)
            if hyp_sim >= threshold or cause_sim >= threshold:
                matched.append(lesson)
        logger.debug("Found %d similar lessons (threshold=%.2f)", len(matched), threshold)
        return matched

    def has_been_tried(self, hypothesis_summary: str, root_cause: str) -> bool:
        """Check if a very similar hypothesis has been attempted before.

        Uses :meth:`find_similar` with a high threshold (0.8).

        Args:
            hypothesis_summary: The hypothesis to check.
            root_cause: The root cause to check.

        Returns:
            ``True`` if a similar lesson exists.
        """
        similar = self.find_similar(hypothesis_summary, root_cause, threshold=0.8)
        return len(similar) > 0

    def get_failed_attempts(self, target_files: list[str] | None = None) -> list[Lesson]:
        """Get all reverted (failed) lessons, optionally filtered by target files.

        Args:
            target_files: optional files to filter by.

        Returns:
            Lessons with ``outcome == "reverted"``.
        """
        pool = self.load_for_target(target_files) if target_files else self.load_all()
        failed = [lesson for lesson in pool if lesson.outcome == "reverted"]
        logger.debug("Found %d failed attempts", len(failed))
        return failed

    def get_successful_lessons(self, target_files: list[str] | None = None) -> list[Lesson]:
        """Get all kept (successful) lessons, optionally filtered by target files.

        Args:
            target_files: optional files to filter by.

        Returns:
            Lessons with ``outcome == "kept"``.
        """
        pool = self.load_for_target(target_files) if target_files else self.load_all()
        successful = [lesson for lesson in pool if lesson.outcome == "kept"]
        logger.debug("Found %d successful lessons", len(successful))
        return successful

    def format_lessons_for_prompt(self, lessons: list[Lesson], max_lessons: int = 10) -> str:
        """Format lessons into a human-readable string for LLM prompt injection.

        Args:
            lessons: Lessons to format.
            max_lessons: Maximum number of lessons to include.

        Returns:
            Formatted string for prompt injection.
        """
        if not lessons:
            return "## Past Lessons\nNo lessons recorded yet."

        shown = lessons[:max_lessons]
        total = len(lessons)
        lines: list[str] = [f"## Past Lessons (showing {len(shown)} of {total})"]

        for lesson in shown:
            lines.append(f"### Lesson {lesson.lesson_id}: {lesson.hypothesis_summary}")
            lines.append(f"- Outcome: {lesson.outcome}")
            lines.append(f"- Score delta: {lesson.score_delta:+.4f}")
            lines.append(f"- Root cause: {lesson.root_cause}")
            files_str = ", ".join(lesson.target_files) if lesson.target_files else "none"
            lines.append(f"- Target files: {files_str}")

        return "\n".join(lines)

    def lesson_count(self) -> int:
        """Return total number of stored lessons."""
        if not self._lessons_dir.is_dir():
            return 0
        return len(list(self._lessons_dir.glob("*.json")))
