from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pydantic as pd

logger = logging.getLogger(__name__)


class Lesson(pd.BaseModel):
    """A lesson learned during the improvement loop."""

    model_config = pd.ConfigDict(extra="forbid")

    lesson_id: str = pd.Field(description="Unique lesson identifier")
    trial_id: str = pd.Field(description="Trial that generated this lesson")
    hypothesis_summary: str = pd.Field(description="Summary of the hypothesis tested")
    root_cause: str = pd.Field(description="Root cause identified")
    target_files: list[str] = pd.Field(
        default_factory=list, description="Files targeted by the fix"
    )
    outcome: str = pd.Field(description="kept or reverted")
    score_before: float = pd.Field(description="Pass rate before the fix")
    score_after: float = pd.Field(description="Pass rate after the fix")
    score_delta: float = pd.Field(description="Change in pass rate")
    iteration: int = pd.Field(description="Which iteration produced this lesson")
    agent_path: str | None = pd.Field(default=None, description="Path to the agent directory")
    created_at: str = pd.Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO timestamp",
    )
    tags: list[str] = pd.Field(default_factory=list, description="Searchable tags")
    metadata: dict[str, object] = pd.Field(default_factory=dict, description="Arbitrary metadata")


def _jaccard_similarity(text_a: str, text_b: str) -> float:
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
        self._lessons_dir = lessons_dir or Path(".ash-hawk/lessons")

    def save(self, lesson: Lesson) -> Path:
        self._lessons_dir.mkdir(parents=True, exist_ok=True)
        path = self._lessons_dir / f"{lesson.lesson_id}.json"
        tmp_path = path.with_suffix(".json.tmp")
        try:
            tmp_path.write_text(
                json.dumps(lesson.model_dump(), indent=2, sort_keys=True),
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
        if not self._lessons_dir.is_dir():
            return []

        lessons: list[Lesson] = []
        for json_file in sorted(self._lessons_dir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                lessons.append(Lesson.model_validate(data))
            except (json.JSONDecodeError, TypeError, ValueError):
                logger.warning("Failed to load lesson from %s", json_file, exc_info=True)
        logger.info("Loaded %d lessons from %s", len(lessons), self._lessons_dir)
        return lessons

    def load_for_target(self, target_files: list[str]) -> list[Lesson]:
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
        similar = self.find_similar(hypothesis_summary, root_cause, threshold=0.8)
        return len(similar) > 0

    def get_failed_attempts(self, target_files: list[str] | None = None) -> list[Lesson]:
        pool = self.load_for_target(target_files) if target_files else self.load_all()
        failed = [lesson for lesson in pool if lesson.outcome == "reverted"]
        logger.debug("Found %d failed attempts", len(failed))
        return failed

    def get_successful_lessons(self, target_files: list[str] | None = None) -> list[Lesson]:
        pool = self.load_for_target(target_files) if target_files else self.load_all()
        successful = [lesson for lesson in pool if lesson.outcome == "kept"]
        logger.debug("Found %d successful lessons", len(successful))
        return successful

    def format_lessons_for_prompt(self, lessons: list[Lesson], max_lessons: int = 10) -> str:
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
        if not self._lessons_dir.is_dir():
            return 0
        return len(list(self._lessons_dir.glob("*.json")))
