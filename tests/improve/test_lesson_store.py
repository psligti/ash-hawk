from __future__ import annotations

from pathlib import Path

import pytest

from ash_hawk.improve.lesson_store import Lesson, LessonStore


def _make_lesson(
    lesson_id: str = "lesson-1",
    trial_id: str = "trial-1",
    hypothesis_summary: str = "increase timeout for API calls",
    root_cause: str = "timeout too short causing flaky failures",
    target_files: list[str] | None = None,
    outcome: str = "kept",
    score_before: float = 0.3,
    score_after: float = 0.8,
    score_delta: float = 0.5,
    iteration: int = 1,
    tags: list[str] | None = None,
) -> Lesson:
    return Lesson(
        lesson_id=lesson_id,
        trial_id=trial_id,
        hypothesis_summary=hypothesis_summary,
        root_cause=root_cause,
        target_files=target_files or ["agents/build.py"],
        outcome=outcome,
        score_before=score_before,
        score_after=score_after,
        score_delta=score_delta,
        iteration=iteration,
        tags=tags or [],
    )


class TestLessonModel:
    def test_model_dump_roundtrip(self):
        lesson = _make_lesson()
        data = lesson.model_dump()
        restored = Lesson.model_validate(data)
        assert restored.lesson_id == lesson.lesson_id
        assert restored.trial_id == lesson.trial_id
        assert restored.target_files == lesson.target_files
        assert restored.tags == lesson.tags
        assert restored.metadata == lesson.metadata

    def test_from_dict_extra_fields_raises(self):
        data = _make_lesson().model_dump()
        data["unknown_field"] = "oops"
        with pytest.raises(Exception):
            Lesson.model_validate(data)

    def test_default_created_at_is_iso(self):
        lesson = _make_lesson()
        assert "T" in lesson.created_at


class TestLessonStoreSaveLoad:
    def test_save_and_load_all(self, tmp_path: Path):
        store = LessonStore(lessons_dir=tmp_path / "lessons")
        lesson = _make_lesson()
        path = store.save(lesson)

        assert path.exists()
        loaded = store.load_all()
        assert len(loaded) == 1
        assert loaded[0].lesson_id == lesson.lesson_id

    def test_save_creates_directory(self, tmp_path: Path):
        store = LessonStore(lessons_dir=tmp_path / "nested" / "lessons")
        store.save(_make_lesson())
        assert (tmp_path / "nested" / "lessons").is_dir()

    def test_load_all_empty_dir(self, tmp_path: Path):
        store = LessonStore(lessons_dir=tmp_path / "lessons")
        assert store.load_all() == []

    def test_load_all_nonexistent_dir(self, tmp_path: Path):
        store = LessonStore(lessons_dir=tmp_path / "nope")
        assert store.load_all() == []

    def test_multiple_lessons(self, tmp_path: Path):
        store = LessonStore(lessons_dir=tmp_path / "lessons")
        for i in range(5):
            store.save(_make_lesson(lesson_id=f"lesson-{i}"))
        assert store.lesson_count() == 5
        assert len(store.load_all()) == 5


class TestLoadForTarget:
    def test_filters_by_target_files(self, tmp_path: Path):
        store = LessonStore(lessons_dir=tmp_path / "lessons")
        store.save(_make_lesson(lesson_id="l1", target_files=["agents/a.py"]))
        store.save(_make_lesson(lesson_id="l2", target_files=["agents/b.py"]))
        store.save(_make_lesson(lesson_id="l3", target_files=["agents/c.py"]))

        result = store.load_for_target(["agents/b.py", "agents/c.py"])
        ids = {lesson.lesson_id for lesson in result}
        assert ids == {"l2", "l3"}

    def test_no_match_returns_empty(self, tmp_path: Path):
        store = LessonStore(lessons_dir=tmp_path / "lessons")
        store.save(_make_lesson(target_files=["agents/a.py"]))
        assert store.load_for_target(["agents/z.py"]) == []


class TestFindSimilar:
    def test_finds_similar_hypothesis(self, tmp_path: Path):
        store = LessonStore(lessons_dir=tmp_path / "lessons")
        store.save(
            _make_lesson(
                hypothesis_summary="increase the timeout for API calls",
                root_cause="network latency causes intermittent failures",
            )
        )

        results = store.find_similar(
            hypothesis_summary="increase timeout for API calls",
            root_cause="completely unrelated cause",
        )
        assert len(results) == 1

    def test_finds_similar_root_cause(self, tmp_path: Path):
        store = LessonStore(lessons_dir=tmp_path / "lessons")
        store.save(
            _make_lesson(
                hypothesis_summary="unrelated hypothesis",
                root_cause="network latency causes intermittent failures",
            )
        )

        results = store.find_similar(
            hypothesis_summary="something totally different",
            root_cause="network latency causes intermittent failures",
        )
        assert len(results) == 1

    def test_threshold_filters_weak_matches(self, tmp_path: Path):
        store = LessonStore(lessons_dir=tmp_path / "lessons")
        store.save(
            _make_lesson(
                hypothesis_summary="increase timeout for API calls",
                root_cause="network latency intermittent failures",
            )
        )

        results = store.find_similar(
            hypothesis_summary="completely unrelated topic about cooking",
            root_cause="completely unrelated topic about gardening",
            threshold=0.6,
        )
        assert len(results) == 0


class TestHasBeenTried:
    def test_returns_true_for_similar(self, tmp_path: Path):
        store = LessonStore(lessons_dir=tmp_path / "lessons")
        store.save(
            _make_lesson(
                hypothesis_summary="increase timeout for API calls to reduce flakes",
                root_cause="timeout too short causing flaky failures",
            )
        )
        assert store.has_been_tried(
            "increase timeout for API calls to reduce flakes",
            "timeout too short causing flaky failures",
        )

    def test_returns_false_for_different(self, tmp_path: Path):
        store = LessonStore(lessons_dir=tmp_path / "lessons")
        store.save(
            _make_lesson(
                hypothesis_summary="increase timeout for API calls",
                root_cause="timeout too short",
            )
        )
        assert not store.has_been_tried(
            "rewrite the entire agent from scratch",
            "fundamental architecture problem",
        )


class TestFailedAndSuccessful:
    def test_get_failed_attempts(self, tmp_path: Path):
        store = LessonStore(lessons_dir=tmp_path / "lessons")
        store.save(_make_lesson(lesson_id="l1", outcome="reverted"))
        store.save(_make_lesson(lesson_id="l2", outcome="kept"))
        store.save(_make_lesson(lesson_id="l3", outcome="reverted"))

        failed = store.get_failed_attempts()
        assert len(failed) == 2
        assert all(lesson.outcome == "reverted" for lesson in failed)

    def test_get_successful_lessons(self, tmp_path: Path):
        store = LessonStore(lessons_dir=tmp_path / "lessons")
        store.save(_make_lesson(lesson_id="l1", outcome="kept"))
        store.save(_make_lesson(lesson_id="l2", outcome="reverted"))
        store.save(_make_lesson(lesson_id="l3", outcome="kept"))

        successful = store.get_successful_lessons()
        assert len(successful) == 2
        assert all(lesson.outcome == "kept" for lesson in successful)

    def test_get_failed_with_target_filter(self, tmp_path: Path):
        store = LessonStore(lessons_dir=tmp_path / "lessons")
        store.save(_make_lesson(lesson_id="l1", outcome="reverted", target_files=["agents/a.py"]))
        store.save(_make_lesson(lesson_id="l2", outcome="reverted", target_files=["agents/b.py"]))

        failed = store.get_failed_attempts(target_files=["agents/a.py"])
        assert len(failed) == 1
        assert failed[0].lesson_id == "l1"

    def test_get_successful_with_target_filter(self, tmp_path: Path):
        store = LessonStore(lessons_dir=tmp_path / "lessons")
        store.save(_make_lesson(lesson_id="l1", outcome="kept", target_files=["agents/a.py"]))
        store.save(_make_lesson(lesson_id="l2", outcome="kept", target_files=["agents/b.py"]))

        successful = store.get_successful_lessons(target_files=["agents/b.py"])
        assert len(successful) == 1
        assert successful[0].lesson_id == "l2"


class TestFormatLessonsForPrompt:
    def test_format_empty(self, tmp_path: Path):
        store = LessonStore(lessons_dir=tmp_path / "lessons")
        result = store.format_lessons_for_prompt([])
        assert "No lessons recorded yet" in result

    def test_format_with_lessons(self, tmp_path: Path):
        store = LessonStore(lessons_dir=tmp_path / "lessons")
        lessons = [
            _make_lesson(
                lesson_id="l1",
                hypothesis_summary="increase timeout",
                outcome="kept",
                score_delta=0.5,
                root_cause="timeout too short",
                target_files=["agents/build.py"],
            ),
        ]
        result = store.format_lessons_for_prompt(lessons)
        assert "showing 1 of 1" in result
        assert "### Lesson l1: increase timeout" in result
        assert "- Outcome: kept" in result
        assert "- Score delta: +0.5000" in result
        assert "- Root cause: timeout too short" in result
        assert "- Target files: agents/build.py" in result

    def test_max_lessons_limits_output(self, tmp_path: Path):
        store = LessonStore(lessons_dir=tmp_path / "lessons")
        lessons = [_make_lesson(lesson_id=f"l{i}") for i in range(20)]
        result = store.format_lessons_for_prompt(lessons, max_lessons=5)
        assert "showing 5 of 20" in result
        assert "### Lesson l4:" in result
        assert "### Lesson l5:" not in result


class TestLessonCount:
    def test_count_empty(self, tmp_path: Path):
        store = LessonStore(lessons_dir=tmp_path / "lessons")
        assert store.lesson_count() == 0

    def test_count_after_saves(self, tmp_path: Path):
        store = LessonStore(lessons_dir=tmp_path / "lessons")
        store.save(_make_lesson(lesson_id="l1"))
        store.save(_make_lesson(lesson_id="l2"))
        assert store.lesson_count() == 2
