from __future__ import annotations

import json
from pathlib import Path

from ash_hawk.auto_research.knowledge_promotion import (
    KnowledgePromoter,
    PromotedLesson,
    PromotionCriteria,
)
from ash_hawk.improve.lesson_store import Lesson


def _lesson(
    *,
    outcome: str = "kept",
    score_delta: float = 0.1,
    target_files: list[str] | None = None,
    iteration: int = 1,
    tags: list[str] | None = None,
) -> Lesson:
    return Lesson(
        lesson_id=f"lesson-{iteration}",
        trial_id=f"trial-{iteration}",
        hypothesis_summary="Test hypothesis",
        root_cause="Test root cause",
        target_files=target_files or ["src/module.py"],
        outcome=outcome,
        score_before=0.5,
        score_after=0.5 + score_delta,
        score_delta=score_delta,
        iteration=iteration,
        tags=tags or [],
    )


class TestShouldPromoteApproved:
    def test_sufficient_delta_and_consecutive_successes(self) -> None:
        promoter = KnowledgePromoter(
            criteria=PromotionCriteria(
                min_improvement=0.05,
                min_consecutive_successes=3,
            ),
            storage_dir=Path("/tmp/test-promoted"),
        )
        target = ["src/module.py"]
        for i in range(3):
            lesson = _lesson(score_delta=0.1, target_files=target, iteration=i)
            approved, reason = promoter.should_promote(lesson)

        assert approved
        assert reason == "all criteria met"


class TestShouldPromoteRejected:
    def test_reverted_outcome(self) -> None:
        promoter = KnowledgePromoter(storage_dir=Path("/tmp/test-promoted"))
        lesson = _lesson(outcome="reverted", score_delta=0.1)
        approved, reason = promoter.should_promote(lesson)
        assert not approved
        assert "reverted" in reason

    def test_insufficient_delta(self) -> None:
        promoter = KnowledgePromoter(
            criteria=PromotionCriteria(min_improvement=0.1),
            storage_dir=Path("/tmp/test-promoted"),
        )
        lesson = _lesson(score_delta=0.03)
        approved, reason = promoter.should_promote(lesson)
        assert not approved
        assert "below threshold" in reason

    def test_needs_consecutive_successes(self) -> None:
        promoter = KnowledgePromoter(
            criteria=PromotionCriteria(
                min_improvement=0.05,
                min_consecutive_successes=5,
            ),
            storage_dir=Path("/tmp/test-promoted"),
        )
        target = ["src/module.py"]
        for i in range(4):
            lesson = _lesson(score_delta=0.1, target_files=target, iteration=i)
            approved, reason = promoter.should_promote(lesson)

        assert not approved
        assert "4/5" in reason

    def test_consecutive_tracked_per_target(self) -> None:
        promoter = KnowledgePromoter(
            criteria=PromotionCriteria(
                min_improvement=0.05,
                min_consecutive_successes=2,
            ),
            storage_dir=Path("/tmp/test-promoted"),
        )
        target_a = ["src/a.py"]
        target_b = ["src/b.py"]

        lesson_a1 = _lesson(score_delta=0.1, target_files=target_a, iteration=1)
        approved_a1, _ = promoter.should_promote(lesson_a1)
        assert not approved_a1

        lesson_b1 = _lesson(score_delta=0.1, target_files=target_b, iteration=2)
        approved_b1, _ = promoter.should_promote(lesson_b1)
        assert not approved_b1

        lesson_a2 = _lesson(score_delta=0.1, target_files=target_a, iteration=3)
        approved_a2, _ = promoter.should_promote(lesson_a2)
        assert approved_a2


class TestPromote:
    def test_saves_to_local_json(self, tmp_path: Path) -> None:
        promoter = KnowledgePromoter(storage_dir=tmp_path / "promoted")
        lesson = _lesson(score_delta=0.15, tags=["auto"])

        result = promoter.promote(lesson, tags=["promoted"])

        assert result is not None
        assert result.score_delta == 0.15
        assert result.source_iteration == lesson.iteration
        assert result.hypothesis_summary == lesson.hypothesis_summary
        assert "auto" in result.tags
        assert "promoted" in result.tags

        saved_files = list((tmp_path / "promoted").glob("*.json"))
        assert len(saved_files) == 1

        data = json.loads(saved_files[0].read_text(encoding="utf-8"))
        assert data["lesson_id"] == result.lesson_id
        assert data["score_delta"] == 0.15


class TestLoadPromoted:
    def test_reads_back_correctly(self, tmp_path: Path) -> None:
        promoter = KnowledgePromoter(storage_dir=tmp_path / "promoted")
        lesson = _lesson(score_delta=0.2, iteration=5)
        promoted = promoter.promote(lesson)
        assert promoted is not None

        loader = KnowledgePromoter(storage_dir=tmp_path / "promoted")
        loaded = loader.load_promoted()

        assert len(loaded) == 1
        assert loaded[0].lesson_id == promoted.lesson_id
        assert loaded[0].score_delta == 0.2
        assert loaded[0].source_iteration == 5

    def test_empty_when_no_files(self, tmp_path: Path) -> None:
        promoter = KnowledgePromoter(storage_dir=tmp_path / "promoted")
        assert promoter.load_promoted() == []


class TestPromotionCount:
    def test_counts_files(self, tmp_path: Path) -> None:
        promoter = KnowledgePromoter(storage_dir=tmp_path / "promoted")
        assert promoter.promotion_count() == 0

        promoter.promote(_lesson(score_delta=0.1, iteration=1))
        assert promoter.promotion_count() == 1

        promoter.promote(_lesson(score_delta=0.2, iteration=2))
        assert promoter.promotion_count() == 2


class TestCustomCriteria:
    def test_low_threshold_promotes_earlier(self) -> None:
        criteria = PromotionCriteria(
            min_improvement=0.01,
            min_consecutive_successes=1,
        )
        promoter = KnowledgePromoter(
            criteria=criteria,
            storage_dir=Path("/tmp/test-promoted"),
        )
        lesson = _lesson(score_delta=0.02)
        approved, reason = promoter.should_promote(lesson)
        assert approved

    def test_high_threshold_requires_more(self) -> None:
        criteria = PromotionCriteria(
            min_improvement=0.5,
            min_consecutive_successes=1,
        )
        promoter = KnowledgePromoter(
            criteria=criteria,
            storage_dir=Path("/tmp/test-promoted"),
        )
        lesson = _lesson(score_delta=0.3)
        approved, reason = promoter.should_promote(lesson)
        assert not approved


class TestResetConsecutive:
    def test_reset_clears_tracking(self) -> None:
        promoter = KnowledgePromoter(
            criteria=PromotionCriteria(min_consecutive_successes=2),
            storage_dir=Path("/tmp/test-promoted"),
        )
        target = ["src/x.py"]
        promoter.should_promote(_lesson(score_delta=0.1, target_files=target, iteration=1))
        promoter.reset_consecutive()

        approved, reason = promoter.should_promote(
            _lesson(score_delta=0.1, target_files=target, iteration=2)
        )
        assert not approved
        assert "1/2" in reason


class TestNoteLarkFlag:
    def test_note_lark_enabled_does_not_error(self, tmp_path: Path) -> None:
        promoter = KnowledgePromoter(
            storage_dir=tmp_path / "promoted",
            note_lark_enabled=True,
        )
        lesson = _lesson(score_delta=0.1)
        result = promoter.promote(lesson)
        assert result is not None


class TestPromotedLessonSerialization:
    def test_roundtrip(self) -> None:
        promoted = PromotedLesson(
            lesson_id="test-id",
            source_iteration=3,
            hypothesis_summary="hyp",
            root_cause="cause",
            target_files=["a.py", "b.py"],
            score_delta=0.12,
            promotion_confidence=0.9,
            tags=["tag1"],
        )
        data = promoted.to_dict()
        restored = PromotedLesson.from_dict(data)
        assert restored.lesson_id == promoted.lesson_id
        assert restored.target_files == ["a.py", "b.py"]
        assert restored.score_delta == 0.12
