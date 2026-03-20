from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ash_hawk.contracts import CuratedLesson
from ash_hawk.cycle.coordinator import CycleRunner
from ash_hawk.strategies import Strategy


def _lesson(index: int, strategy: Strategy | None = None) -> CuratedLesson:
    created_at = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index)
    return CuratedLesson(
        lesson_id=f"lesson-{index}",
        source_proposal_id=f"proposal-{index}",
        applies_to_agents=["bolt-merlin"],
        lesson_type="policy",
        title=f"Lesson {index}",
        description=f"Description {index}",
        lesson_payload={"index": index},
        validation_status="approved",
        version=1,
        created_at=created_at,
        strategy=strategy,
    )


class TestCycleLessonSelection:
    def test_selection_respects_max_lessons(self) -> None:
        lessons = [_lesson(i, Strategy.TOOL_QUALITY) for i in range(1, 9)]
        selected = CycleRunner.select_lessons_for_iteration(lessons, iteration_num=3, max_lessons=4)
        assert len(selected) == 4

    def test_selection_rotates_with_iteration(self) -> None:
        lessons = [
            _lesson(1, Strategy.POLICY_QUALITY),
            _lesson(2, Strategy.POLICY_QUALITY),
            _lesson(3, Strategy.SKILL_QUALITY),
            _lesson(4, Strategy.SKILL_QUALITY),
            _lesson(5, Strategy.TOOL_QUALITY),
            _lesson(6, Strategy.TOOL_QUALITY),
        ]

        first = CycleRunner.select_lessons_for_iteration(lessons, iteration_num=2, max_lessons=4)
        second = CycleRunner.select_lessons_for_iteration(lessons, iteration_num=3, max_lessons=4)

        first_ids = [lesson.lesson_id for lesson in first]
        second_ids = [lesson.lesson_id for lesson in second]
        assert first_ids != second_ids
