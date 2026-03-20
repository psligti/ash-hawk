from __future__ import annotations

from uuid import uuid4

from ash_hawk.improve_cycle.models import AppliedChange, ChangeSet, CuratedLesson, ExperimentPlan
from ash_hawk.improve_cycle.roles.base import BaseRoleAgent


class ApplierRole(BaseRoleAgent[tuple[list[CuratedLesson], ExperimentPlan], ChangeSet]):
    def __init__(self) -> None:
        super().__init__(
            "applier", "Convert lessons and plans into reversible change sets", "deterministic", 0.0
        )

    def run(self, payload: tuple[list[CuratedLesson], ExperimentPlan]) -> ChangeSet:
        lessons, plan = payload
        touched: list[AppliedChange] = [
            AppliedChange(
                path=f"surface/{lesson.target_surface.replace(' ', '_')}.md",
                surface=lesson.target_surface,
                change_kind="update",
                description=f"Applied lesson {lesson.lesson_id}",
            )
            for lesson in lessons
            if lesson.lesson_id in plan.lesson_ids
        ]
        return ChangeSet(
            change_set_id=f"changeset-{uuid4().hex[:8]}",
            lesson_ids=plan.lesson_ids,
            applied_changes=touched,
            rollback_plan=[f"revert {change.path}" for change in touched],
            temp_only=False,
        )
