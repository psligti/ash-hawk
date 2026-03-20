from __future__ import annotations

import re
from uuid import uuid4

from ash_hawk.improve_cycle.models import (
    AppliedChange,
    ChangeSet,
    CuratedLesson,
    ExperimentPlan,
    ProposalType,
)
from ash_hawk.improve_cycle.roles.base import BaseRoleAgent


class ApplierRole(BaseRoleAgent[tuple[list[CuratedLesson], ExperimentPlan], ChangeSet]):
    def __init__(self) -> None:
        super().__init__(
            "applier", "Convert lessons and plans into reversible change sets", "deterministic", 0.0
        )

    def run(self, payload: tuple[list[CuratedLesson], ExperimentPlan]) -> ChangeSet:
        lessons, plan = payload
        selected_lessons = [lesson for lesson in lessons if lesson.lesson_id in plan.lesson_ids]

        def _change_kind(lesson: CuratedLesson) -> str:
            if lesson.proposal_type in {ProposalType.TOOL_CREATE, ProposalType.SKILL_CREATE}:
                return "create"
            if lesson.proposal_type in {
                ProposalType.EVAL_EXPANSION,
                ProposalType.CONFIG_ADJUSTMENT,
            }:
                return "augment"
            return "update"

        touched: list[AppliedChange] = []
        for lesson in selected_lessons:
            surface_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", lesson.target_surface.strip()).strip("_")
            if not surface_slug:
                surface_slug = "unknown_surface"
            path = f"surface/{surface_slug}.md"
            touched.append(
                AppliedChange(
                    path=path,
                    surface=lesson.target_surface,
                    change_kind=_change_kind(lesson),
                    description=f"Applied {lesson.proposal_type.value} from {lesson.lesson_id}",
                )
            )

        return ChangeSet(
            change_set_id=f"changeset-{uuid4().hex[:8]}",
            lesson_ids=plan.lesson_ids,
            applied_changes=touched,
            rollback_plan=[f"revert {change.path} for lesson mapping" for change in touched],
            temp_only=plan.mode in {"ab", "adversarial", "isolated"},
        )
