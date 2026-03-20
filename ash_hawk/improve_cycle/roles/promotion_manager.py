from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from ash_hawk.improve_cycle.models import (
    CuratedLesson,
    PromotionDecision,
    PromotionStatus,
    VerificationReport,
)
from ash_hawk.improve_cycle.roles.base import BaseRoleAgent


class PromotionManagerRole(
    BaseRoleAgent[tuple[VerificationReport, list[CuratedLesson]], list[PromotionDecision]]
):
    def __init__(self, *, default_scope: str = "agent-specific") -> None:
        super().__init__(
            "promotion_manager", "Make scoped promotion decisions", "deterministic", 0.0
        )
        self._default_scope = default_scope

    def run(
        self,
        payload: tuple[VerificationReport, list[CuratedLesson]],
    ) -> list[PromotionDecision]:
        report, lessons = payload
        decisions: list[PromotionDecision] = []
        status = (
            PromotionStatus.PROMOTE_AGENT_SPECIFIC
            if report.recommendation == "promote"
            else PromotionStatus.HOLD_FOR_MORE_DATA
        )
        for lesson in lessons:
            decisions.append(
                PromotionDecision(
                    decision_id=f"decision-{uuid4().hex[:8]}",
                    lesson_id=lesson.lesson_id,
                    status=status,
                    scope=self._default_scope,
                    reason=report.overall_summary,
                    effective_version=datetime.now(UTC).strftime("%Y.%m.%d"),
                    rollback_trigger="regression_count>0",
                )
            )
        return decisions
