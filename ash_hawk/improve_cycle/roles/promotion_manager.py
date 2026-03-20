from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from ash_hawk.improve_cycle.configuration import ImproveCyclePromotionConfig
from ash_hawk.improve_cycle.models import (
    CuratedLesson,
    PromotionDecision,
    VerificationReport,
)
from ash_hawk.improve_cycle.promotion import (
    PromotionContext,
    PromotionPolicy,
    decide_promotion_status,
    decision_scope,
)
from ash_hawk.improve_cycle.roles.base import BaseRoleAgent


class PromotionManagerRole(
    BaseRoleAgent[
        tuple[VerificationReport, list[CuratedLesson], PromotionContext],
        list[PromotionDecision],
    ]
):
    def __init__(
        self,
        *,
        config: ImproveCyclePromotionConfig,
        default_scope: str = "agent-specific",
    ) -> None:
        super().__init__(
            "promotion_manager", "Make scoped promotion decisions", "deterministic", 0.0
        )
        self._default_scope = default_scope
        self._policy = PromotionPolicy(
            low_risk_success_threshold=config.low_risk_success_threshold,
            medium_risk_success_threshold=config.medium_risk_success_threshold,
            min_score_delta_for_global=config.min_score_delta_for_global,
            retire_after_failures=config.retire_after_failures,
        )

    def run(
        self,
        payload: tuple[VerificationReport, list[CuratedLesson], PromotionContext],
    ) -> list[PromotionDecision]:
        report, lessons, context = payload
        decisions: list[PromotionDecision] = []
        for lesson in lessons:
            status = decide_promotion_status(
                report,
                lesson,
                policy=self._policy,
                context=context,
            )
            decisions.append(
                PromotionDecision(
                    decision_id=f"decision-{uuid4().hex[:8]}",
                    lesson_id=lesson.lesson_id,
                    status=status,
                    scope=decision_scope(status, self._default_scope),
                    reason=report.overall_summary,
                    effective_version=datetime.now(UTC).strftime("%Y.%m.%d"),
                    rollback_trigger="regression_count>0",
                )
            )
        return decisions
