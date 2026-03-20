from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from ash_hawk.improve_cycle.models import (
    CuratedLesson,
    PromotionDecision,
    PromotionStatus,
    RiskLevel,
    VerificationReport,
)

PROMOTED_STATUSES: set[PromotionStatus] = {
    PromotionStatus.PROMOTE_GLOBAL,
    PromotionStatus.PROMOTE_AGENT_SPECIFIC,
    PromotionStatus.PROMOTE_PACK_SPECIFIC,
}


@dataclass(frozen=True)
class PromotionPolicy:
    low_risk_success_threshold: int = 3
    medium_risk_success_threshold: int = 5
    min_score_delta_for_global: float = 0.05
    retire_after_failures: int = 3


@dataclass(frozen=True)
class PromotionContext:
    consecutive_successes: dict[str, int] = field(default_factory=lambda: cast(dict[str, int], {}))
    consecutive_failures: dict[str, int] = field(default_factory=lambda: cast(dict[str, int], {}))
    cross_pack_validated_lesson_ids: set[str] = field(default_factory=lambda: cast(set[str], set()))
    conflicting_lesson_ids: set[str] = field(default_factory=lambda: cast(set[str], set()))


def decide_promotion_status(
    report: VerificationReport,
    lesson: CuratedLesson,
    *,
    policy: PromotionPolicy,
    context: PromotionContext,
) -> PromotionStatus:
    if report.regression_count > 0:
        return PromotionStatus.ROLLBACK
    if lesson.lesson_id in context.conflicting_lesson_ids:
        return PromotionStatus.HOLD_FOR_MORE_DATA
    if context.consecutive_failures.get(lesson.lesson_id, 0) >= policy.retire_after_failures:
        return PromotionStatus.RETIRE
    if report.recommendation == "hold":
        return PromotionStatus.HOLD_FOR_MORE_DATA
    if not report.passed or report.recommendation == "reject":
        return PromotionStatus.DEMOTE

    threshold = (
        policy.low_risk_success_threshold
        if lesson.risk_level == RiskLevel.LOW
        else policy.medium_risk_success_threshold
    )
    if context.consecutive_successes.get(lesson.lesson_id, 0) < threshold:
        return PromotionStatus.HOLD_FOR_MORE_DATA

    if lesson.lesson_id in context.cross_pack_validated_lesson_ids:
        if report.score_delta is None or report.score_delta >= policy.min_score_delta_for_global:
            return PromotionStatus.PROMOTE_GLOBAL
    if report.score_delta is not None and report.score_delta < policy.min_score_delta_for_global:
        return PromotionStatus.PROMOTE_PACK_SPECIFIC
    return PromotionStatus.PROMOTE_AGENT_SPECIFIC


def decision_scope(status: PromotionStatus, default_scope: str) -> str:
    if status == PromotionStatus.PROMOTE_GLOBAL:
        return "global"
    if status == PromotionStatus.PROMOTE_PACK_SPECIFIC:
        return "pack-specific"
    return default_scope


def has_promoted(decisions: list[PromotionDecision]) -> bool:
    return any(decision.status in PROMOTED_STATUSES for decision in decisions)
