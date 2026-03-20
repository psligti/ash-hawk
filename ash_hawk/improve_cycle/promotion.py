from __future__ import annotations

from ash_hawk.improve_cycle.models import PromotionDecision, PromotionStatus, VerificationReport


def decide_promotion_status(report: VerificationReport) -> PromotionStatus:
    if report.recommendation == "promote" and report.passed and report.regression_count == 0:
        return PromotionStatus.PROMOTE_AGENT_SPECIFIC
    if report.recommendation == "hold":
        return PromotionStatus.HOLD_FOR_MORE_DATA
    return PromotionStatus.DEMOTE


def has_promoted(decisions: list[PromotionDecision]) -> bool:
    return any(
        decision.status
        in {
            PromotionStatus.PROMOTE_GLOBAL,
            PromotionStatus.PROMOTE_AGENT_SPECIFIC,
            PromotionStatus.PROMOTE_PACK_SPECIFIC,
        }
        for decision in decisions
    )
