from __future__ import annotations

from ash_hawk.improve_cycle.models import (
    CuratedLesson,
    ImprovementProposal,
    PromotionStatus,
    RiskLevel,
    VerificationReport,
)
from ash_hawk.improve_cycle.promotion import (
    PromotionContext,
    PromotionPolicy,
    decide_promotion_status,
)


def curator_passes(
    proposal: ImprovementProposal,
    min_confidence: float = 0.7,
    require_evidence: bool = True,
) -> bool:
    if proposal.confidence < min_confidence:
        return False
    if require_evidence and not proposal.evidence:
        return False
    if proposal.target_surface.strip() == "":
        return False
    if proposal.risk_level in {RiskLevel.MEDIUM, RiskLevel.HIGH} and not proposal.rollback_notes:
        return False
    return True


def verifier_passes(report: VerificationReport) -> bool:
    if not report.passed:
        return False
    if report.regression_count > 0:
        return False
    if report.variance is not None and report.variance > 0.02:
        return False
    if report.score_delta is not None and report.score_delta <= 0:
        return False
    return True


def promotion_eligible(
    lesson: CuratedLesson,
    report: VerificationReport,
    *,
    policy: PromotionPolicy | None = None,
    context: PromotionContext | None = None,
) -> bool:
    if not lesson.approved:
        return False
    if not verifier_passes(report):
        return False
    effective_policy = policy or PromotionPolicy()
    effective_context = context or PromotionContext()
    status = decide_promotion_status(
        report,
        lesson,
        policy=effective_policy,
        context=effective_context,
    )
    return status in {
        PromotionStatus.PROMOTE_GLOBAL,
        PromotionStatus.PROMOTE_AGENT_SPECIFIC,
        PromotionStatus.PROMOTE_PACK_SPECIFIC,
    }
