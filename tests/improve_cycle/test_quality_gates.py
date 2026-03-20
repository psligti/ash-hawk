from __future__ import annotations

from ash_hawk.improve_cycle.models import (
    CuratedLesson,
    EvidenceRef,
    ImprovementProposal,
    ProposalType,
    RiskLevel,
    VerificationReport,
)
from ash_hawk.improve_cycle.promotion import PromotionContext, PromotionPolicy
from ash_hawk.improve_cycle.quality_gates import curator_passes, promotion_eligible, verifier_passes


def _proposal(*, confidence: float = 0.8, risk: RiskLevel = RiskLevel.LOW) -> ImprovementProposal:
    return ImprovementProposal(
        proposal_id="p-1",
        source_role="coach",
        proposal_type=ProposalType.PLAYBOOK_UPDATE,
        title="title",
        summary="summary",
        rationale="why",
        target_surface="policy",
        confidence=confidence,
        risk_level=risk,
        evidence=[],
        rollback_notes="rollback",
    )


def _lesson() -> CuratedLesson:
    return CuratedLesson(
        lesson_id="lesson-1",
        proposal_id="p-1",
        proposal_type=ProposalType.PLAYBOOK_UPDATE,
        title="t",
        summary="s",
        target_surface="policy",
        approved=True,
        curation_notes="ok",
        confidence=0.8,
        risk_level=RiskLevel.LOW,
    )


def _report(*, score_delta: float | None = 0.1, passed: bool = True) -> VerificationReport:
    return VerificationReport(
        verification_id="v-1",
        change_set_id="c-1",
        passed=passed,
        overall_summary="summary",
        score_delta=score_delta,
        variance=0.01,
        regression_count=0,
        recommendation="promote" if passed else "reject",
    )


def test_curator_passes_valid_proposal() -> None:
    proposal = _proposal()
    proposal.evidence = []
    assert curator_passes(proposal, require_evidence=False) is True


def test_curator_passes_rejects_missing_rollback_for_high_risk() -> None:
    proposal = _proposal(risk=RiskLevel.HIGH)
    proposal.rollback_notes = None
    proposal.evidence = [EvidenceRef(artifact_id="x", kind="k")]
    assert curator_passes(proposal) is False


def test_verifier_passes_valid_report() -> None:
    assert verifier_passes(_report()) is True


def test_verifier_passes_rejects_non_positive_delta() -> None:
    assert verifier_passes(_report(score_delta=0.0)) is False


def test_promotion_eligible_uses_policy_and_context() -> None:
    eligible = promotion_eligible(
        _lesson(),
        _report(score_delta=0.2),
        policy=PromotionPolicy(low_risk_success_threshold=1),
        context=PromotionContext(consecutive_successes={"lesson-1": 1}),
    )
    assert eligible is True


def test_promotion_eligible_false_when_not_approved() -> None:
    lesson = _lesson()
    lesson.approved = False
    assert promotion_eligible(lesson, _report()) is False
