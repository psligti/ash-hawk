from __future__ import annotations

from ash_hawk.improve_cycle.configuration import ImproveCyclePromotionConfig
from ash_hawk.improve_cycle.models import (
    CuratedLesson,
    PromotionStatus,
    ProposalType,
    RiskLevel,
    VerificationReport,
)
from ash_hawk.improve_cycle.promotion import PromotionContext
from ash_hawk.improve_cycle.roles.promotion_manager import PromotionManagerRole


def _lesson() -> CuratedLesson:
    return CuratedLesson(
        lesson_id="lesson-1",
        proposal_id="proposal-1",
        proposal_type=ProposalType.PLAYBOOK_UPDATE,
        title="Title",
        summary="Summary",
        target_surface="policy",
        approved=True,
        curation_notes="ok",
        confidence=0.9,
        risk_level=RiskLevel.LOW,
    )


def _report(*, score_delta: float = 0.1, recommendation: str = "promote") -> VerificationReport:
    return VerificationReport(
        verification_id="verify-1",
        change_set_id="changeset-1",
        passed=recommendation != "reject",
        overall_summary="summary",
        score_delta=score_delta,
        variance=0.01,
        regression_count=0,
        recommendation=recommendation,
    )


def test_promotion_manager_promotes_global_with_cross_pack_context() -> None:
    role = PromotionManagerRole(
        config=ImproveCyclePromotionConfig(low_risk_success_threshold=1),
        default_scope="agent-specific",
    )
    decisions = role.run(
        (
            _report(score_delta=0.2),
            [_lesson()],
            PromotionContext(
                consecutive_successes={"lesson-1": 1},
                cross_pack_validated_lesson_ids={"lesson-1"},
            ),
        )
    )
    assert decisions
    assert decisions[0].status == PromotionStatus.PROMOTE_GLOBAL
    assert decisions[0].scope == "global"


def test_promotion_manager_holds_when_threshold_not_met() -> None:
    role = PromotionManagerRole(
        config=ImproveCyclePromotionConfig(low_risk_success_threshold=2),
        default_scope="agent-specific",
    )
    decisions = role.run(
        (
            _report(score_delta=0.2),
            [_lesson()],
            PromotionContext(consecutive_successes={"lesson-1": 1}),
        )
    )
    assert decisions[0].status == PromotionStatus.HOLD_FOR_MORE_DATA


def test_promotion_manager_rolls_back_on_regression() -> None:
    role = PromotionManagerRole(
        config=ImproveCyclePromotionConfig(), default_scope="agent-specific"
    )
    report = VerificationReport(
        verification_id="verify-1",
        change_set_id="changeset-1",
        passed=False,
        overall_summary="regressed",
        score_delta=-0.1,
        variance=0.01,
        regression_count=2,
        recommendation="reject",
    )
    decisions = role.run((report, [_lesson()], PromotionContext()))
    assert decisions[0].status == PromotionStatus.ROLLBACK
