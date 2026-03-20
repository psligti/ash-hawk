from __future__ import annotations

from typing import Literal

from ash_hawk.improve_cycle.models import (
    CuratedLesson,
    PromotionDecision,
    PromotionStatus,
    ProposalType,
    RiskLevel,
    VerificationReport,
)
from ash_hawk.improve_cycle.promotion import (
    PromotionContext,
    PromotionPolicy,
    decide_promotion_status,
    decision_scope,
    has_promoted,
)


def _lesson(*, risk_level: RiskLevel = RiskLevel.LOW) -> CuratedLesson:
    return CuratedLesson(
        lesson_id="lesson-1",
        proposal_id="proposal-1",
        proposal_type=ProposalType.PLAYBOOK_UPDATE,
        title="Title",
        summary="Summary",
        target_surface="policy",
        approved=True,
        curation_notes="ok",
        confidence=0.8,
        risk_level=risk_level,
    )


def _report(
    *,
    passed: bool = True,
    recommendation: Literal["reject", "hold", "promote"] = "promote",
    regression_count: int = 0,
    score_delta: float | None = 0.1,
) -> VerificationReport:
    return VerificationReport(
        verification_id="verify-1",
        change_set_id="changeset-1",
        passed=passed,
        overall_summary="report",
        score_delta=score_delta,
        variance=0.01,
        regression_count=regression_count,
        recommendation=recommendation,
    )


def test_decide_promotion_status_rollback_on_regression() -> None:
    status = decide_promotion_status(
        _report(regression_count=1),
        _lesson(),
        policy=PromotionPolicy(),
        context=PromotionContext(),
    )
    assert status == PromotionStatus.ROLLBACK


def test_decide_promotion_status_retire_after_failures() -> None:
    status = decide_promotion_status(
        _report(passed=False, recommendation="reject"),
        _lesson(),
        policy=PromotionPolicy(retire_after_failures=2),
        context=PromotionContext(consecutive_failures={"lesson-1": 2}),
    )
    assert status == PromotionStatus.RETIRE


def test_decide_promotion_status_global_when_cross_pack_and_threshold_met() -> None:
    status = decide_promotion_status(
        _report(score_delta=0.2),
        _lesson(),
        policy=PromotionPolicy(low_risk_success_threshold=2, min_score_delta_for_global=0.05),
        context=PromotionContext(
            consecutive_successes={"lesson-1": 2},
            cross_pack_validated_lesson_ids={"lesson-1"},
        ),
    )
    assert status == PromotionStatus.PROMOTE_GLOBAL


def test_decide_promotion_status_pack_specific_for_small_gain() -> None:
    status = decide_promotion_status(
        _report(score_delta=0.02),
        _lesson(),
        policy=PromotionPolicy(low_risk_success_threshold=1, min_score_delta_for_global=0.05),
        context=PromotionContext(consecutive_successes={"lesson-1": 1}),
    )
    assert status == PromotionStatus.PROMOTE_PACK_SPECIFIC


def test_decide_promotion_status_hold_until_threshold_met() -> None:
    status = decide_promotion_status(
        _report(score_delta=0.2),
        _lesson(risk_level=RiskLevel.MEDIUM),
        policy=PromotionPolicy(medium_risk_success_threshold=3),
        context=PromotionContext(consecutive_successes={"lesson-1": 2}),
    )
    assert status == PromotionStatus.HOLD_FOR_MORE_DATA


def test_decide_promotion_status_holds_on_conflict() -> None:
    status = decide_promotion_status(
        _report(score_delta=0.2),
        _lesson(),
        policy=PromotionPolicy(low_risk_success_threshold=1),
        context=PromotionContext(
            consecutive_successes={"lesson-1": 1},
            conflicting_lesson_ids={"lesson-1"},
        ),
    )
    assert status == PromotionStatus.HOLD_FOR_MORE_DATA


def test_decision_scope_maps_global_and_pack_specific() -> None:
    assert decision_scope(PromotionStatus.PROMOTE_GLOBAL, "agent-specific") == "global"
    assert (
        decision_scope(PromotionStatus.PROMOTE_PACK_SPECIFIC, "agent-specific") == "pack-specific"
    )
    assert decision_scope(PromotionStatus.DEMOTE, "agent-specific") == "agent-specific"


def test_has_promoted_detects_promoted_statuses() -> None:
    promoted = PromotionDecision(
        decision_id="d-1",
        lesson_id="lesson-1",
        status=PromotionStatus.PROMOTE_AGENT_SPECIFIC,
        scope="agent-specific",
        reason="ok",
    )
    held = PromotionDecision(
        decision_id="d-2",
        lesson_id="lesson-2",
        status=PromotionStatus.HOLD_FOR_MORE_DATA,
        scope="agent-specific",
        reason="hold",
    )
    assert has_promoted([held, promoted]) is True
    assert has_promoted([held]) is False
