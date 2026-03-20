from __future__ import annotations

from typing import Literal
from uuid import uuid4

from ash_hawk.improve_cycle.models import (
    CuratedLesson,
    ExperimentPlan,
    ProposalType,
    RiskLevel,
    RunArtifactBundle,
)
from ash_hawk.improve_cycle.roles.base import BaseRoleAgent


class ExperimentDesignerRole(
    BaseRoleAgent[tuple[list[CuratedLesson], RunArtifactBundle], list[ExperimentPlan]]
):
    def __init__(
        self,
        *,
        min_verification_runs: int = 3,
        max_latency_delta_pct: float = 15.0,
        max_token_delta_pct: float = 10.0,
        cross_pack_eval_pack_ids: list[str] | None = None,
    ) -> None:
        super().__init__(
            "experiment_designer", "Produce explicit experiment plans", "deterministic", 0.0
        )
        self._min_verification_runs = max(1, min_verification_runs)
        self._max_latency_delta_pct = max_latency_delta_pct
        self._max_token_delta_pct = max_token_delta_pct
        self._cross_pack_eval_pack_ids = cross_pack_eval_pack_ids or []

    def run(self, payload: tuple[list[CuratedLesson], RunArtifactBundle]) -> list[ExperimentPlan]:
        lessons, run_bundle = payload
        plans: list[ExperimentPlan] = []
        for lesson in lessons:
            mode: Literal["isolated", "bundled", "ab", "adversarial", "regression", "cross_pack"]
            mode = "isolated"
            if lesson.risk_level in {RiskLevel.HIGH, RiskLevel.BLOCKED}:
                mode = "ab"
            if lesson.proposal_type in {ProposalType.EVAL_PATCH, ProposalType.EVAL_EXPANSION}:
                mode = "cross_pack"
            if lesson.risk_level == RiskLevel.BLOCKED:
                mode = "adversarial"

            acceptance_criteria = ["score_delta>0", "regression_count==0"]
            if lesson.risk_level in {RiskLevel.HIGH, RiskLevel.BLOCKED}:
                acceptance_criteria.append("variance<=0.015")

            rejection_criteria = ["regression_count>0", "variance>0.02"]
            if lesson.risk_level == RiskLevel.BLOCKED:
                rejection_criteria.append("score_delta<=0")

            plans.append(
                ExperimentPlan(
                    experiment_plan_id=f"plan-{uuid4().hex[:8]}",
                    lesson_ids=[lesson.lesson_id],
                    mode=mode,
                    scenario_ids=run_bundle.scenario_ids,
                    eval_pack_ids=[run_bundle.eval_pack_id] + self._cross_pack_eval_pack_ids,
                    repeat_count=self._min_verification_runs,
                    acceptance_criteria=acceptance_criteria,
                    rejection_criteria=rejection_criteria,
                    rollback_criteria=[
                        f"latency_delta_pct>{self._max_latency_delta_pct}",
                        f"token_delta_pct>{self._max_token_delta_pct}",
                    ],
                    max_latency_delta_pct=self._max_latency_delta_pct,
                    max_token_delta_pct=self._max_token_delta_pct,
                    notes=f"Spec-aligned {mode} validation",
                )
            )
        return plans
