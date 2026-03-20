from __future__ import annotations

from ash_hawk.improve_cycle.models import (
    AnalystOutput,
    CuratedLesson,
    ExperimentHistorySummary,
    PromotionDecision,
    PromotionStatus,
    RunArtifactBundle,
    VerificationReport,
)
from ash_hawk.improve_cycle.roles.base import BaseRoleAgent


class HistorianRole(
    BaseRoleAgent[
        tuple[
            RunArtifactBundle,
            AnalystOutput,
            list[CuratedLesson],
            list[VerificationReport],
            list[PromotionDecision],
        ],
        ExperimentHistorySummary,
    ]
):
    def __init__(self) -> None:
        super().__init__("historian", "Record lineage and trend telemetry", "reasoning", 0.1)

    def run(
        self,
        payload: tuple[
            RunArtifactBundle,
            AnalystOutput,
            list[CuratedLesson],
            list[VerificationReport],
            list[PromotionDecision],
        ],
    ) -> ExperimentHistorySummary:
        run_bundle, analyst_output, _lessons, reports, decisions = payload
        promoted = sum(
            1
            for decision in decisions
            if decision.status
            in {
                PromotionStatus.PROMOTE_AGENT_SPECIFIC,
                PromotionStatus.PROMOTE_GLOBAL,
                PromotionStatus.PROMOTE_PACK_SPECIFIC,
            }
        )
        retired = sum(1 for decision in decisions if decision.status == PromotionStatus.RETIRE)
        common = sorted(
            {
                finding.category.value
                for finding in analyst_output.findings
                if finding.category is not None
            }
        )
        return ExperimentHistorySummary(
            history_id=f"history-{run_bundle.run_id}",
            agent_id=run_bundle.agent_id,
            experiment_count=1,
            promoted_lessons=promoted,
            retired_lessons=retired,
            common_failure_categories=common,
            recurring_regressions=[
                report.overall_summary for report in reports if report.regression_count > 0
            ],
            trend_notes=["Lineage captured for improve cycle"],
        )
