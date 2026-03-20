from __future__ import annotations

from uuid import uuid4

from ash_hawk.improve_cycle.models import AdversarialScenario, AnalystOutput, VerificationReport
from ash_hawk.improve_cycle.roles.base import BaseRoleAgent


class AdversaryRole(
    BaseRoleAgent[tuple[AnalystOutput, list[VerificationReport]], list[AdversarialScenario]]
):
    def __init__(self) -> None:
        super().__init__(
            "adversary", "Generate adversarial scenarios to prevent overfitting", "creative", 0.4
        )

    def run(
        self,
        payload: tuple[AnalystOutput, list[VerificationReport]],
    ) -> list[AdversarialScenario]:
        analyst_output, reports = payload
        if not reports and not analyst_output.findings:
            return []
        scenarios: list[AdversarialScenario] = []
        target = analyst_output.risk_areas[0] if analyst_output.risk_areas else "unknown_weakness"

        high_variance = any(
            report.variance is not None and report.variance > 0.015 for report in reports
        )
        has_regressions = any(report.regression_count > 0 for report in reports)
        has_multicausal = any(
            finding.category is not None and finding.category.value == "multi_causal"
            for finding in analyst_output.findings
        )

        scenarios.append(
            AdversarialScenario(
                scenario_id=f"adv-{uuid4().hex[:8]}",
                title="Contradictory evidence stress test",
                target_weakness=target,
                description="Inject conflicting tool outputs and require conservative evidence handling",
                expected_failure_mode="premature confident action",
                evaluation_hooks=["evidence_consistency", "rollback_guardrails"],
            )
        )
        if high_variance:
            scenarios.append(
                AdversarialScenario(
                    scenario_id=f"adv-{uuid4().hex[:8]}",
                    title="Repeatability pressure test",
                    target_weakness="stability",
                    description="Repeat near-identical tasks under jittered timing and tool response order",
                    expected_failure_mode="non-deterministic decision drift",
                    evaluation_hooks=["variance_guard", "decision_consistency"],
                )
            )
        if has_regressions or has_multicausal:
            scenarios.append(
                AdversarialScenario(
                    scenario_id=f"adv-{uuid4().hex[:8]}",
                    title="Cross-surface conflict scenario",
                    target_weakness="change interaction",
                    description="Combine policy and tool perturbations to expose hidden coupling",
                    expected_failure_mode="regression masked by narrow gains",
                    evaluation_hooks=["cross_pack_regression", "rollback_trigger_alignment"],
                )
            )
        return scenarios
