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
        target = analyst_output.risk_areas[0] if analyst_output.risk_areas else "unknown_weakness"
        suspicious = any(
            report.variance is not None and report.variance > 0.015 for report in reports
        )
        if not suspicious and not analyst_output.findings:
            return []
        return [
            AdversarialScenario(
                scenario_id=f"adv-{uuid4().hex[:8]}",
                title="Contradictory evidence stress test",
                target_weakness=target,
                description="Inject conflicting tool outputs and require conservative evidence handling",
                expected_failure_mode="premature confident action",
                evaluation_hooks=["evidence_consistency", "rollback_guardrails"],
            )
        ]
