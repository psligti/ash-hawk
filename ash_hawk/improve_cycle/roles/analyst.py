from __future__ import annotations

from ash_hawk.improve_cycle.models import AnalystOutput, MetricValue, TranslatorOutput
from ash_hawk.improve_cycle.roles.base import BaseRoleAgent


class AnalystRole(BaseRoleAgent[TranslatorOutput, AnalystOutput]):
    def __init__(self) -> None:
        super().__init__(
            "analyst", "Identify patterns, risk areas, and measurable weaknesses", "reasoning", 0.1
        )

    def run(self, payload: TranslatorOutput) -> AnalystOutput:
        categories = [
            finding.category.value
            for finding in payload.normalized_findings
            if finding.category is not None
        ]
        recurring = sorted(set(categories))
        metrics = [MetricValue(name="finding_count", value=float(len(payload.normalized_findings)))]
        risk_areas = (
            ["tooling"] if any("tool" in category for category in categories) else ["behavior"]
        )
        return AnalystOutput(
            findings=payload.normalized_findings,
            risk_areas=risk_areas,
            recurring_patterns=recurring,
            efficiency_metrics=metrics,
            summary=f"Analyzed {len(payload.normalized_findings)} findings",
        )
