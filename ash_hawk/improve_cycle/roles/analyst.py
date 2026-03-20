from __future__ import annotations

from collections import Counter

from ash_hawk.improve_cycle.models import AnalystOutput, MetricValue, Severity, TranslatorOutput
from ash_hawk.improve_cycle.roles.base import BaseRoleAgent


class AnalystRole(BaseRoleAgent[TranslatorOutput, AnalystOutput]):
    def __init__(self) -> None:
        super().__init__(
            "analyst", "Identify patterns, risk areas, and measurable weaknesses", "reasoning", 0.1
        )

    def run(self, payload: TranslatorOutput) -> AnalystOutput:
        category_values = [
            finding.category.value
            for finding in payload.normalized_findings
            if finding.category is not None
        ]
        category_counts = Counter(category_values)
        recurring = sorted(
            category
            for category, count in category_counts.items()
            if count > 1 or len(category_counts) == 1
        )

        risk_areas: set[str] = set()
        for category in category_values:
            if category.startswith("tool") or category in {"harness_limitation", "eval_gap"}:
                risk_areas.add("tooling")
            if category.startswith("policy") or category.startswith("skill"):
                risk_areas.add("behavior")
            if category in {"nondeterminism", "environmental_flake", "multi_causal"}:
                risk_areas.add("reliability")
        if not risk_areas:
            risk_areas = {"behavior"}

        metrics = [
            MetricValue(name="finding_count", value=float(len(payload.normalized_findings))),
            MetricValue(
                name="high_severity_count",
                value=float(
                    sum(
                        1
                        for finding in payload.normalized_findings
                        if finding.severity in {Severity.HIGH, Severity.CRITICAL}
                    )
                ),
            ),
            MetricValue(name="category_diversity", value=float(len(category_counts))),
        ]

        dominant = category_counts.most_common(1)[0][0] if category_counts else "unknown"
        return AnalystOutput(
            findings=payload.normalized_findings,
            risk_areas=sorted(risk_areas),
            recurring_patterns=recurring,
            efficiency_metrics=metrics,
            summary=(
                f"Analyzed {len(payload.normalized_findings)} findings; "
                f"dominant category={dominant}"
            ),
        )
