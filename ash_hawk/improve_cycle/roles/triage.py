from __future__ import annotations

from typing import Literal

from ash_hawk.improve_cycle.models import (
    AnalystOutput,
    EvidenceRef,
    FailureCategory,
    FailureClassification,
    TriageOutput,
)
from ash_hawk.improve_cycle.roles.base import BaseRoleAgent


class TriageRole(BaseRoleAgent[AnalystOutput, TriageOutput]):
    def __init__(self) -> None:
        super().__init__(
            "triage", "Classify primary cause and route ownership", "deterministic", 0.0
        )

    def run(self, payload: AnalystOutput) -> TriageOutput:
        categories = [
            finding.category for finding in payload.findings if finding.category is not None
        ]
        if not categories:
            primary_category = FailureCategory.MULTI_CAUSAL
        elif len(set(categories)) > 1:
            primary_category = FailureCategory.MULTI_CAUSAL
        else:
            primary_category = categories[0]

        owner: Literal["coach", "architect", "both", "block"] = "coach"
        if primary_category in {
            FailureCategory.TOOL_MISSING,
            FailureCategory.TOOL_INTERFACE_POOR,
            FailureCategory.TOOL_OBSERVABILITY_POOR,
            FailureCategory.HARNESS_LIMITATION,
            FailureCategory.EVAL_GAP,
        }:
            owner = "architect"
        elif primary_category == FailureCategory.MULTI_CAUSAL:
            owner = "both"

        primary = FailureClassification(
            category=primary_category,
            confidence=0.75,
            rationale="Derived from analyst finding categories",
            evidence=[EvidenceRef(artifact_id="analyst", kind="summary", note=payload.summary)],
        )
        secondaries = [
            FailureClassification(
                category=category,
                confidence=0.55,
                rationale="Secondary inferred category",
                evidence=[],
            )
            for category in sorted(set(categories))[:2]
            if category != primary_category
        ]
        return TriageOutput(
            primary_cause=primary,
            secondary_causes=secondaries,
            primary_owner=owner,
            recommended_actions=[
                "route_to_coach" if owner in {"coach", "both"} else "route_to_architect"
            ],
            notes="Escalate multi-causal cases to both roles",
        )
