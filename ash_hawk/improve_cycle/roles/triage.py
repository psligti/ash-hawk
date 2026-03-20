from __future__ import annotations

from collections import defaultdict
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
        severity_weight = {"low": 1.0, "medium": 2.0, "high": 3.0, "critical": 4.0}
        weights: dict[FailureCategory, float] = defaultdict(float)
        evidence_by_category: dict[FailureCategory, list[EvidenceRef]] = defaultdict(list)
        for finding in payload.findings:
            if finding.category is None:
                continue
            weights[finding.category] += severity_weight[finding.severity.value]
            evidence_by_category[finding.category].extend(finding.evidence)

        if not weights:
            primary_category = FailureCategory.MULTI_CAUSAL
            confidence = 0.5
        else:
            ranked = sorted(weights.items(), key=lambda item: item[1], reverse=True)
            if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                primary_category = FailureCategory.MULTI_CAUSAL
                confidence = 0.6
            else:
                primary_category = ranked[0][0]
                total_weight = sum(weights.values())
                confidence = min(0.95, max(0.55, ranked[0][1] / total_weight))

        owner: Literal["coach", "architect", "both", "block"] = "coach"
        if primary_category in {
            FailureCategory.TOOL_MISSING,
            FailureCategory.TOOL_INTERFACE_POOR,
            FailureCategory.TOOL_OBSERVABILITY_POOR,
            FailureCategory.HARNESS_LIMITATION,
            FailureCategory.EVAL_GAP,
        }:
            owner = "architect"
        elif primary_category in {
            FailureCategory.NONDETERMINISM,
            FailureCategory.ENVIRONMENTAL_FLAKE,
        }:
            owner = "block"
        elif primary_category == FailureCategory.MULTI_CAUSAL:
            owner = "both"

        primary = FailureClassification(
            category=primary_category,
            confidence=confidence,
            rationale="Derived from analyst finding categories",
            evidence=evidence_by_category.get(primary_category)
            or [EvidenceRef(artifact_id="analyst", kind="summary", note=payload.summary)],
        )
        secondary_candidates = sorted(
            (item for item in weights.items() if item[0] != primary_category),
            key=lambda item: item[1],
            reverse=True,
        )
        secondaries = [
            FailureClassification(
                category=category,
                confidence=min(0.8, max(0.45, weight / max(1.0, sum(weights.values())))),
                rationale="Secondary inferred category",
                evidence=evidence_by_category.get(category, []),
            )
            for category, weight in secondary_candidates[:2]
        ]

        recommended_actions: list[str] = []
        if owner in {"coach", "both"}:
            recommended_actions.append("route_to_coach")
        if owner in {"architect", "both"}:
            recommended_actions.append("route_to_architect")
        if owner == "block":
            recommended_actions.append("block_promotion_pending_stability")

        return TriageOutput(
            primary_cause=primary,
            secondary_causes=secondaries,
            primary_owner=owner,
            recommended_actions=recommended_actions,
            notes="Escalate multi-causal cases to both roles",
        )
