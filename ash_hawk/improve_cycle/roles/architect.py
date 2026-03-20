from __future__ import annotations

from uuid import uuid4

from ash_hawk.improve_cycle.models import (
    AnalystOutput,
    EvidenceRef,
    FailureCategory,
    ImprovementProposal,
    ProposalType,
    RiskLevel,
    TriageOutput,
)
from ash_hawk.improve_cycle.roles.base import BaseRoleAgent


class ArchitectRole(BaseRoleAgent[tuple[AnalystOutput, TriageOutput], list[ImprovementProposal]]):
    def __init__(self) -> None:
        super().__init__("architect", "Generate infra-scoped proposals", "reasoning", 0.2)

    def run(self, payload: tuple[AnalystOutput, TriageOutput]) -> list[ImprovementProposal]:
        analyst_output, triage_output = payload
        if triage_output.primary_owner not in {"architect", "both"}:
            return []
        category = triage_output.primary_cause.category
        confidence = min(0.95, max(0.55, triage_output.primary_cause.confidence + 0.05))

        specs: list[tuple[ProposalType, str, str, RiskLevel]] = []
        if category in {
            FailureCategory.TOOL_MISSING,
            FailureCategory.TOOL_INTERFACE_POOR,
            FailureCategory.TOOL_OBSERVABILITY_POOR,
        }:
            specs.append(
                (
                    ProposalType.TOOL_WRAPPER_UPDATE,
                    "tool interface layer",
                    "Harden tool contracts and observability fields",
                    RiskLevel.MEDIUM,
                )
            )
        if category == FailureCategory.HARNESS_LIMITATION:
            specs.append(
                (
                    ProposalType.HARNESS_PATCH,
                    "evaluation harness",
                    "Patch harness behavior to remove blocking execution limits",
                    RiskLevel.MEDIUM,
                )
            )
        if category == FailureCategory.EVAL_GAP:
            specs.append(
                (
                    ProposalType.EVAL_EXPANSION,
                    "eval pack configuration",
                    "Expand eval pack coverage for uncovered failure slice",
                    RiskLevel.LOW,
                )
            )
        if not specs:
            specs.append(
                (
                    ProposalType.OBSERVABILITY_IMPROVEMENT,
                    "observability config",
                    "Improve telemetry granularity for faster diagnosis",
                    RiskLevel.LOW,
                )
            )

        evidence = [
            EvidenceRef(
                artifact_id="analyst",
                kind="risk_area",
                note=", ".join(analyst_output.risk_areas),
            )
        ]
        proposals: list[ImprovementProposal] = []
        for proposal_type, surface, rationale, risk_level in specs[:2]:
            proposals.append(
                ImprovementProposal(
                    proposal_id=f"architect-{uuid4().hex[:8]}",
                    source_role="architect",
                    proposal_type=proposal_type,
                    title=f"Architect proposal: {proposal_type.value.replace('_', ' ')}",
                    summary=analyst_output.summary,
                    rationale=rationale,
                    target_surface=surface,
                    confidence=confidence,
                    risk_level=risk_level,
                    evidence=evidence,
                    expected_benefits=["faster root-cause diagnosis", "lower operational risk"],
                    expected_tradeoffs=["integration complexity"],
                    experiment_hints=["verify latency and token overhead thresholds"],
                    rollback_notes="Disable infra patch via configuration fallback",
                )
            )
        return proposals
