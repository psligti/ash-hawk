from __future__ import annotations

from uuid import uuid4

from ash_hawk.improve_cycle.models import (
    AnalystOutput,
    EvidenceRef,
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
        proposal = ImprovementProposal(
            proposal_id=f"architect-{uuid4().hex[:8]}",
            source_role="architect",
            proposal_type=ProposalType.OBSERVABILITY_IMPROVEMENT,
            title="Add stage-level telemetry fields",
            summary=analyst_output.summary,
            rationale="Infra observability gaps hinder durable verification",
            target_surface="observability config",
            confidence=0.8,
            risk_level=RiskLevel.LOW,
            evidence=[
                EvidenceRef(
                    artifact_id="analyst",
                    kind="risk_area",
                    note=", ".join(analyst_output.risk_areas),
                )
            ],
            expected_benefits=["faster root-cause diagnosis", "better lineage"],
            expected_tradeoffs=["small event payload increase"],
            experiment_hints=["verify latency overhead stays below threshold"],
            rollback_notes="Disable added fields via config toggle",
        )
        return [proposal]
