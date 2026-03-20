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


class CoachRole(BaseRoleAgent[tuple[AnalystOutput, TriageOutput], list[ImprovementProposal]]):
    def __init__(self) -> None:
        super().__init__("coach", "Generate behavior-scoped proposals", "reasoning", 0.2)

    def run(self, payload: tuple[AnalystOutput, TriageOutput]) -> list[ImprovementProposal]:
        analyst_output, triage_output = payload
        if triage_output.primary_owner not in {"coach", "both"}:
            return []
        proposal = ImprovementProposal(
            proposal_id=f"coach-{uuid4().hex[:8]}",
            source_role="coach",
            proposal_type=ProposalType.PLAYBOOK_UPDATE,
            title="Tighten investigation order and summary discipline",
            summary=analyst_output.summary,
            rationale="Recurring behavior failures indicate missing ordering discipline",
            target_surface="agent policy file",
            confidence=0.78,
            risk_level=RiskLevel.MEDIUM,
            evidence=[
                EvidenceRef(
                    artifact_id="triage",
                    kind="classification",
                    note=triage_output.primary_cause.category.value,
                )
            ],
            expected_benefits=["more consistent investigation", "fewer missed constraints"],
            expected_tradeoffs=["slightly longer first pass"],
            experiment_hints=["run isolated policy update test"],
            rollback_notes="Revert policy section if regression count increases",
        )
        return [proposal]
