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


class CoachRole(BaseRoleAgent[tuple[AnalystOutput, TriageOutput], list[ImprovementProposal]]):
    def __init__(self) -> None:
        super().__init__("coach", "Generate behavior-scoped proposals", "reasoning", 0.2)

    def run(self, payload: tuple[AnalystOutput, TriageOutput]) -> list[ImprovementProposal]:
        analyst_output, triage_output = payload
        if triage_output.primary_owner not in {"coach", "both"}:
            return []
        base_confidence = min(0.95, max(0.55, triage_output.primary_cause.confidence + 0.1))
        primary_category = triage_output.primary_cause.category

        proposal_specs: list[tuple[ProposalType, str, str, RiskLevel]] = []
        if primary_category in {
            FailureCategory.POLICY_ORDERING,
            FailureCategory.POLICY_GUARDRAIL,
        }:
            proposal_specs.append(
                (
                    ProposalType.POLICY_REORDER,
                    "agent policy file",
                    "Reorder policy checks to enforce guardrails before action",
                    RiskLevel.MEDIUM,
                )
            )
        if primary_category in {FailureCategory.SKILL_MISSING, FailureCategory.SKILL_WEAK}:
            proposal_specs.append(
                (
                    ProposalType.SKILL_REVISE,
                    "skill catalog",
                    "Revise skill guidance for missing/weak behavior pattern",
                    RiskLevel.MEDIUM,
                )
            )
        if not proposal_specs:
            proposal_specs.append(
                (
                    ProposalType.BEHAVIORAL_CHECKLIST,
                    "agent playbook",
                    "Add deterministic behavior checklist for repeated failure mode",
                    RiskLevel.LOW,
                )
            )

        evidence = [
            EvidenceRef(
                artifact_id="triage",
                kind="classification",
                note=triage_output.primary_cause.category.value,
            )
        ]
        proposals: list[ImprovementProposal] = []
        for proposal_type, surface, rationale, risk_level in proposal_specs[:2]:
            proposals.append(
                ImprovementProposal(
                    proposal_id=f"coach-{uuid4().hex[:8]}",
                    source_role="coach",
                    proposal_type=proposal_type,
                    title=f"Coach proposal: {proposal_type.value.replace('_', ' ')}",
                    summary=analyst_output.summary,
                    rationale=rationale,
                    target_surface=surface,
                    confidence=base_confidence,
                    risk_level=risk_level,
                    evidence=evidence,
                    expected_benefits=[
                        "more consistent behavior",
                        "clearer adherence to instructions",
                    ],
                    expected_tradeoffs=["slightly higher prompt verbosity"],
                    experiment_hints=["run isolated behavior validation scenarios"],
                    rollback_notes="Revert behavior rule changes if regressions increase",
                )
            )
        return proposals
