from __future__ import annotations

from uuid import uuid4

from ash_hawk.improve_cycle.models import CuratedLesson, ImprovementProposal, RiskLevel
from ash_hawk.improve_cycle.roles.base import BaseRoleAgent


class CuratorRole(BaseRoleAgent[list[ImprovementProposal], list[CuratedLesson]]):
    def __init__(self, min_confidence: float = 0.7) -> None:
        super().__init__("curator", "Gate proposals before experimentation", "deterministic", 0.0)
        self._min_confidence = min_confidence

    def run(self, payload: list[ImprovementProposal]) -> list[CuratedLesson]:
        best_by_surface: dict[tuple[str, str], ImprovementProposal] = {}
        for proposal in payload:
            key = (proposal.target_surface.strip().lower(), proposal.proposal_type.value)
            current = best_by_surface.get(key)
            if current is None or proposal.confidence > current.confidence:
                best_by_surface[key] = proposal

        lessons: list[CuratedLesson] = []
        for proposal in best_by_surface.values():
            if proposal.confidence < self._min_confidence:
                continue
            if not proposal.evidence:
                continue
            if (
                proposal.risk_level in {RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.BLOCKED}
                and not proposal.rollback_notes
            ):
                continue
            lessons.append(
                CuratedLesson(
                    lesson_id=f"lesson-{uuid4().hex[:8]}",
                    proposal_id=proposal.proposal_id,
                    proposal_type=proposal.proposal_type,
                    title=proposal.title,
                    summary=proposal.summary,
                    target_surface=proposal.target_surface,
                    approved=True,
                    curation_notes=(
                        "Passed confidence/evidence gates and selected as best proposal "
                        "for target surface"
                    ),
                    confidence=proposal.confidence,
                    risk_level=proposal.risk_level,
                    lineage=[proposal.proposal_id],
                )
            )
        return lessons
