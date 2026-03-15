"""Curator role for proposal curation and lesson creation."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from ash_hawk.contracts import CuratedLesson, ImprovementProposal


class CuratorRole:
    def curate(
        self, proposals: list[ImprovementProposal], auto_appro: bool = True
    ) -> list[CuratedLesson]:
        lessons = []

        for proposal in proposals:
            if not auto_appro:
                continue

            if proposal.status != "pending":
                continue

            lesson = CuratedLesson(
                lesson_id=f"lesson-{uuid4().hex[:8]}",
                source_proposal_id=proposal.proposal_id,
                applies_to_agents=[proposal.target_agent],
                lesson_type=proposal.proposal_type,
                title=proposal.title,
                description=proposal.rationale,
                lesson_payload=proposal.diff_payload,
                validation_status="approved",
                version=1,
                created_at=datetime.now(UTC),
            )
            lessons.append(lesson)

        return lessons

    def reject(self, proposal: ImprovementProposal, reason: str) -> ImprovementProposal:
        proposal.status = "rejected"
        proposal.rejection_reason = reason
        proposal.reviewed_at = datetime.now(UTC)
        return proposal

    def defer(self, proposal: ImprovementProposal) -> ImprovementProposal:
        proposal.status = "pending"
        proposal.reviewed_at = datetime.now(UTC)
        return proposal
