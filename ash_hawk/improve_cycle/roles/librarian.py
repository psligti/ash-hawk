from __future__ import annotations

from uuid import uuid4

from ash_hawk.improve_cycle.models import (
    CuratedLesson,
    EvidenceRef,
    KnowledgeEntry,
    PromotionDecision,
    PromotionStatus,
)
from ash_hawk.improve_cycle.roles.base import BaseRoleAgent


class LibrarianRole(
    BaseRoleAgent[tuple[list[PromotionDecision], list[CuratedLesson]], list[KnowledgeEntry]]
):
    def __init__(self) -> None:
        super().__init__(
            "librarian", "Convert promoted lessons into reusable knowledge", "reasoning", 0.1
        )

    def run(
        self,
        payload: tuple[list[PromotionDecision], list[CuratedLesson]],
    ) -> list[KnowledgeEntry]:
        decisions, lessons = payload
        promoted_ids = {
            decision.lesson_id
            for decision in decisions
            if decision.status
            in {
                PromotionStatus.PROMOTE_AGENT_SPECIFIC,
                PromotionStatus.PROMOTE_GLOBAL,
                PromotionStatus.PROMOTE_PACK_SPECIFIC,
            }
        }
        entries: list[KnowledgeEntry] = []
        for lesson in lessons:
            if lesson.lesson_id not in promoted_ids:
                continue
            entries.append(
                KnowledgeEntry(
                    knowledge_id=f"knowledge-{uuid4().hex[:8]}",
                    lesson_id=lesson.lesson_id,
                    kind="strategy_playbook_entry",
                    title=lesson.title,
                    summary=lesson.summary,
                    applicability_conditions=[f"target_surface={lesson.target_surface}"],
                    anti_patterns=["promote without verification"],
                    references=[EvidenceRef(artifact_id=lesson.proposal_id, kind="lineage")],
                )
            )
        return entries
