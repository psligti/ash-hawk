from __future__ import annotations

from ash_hawk.improve_cycle.models import PromotionDecision


class LineageService:
    def lesson_ids(self, decisions: list[PromotionDecision]) -> list[str]:
        return [decision.lesson_id for decision in decisions]
