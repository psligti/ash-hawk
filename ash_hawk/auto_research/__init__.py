"""Auto-research modules for Ash Hawk improvement cycles."""

from ash_hawk.auto_research.convergence import (
    ConvergenceDetector,
    ConvergenceReason,
    ConvergenceResult,
    ScoreRecord,
)
from ash_hawk.auto_research.cycle_runner import (
    CycleConfig,
    CycleResult,
    CycleStatus,
    run_cycle,
)
from ash_hawk.auto_research.knowledge_promotion import (
    KnowledgePromoter,
    PromotedLesson,
    PromotionCriteria,
)

__all__ = [
    "ConvergenceDetector",
    "ConvergenceReason",
    "ConvergenceResult",
    "CycleConfig",
    "CycleResult",
    "CycleStatus",
    "KnowledgePromoter",
    "PromotedLesson",
    "PromotionCriteria",
    "ScoreRecord",
    "run_cycle",
]
