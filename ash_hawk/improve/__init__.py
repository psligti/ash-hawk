"""Basic improvement loop for Ash Hawk evals."""

from ash_hawk.improve.diagnose import Diagnosis, diagnose_failures
from ash_hawk.improve.hypothesis_ranker import (
    HypothesisRanker,
    HypothesisRanking,
    RankedHypothesis,
)
from ash_hawk.improve.lesson_store import Lesson, LessonStore
from ash_hawk.improve.loop import ImprovementResult, improve
from ash_hawk.improve.patch import ProposedPatch, propose_patch, write_patch

__all__ = [
    "Diagnosis",
    "HypothesisRanker",
    "HypothesisRanking",
    "ImprovementResult",
    "Lesson",
    "LessonStore",
    "ProposedPatch",
    "RankedHypothesis",
    "diagnose_failures",
    "improve",
    "propose_patch",
    "write_patch",
]
