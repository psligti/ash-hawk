from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pydantic as pd

from ash_hawk.improve.diagnose import Diagnosis

if TYPE_CHECKING:
    from ash_hawk.improve.lesson_store import LessonStore

logger = logging.getLogger(__name__)


class RankedHypothesis(pd.BaseModel):
    model_config = pd.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    diagnosis: Diagnosis = pd.Field(description="The underlying diagnosis")
    rank: int = pd.Field(description="Rank position (1 = best)")
    novelty_score: float = pd.Field(ge=0.0, le=1.0, description="Novelty score")
    confidence: float = pd.Field(ge=0.0, le=1.0, description="LLM confidence")
    estimated_impact: float = pd.Field(ge=0.0, le=1.0, description="Estimated impact")
    already_tried: bool = pd.Field(description="Whether this approach was tried before")
    similar_lesson_ids: list[str] = pd.Field(
        default_factory=list, description="IDs of similar past lessons"
    )


class HypothesisRanking(pd.BaseModel):
    model_config = pd.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    hypotheses: list[RankedHypothesis] = pd.Field(
        default_factory=list, description="Ranked hypotheses"
    )
    total_candidates: int = pd.Field(description="Total before filtering")
    filtered_as_tried: int = pd.Field(description="Number filtered as already tried")
    ranked_count: int = pd.Field(description="Number after ranking")


class HypothesisRanker:
    """Rank and filter diagnosis hypotheses for the improvement loop.

    Uses measured delta from past lessons instead of heuristic scoring.
    Falls back to LLM confidence when no past data exists.
    """

    def __init__(self, lesson_store: LessonStore | None = None) -> None:
        self._lesson_store = lesson_store

    def rank(
        self,
        diagnoses: list[Diagnosis],
        filter_tried: bool = True,
    ) -> HypothesisRanking:
        logger.info("Ranking %d hypotheses", len(diagnoses))

        candidates: list[RankedHypothesis] = []
        for diagnosis in diagnoses:
            novelty_score, already_tried, similar_ids = self._compute_novelty(diagnosis)
            impact = self._compute_impact(diagnosis)

            candidates.append(
                RankedHypothesis(
                    diagnosis=diagnosis,
                    rank=0,
                    novelty_score=novelty_score,
                    confidence=diagnosis.confidence,
                    estimated_impact=impact,
                    already_tried=already_tried,
                    similar_lesson_ids=similar_ids,
                )
            )
            logger.debug(
                "Hypothesis trial=%s impact=%.2f novel=%.2f tried=%s",
                diagnosis.trial_id,
                impact,
                novelty_score,
                already_tried,
            )

        total_candidates = len(candidates)
        filtered_as_tried = 0
        if filter_tried:
            before = len(candidates)
            candidates = [c for c in candidates if not c.already_tried]
            filtered_as_tried = before - len(candidates)

        if filtered_as_tried > 0:
            logger.info("Filtered %d already-tried hypotheses", filtered_as_tried)

        candidates.sort(
            key=lambda h: (h.estimated_impact, h.novelty_score, h.confidence),
            reverse=True,
        )

        candidates = self._ensure_diversity(candidates)

        for i, hyp in enumerate(candidates):
            hyp.rank = i + 1

        if candidates:
            top = candidates[0]
            logger.info(
                "Top hypothesis: rank=%d trial=%s impact=%.2f novel=%.2f",
                top.rank,
                top.diagnosis.trial_id,
                top.estimated_impact,
                top.novelty_score,
            )

        return HypothesisRanking(
            hypotheses=candidates,
            total_candidates=total_candidates,
            filtered_as_tried=filtered_as_tried,
            ranked_count=len(candidates),
        )

    def _compute_novelty(self, diagnosis: Diagnosis) -> tuple[float, bool, list[str]]:
        if self._lesson_store is None:
            return (1.0, False, [])

        summary = diagnosis.failure_summary
        root_cause = diagnosis.root_cause

        try:
            already_tried = self._lesson_store.has_been_tried(summary, root_cause)
        except Exception:
            logger.warning(
                "lesson_store.has_been_tried failed for trial=%s",
                diagnosis.trial_id,
                exc_info=True,
            )
            already_tried = False

        try:
            similar = self._lesson_store.find_similar(summary, root_cause)
            similar_ids = [str(getattr(s, "lesson_id", s)) for s in similar]
        except Exception:
            logger.warning(
                "lesson_store.find_similar failed for trial=%s",
                diagnosis.trial_id,
                exc_info=True,
            )
            similar_ids = []

        if already_tried:
            novelty_score = 0.0
        elif similar_ids:
            novelty_score = max(0.1, 1.0 - len(similar_ids) * 0.2)
        else:
            novelty_score = 1.0

        return (novelty_score, already_tried, similar_ids)

    def _compute_impact(self, diagnosis: Diagnosis) -> float:
        """Estimate impact from past measured deltas on similar files.

        Falls back to LLM confidence when no past data exists.
        """
        if self._lesson_store is None:
            return diagnosis.confidence

        try:
            past = self._lesson_store.load_for_target(diagnosis.target_files)
        except Exception:
            logger.warning(
                "lesson_store.load_for_target failed for trial=%s",
                diagnosis.trial_id,
                exc_info=True,
            )
            return diagnosis.confidence

        if not past:
            return diagnosis.confidence

        kept = [lesson for lesson in past if lesson.outcome == "kept"]
        if kept:
            return max(0.0, min(1.0, sum(lesson.score_delta for lesson in kept) / len(kept)))

        return max(0.0, diagnosis.confidence - 0.2)

    def _ensure_diversity(self, hypotheses: list[RankedHypothesis]) -> list[RankedHypothesis]:
        seen_files: set[str] = set()
        kept: list[RankedHypothesis] = []
        demoted: list[RankedHypothesis] = []

        for hyp in hypotheses:
            primary_file = hyp.diagnosis.target_files[0] if hyp.diagnosis.target_files else ""
            if primary_file and primary_file in seen_files:
                logger.debug(
                    "Demoting hypothesis trial=%s (duplicate file=%s)",
                    hyp.diagnosis.trial_id,
                    primary_file,
                )
                demoted.append(hyp)
            else:
                if primary_file:
                    seen_files.add(primary_file)
                kept.append(hyp)

        return kept + demoted
