from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pydantic as pd

from ash_hawk.improve.diagnose import Diagnosis
from ash_hawk.improve.targeting import diagnosis_targets_allowed

if TYPE_CHECKING:
    from ash_hawk.improve.lesson_store import LessonStore
    from ash_hawk.improve.memory_store import MemoryStore

logger = logging.getLogger(__name__)


class RankedHypothesis(pd.BaseModel):
    model_config = pd.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    diagnosis: Diagnosis = pd.Field(description="The underlying diagnosis")
    rank: int = pd.Field(description="Rank position (1 = best)")
    novelty_score: float = pd.Field(ge=0.0, le=1.0, description="Novelty score")
    confidence: float = pd.Field(ge=0.0, le=1.0, description="LLM confidence")
    estimated_impact: float = pd.Field(ge=0.0, le=1.0, description="Estimated impact")
    breadth_penalty: float = pd.Field(ge=0.0, le=1.0, description="Penalty for broad changes")
    overlap_penalty: float = pd.Field(
        ge=0.0, le=1.0, description="Penalty for overlap with resolved work"
    )
    total_score: float = pd.Field(ge=0.0, le=1.0, description="Composite ranking score")
    already_tried: bool = pd.Field(description="Whether this approach was tried before")
    similar_lesson_ids: list[str] = pd.Field(
        default_factory=list, description="IDs of similar past lessons"
    )
    semantic_penalty: float = pd.Field(default=0.0, ge=0.0, le=1.0)
    semantic_boost: float = pd.Field(default=0.0, ge=0.0, le=1.0)
    calibration_factor: float = pd.Field(default=1.0, ge=0.0)


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

    def __init__(
        self,
        lesson_store: LessonStore | None = None,
        memory_store: MemoryStore | None = None,
    ) -> None:
        self._lesson_store = lesson_store
        self._memory_store = memory_store

    def rank(
        self,
        diagnoses: list[Diagnosis],
        filter_tried: bool = True,
        suppress_target_files: set[str] | None = None,
        suppress_families: set[str] | None = None,
        regression_target_files: set[str] | None = None,
        regression_families: set[str] | None = None,
        allowed_target_files: set[str] | None = None,
    ) -> HypothesisRanking:
        logger.info("Ranking %d hypotheses", len(diagnoses))

        suppressed_files = suppress_target_files or set()
        suppressed_families = suppress_families or set()
        regressed_files = regression_target_files or set()
        regressed_families = regression_families or set()
        allowed_files = allowed_target_files or set()
        candidates: list[RankedHypothesis] = []
        for diagnosis in diagnoses:
            if allowed_files and diagnosis.target_files:
                if not diagnosis_targets_allowed(diagnosis, allowed_files):
                    continue
            novelty_score, already_tried, similar_ids = self._compute_novelty(diagnosis)
            impact, calibration_factor = self._compute_impact(diagnosis)
            breadth_penalty = self._compute_breadth_penalty(diagnosis)
            overlap_penalty = self._compute_overlap_penalty(
                diagnosis,
                suppressed_files=suppressed_files,
                suppressed_families=suppressed_families,
                regression_files=regressed_files,
                regression_families=regressed_families,
            )
            semantic_penalty, semantic_boost = self._memory_adjustment(diagnosis)
            total_score = max(
                0.0,
                min(
                    1.0,
                    impact * 0.55
                    + novelty_score * 0.25
                    + diagnosis.confidence * 0.20
                    + semantic_boost
                    - breadth_penalty
                    - overlap_penalty
                    - semantic_penalty,
                ),
            )

            candidates.append(
                RankedHypothesis(
                    diagnosis=diagnosis,
                    rank=0,
                    novelty_score=novelty_score,
                    confidence=diagnosis.confidence,
                    estimated_impact=impact,
                    breadth_penalty=breadth_penalty,
                    overlap_penalty=overlap_penalty,
                    total_score=total_score,
                    already_tried=already_tried,
                    similar_lesson_ids=similar_ids,
                    semantic_penalty=semantic_penalty,
                    semantic_boost=semantic_boost,
                    calibration_factor=calibration_factor,
                )
            )
            logger.debug(
                "Hypothesis trial=%s impact=%.2f novel=%.2f tried=%s",
                diagnosis.trial_id,
                total_score,
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
            key=lambda h: (h.total_score, h.estimated_impact, h.novelty_score, h.confidence),
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
                top.total_score,
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

    def _compute_impact(self, diagnosis: Diagnosis) -> tuple[float, float]:
        """Estimate impact from past measured deltas on similar files.

        Falls back to LLM confidence when no past data exists.
        """
        calibration = 1.0
        if self._memory_store is not None:
            calibration = self._memory_store.calibration_factor(
                diagnosis.family,
                agent_name="default",
            )

        if self._lesson_store is None:
            return (max(0.0, min(1.0, diagnosis.confidence * calibration)), calibration)

        try:
            past = self._lesson_store.load_for_target(diagnosis.target_files)
        except Exception:
            logger.warning(
                "lesson_store.load_for_target failed for trial=%s",
                diagnosis.trial_id,
                exc_info=True,
            )
            return (max(0.0, min(1.0, diagnosis.confidence * calibration)), calibration)

        if not past:
            return (max(0.0, min(1.0, diagnosis.confidence * calibration)), calibration)

        kept = [lesson for lesson in past if lesson.outcome == "kept"]
        if kept:
            base_impact = max(0.0, min(1.0, sum(lesson.score_delta for lesson in kept) / len(kept)))
            return (max(0.0, min(1.0, base_impact * calibration)), calibration)

        return (max(0.0, min(1.0, (diagnosis.confidence - 0.2) * calibration)), calibration)

    def _memory_adjustment(self, diagnosis: Diagnosis) -> tuple[float, float]:
        if self._memory_store is None:
            return (0.0, 0.0)
        try:
            return self._memory_store.semantic_adjustment(
                diagnosis.family,
                diagnosis.target_files,
            )
        except Exception:
            logger.warning(
                "memory_store.semantic_adjustment failed for trial=%s",
                diagnosis.trial_id,
                exc_info=True,
            )
            return (0.0, 0.0)

    def _ensure_diversity(self, hypotheses: list[RankedHypothesis]) -> list[RankedHypothesis]:
        seen_keys: set[tuple[str, str]] = set()
        kept: list[RankedHypothesis] = []
        demoted: list[RankedHypothesis] = []

        for hyp in hypotheses:
            primary_file = hyp.diagnosis.target_files[0] if hyp.diagnosis.target_files else ""
            key = (hyp.diagnosis.family, primary_file)
            if primary_file and key in seen_keys:
                logger.debug(
                    "Demoting hypothesis trial=%s (duplicate file=%s)",
                    hyp.diagnosis.trial_id,
                    primary_file,
                )
                demoted.append(hyp)
            else:
                if primary_file:
                    seen_keys.add(key)
                kept.append(hyp)

        return kept + demoted

    def _compute_breadth_penalty(self, diagnosis: Diagnosis) -> float:
        target_count = len(diagnosis.target_files)
        penalty = 0.0
        if target_count == 0:
            penalty += 0.05
        if target_count > 1:
            penalty += min(0.3, (target_count - 1) * 0.08)
        broad_markers = ("prompt", "skill", "system", "coding.md", "general-coding", "verification")
        generic_targets = sum(
            1
            for path in diagnosis.target_files
            if any(marker in path.lower() for marker in broad_markers)
        )
        if generic_targets >= 2:
            penalty += 0.18
        elif generic_targets == 1:
            penalty += 0.08
        return min(0.5, penalty)

    def _compute_overlap_penalty(
        self,
        diagnosis: Diagnosis,
        *,
        suppressed_files: set[str],
        suppressed_families: set[str],
        regression_files: set[str],
        regression_families: set[str],
    ) -> float:
        penalty = 0.0
        if diagnosis.family in suppressed_families:
            penalty += 0.2
        if suppressed_files and set(diagnosis.target_files).intersection(suppressed_files):
            penalty += 0.2
        if diagnosis.family in regression_families:
            penalty += 0.15
        if regression_files and set(diagnosis.target_files).intersection(regression_files):
            penalty += 0.15
        return min(0.4, penalty)
