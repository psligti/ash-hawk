from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ash_hawk.improve.diagnose import Diagnosis
    from ash_hawk.improve.lesson_store import LessonStore

logger = logging.getLogger(__name__)


@dataclass
class RankedHypothesis:
    """A diagnosis ranked by estimated impact and novelty."""

    diagnosis: Diagnosis
    rank: int
    novelty_score: float
    confidence: float
    estimated_impact: float
    already_tried: bool
    similar_lesson_ids: list[str] = field(default_factory=list)


@dataclass
class HypothesisRanking:
    """Result of ranking multiple hypotheses."""

    hypotheses: list[RankedHypothesis]
    total_candidates: int
    filtered_as_tried: int
    ranked_count: int


class HypothesisRanker:
    """Rank and filter diagnosis hypotheses for the improvement loop.

    Ranking considers:
    1. Novelty — has a similar approach been tried before?
    2. Confidence — the LLM's own confidence in the diagnosis
    3. Estimated impact — heuristic based on target files and root cause specificity
    4. Diversity — prefer hypotheses targeting different files/root causes
    """

    def __init__(self, lesson_store: LessonStore | None = None) -> None:
        """Initialize with optional lesson store for novelty checking.

        Args:
            lesson_store: LessonStore instance for checking past attempts.
                         If None, novelty checking is skipped.
        """
        self._lesson_store = lesson_store

    def rank(
        self,
        diagnoses: list[Diagnosis],
        filter_tried: bool = True,
    ) -> HypothesisRanking:
        """Rank diagnoses into ordered hypotheses.

        Steps:
        1. For each diagnosis, check if similar approach was tried (via lesson_store)
        2. Compute novelty_score (1.0 if never tried, decreasing with similarity)
        3. Compute estimated_impact from confidence + target_files count
        4. Filter out already-tried if filter_tried=True
        5. Sort by: estimated_impact DESC, novelty_score DESC, confidence DESC

        Args:
            diagnoses: List of diagnoses to rank.
            filter_tried: Whether to filter out already-tried hypotheses.

        Returns:
            HypothesisRanking with ordered hypotheses.
        """
        logger.info("Ranking %d hypotheses", len(diagnoses))

        # Step 1-3: Score each diagnosis
        candidates: list[RankedHypothesis] = []
        for diagnosis in diagnoses:
            novelty_score, already_tried, similar_ids = self._compute_novelty(diagnosis)
            impact = self._compute_impact(diagnosis)

            candidates.append(
                RankedHypothesis(
                    diagnosis=diagnosis,
                    rank=0,  # assigned after sorting
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

        # Step 4: Filter
        total_candidates = len(candidates)
        filtered_as_tried = 0
        if filter_tried:
            before = len(candidates)
            candidates = [c for c in candidates if not c.already_tried]
            filtered_as_tried = before - len(candidates)

        if filtered_as_tried > 0:
            logger.info("Filtered %d already-tried hypotheses", filtered_as_tried)

        # Step 5: Sort
        candidates.sort(
            key=lambda h: (h.estimated_impact, h.novelty_score, h.confidence),
            reverse=True,
        )

        # Apply diversity pass
        candidates = self._ensure_diversity(candidates)

        # Assign ranks
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
        """Compute novelty score for a diagnosis.

        If lesson_store is available, uses has_been_tried() and find_similar().
        Returns (novelty_score, already_tried, similar_lesson_ids).

        Args:
            diagnosis: The diagnosis to evaluate.

        Returns:
            Tuple of (novelty_score, already_tried, similar_lesson_ids).
        """
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
            similar_ids = [str(getattr(s, "id", s)) for s in similar]
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
            # More similar lessons = lower novelty
            novelty_score = max(0.1, 1.0 - len(similar_ids) * 0.2)
        else:
            novelty_score = 1.0

        return (novelty_score, already_tried, similar_ids)

    def _compute_impact(self, diagnosis: Diagnosis) -> float:
        """Estimate impact of a diagnosis.

        Heuristic:
        - Base: diagnosis.confidence (0.0-1.0)
        - More target files = higher impact (up to +0.2 for 3+ files)
        - Longer root_cause = more specific = higher impact (up to +0.1 for >200 chars)
        - Clamp to [0.0, 1.0]

        Args:
            diagnosis: The diagnosis to evaluate.

        Returns:
            Estimated impact score between 0.0 and 1.0.
        """
        base = diagnosis.confidence

        # Target files bonus
        file_bonus = min(len(diagnosis.target_files) / 3.0, 1.0) * 0.2

        # Root cause specificity bonus
        rc_len = len(diagnosis.root_cause)
        rc_bonus = min(rc_len / 200.0, 1.0) * 0.1

        impact = base + file_bonus + rc_bonus
        return max(0.0, min(1.0, impact))

    def _ensure_diversity(self, hypotheses: list[RankedHypothesis]) -> list[RankedHypothesis]:
        """Ensure hypotheses target diverse files.

        If multiple hypotheses target the same file, keep only the highest-ranked
        one and demote others. Log when demoting.

        Args:
            hypotheses: Already-sorted list of hypotheses.

        Returns:
            List with duplicates demoted to the end.
        """
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
