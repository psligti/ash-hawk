from __future__ import annotations

import logging
import statistics
from typing import TYPE_CHECKING

from ash_hawk.cycle.types import ConvergenceStatus

if TYPE_CHECKING:
    from ash_hawk.cycle.types import CycleConfig

logger = logging.getLogger(__name__)


class ConvergenceChecker:
    """Checks for convergence in improvement cycles.

    Detects when scores have plateaued or when the agent is still
    making meaningful improvements.
    """

    def __init__(self, config: CycleConfig) -> None:
        self._config = config
        self._scores: list[float] = []
        self._consecutive_improvements = 0

    def add_score(self, score: float) -> None:
        self._scores.append(score)

        if len(self._scores) >= 2:
            delta = score - self._scores[-2]
            if delta >= self._config.min_score_improvement:
                self._consecutive_improvements += 1
            else:
                self._consecutive_improvements = 0

    def check_convergence(self) -> ConvergenceStatus:
        if len(self._scores) < 2:
            return ConvergenceStatus.IMPROVING

        if len(self._scores) < self._config.convergence_window:
            return ConvergenceStatus.IMPROVING

        window = min(self._config.convergence_window, len(self._scores))
        recent_scores = self._scores[-window:]

        if len(recent_scores) >= 3:
            variance = statistics.variance(recent_scores)
        else:
            return ConvergenceStatus.IMPROVING

        if len(self._scores) >= 3:
            recent_avg = statistics.mean(recent_scores)
            earlier_avg = (
                statistics.mean(self._scores[:-window])
                if len(self._scores) > window
                else recent_avg
            )
            if recent_avg < earlier_avg - self._config.min_score_improvement:
                return ConvergenceStatus.REGRESSING

        if variance < self._config.convergence_threshold:
            logger.info(
                f"Converged: variance={variance:.4f} < threshold={self._config.convergence_threshold}"
            )
            return ConvergenceStatus.CONVERGED

        if self._is_still_improving(recent_scores):
            return ConvergenceStatus.IMPROVING

        return ConvergenceStatus.STAGNANT

    def _is_still_improving(self, scores: list[float]) -> bool:
        if len(scores) < 3:
            return True

        mid = len(scores) // 2
        first_half_avg = statistics.mean(scores[:mid])
        second_half_avg = statistics.mean(scores[mid:])

        return second_half_avg > first_half_avg + self._config.min_score_improvement * 0.5

    def should_promote_lessons(self) -> bool:
        return self._consecutive_improvements >= self._config.promotion_success_threshold

    def get_consecutive_improvements(self) -> int:
        return self._consecutive_improvements

    def reset_improvement_counter(self) -> None:
        self._consecutive_improvements = 0

    def get_scores(self) -> list[float]:
        return self._scores.copy()

    def get_best_score(self) -> float:
        return max(self._scores) if self._scores else 0.0

    def get_latest_score(self) -> float | None:
        return self._scores[-1] if self._scores else None
