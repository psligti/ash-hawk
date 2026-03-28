"""Convergence detection for auto-research improvement cycles."""

from __future__ import annotations

from ash_hawk.auto_research.types import ConvergenceReason, ConvergenceResult, IterationResult

_REGRESSION_WINDOW = 3


class ConvergenceDetector:
    """Detects when an improvement cycle has converged.

    Checks three convergence modes:
    1. Plateau — score variance below threshold over a sliding window.
    2. No-improvement — no meaningful gain for N consecutive iterations.
    3. Regression — score decreased for 3+ consecutive iterations.
    """

    def __init__(
        self,
        window_size: int = 5,
        variance_threshold: float = 0.001,
        min_improvement: float = 0.005,
        max_iterations_without_improvement: int = 10,
    ) -> None:
        self._window_size = window_size
        self._variance_threshold = variance_threshold
        self._min_improvement = min_improvement
        self._max_iterations_without_improvement = max_iterations_without_improvement

    def check(self, iterations: list[IterationResult]) -> ConvergenceResult:
        """Check if improvement has converged.

        Convergence criteria:
        1. Score variance < variance_threshold for window_size iterations (PLATEAU)
        2. No improvement > min_improvement for max_iterations_without_improvement (NO_IMPROVEMENT)
        3. Score decreased for 3+ consecutive iterations (REGRESSION)
        """
        if len(iterations) < self._window_size:
            recent = [it.score_after for it in iterations]
            return ConvergenceResult(
                converged=False,
                recent_scores=recent,
                score_variance=self._compute_variance(recent) if recent else 0.0,
            )

        recent_scores = [it.score_after for it in iterations[-self._window_size :]]
        variance = self._compute_variance(recent_scores)

        if variance < self._variance_threshold:
            confidence = max(0.0, min(1.0, 1.0 - (variance / self._variance_threshold)))
            return ConvergenceResult(
                converged=True,
                reason=ConvergenceReason.PLATEAU,
                score_variance=variance,
                confidence=confidence,
                recent_scores=recent_scores,
                iterations_since_improvement=self._iterations_since_improvement(
                    iterations, self._min_improvement
                ),
            )

        iters_since = self._iterations_since_improvement(iterations, self._min_improvement)
        if iters_since >= self._max_iterations_without_improvement:
            return ConvergenceResult(
                converged=True,
                reason=ConvergenceReason.NO_IMPROVEMENT,
                iterations_since_improvement=iters_since,
                score_variance=variance,
                confidence=min(1.0, iters_since / (self._max_iterations_without_improvement * 2)),
                recent_scores=recent_scores,
            )

        if self._detect_regression(iterations):
            return ConvergenceResult(
                converged=True,
                reason=ConvergenceReason.REGRESSION,
                score_variance=variance,
                confidence=1.0,
                recent_scores=recent_scores,
                iterations_since_improvement=iters_since,
            )

        return ConvergenceResult(
            converged=False,
            score_variance=variance,
            recent_scores=recent_scores,
            iterations_since_improvement=iters_since,
        )

    def _compute_variance(self, scores: list[float]) -> float:
        if len(scores) < 2:
            return 0.0
        mean = sum(scores) / len(scores)
        return sum((s - mean) ** 2 for s in scores) / len(scores)

    def _iterations_since_improvement(
        self, iterations: list[IterationResult], threshold: float
    ) -> int:
        best_score = -float("inf")
        last_improvement_idx = -1
        for idx, it in enumerate(iterations):
            if it.score_after > best_score + threshold:
                best_score = it.score_after
                last_improvement_idx = idx
        if last_improvement_idx < 0:
            return len(iterations)
        return len(iterations) - 1 - last_improvement_idx

    def _detect_regression(self, iterations: list[IterationResult]) -> bool:
        if len(iterations) < _REGRESSION_WINDOW:
            return False
        tail = iterations[-_REGRESSION_WINDOW:]
        return all(it.delta < 0 for it in tail)
