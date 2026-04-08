from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class ConvergenceReason(StrEnum):
    PLATEAU = "plateau"
    NO_IMPROVEMENT = "no_improvement"
    REGRESSION = "regression"


@dataclass
class ConvergenceResult:
    converged: bool
    reason: ConvergenceReason | None = None
    iterations_since_improvement: int = 0
    score_variance: float = 0.0
    confidence: float = 0.0
    message: str = ""


@dataclass
class ScoreRecord:
    iteration: int
    score: float
    applied: bool
    delta: float = 0.0


class ConvergenceDetector:
    def __init__(
        self,
        window_size: int = 5,
        variance_threshold: float = 0.001,
        min_improvement: float = 0.005,
        max_iterations_without_improvement: int = 10,
        max_consecutive_regressions: int = 3,
    ) -> None:
        self._window_size = window_size
        self._variance_threshold = variance_threshold
        self._min_improvement = min_improvement
        self._max_iterations_without_improvement = max_iterations_without_improvement
        self._max_consecutive_regressions = max_consecutive_regressions
        self._history: list[ScoreRecord] = []
        self._best_score: float = 0.0
        self._iterations_since_best: int = 0

    def record(self, record: ScoreRecord) -> ConvergenceResult:
        if not self._history or record.score > self._best_score:
            self._best_score = record.score
            self._iterations_since_best = 0
        else:
            self._iterations_since_best += 1

        self._history.append(record)

        logger.info(
            "Convergence check: iter=%d score=%.4f delta=%.4f applied=%s",
            record.iteration,
            record.score,
            record.delta,
            record.applied,
        )

        result = self.check()

        if result.converged:
            logger.warning(
                "Convergence detected: reason=%s confidence=%.2f",
                result.reason,
                result.confidence,
            )

        return result

    def check(self) -> ConvergenceResult:
        if len(self._history) < 2:
            return ConvergenceResult(converged=False)

        checks = [self._check_regression(), self._check_no_improvement(), self._check_plateau()]

        for result in checks:
            if result is not None:
                return result

        return ConvergenceResult(converged=False)

    def _check_plateau(self) -> ConvergenceResult | None:
        if len(self._history) < self._window_size:
            return None

        window = self._history[-self._window_size :]
        scores = [r.score for r in window]
        variance = self._compute_variance(scores)

        if variance >= self._variance_threshold:
            return None

        total_delta = sum(r.delta for r in window)
        if total_delta > self._min_improvement:
            return None

        confidence = min(1.0, self._variance_threshold / max(variance, 1e-10))
        return ConvergenceResult(
            converged=True,
            reason=ConvergenceReason.PLATEAU,
            score_variance=variance,
            iterations_since_improvement=self._iterations_since_best,
            confidence=confidence,
            message=f"Score variance {variance:.6f} below threshold {self._variance_threshold}",
        )

    def _check_no_improvement(self) -> ConvergenceResult | None:
        if self._iterations_since_best < self._max_iterations_without_improvement:
            return None

        applied_deltas = [r.delta for r in self._history if r.applied and r.delta > 0]
        best_delta = max(applied_deltas) if applied_deltas else 0.0
        confidence = min(
            1.0,
            self._iterations_since_best / self._max_iterations_without_improvement,
        )

        return ConvergenceResult(
            converged=True,
            reason=ConvergenceReason.NO_IMPROVEMENT,
            iterations_since_improvement=self._iterations_since_best,
            confidence=confidence,
            message=(
                f"No improvement above {self._min_improvement} for "
                f"{self._iterations_since_best} iterations"
            ),
        )

    def _check_regression(self) -> ConvergenceResult | None:
        if len(self._history) < self._max_consecutive_regressions:
            return None

        recent = self._history[-self._max_consecutive_regressions :]
        all_decreasing = all(r.delta < 0 for r in recent)

        if not all_decreasing:
            return None

        total_drop = sum(abs(r.delta) for r in recent)
        confidence = min(1.0, total_drop)

        return ConvergenceResult(
            converged=True,
            reason=ConvergenceReason.REGRESSION,
            confidence=confidence,
            iterations_since_improvement=self._iterations_since_best,
            message=(
                f"Score decreased for {self._max_consecutive_regressions} "
                f"consecutive iterations (total drop: {total_drop:.4f})"
            ),
        )

    def _compute_variance(self, scores: list[float]) -> float:
        if len(scores) < 2:
            return 0.0
        mean = sum(scores) / len(scores)
        return sum((s - mean) ** 2 for s in scores) / len(scores)

    @property
    def history(self) -> list[ScoreRecord]:
        return list(self._history)

    def reset(self) -> None:
        self._history.clear()
        self._best_score = 0.0
        self._iterations_since_best = 0
