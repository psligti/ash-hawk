"""Tests for ConvergenceDetector."""

from __future__ import annotations

import pytest

from ash_hawk.auto_research.convergence import _REGRESSION_WINDOW, ConvergenceDetector
from ash_hawk.auto_research.types import ConvergenceReason, IterationResult


def _make_iteration(
    num: int,
    before: float,
    after: float,
    applied: bool = True,
) -> IterationResult:
    return IterationResult(
        iteration_num=num,
        score_before=before,
        score_after=after,
        improvement_text=f"Iteration {num}",
        applied=applied,
    )


def _make_iterations(scores: list[tuple[float, float]]) -> list[IterationResult]:
    return [_make_iteration(i, before, after) for i, (before, after) in enumerate(scores)]


@pytest.fixture
def detector() -> ConvergenceDetector:
    return ConvergenceDetector(
        window_size=5,
        variance_threshold=0.001,
        min_improvement=0.005,
        max_iterations_without_improvement=10,
    )


class TestCheck:
    """Tests for the check() method."""

    def test_returns_not_converged_with_insufficient_data(
        self, detector: ConvergenceDetector
    ) -> None:
        iterations = _make_iterations([(0.5, 0.55), (0.55, 0.58)])
        result = detector.check(iterations)

        assert result.converged is False
        assert result.reason is None
        assert len(result.recent_scores) == 2

    def test_detects_plateau_when_variance_below_threshold(
        self, detector: ConvergenceDetector
    ) -> None:
        scores = [
            (0.70, 0.7001),
            (0.7001, 0.7002),
            (0.7002, 0.7001),
            (0.7001, 0.7000),
            (0.7000, 0.7001),
        ]
        iterations = _make_iterations(scores)
        result = detector.check(iterations)

        assert result.converged is True
        assert result.reason == ConvergenceReason.PLATEAU
        assert result.score_variance < detector._variance_threshold
        assert result.confidence > 0.0

    def test_detects_no_improvement_after_threshold_iterations(
        self, detector: ConvergenceDetector
    ) -> None:
        no_plateau_detector = ConvergenceDetector(
            window_size=5,
            variance_threshold=1e-8,
            min_improvement=0.005,
            max_iterations_without_improvement=10,
        )
        scores = [
            (0.50, 0.55),
            (0.55, 0.60),
            (0.60, 0.65),
            (0.65, 0.70),
            (0.70, 0.75),
            (0.75, 0.752),
            (0.752, 0.754),
            (0.754, 0.756),
            (0.756, 0.757),
            (0.757, 0.758),
            (0.758, 0.7585),
            (0.7585, 0.7588),
            (0.7588, 0.759),
            (0.759, 0.7592),
            (0.7592, 0.7593),
            (0.7593, 0.7594),
            (0.7594, 0.75945),
            (0.75945, 0.7595),
        ]
        iterations = _make_iterations(scores)
        result = no_plateau_detector.check(iterations)

        assert result.converged is True
        assert result.reason == ConvergenceReason.NO_IMPROVEMENT
        assert result.iterations_since_improvement >= 10

    def test_detects_regression_with_consecutive_decreases(
        self, detector: ConvergenceDetector
    ) -> None:
        scores = [
            (0.50, 0.55),
            (0.55, 0.60),
            (0.60, 0.65),
            (0.65, 0.70),
            (0.70, 0.75),
            (0.75, 0.70),
            (0.70, 0.65),
            (0.65, 0.60),
        ]
        iterations = _make_iterations(scores)
        result = detector.check(iterations)

        assert result.converged is True
        assert result.reason == ConvergenceReason.REGRESSION
        assert result.confidence == 1.0

    def test_returns_not_converged_when_still_improving(
        self, detector: ConvergenceDetector
    ) -> None:
        scores = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.75)]
        iterations = _make_iterations(scores)
        result = detector.check(iterations)

        assert result.converged is False

    def test_prioritizes_plateau_over_no_improvement(self, detector: ConvergenceDetector) -> None:
        scores = [
            (0.70, 0.7001),
            (0.7001, 0.7002),
            (0.7002, 0.7001),
            (0.7001, 0.7000),
            (0.7000, 0.7001),
        ]
        iterations = _make_iterations(scores)
        result = detector.check(iterations)

        assert result.reason == ConvergenceReason.PLATEAU


class TestComputeVariance:
    """Tests for _compute_variance."""

    def test_returns_zero_for_single_score(self, detector: ConvergenceDetector) -> None:
        assert detector._compute_variance([0.5]) == 0.0

    def test_returns_zero_for_empty_list(self, detector: ConvergenceDetector) -> None:
        assert detector._compute_variance([]) == 0.0

    def test_computes_population_variance(self, detector: ConvergenceDetector) -> None:
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        variance = detector._compute_variance(scores)
        expected = 2.0
        assert variance == pytest.approx(expected)

    def test_returns_zero_for_identical_scores(self, detector: ConvergenceDetector) -> None:
        scores = [0.75, 0.75, 0.75, 0.75]
        variance = detector._compute_variance(scores)
        assert variance == 0.0


class TestIterationsSinceImprovement:
    """Tests for _iterations_since_improvement."""

    def test_returns_zero_when_just_improved(self, detector: ConvergenceDetector) -> None:
        iterations = _make_iterations([(0.5, 0.6), (0.6, 0.7), (0.7, 0.8)])
        result = detector._iterations_since_improvement(iterations, 0.005)
        assert result == 0

    def test_counts_iterations_since_last_significant_gain(
        self, detector: ConvergenceDetector
    ) -> None:
        iterations = _make_iterations(
            [(0.5, 0.6), (0.6, 0.7), (0.7, 0.71), (0.71, 0.715), (0.715, 0.718)]
        )
        result = detector._iterations_since_improvement(iterations, 0.05)
        assert result == 3

    def test_returns_total_count_when_no_improvement(self, detector: ConvergenceDetector) -> None:
        iterations = _make_iterations([(0.5, 0.51), (0.51, 0.511), (0.511, 0.5115)])
        result = detector._iterations_since_improvement(iterations, 0.1)
        assert result == 2


class TestDetectRegression:
    """Tests for _detect_regression."""

    def test_returns_false_with_insufficient_data(self, detector: ConvergenceDetector) -> None:
        iterations = _make_iterations([(0.5, 0.4), (0.4, 0.3)])
        assert detector._detect_regression(iterations) is False

    def test_returns_true_when_all_recent_deltas_negative(
        self, detector: ConvergenceDetector
    ) -> None:
        iterations = _make_iterations([(0.5, 0.6), (0.6, 0.7), (0.7, 0.6), (0.6, 0.5), (0.5, 0.4)])
        assert detector._detect_regression(iterations) is True

    def test_returns_false_when_any_recent_delta_positive(
        self, detector: ConvergenceDetector
    ) -> None:
        iterations = _make_iterations([(0.5, 0.6), (0.6, 0.7), (0.7, 0.6), (0.6, 0.5), (0.5, 0.6)])
        assert detector._detect_regression(iterations) is False

    def test_returns_false_when_delta_is_zero(self, detector: ConvergenceDetector) -> None:
        iterations = _make_iterations([(0.5, 0.6), (0.6, 0.7), (0.7, 0.6), (0.6, 0.5), (0.5, 0.5)])
        assert detector._detect_regression(iterations) is False


class TestRegressionWindowConstant:
    """Tests for the regression window constant."""

    def test_regression_window_is_three(self) -> None:
        assert _REGRESSION_WINDOW == 3


class TestConfidenceCalculation:
    """Tests for confidence score calculation."""

    def test_plateau_confidence_decreases_with_higher_variance(
        self, detector: ConvergenceDetector
    ) -> None:
        low_var_scores = [
            (0.70, 0.70001),
            (0.70001, 0.70002),
            (0.70002, 0.70001),
            (0.70001, 0.70000),
            (0.70000, 0.70001),
        ]
        high_var_scores = [
            (0.70, 0.705),
            (0.705, 0.695),
            (0.695, 0.704),
            (0.704, 0.696),
            (0.696, 0.703),
        ]

        low_var_result = detector.check(_make_iterations(low_var_scores))
        high_var_result = detector.check(_make_iterations(high_var_scores))

        if low_var_result.reason == ConvergenceReason.PLATEAU:
            assert low_var_result.confidence > 0.9

    def test_no_improvement_confidence_scales_with_iterations(
        self,
    ) -> None:
        detector = ConvergenceDetector(
            window_size=3,
            variance_threshold=0.1,
            min_improvement=0.01,
            max_iterations_without_improvement=5,
        )
        scores = [(0.50 + i * 0.001, 0.51 + i * 0.001) for i in range(8)]
        result = detector.check(_make_iterations(scores))

        if result.reason == ConvergenceReason.NO_IMPROVEMENT:
            assert result.confidence >= 0.5


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_iterations_list(self, detector: ConvergenceDetector) -> None:
        result = detector.check([])
        assert result.converged is False
        assert result.recent_scores == []

    def test_single_iteration(self, detector: ConvergenceDetector) -> None:
        iterations = _make_iterations([(0.5, 0.6)])
        result = detector.check(iterations)
        assert result.converged is False

    def test_exactly_window_size_iterations(self, detector: ConvergenceDetector) -> None:
        scores = [
            (0.70, 0.7001),
            (0.7001, 0.7002),
            (0.7002, 0.7001),
            (0.7001, 0.7000),
            (0.7000, 0.7001),
        ]
        iterations = _make_iterations(scores)
        result = detector.check(iterations)

        assert len(result.recent_scores) == 5
        assert result.converged is True
        assert result.reason == ConvergenceReason.PLATEAU

    def test_negative_scores(self, detector: ConvergenceDetector) -> None:
        iterations = _make_iterations(
            [(-0.1, -0.05), (-0.05, 0.0), (0.0, 0.05), (0.05, 0.1), (0.1, 0.15)]
        )
        result = detector.check(iterations)
        assert result.converged is False
