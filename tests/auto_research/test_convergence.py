from __future__ import annotations

from ash_hawk.auto_research.convergence import (
    ConvergenceDetector,
    ConvergenceReason,
    ScoreRecord,
)


def _record(iteration: int, score: float, applied: bool = True, delta: float = 0.0) -> ScoreRecord:
    return ScoreRecord(iteration=iteration, score=score, applied=applied, delta=delta)


class TestPlateauDetection:
    def test_low_variance_triggers_plateau(self) -> None:
        detector = ConvergenceDetector(window_size=5, variance_threshold=0.001)
        for i in range(5):
            detector.record(_record(i, 0.75 + i * 0.0001, delta=0.0001))
        result = detector.check()
        assert result.converged
        assert result.reason == ConvergenceReason.PLATEAU

    def test_high_variance_no_plateau(self) -> None:
        detector = ConvergenceDetector(window_size=5, variance_threshold=0.001)
        for i in range(5):
            detector.record(_record(i, 0.5 + i * 0.1, delta=0.1))
        result = detector.check()
        assert not result.converged


class TestNoImprovementDetection:
    def test_triggers_after_max_iterations(self) -> None:
        detector = ConvergenceDetector(
            max_iterations_without_improvement=10,
            max_consecutive_regressions=20,
        )
        detector.record(_record(0, 0.9, applied=True, delta=0.1))
        for i in range(1, 11):
            detector.record(_record(i, 0.9, applied=True, delta=0.0))

        result = detector.check()
        assert result.converged
        assert result.reason == ConvergenceReason.NO_IMPROVEMENT

    def test_improving_scores_no_trigger(self) -> None:
        detector = ConvergenceDetector(max_iterations_without_improvement=5)
        for i in range(6):
            detector.record(_record(i, 0.5 + i * 0.05, delta=0.05))
        result = detector.check()
        assert not result.converged


class TestRegressionDetection:
    def test_consecutive_decreases_triggers_regression(self) -> None:
        detector = ConvergenceDetector(max_consecutive_regressions=3)
        detector.record(_record(0, 0.8, delta=0.1))
        detector.record(_record(1, 0.7, delta=-0.1))
        detector.record(_record(2, 0.6, delta=-0.1))
        result = detector.record(_record(3, 0.5, delta=-0.1))
        assert result.converged
        assert result.reason == ConvergenceReason.REGRESSION

    def test_mixed_deltas_no_regression(self) -> None:
        detector = ConvergenceDetector(max_consecutive_regressions=3)
        detector.record(_record(0, 0.8, delta=0.1))
        detector.record(_record(1, 0.7, delta=-0.1))
        detector.record(_record(2, 0.75, delta=0.05))
        detector.record(_record(3, 0.65, delta=-0.1))
        result = detector.check()
        assert not result.converged or result.reason != ConvergenceReason.REGRESSION


class TestNoFalseConvergence:
    def test_steady_improvement_no_convergence(self) -> None:
        detector = ConvergenceDetector()
        for i in range(15):
            detector.record(_record(i, 0.5 + i * 0.02, delta=0.02))
        assert not detector.check().converged


class TestReset:
    def test_reset_clears_history(self) -> None:
        detector = ConvergenceDetector()
        for i in range(5):
            detector.record(_record(i, 0.75, delta=0.0))
        assert len(detector.history) == 5
        detector.reset()
        assert len(detector.history) == 0

    def test_reset_allows_fresh_detection(self) -> None:
        detector = ConvergenceDetector(max_consecutive_regressions=3)
        detector.record(_record(0, 0.8, delta=0.1))
        detector.record(_record(1, 0.7, delta=-0.1))
        detector.record(_record(2, 0.6, delta=-0.1))
        detector.record(_record(3, 0.5, delta=-0.1))
        assert detector.check().converged
        detector.reset()
        detector.record(_record(0, 0.9, delta=0.1))
        assert not detector.check().converged


class TestComputeVariance:
    def test_uniform_scores_zero_variance(self) -> None:
        detector = ConvergenceDetector()
        assert detector._compute_variance([1.0, 1.0, 1.0]) == 0.0

    def test_known_variance(self) -> None:
        detector = ConvergenceDetector()
        scores = [0.0, 1.0]
        expected = 0.25
        assert abs(detector._compute_variance(scores) - expected) < 1e-10

    def test_single_score_zero_variance(self) -> None:
        detector = ConvergenceDetector()
        assert detector._compute_variance([0.5]) == 0.0


class TestCustomParameters:
    def test_custom_window_size(self) -> None:
        detector = ConvergenceDetector(window_size=3, variance_threshold=0.001)
        for i in range(2):
            detector.record(_record(i, 0.75, delta=0.0))
        assert not detector.check().converged
        detector.record(_record(2, 0.75, delta=0.0))
        assert detector.check().converged
        assert detector.check().reason == ConvergenceReason.PLATEAU

    def test_custom_regression_threshold(self) -> None:
        detector = ConvergenceDetector(max_consecutive_regressions=2)
        detector.record(_record(0, 0.8, delta=0.1))
        detector.record(_record(1, 0.7, delta=-0.1))
        result = detector.record(_record(2, 0.6, delta=-0.1))
        assert result.converged
        assert result.reason == ConvergenceReason.REGRESSION


class TestHistoryProperty:
    def test_history_is_copy(self) -> None:
        detector = ConvergenceDetector()
        detector.record(_record(0, 0.5, delta=0.0))
        h = detector.history
        h.clear()
        assert len(detector.history) == 1
