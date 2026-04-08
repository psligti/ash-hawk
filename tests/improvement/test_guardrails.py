from __future__ import annotations

import pytest

from ash_hawk.improvement.guardrails import (
    GuardrailChecker,
    GuardrailConfig,
    IterationRecord,
)


def _record(
    iteration: int,
    score: float,
    applied: bool = True,
    holdout_score: float | None = None,
) -> IterationRecord:
    return IterationRecord(
        iteration=iteration,
        score=score,
        applied=applied,
        holdout_score=holdout_score,
    )


class TestGuardrailConfig:
    def test_defaults(self) -> None:
        cfg = GuardrailConfig()
        assert cfg.max_consecutive_holdout_drops == 3
        assert cfg.max_reverts == 5
        assert cfg.plateau_window == 5
        assert cfg.plateau_threshold == 0.02

    def test_custom(self) -> None:
        cfg = GuardrailConfig(
            max_consecutive_holdout_drops=1,
            max_reverts=2,
            plateau_window=3,
            plateau_threshold=0.01,
        )
        assert cfg.max_consecutive_holdout_drops == 1
        assert cfg.max_reverts == 2


class TestHoldoutDropDetection:
    def test_three_consecutive_drops_triggers_stop(self) -> None:
        checker = GuardrailChecker()
        # Baseline
        checker.record_iteration(_record(1, 0.7, holdout_score=0.70))
        checker.record_iteration(_record(2, 0.72, holdout_score=0.69))
        checker.record_iteration(_record(3, 0.74, holdout_score=0.68))
        result = checker.record_iteration(_record(4, 0.75, holdout_score=0.67))
        assert result.should_stop is True
        assert result.consecutive_drops == 3
        assert "Holdout score dropped" in (result.reason or "")

    def test_recovery_resets_counter(self) -> None:
        checker = GuardrailChecker()
        checker.record_iteration(_record(1, 0.7, holdout_score=0.70))
        checker.record_iteration(_record(2, 0.72, holdout_score=0.69))
        # Recovery — holdout goes up
        result = checker.record_iteration(_record(3, 0.74, holdout_score=0.71))
        assert result.should_stop is False
        assert result.consecutive_drops == 0

    def test_none_holdout_skips_check(self) -> None:
        checker = GuardrailChecker()
        checker.record_iteration(_record(1, 0.7, holdout_score=None))
        result = checker.record_iteration(_record(2, 0.72, holdout_score=None))
        assert result.consecutive_drops == 0


class TestMaxRevertsDetection:
    def test_exceeding_limit_triggers_stop(self) -> None:
        cfg = GuardrailConfig(max_reverts=3)
        checker = GuardrailChecker(cfg)
        checker.record_iteration(_record(1, 0.5, applied=False))
        checker.record_iteration(_record(2, 0.6, applied=False))
        result = checker.record_iteration(_record(3, 0.55, applied=False))
        assert result.should_stop is True
        assert result.total_reverts == 3
        assert "reverts" in (result.reason or "").lower()

    def test_mixed_applied_and_reverted(self) -> None:
        cfg = GuardrailConfig(max_reverts=2)
        checker = GuardrailChecker(cfg)
        checker.record_iteration(_record(1, 0.5, applied=True))
        checker.record_iteration(_record(2, 0.6, applied=False))
        result = checker.record_iteration(_record(3, 0.55, applied=False))
        assert result.should_stop is True
        assert result.total_reverts == 2


class TestPlateauDetection:
    def test_plateau_triggers_stop(self) -> None:
        cfg = GuardrailConfig(plateau_window=5, plateau_threshold=0.02)
        checker = GuardrailChecker(cfg)
        scores = [0.70, 0.705, 0.708, 0.710, 0.712]
        for i, s in enumerate(scores, 1):
            result = checker.record_iteration(_record(i, s))
        assert result.plateau_detected is True
        assert result.should_stop is True
        assert "plateau" in (result.reason or "").lower()

    def test_improvement_avoids_plateau(self) -> None:
        cfg = GuardrailConfig(plateau_window=5, plateau_threshold=0.02)
        checker = GuardrailChecker(cfg)
        scores = [0.60, 0.65, 0.70, 0.75, 0.80]
        for i, s in enumerate(scores, 1):
            result = checker.record_iteration(_record(i, s))
        assert result.plateau_detected is False

    def test_too_few_iterations_skips_plateau(self) -> None:
        cfg = GuardrailConfig(plateau_window=5, plateau_threshold=0.02)
        checker = GuardrailChecker(cfg)
        result = checker.record_iteration(_record(1, 0.5))
        assert result.plateau_detected is False


class TestCheckWithoutRecording:
    def test_check_returns_current_state(self) -> None:
        cfg = GuardrailConfig(max_reverts=2)
        checker = GuardrailChecker(cfg)
        checker.record_iteration(_record(1, 0.5, applied=False))
        checker.record_iteration(_record(2, 0.6, applied=False))
        result = checker.check()
        assert result.should_stop is True
        assert len(checker.history) == 2


class TestReset:
    def test_reset_clears_history(self) -> None:
        checker = GuardrailChecker()
        checker.record_iteration(_record(1, 0.5))
        checker.record_iteration(_record(2, 0.6))
        assert len(checker.history) == 2
        checker.reset()
        assert len(checker.history) == 0
        assert checker.stop_reason is None


class TestStopReasonProperty:
    def test_no_reason_before_trigger(self) -> None:
        checker = GuardrailChecker()
        assert checker.stop_reason is None

    def test_reason_set_after_trigger(self) -> None:
        cfg = GuardrailConfig(max_reverts=1)
        checker = GuardrailChecker(cfg)
        checker.record_iteration(_record(1, 0.5, applied=False))
        assert checker.stop_reason is not None


class TestHistoryReadOnly:
    def test_history_copy_not_mutable(self) -> None:
        checker = GuardrailChecker()
        checker.record_iteration(_record(1, 0.5))
        h = checker.history
        assert len(h) == 1
        h.clear()
        assert len(checker.history) == 1


class TestNoStopWhenConditionsMet:
    def test_healthy_iterations_no_stop(self) -> None:
        cfg = GuardrailConfig(
            max_consecutive_holdout_drops=3,
            max_reverts=5,
            plateau_window=5,
            plateau_threshold=0.02,
        )
        checker = GuardrailChecker(cfg)
        for i, s in enumerate([0.50, 0.55, 0.60], 1):
            result = checker.record_iteration(
                _record(i, s, applied=True, holdout_score=s + 0.01),
            )
        assert result.should_stop is False
        assert result.reason is None
