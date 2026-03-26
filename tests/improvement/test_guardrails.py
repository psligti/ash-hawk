"""Tests for guardrails module."""

from __future__ import annotations

import pytest

from ash_hawk.improvement.guardrails import (
    GuardrailChecker,
    GuardrailConfig,
    GuardrailState,
)


class TestGuardrailConfig:
    def test_defaults(self):
        config = GuardrailConfig()

        assert config.max_consecutive_holdout_drops == 3
        assert config.max_reverts == 5
        assert config.plateau_window == 5
        assert config.plateau_threshold == 0.02

    def test_custom_config(self):
        config = GuardrailConfig(
            max_consecutive_holdout_drops=5,
            max_reverts=10,
            plateau_window=3,
            plateau_threshold=0.05,
        )

        assert config.max_consecutive_holdout_drops == 5
        assert config.max_reverts == 10
        assert config.plateau_window == 3
        assert config.plateau_threshold == 0.05


class TestGuardrailState:
    def test_defaults(self):
        state = GuardrailState()

        assert state.consecutive_holdout_drops == 0
        assert state.total_reverts == 0
        assert state.recent_scores == []
        assert state.stop_reason is None


class TestGuardrailChecker:
    @pytest.fixture
    def config(self):
        return GuardrailConfig(
            max_consecutive_holdout_drops=3,
            max_reverts=5,
            plateau_window=5,
            plateau_threshold=0.02,
        )

    @pytest.fixture
    def checker(self, config):
        return GuardrailChecker(config)

    @pytest.fixture
    def default_checker(self):
        return GuardrailChecker()

    def test_initial_state(self, default_checker):
        state = default_checker.state

        assert state.consecutive_holdout_drops == 0
        assert state.total_reverts == 0
        assert state.recent_scores == []
        assert state.stop_reason is None

    def test_record_iteration_applied(self, checker):
        for i in range(10):
            checker.record_iteration(0.5 + i * 0.1, applied=True)

        state = checker.state
        assert state.consecutive_holdout_drops == 0
        assert len(state.recent_scores) == 5
        assert state.stop_reason is None

    def test_record_iteration_not_applied(self, checker):
        for i in range(10):
            checker.record_iteration(0.5 + i * 0.1, applied=False)

        state = checker.state
        assert state.consecutive_holdout_drops == 0
        assert state.recent_scores == []

    def test_record_revert(self, checker):
        checker.record_iteration(0.5, applied=True, reverted=True)
        state = checker.state
        assert state.total_reverts == 1

    def test_consecutive_drops_trigger_stop(self, checker):
        scores = [0.8, 0.7, 0.6, 0.5]
        for i, score in enumerate(scores):
            checker.record_iteration(score, applied=True)

            if i < 3:
                assert not checker.should_stop()
                assert checker.stop_reason is None

        assert checker.should_stop()
        assert checker.stop_reason is not None

    def test_score_improvement_resets_drops(self, checker):
        scores = [0.8, 0.7, 0.6, 0.5]
        for score in scores:
            checker.record_iteration(score, applied=True)

        assert checker.should_stop()

        new_checker = GuardrailChecker(checker.config)
        new_checker.record_iteration(0.9, applied=True)
        assert not new_checker.should_stop()
        assert new_checker.state.consecutive_holdout_drops == 0
        new_checker.record_iteration(0.95, applied=True)
        assert not new_checker.should_stop()
        assert new_checker.state.consecutive_holdout_drops == 0

    def test_plateau_detection(self, checker):
        for i in range(5):
            score = 0.5 + i * 0.001
            checker.record_iteration(score, applied=True)
        assert checker.should_stop()
        assert "plateau" in checker.stop_reason.lower()

    def test_max_reverts_trigger_stop(self, checker):
        for i in range(5):
            checker.record_iteration(0.5, applied=True, reverted=True)

        assert checker.should_stop()
        assert "reverts" in checker.stop_reason.lower()

    def test_reset(self, checker):
        for i in range(3):
            score = 0.8 - i * 0.1
            checker.record_iteration(score, applied=True)

        state_before = checker.state
        assert state_before.consecutive_holdout_drops == 2
        assert state_before.total_reverts == 0

        checker.reset()

        state_after = checker.state
        assert state_after.consecutive_holdout_drops == 0
        assert state_after.total_reverts == 0
        assert state_after.recent_scores == []
        assert state_after.stop_reason is None

    def test_state_snapshot(self, checker):
        checker.record_iteration(0.5, applied=True)
        state1 = checker.state
        state2 = checker.state

        assert state1.recent_scores == [0.5]
        assert state2.recent_scores == [0.5]
        assert state1 is not state2
