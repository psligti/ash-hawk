"""Integration tests for self-improvement loop."""

from __future__ import annotations
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ash_hawk.improvement.fixture_splitter import FixtureSplit, FixtureSplitter
from ash_hawk.improvement.guardrails import (
    GuardrailChecker,
    GuardrailConfig,
    GuardrailState,
)


from ash_hawk.auto_research.cycle_runner import run_cycle


class TestSelfImprovementLoopIntegration:
    @pytest.fixture
    def fixtures_dir(self, tmp_path: Path) -> Path:
        for i in range(20):
            fixture_dir = tmp_path / f"fixture-{i:02}"
            fixture_dir.mkdir(parents=True, exist_ok=False)
        return tmp_path

    @pytest.fixture
    def fixtures(self, fixtures_dir: Path) -> list[Path]:
        return sorted([fixtures_dir / f"fixture-{i:02}" for i in range(20)])

    @pytest.fixture
    def splitter(self):
        return FixtureSplitter(seed=42, train_ratio=0.7)

    @pytest.fixture
    def guardrail_config(self):
        return GuardrailConfig(
            max_consecutive_holdout_drops=3,
            max_reverts=5,
            plateau_window=5,
            plateau_threshold=0.02,
        )

    def test_fixture_split_creates_disjoint_sets(self, splitter, fixtures):
        split = splitter.split(fixtures)

        assert split.train_count == 14
        assert split.heldout_count == 6
        assert len(split.train) == 14
        assert len(split.heldout) == 6

        train_set = set(split.train)
        heldout_set = set(split.heldout)
        assert train_set.isdisjoint(heldout_set)

    def test_guardrail_checker_with_split(self, splitter, fixtures, guardrail_config):
        split = splitter.split(fixtures)
        checker = GuardrailChecker(guardrail_config)

        for i, score in enumerate([0.5 + i * 0.05 for i in range(5)]):
            checker.record_iteration(score, applied=True)

        assert not checker.should_stop()

        for i in range(4):
            score = 0.8 - i * 0.1
            checker.record_iteration(score, applied=True)

        assert checker.should_stop()

    def test_guardrail_reset_allows_new_cycle(self, splitter, fixtures, guardrail_config):
        split = splitter.split(fixtures)
        checker = GuardrailChecker(guardrail_config)

        for i in range(4):
            score = 0.8 - i * 0.1
            checker.record_iteration(score, applied=True)

        assert checker.should_stop()
        assert checker.stop_reason is not None

        checker.reset()

        assert not checker.should_stop()
        assert checker.state.consecutive_holdout_drops == 0

        checker.record_iteration(0.9, applied=True)
        assert not checker.should_stop()
