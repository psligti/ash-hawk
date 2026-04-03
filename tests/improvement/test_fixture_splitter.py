from __future__ import annotations

import math
from pathlib import Path

import pytest

from ash_hawk.improvement.fixture_splitter import FixtureSplit, FixtureSplitter
from ash_hawk.improvement.guardrails import GuardrailChecker, GuardrailConfig, GuardrailState


class TestFixtureSplitter:
    @pytest.fixture
    def fixtures(self, tmp_path: Path) -> list[Path]:
        for i in range(20):
            fixture_dir = tmp_path / f"fixture-{i:02}"
            fixture_dir.mkdir(parents=True, exist_ok=False)

        return sorted([tmp_path / f"fixture-{i:02}" for i in range(20)])

    @pytest.fixture
    def splitter(self):
        return FixtureSplitter(seed=42, train_ratio=0.7)

    @pytest.fixture
    def small_fixtures(self, tmp_path: Path) -> list[Path]:
        for i in range(4):
            fixture_dir = tmp_path / f"fixture-{i:02}"
            fixture_dir.mkdir(parents=True, exist_ok=False)

        return sorted([tmp_path / f"fixture-{i:02}" for i in range(4)])

    @pytest.fixture
    def small_splitter(self):
        return FixtureSplitter(seed=42, train_ratio=0.7)

    def test_default_split(self, splitter, fixtures):
        result = splitter.split(fixtures)

        assert isinstance(result, FixtureSplit)
        assert len(result.train) > 0
        assert len(result.heldout) > 0
        assert result.seed == 42
        assert 0.6 < result.ratio < 0.8

        expected_train = int(len(fixtures) * 0.7)
        assert result.train_count == expected_train

        expected_heldout = len(fixtures) - expected_train
        assert result.heldout_count == expected_heldout

    def test_split_with_counts(self, splitter, fixtures):
        result = splitter.split_with_counts(fixtures, train_count=14)

        assert isinstance(result, FixtureSplit)
        assert len(result.train) == 14
        assert len(result.heldout) == 6
        assert result.ratio == 14 / 20

        expected_train = 14
        expected_heldout = 6

    def test_small_fixtures_all_to_training(self, small_splitter, small_fixtures):
        result = small_splitter.split(small_fixtures)

        assert len(result.train) == 4
        assert len(result.heldout) == 0

    def test_empty_fixtures(self, splitter):
        result = splitter.split([])

        assert isinstance(result, FixtureSplit)
        assert len(result.train) == 0
        assert len(result.heldout) == 0
        assert result.seed == 42

        assert result.ratio == 0.7

    def test_invalid_ratio_raises(self, splitter):
        with pytest.raises(ValueError, match="between 0 and 1"):
            FixtureSplitter(seed=42, train_ratio=0.0)

    def test_invalid_ratio_upper(self, splitter):
        with pytest.raises(ValueError, match="between 0 and 1"):
            FixtureSplitter(seed=42, train_ratio=1.0)

    def test_deterministic_split(self, splitter, fixtures):
        result1 = splitter.split(fixtures)
        result2 = splitter.split(fixtures)

        assert result1.train == result2.train
        assert result1.heldout == result2.heldout
