from __future__ import annotations

from pathlib import Path

import pytest

from ash_hawk.improvement.fixture_splitter import FixtureSplitter


class TestFixtureSplitter:
    def _make_fixtures(self, tmp_path: Path, n: int) -> list[Path]:
        return [tmp_path / f"fixture_{i}" for i in range(n)]

    def test_split_respects_train_ratio(self, tmp_path: Path) -> None:
        fixtures = self._make_fixtures(tmp_path, 100)
        splitter = FixtureSplitter(seed=42, train_ratio=0.7)
        result = splitter.split(fixtures)

        assert result.total == 100
        train_ratio = len(result.train) / result.total
        assert 0.6 <= train_ratio <= 0.8

    def test_determinism(self, tmp_path: Path) -> None:
        fixtures = self._make_fixtures(tmp_path, 20)
        splitter = FixtureSplitter(seed=42, train_ratio=0.7)

        r1 = splitter.split(fixtures)
        r2 = splitter.split(fixtures)

        assert r1.train == r2.train
        assert r1.heldout == r2.heldout

    def test_different_seeds_different_splits(self, tmp_path: Path) -> None:
        fixtures = self._make_fixtures(tmp_path, 30)
        s1 = FixtureSplitter(seed=1, train_ratio=0.7)
        s2 = FixtureSplitter(seed=999, train_ratio=0.7)

        r1 = s1.split(fixtures)
        r2 = s2.split(fixtures)

        assert r1.train != r2.train or r1.heldout != r2.heldout

    def test_single_fixture(self, tmp_path: Path) -> None:
        fixtures = [tmp_path / "only_one"]
        splitter = FixtureSplitter(seed=42, train_ratio=0.7)
        result = splitter.split(fixtures)

        assert result.total == 1
        assert len(result.train) + len(result.heldout) == 1

    def test_empty_list(self) -> None:
        splitter = FixtureSplitter(seed=42, train_ratio=0.7)
        result = splitter.split([])

        assert result.train == []
        assert result.heldout == []
        assert result.total == 0

    def test_invalid_train_ratio_zero(self) -> None:
        with pytest.raises(ValueError, match="train_ratio"):
            FixtureSplitter(seed=42, train_ratio=0.0)

    def test_invalid_train_ratio_one(self) -> None:
        with pytest.raises(ValueError, match="train_ratio"):
            FixtureSplitter(seed=42, train_ratio=1.0)

    def test_invalid_train_ratio_negative(self) -> None:
        with pytest.raises(ValueError, match="train_ratio"):
            FixtureSplitter(seed=42, train_ratio=-0.5)

    def test_hash_fixture_in_range(self, tmp_path: Path) -> None:
        splitter = FixtureSplitter(seed=42, train_ratio=0.7)
        for i in range(50):
            val = splitter._hash_fixture(tmp_path / f"f_{i}")
            assert 0.0 <= val < 1.0

    def test_split_metadata(self, tmp_path: Path) -> None:
        fixtures = self._make_fixtures(tmp_path, 10)
        splitter = FixtureSplitter(seed=7, train_ratio=0.6)
        result = splitter.split(fixtures)

        assert result.seed == 7
        assert result.train_ratio == 0.6
        assert result.total == 10
