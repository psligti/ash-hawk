"""Tests for Brier score computation."""

from __future__ import annotations

import pytest

from ash_hawk.calibration.brier import compute_brier_score, compute_brier_skill_score


class TestComputeBrierScore:
    """Tests for compute_brier_score function."""

    def test_perfect_score(self) -> None:
        predicted = [1.0, 1.0, 0.0, 0.0]
        actual = [True, True, False, False]
        brier = compute_brier_score(predicted, actual)
        assert brier == 0.0

    def test_worst_score(self) -> None:
        predicted = [0.0, 0.0, 1.0, 1.0]
        actual = [True, True, False, False]
        brier = compute_brier_score(predicted, actual)
        assert brier == 1.0

    def test_partial_score(self) -> None:
        predicted = [0.8, 0.6]
        actual = [True, False]
        brier = compute_brier_score(predicted, actual)
        expected = ((0.8 - 1) ** 2 + (0.6 - 0) ** 2) / 2
        assert abs(brier - expected) < 0.001

    def test_length_mismatch_raises(self) -> None:
        predicted = [0.5]
        actual = [True, False]
        with pytest.raises(ValueError, match="same length"):
            compute_brier_score(predicted, actual)

    def test_empty_data_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_brier_score([], [])


class TestComputeBrierSkillScore:
    """Tests for compute_brier_skill_score function."""

    def test_perfect_bss(self) -> None:
        predicted = [1.0, 1.0, 0.0, 0.0]
        actual = [True, True, False, False]
        bss = compute_brier_skill_score(predicted, actual)
        assert bss == 1.0

    def test_negative_bss(self) -> None:
        predicted = [0.0, 0.0, 1.0, 1.0]
        actual = [True, True, False, False]
        bss = compute_brier_skill_score(predicted, actual)
        assert bss < 0

    def test_baseline_equals_zero_bss(self) -> None:
        predicted = [0.5, 0.5]
        actual = [True, False]
        bss = compute_brier_skill_score(predicted, actual)
        assert abs(bss) < 0.01

    def test_better_than_baseline(self) -> None:
        predicted = [0.9, 0.1]
        actual = [True, False]
        bss = compute_brier_skill_score(predicted, actual)
        assert bss > 0
