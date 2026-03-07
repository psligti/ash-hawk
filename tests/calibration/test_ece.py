"""Tests for ECE computation."""

from __future__ import annotations

import pytest

from ash_hawk.calibration.ece import compute_ece, compute_ece_with_bins


class TestComputeECE:
    """Tests for compute_ece function."""

    def test_perfect_calibration(self) -> None:
        predicted = [0.0, 0.5, 1.0, 0.5]
        actual = [False, True, True, False]
        ece = compute_ece(predicted, actual)
        assert ece < 0.2

    def test_zero_ece_for_perfect_predictions(self) -> None:
        predicted = [1.0, 1.0, 0.0, 0.0]
        actual = [True, True, False, False]
        ece = compute_ece(predicted, actual)
        assert ece == 0.0

    def test_length_mismatch_raises(self) -> None:
        predicted = [0.5, 0.8]
        actual = [True]
        with pytest.raises(ValueError, match="same length"):
            compute_ece(predicted, actual)

    def test_empty_data_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_ece([], [])

    def test_uniform_predictions(self) -> None:
        predicted = [0.7] * 10
        actual = [True] * 7 + [False] * 3
        ece = compute_ece(predicted, actual)
        assert abs(ece - 0.0) < 0.01

    def test_custom_n_bins(self) -> None:
        predicted = [0.25, 0.75]
        actual = [False, True]
        ece = compute_ece(predicted, actual, n_bins=5)
        assert isinstance(ece, float)


class TestComputeECEWithBins:
    """Tests for compute_ece_with_bins function."""

    def test_returns_bin_details(self) -> None:
        predicted = [0.05, 0.15, 0.55, 0.95]
        actual = [False, False, True, True]
        ece, bins = compute_ece_with_bins(predicted, actual, n_bins=10)

        assert isinstance(ece, float)
        assert len(bins) == 10
        # 0.05 -> bin 0, 0.15 -> bin 1, 0.55 -> bin 5, 0.95 -> bin 9
        assert bins[0][2] == 1
        assert bins[1][2] == 1
        assert bins[5][2] == 1
        assert bins[9][2] == 1

    def test_empty_bins_have_zero_count(self) -> None:
        predicted = [0.5, 0.6]
        actual = [True, True]
        _, bins = compute_ece_with_bins(predicted, actual, n_bins=10)

        empty_bins = [b for b in bins if b[2] == 0]
        assert len(empty_bins) > 0
