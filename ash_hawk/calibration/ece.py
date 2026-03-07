"""Expected Calibration Error (ECE) computation."""

from __future__ import annotations

from typing import Sequence


def compute_ece(
    predicted_scores: Sequence[float],
    actual_outcomes: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    ECE measures the difference between predicted probabilities and actual outcomes.
    Lower is better; 0.0 indicates perfect calibration.

    Args:
        predicted_scores: Predicted scores in [0.0, 1.0]
        actual_outcomes: Actual pass/fail outcomes
        n_bins: Number of equal-width bins for reliability diagram

    Returns:
        ECE value (0.0 = perfectly calibrated)

    Raises:
        ValueError: If inputs are empty or have different lengths
    """
    if len(predicted_scores) != len(actual_outcomes):
        raise ValueError("predicted_scores and actual_outcomes must have same length")
    if len(predicted_scores) == 0:
        raise ValueError("Cannot compute ECE on empty data")

    # Create bins
    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]

    # Assign predictions to bins
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for pred, actual in zip(predicted_scores, actual_outcomes):
        # Find bin index (handle edge case of score == 1.0)
        bin_idx = min(int(pred * n_bins), n_bins - 1)
        bins[bin_idx].append((pred, actual))

    # Compute ECE
    total_samples = len(predicted_scores)
    ece = 0.0

    for bin_data in bins:
        if not bin_data:
            continue

        bin_count = len(bin_data)
        predicted_mean = sum(p for p, _ in bin_data) / bin_count
        actual_accuracy = sum(1 for _, a in bin_data if a) / bin_count

        # Weighted contribution to ECE
        ece += (bin_count / total_samples) * abs(predicted_mean - actual_accuracy)

    return ece


def compute_ece_with_bins(
    predicted_scores: Sequence[float],
    actual_outcomes: Sequence[bool],
    n_bins: int = 10,
) -> tuple[float, list[tuple[float, float, int]]]:
    """Compute ECE with per-bin accuracy details.

    Returns:
        Tuple of (ECE, list of (predicted_mean, actual_accuracy, count) per bin)
    """
    if len(predicted_scores) != len(actual_outcomes):
        raise ValueError("predicted_scores and actual_outcomes must have same length")
    if len(predicted_scores) == 0:
        raise ValueError("Cannot compute ECE on empty data")

    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for pred, actual in zip(predicted_scores, actual_outcomes):
        bin_idx = min(int(pred * n_bins), n_bins - 1)
        bins[bin_idx].append((pred, actual))

    total_samples = len(predicted_scores)
    ece = 0.0
    bin_details: list[tuple[float, float, int]] = []

    for bin_data in bins:
        if not bin_data:
            bin_details.append((0.0, 0.0, 0))
            continue

        bin_count = len(bin_data)
        predicted_mean = sum(p for p, _ in bin_data) / bin_count
        actual_accuracy = sum(1 for _, a in bin_data if a) / bin_count

        ece += (bin_count / total_samples) * abs(predicted_mean - actual_accuracy)
        bin_details.append((predicted_mean, actual_accuracy, bin_count))

    return ece, bin_details


__all__ = ["compute_ece", "compute_ece_with_bins"]
