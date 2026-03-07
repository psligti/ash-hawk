"""Brier score computation for calibration."""

from __future__ import annotations

from typing import Sequence


def compute_brier_score(
    predicted_scores: Sequence[float],
    actual_outcomes: Sequence[bool],
) -> float:
    """Compute Brier score.

    Brier score = mean((predicted - actual)^2)
    Lower is better; 0.0 indicates perfect predictions.

    Args:
        predicted_scores: Predicted scores in [0.0, 1.0]
        actual_outcomes: Actual pass/fail outcomes (converted to 0/1)

    Returns:
        Brier score (0.0 = perfect)

    Raises:
        ValueError: If inputs are empty or have different lengths
    """
    if len(predicted_scores) != len(actual_outcomes):
        raise ValueError("predicted_scores and actual_outcomes must have same length")
    if len(predicted_scores) == 0:
        raise ValueError("Cannot compute Brier score on empty data")

    total = 0.0
    for pred, actual in zip(predicted_scores, actual_outcomes):
        actual_value = 1.0 if actual else 0.0
        total += (pred - actual_value) ** 2

    return total / len(predicted_scores)


def compute_brier_skill_score(
    predicted_scores: Sequence[float],
    actual_outcomes: Sequence[bool],
) -> float:
    """Compute Brier Skill Score (BSS).

    BSS = 1 - (Brier_score / Brier_ref)
    where Brier_ref is the Brier score of a baseline that always predicts
    the mean actual outcome.

    BSS > 0: Better than baseline
    BSS = 0: Same as baseline
    BSS < 0: Worse than baseline

    Args:
        predicted_scores: Predicted scores in [0.0, 1.0]
        actual_outcomes: Actual pass/fail outcomes

    Returns:
        Brier Skill Score
    """
    brier_model = compute_brier_score(predicted_scores, actual_outcomes)

    # Baseline: always predict mean actual outcome
    mean_actual = sum(1 for a in actual_outcomes if a) / len(actual_outcomes)
    brier_ref = mean_actual * (1 - mean_actual)  # Variance of Bernoulli

    if brier_ref == 0:
        return 1.0 if brier_model == 0 else float("-inf")

    return 1.0 - (brier_model / brier_ref)


__all__ = ["compute_brier_score", "compute_brier_skill_score"]
