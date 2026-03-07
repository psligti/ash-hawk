"""Score normalization utilities for consistent handling of grader outputs.

This module provides utilities to normalize scores from various sources
(LLM judges, heuristics, etc.) to a consistent 0.0-1.0 scale.

Extracted from iron-rook eval infrastructure.
"""

from __future__ import annotations

import math
from typing import Any


def normalize_score(value: Any) -> float:
    """Normalize a score value to the 0.0-1.0 range.

    Handles:
    - Already normalized scores (0.0-1.0)
    - Scores on 0-10 scale
    - Scores on 0-100 scale
    - NaN and None values
    - String representations of numbers

    Args:
        value: The score value to normalize (any type)

    Returns:
        Normalized score in [0.0, 1.0] range

    Examples:
        >>> normalize_score(0.8)
        0.8
        >>> normalize_score(8)
        0.8
        >>> normalize_score(80)
        0.8
        >>> normalize_score(None)
        0.0
        >>> normalize_score(float('nan'))
        0.0
    """
    # Handle None
    if value is None:
        return 0.0

    # Try to convert to float
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0

    # Handle NaN
    if math.isnan(score):
        return 0.0

    # Handle infinity
    if math.isinf(score):
        return 1.0 if score > 0 else 0.0

    # Negative scores clamp to 0.0
    if score < 0.0:
        return 0.0

    # Already normalized (0.0-1.0)
    if score <= 1.0:
        return score

    # 1-10 scale
    if score <= 10.0:
        return score / 10.0

    # 1-100 scale (percentage)
    if score <= 100.0:
        return score / 100.0

    # Anything larger, clamp to 1.0
    return 1.0


def normalize_grader_scores(
    grader_results: list[Any],
    score_attr: str = "score",
    grader_type_attr: str = "grader_type",
) -> dict[str, float]:
    """Extract and normalize scores from a list of grader results.

    Args:
        grader_results: List of grader result objects
        score_attr: Attribute name for the score value
        grader_type_attr: Attribute name for the grader type

    Returns:
        Dict mapping grader_type to normalized score

    Examples:
        >>> results = [
        ...     type('Result', (), {'grader_type': 'llm_judge', 'score': 0.8})(),
        ...     type('Result', (), {'grader_type': 'string_match', 'score': 1})(),
        ... ]
        >>> normalize_grader_scores(results)
        {'llm_judge': 0.8, 'string_match': 1.0}
    """
    normalized: dict[str, float] = {}
    for result in grader_results:
        grader_type = getattr(result, grader_type_attr, None)
        if not isinstance(grader_type, str):
            continue
        raw_score = getattr(result, score_attr, 0.0)
        normalized[grader_type] = normalize_score(raw_score)
    return normalized


def compute_weighted_score(
    scores: dict[str, float],
    weights: dict[str, float],
    required_graders: set[str] | None = None,
) -> float:
    """Compute a weighted average of scores.

    Args:
        scores: Dict mapping grader type to score
        weights: Dict mapping grader type to weight
        required_graders: If provided, return 0.0 if any required grader is missing

    Returns:
        Weighted average score in [0.0, 1.0]

    Examples:
        >>> scores = {'llm_judge': 0.8, 'string_match': 0.5}
        >>> weights = {'llm_judge': 0.7, 'string_match': 0.3}
        >>> compute_weighted_score(scores, weights)
        0.71
    """
    # Check required graders
    if required_graders:
        missing = required_graders - set(scores.keys())
        if missing:
            return 0.0

    total_weight = 0.0
    weighted_sum = 0.0

    for grader_type, weight in weights.items():
        if grader_type in scores:
            weighted_sum += scores[grader_type] * weight
            total_weight += weight

    if total_weight == 0:
        return 0.0

    return weighted_sum / total_weight


def score_to_grade(score: float) -> str:
    """Convert a normalized score to a letter grade.

    Args:
        score: Normalized score in [0.0, 1.0]

    Returns:
        Letter grade: A, B, C, D, or F

    Examples:
        >>> score_to_grade(0.95)
        'A'
        >>> score_to_grade(0.65)
        'C'
    """
    if score >= 0.9:
        return "A"
    if score >= 0.8:
        return "B"
    if score >= 0.7:
        return "C"
    if score >= 0.6:
        return "D"
    return "F"


__all__ = [
    "normalize_score",
    "normalize_grader_scores",
    "compute_weighted_score",
    "score_to_grade",
]
