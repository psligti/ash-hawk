"""Tests for score_normalizer module."""

from __future__ import annotations

import math

import pytest

from ash_hawk.graders.score_normalizer import (
    compute_weighted_score,
    normalize_grader_scores,
    normalize_score,
    score_to_grade,
)


class TestNormalizeScore:
    """Tests for normalize_score function."""

    def test_already_normalized(self) -> None:
        assert normalize_score(0.0) == 0.0
        assert normalize_score(0.5) == 0.5
        assert normalize_score(1.0) == 1.0

    def test_ten_scale(self) -> None:
        assert normalize_score(5) == 0.5
        assert normalize_score(8) == 0.8
        assert normalize_score(10) == 1.0

    def test_hundred_scale(self) -> None:
        assert normalize_score(50) == 0.5
        assert normalize_score(80) == 0.8
        assert normalize_score(100) == 1.0

    def test_negative_clamps_to_zero(self) -> None:
        assert normalize_score(-1) == 0.0
        assert normalize_score(-0.5) == 0.0
        assert normalize_score(-100) == 0.0

    def test_over_100_clamps_to_one(self) -> None:
        assert normalize_score(150) == 1.0
        assert normalize_score(1000) == 1.0

    def test_none_returns_zero(self) -> None:
        assert normalize_score(None) == 0.0

    def test_nan_returns_zero(self) -> None:
        assert normalize_score(float("nan")) == 0.0

    def test_infinity(self) -> None:
        assert normalize_score(float("inf")) == 1.0
        assert normalize_score(float("-inf")) == 0.0

    def test_string_numbers(self) -> None:
        assert normalize_score("0.8") == 0.8
        assert normalize_score("80") == 0.8

    def test_invalid_string_returns_zero(self) -> None:
        assert normalize_score("invalid") == 0.0
        assert normalize_score("") == 0.0


class TestNormalizeGraderScores:
    """Tests for normalize_grader_scores function."""

    def test_basic_normalization(self) -> None:
        class Result:
            grader_type = "llm_judge"
            score = 0.8

        results = [Result()]
        normalized = normalize_grader_scores(results)
        assert normalized == {"llm_judge": 0.8}

    def test_multiple_graders(self) -> None:
        class Result1:
            grader_type = "llm_judge"
            score = 0.8

        class Result2:
            grader_type = "string_match"
            score = 1

        results = [Result1(), Result2()]
        normalized = normalize_grader_scores(results)
        assert normalized == {"llm_judge": 0.8, "string_match": 1.0}

    def test_skips_non_string_grader_type(self) -> None:
        class Result:
            grader_type = 123  # Not a string
            score = 0.8

        results = [Result()]
        normalized = normalize_grader_scores(results)
        assert normalized == {}

    def test_uses_custom_attributes(self) -> None:
        class Result:
            type_ = "custom"
            value = 80

        results = [Result()]
        normalized = normalize_grader_scores(results, score_attr="value", grader_type_attr="type_")
        assert normalized == {"custom": 0.8}


class TestComputeWeightedScore:
    """Tests for compute_weighted_score function."""

    def test_basic_weighted_average(self) -> None:
        scores = {"llm_judge": 0.8, "string_match": 0.5}
        weights = {"llm_judge": 0.7, "string_match": 0.3}
        result = compute_weighted_score(scores, weights)
        expected = (0.8 * 0.7 + 0.5 * 0.3) / 1.0
        assert math.isclose(result, expected)

    def test_missing_grader_excluded_from_weight(self) -> None:
        scores = {"llm_judge": 0.8}
        weights = {"llm_judge": 0.7, "string_match": 0.3}
        result = compute_weighted_score(scores, weights)
        assert math.isclose(result, 0.8)  # Only llm_judge counted

    def test_empty_scores_returns_zero(self) -> None:
        scores: dict[str, float] = {}
        weights = {"llm_judge": 0.7}
        assert compute_weighted_score(scores, weights) == 0.0

    def test_required_graders_missing(self) -> None:
        scores = {"llm_judge": 0.8}
        weights = {"llm_judge": 0.7}
        required = {"llm_judge", "string_match"}
        assert compute_weighted_score(scores, weights, required) == 0.0

    def test_required_graders_present(self) -> None:
        scores = {"llm_judge": 0.8, "string_match": 0.5}
        weights = {"llm_judge": 0.7, "string_match": 0.3}
        required = {"llm_judge", "string_match"}
        result = compute_weighted_score(scores, weights, required)
        expected = (0.8 * 0.7 + 0.5 * 0.3) / 1.0
        assert math.isclose(result, expected)


class TestScoreToGrade:
    """Tests for score_to_grade function."""

    def test_a_grade(self) -> None:
        assert score_to_grade(0.95) == "A"
        assert score_to_grade(0.9) == "A"

    def test_b_grade(self) -> None:
        assert score_to_grade(0.85) == "B"
        assert score_to_grade(0.8) == "B"

    def test_c_grade(self) -> None:
        assert score_to_grade(0.75) == "C"
        assert score_to_grade(0.7) == "C"

    def test_d_grade(self) -> None:
        assert score_to_grade(0.65) == "D"
        assert score_to_grade(0.6) == "D"

    def test_f_grade(self) -> None:
        assert score_to_grade(0.5) == "F"
        assert score_to_grade(0.0) == "F"
