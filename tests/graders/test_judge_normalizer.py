"""Tests for judge_normalizer module."""

from __future__ import annotations

import pytest

from ash_hawk.graders.judge_normalizer import (
    NormalizedJudgeOutput,
    _extract_score_from_text,
    _normalize_score_value,
    _parse_json_string,
    normalize_judge_output,
)


class TestNormalizedJudgeOutput:
    """Tests for NormalizedJudgeOutput model."""

    def test_valid_output(self) -> None:
        output = NormalizedJudgeOutput(
            score=0.8, passed=True, reasoning="Good work", issues=[], strengths=[]
        )
        assert output.score == 0.8
        assert output.passed is True
        assert output.reasoning == "Good work"

    def test_score_out_of_range(self) -> None:
        with pytest.raises(Exception):  # Pydantic validation error
            NormalizedJudgeOutput(score=1.5, passed=True, reasoning="test", issues=[], strengths=[])


class TestNormalizeJudgeOutput:
    """Tests for normalize_judge_output function."""

    def test_direct_format(self) -> None:
        raw = {"score": 0.8, "passed": True, "reasoning": "Good work"}
        result = normalize_judge_output(raw)
        assert result.score == 0.8
        assert result.passed is True
        assert result.reasoning == "Good work"

    def test_nested_answer_overall_score(self) -> None:
        raw = {"answer": {"overall_score": 0.75, "reasoning": "Decent"}}
        result = normalize_judge_output(raw)
        assert result.score == 0.75
        assert result.reasoning == "Decent"

    def test_nested_answer_score(self) -> None:
        raw = {"answer": {"score": 0.9}}
        result = normalize_judge_output(raw)
        assert result.score == 0.9

    def test_overall_assessment_float(self) -> None:
        raw = {"answer": {"overall_assessment": 0.85}}
        result = normalize_judge_output(raw)
        assert result.score == 0.85

    def test_overall_assessment_dict(self) -> None:
        raw = {"answer": {"overall_assessment": {"score": 0.7}}}
        result = normalize_judge_output(raw)
        assert result.score == 0.7

    def test_dimension_scores(self) -> None:
        raw = {
            "factual_accuracy": {"score": 0.8},
            "logical_soundness": {"score": 0.6},
        }
        result = normalize_judge_output(raw)
        assert result.score == 0.7  # Average of 0.8 and 0.6

    def test_extracted_issues(self) -> None:
        raw = {
            "score": 0.5,
            "issues": ["Missing error handling", "No tests"],
        }
        result = normalize_judge_output(raw)
        assert "Missing error handling" in result.issues
        assert "No tests" in result.issues

    def test_extracted_strengths(self) -> None:
        raw = {
            "score": 0.8,
            "strengths": ["Good documentation", "Clean code"],
        }
        result = normalize_judge_output(raw)
        assert "Good documentation" in result.strengths

    def test_breakdown_extraction(self) -> None:
        raw = {
            "breakdown": {"clarity": 0.8, "accuracy": 0.9},
        }
        result = normalize_judge_output(raw)
        assert result.breakdown == {"clarity": 0.8, "accuracy": 0.9}

    def test_json_string_input(self) -> None:
        raw = '{"score": 0.75, "passed": true, "reasoning": "test"}'
        result = normalize_judge_output(raw)
        assert result.score == 0.75

    def test_markdown_json_block(self) -> None:
        raw = '```json\n{"score": 0.8}\n```'
        result = normalize_judge_output(raw)
        assert result.score == 0.8

    def test_invalid_json_returns_failed(self) -> None:
        result = normalize_judge_output("not valid json")
        assert result.score == 0.0
        assert result.passed is False
        assert "No JSON" in result.reasoning or "error" in result.reasoning.lower()

    def test_non_dict_returns_failed(self) -> None:
        result = normalize_judge_output(12345)
        assert result.score == 0.0
        assert "not a string or dict" in result.reasoning

    def test_passed_from_score_threshold(self) -> None:
        raw = {"score": 0.8}  # Above default threshold of 0.7
        result = normalize_judge_output(raw, pass_threshold=0.7)
        assert result.passed is True

        raw = {"score": 0.6}  # Below threshold
        result = normalize_judge_output(raw, pass_threshold=0.7)
        assert result.passed is False

    def test_is_correct_as_passed(self) -> None:
        raw = {"score": 0.5, "is_correct": True}
        result = normalize_judge_output(raw)
        assert result.passed is True

    def test_answer_string_as_reasoning(self) -> None:
        raw = {"answer": "This is the analysis text"}
        result = normalize_judge_output(raw)
        assert result.reasoning == "This is the analysis text"


class TestParseJsonString:
    """Tests for _parse_json_string function."""

    def test_plain_json(self) -> None:
        text = '{"key": "value"}'
        result = _parse_json_string(text)
        assert result == {"key": "value"}

    def test_json_in_markdown_block(self) -> None:
        text = '```json\n{"key": "value"}\n```'
        result = _parse_json_string(text)
        assert result == {"key": "value"}

    def test_json_in_generic_markdown_block(self) -> None:
        text = '```\n{"key": "value"}\n```'
        result = _parse_json_string(text)
        assert result == {"key": "value"}

    def test_embedded_json(self) -> None:
        text = 'Some text before {"key": "value"} some text after'
        result = _parse_json_string(text)
        assert result == {"key": "value"}

    def test_nested_json(self) -> None:
        text = '{"outer": {"inner": "value"}}'
        result = _parse_json_string(text)
        assert result == {"outer": {"inner": "value"}}

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="No JSON object"):
            _parse_json_string("no json here")


class TestExtractScoreFromText:
    """Tests for _extract_score_from_text function."""

    def test_score_pattern(self) -> None:
        assert _extract_score_from_text("score: 0.8") == 0.8
        assert _extract_score_from_text("Score: 0.75") == 0.75
        assert _extract_score_from_text("SCORE 0.9") == 0.9

    def test_fraction_pattern(self) -> None:
        assert _extract_score_from_text("8/10") == 0.8
        # Note: "7.5/10" doesn't match the pattern (only integer fractions supported)

    def test_percentage_pattern(self) -> None:
        assert _extract_score_from_text("80%") == 0.8
        assert _extract_score_from_text("65%") == 0.65

    def test_no_pattern_returns_zero(self) -> None:
        assert _extract_score_from_text("no score here") == 0.0


class TestNormalizeScoreValue:
    """Tests for _normalize_score_value function."""

    def test_already_normalized(self) -> None:
        assert _normalize_score_value(0.5) == 0.5

    def test_ten_scale(self) -> None:
        assert _normalize_score_value(8) == 0.8

    def test_hundred_scale(self) -> None:
        assert _normalize_score_value(80) == 0.8

    def test_negative_clamped(self) -> None:
        assert _normalize_score_value(-0.5) == 0.0

    def test_over_100_clamped(self) -> None:
        assert _normalize_score_value(150) == 1.0
