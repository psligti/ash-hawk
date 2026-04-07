# type-hygiene: skip-file
"""Judge output normalization for handling various LLM response formats.

The LLM judge returns responses in various formats:
- {"score": 0.8, "passed": true, "reasoning": "..."}
- {"answer": {"overall_score": 0.8, ...}}
- {"answer": {"score": 0.8, ...}}
- Dimension-based scores: {"factual_accuracy": {"score": 0.8}, ...}
- Text patterns: "8/10", "80%", "score: 0.8"

This module normalizes these to a consistent NormalizedJudgeOutput schema
with robust fallback handling.

Extracted from iron-rook eval infrastructure (434 lines).
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

import pydantic as pd

logger = logging.getLogger(__name__)


class NormalizedJudgeOutput(pd.BaseModel):
    """Normalized judge output in consistent schema."""

    model_config = {"extra": "allow"}

    score: float = pd.Field(ge=0.0, le=1.0, description="Overall score from 0.0 to 1.0")
    passed: bool = pd.Field(description="Whether the response passed evaluation")
    reasoning: str = pd.Field(description="Explanation of the evaluation")
    issues: list[str] = pd.Field(default_factory=list, description="Issues found")
    strengths: list[str] = pd.Field(default_factory=list, description="Strengths found")
    breakdown: dict[str, float] | None = pd.Field(
        default=None, description="Breakdown by dimension"
    )


# Known dimension keys for multi-dimensional scoring
DIMENSION_KEYS = [
    "factual_accuracy",
    "logical_soundness",
    "completeness",
    "vulnerability_detection",
    "severity_accuracy",
    "evidence_quality",
    "boundary_detection",
    "layer_violation",
    "task_alignment",
    "relevance",
    "novelty",
    "actionability",
]


def normalize_judge_output(
    raw_output: str | dict[str, Any] | Any,
    pass_threshold: float = 0.7,
) -> NormalizedJudgeOutput:
    """Normalize various LLM response formats to JudgeOutput schema.

    Handles:
    1. Direct format: {"score": 0.8, "passed": true, "reasoning": "..."}
    2. Nested in "answer": {"answer": {"overall_score": 0.8, ...}}
    3. Nested in "answer": {"answer": {"score": 0.8, ...}}
    4. Nested with overall_assessment: {"answer": {"overall_assessment": 0.8}}
    5. Dimension scores: {"factual_accuracy": {"score": 0.8}, ...}
    6. Text patterns: "8/10", "80%", "score: 0.8"
    7. Missing fields with intelligent defaults

    Args:
        raw_output: Raw JSON string from LLM or dict
        pass_threshold: Threshold for passing when not explicit (default 0.7)

    Returns:
        NormalizedJudgeOutput with normalized fields
    """
    try:
        # Parse JSON if needed
        if isinstance(raw_output, str):
            data = _parse_json_string(raw_output)
        elif isinstance(raw_output, dict):
            data = raw_output
        else:
            logger.warning(f"Judge output is unexpected type: {type(raw_output)}")
            return _create_failed_output("Judge output was not a string or dict")

        # Validate we have a dict
        if not isinstance(data, dict):
            logger.warning(f"Judge output is not a dict: {type(data)}")
            return _create_failed_output("Judge output is not a dictionary")

        # Normalize the data
        return _normalize_data_dict(data, pass_threshold)

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse judge output as JSON: {e}")
        return _create_failed_output("Judge output was not valid JSON")
    except Exception as e:
        logger.error(f"Error normalizing judge output: {e}")
        return _create_failed_output(f"Normalization error: {e}")


def _parse_json_string(text: str) -> dict[str, Any]:
    """Extract JSON from text that may contain markdown code blocks."""
    text = text.strip()

    # Handle markdown code blocks
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        # Skip language identifier if present
        newline_pos = text.find("\n", start)
        if newline_pos != -1 and newline_pos < text.find("```", start):
            start = newline_pos + 1
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        # If it parsed to something else, wrap it
        return {"answer": result}
    except json.JSONDecodeError:
        pass

    # Try to extract first {...} block
    start = text.find("{")
    if start != -1:
        # Find matching closing brace
        depth = 0
        in_string = False
        escape_next = False
        for i, char in enumerate(text[start:]):
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    result = json.loads(text[start : start + i + 1])
                    if isinstance(result, dict):
                        return result
                    return {"answer": result}

    raise ValueError("No JSON object found in text")


def _normalize_data_dict(data: dict[str, Any], pass_threshold: float) -> NormalizedJudgeOutput:
    """Normalize a data dict into JudgeOutput format."""
    score = _extract_score(data)
    passed = _extract_passed(data, score, pass_threshold)
    reasoning = _extract_reasoning(data)
    issues = _extract_list(data, "issues")
    strengths = _extract_list(data, "strengths")
    breakdown = _extract_breakdown(data)

    return NormalizedJudgeOutput(
        score=score,
        passed=passed,
        reasoning=reasoning,
        issues=issues,
        strengths=strengths,
        breakdown=breakdown,
    )


def _extract_score(data: dict[str, Any]) -> float:
    """Extract score from various nested locations."""
    # Direct score field
    if "score" in data:
        try:
            return _normalize_score_value(float(data["score"]))
        except (TypeError, ValueError):
            pass

    # overall_score field
    if "overall_score" in data:
        try:
            return _normalize_score_value(float(data["overall_score"]))
        except (TypeError, ValueError):
            pass

    # Nested in "answer"
    if "answer" in data:
        answer = data["answer"]
        if isinstance(answer, dict):
            # answer.score
            if "score" in answer:
                try:
                    return _normalize_score_value(float(answer["score"]))
                except (TypeError, ValueError):
                    pass

            # answer.overall_score
            if "overall_score" in answer:
                try:
                    return _normalize_score_value(float(answer["overall_score"]))
                except (TypeError, ValueError):
                    pass

            # answer.overall_assessment (can be float or dict)
            if "overall_assessment" in answer:
                overall_assessment = answer["overall_assessment"]
                if isinstance(overall_assessment, dict) and "score" in overall_assessment:
                    try:
                        return _normalize_score_value(float(overall_assessment["score"]))
                    except (TypeError, ValueError):
                        pass
                elif isinstance(overall_assessment, int | float):
                    try:
                        return _normalize_score_value(float(overall_assessment))
                    except (TypeError, ValueError):
                        pass

        # answer is a string - try to extract number
        if isinstance(answer, str):
            return _extract_score_from_text(answer)

        # answer can also be a list of scores
        if isinstance(answer, list):
            valid_scores = [s for s in answer if isinstance(s, int | float)]
            if valid_scores:
                return _normalize_score_value(sum(valid_scores) / len(valid_scores))

    # Dimension scores (e.g., {"factual_accuracy": {"score": 0.8}})
    dimension_scores = _extract_dimension_scores(data)
    if dimension_scores:
        return _normalize_score_value(sum(dimension_scores) / len(dimension_scores))

    # Default fallback
    return 0.0


def _extract_dimension_scores(data: dict[str, Any]) -> list[float]:
    """Extract scores from dimension fields."""
    scores = []
    for key in DIMENSION_KEYS:
        if key in data:
            if isinstance(data[key], dict) and "score" in data[key]:
                try:
                    scores.append(_normalize_score_value(float(data[key]["score"])))
                except (TypeError, ValueError):
                    pass
            elif isinstance(data[key], int | float):
                scores.append(_normalize_score_value(float(data[key])))

    return scores


def _extract_score_from_text(text: str) -> float:
    """Extract a score from text that might contain a number.

    Looks for patterns like:
    - "score: 0.8"
    - "8/10"
    - "80%"
    """
    # Look for "score: X.X" pattern
    score_match = re.search(r"score[:\s]*(\d+\.?\d*)", text, re.IGNORECASE)
    if score_match:
        try:
            return _normalize_score_value(float(score_match.group(1)))
        except (TypeError, ValueError):
            pass

    # Look for fraction pattern (e.g., "8/10")
    fraction_match = re.search(r"(\d+)/(\d+)", text)
    if fraction_match:
        try:
            num = fraction_match.group(1)
            denom = fraction_match.group(2)
            return _normalize_score_value(float(num) / float(denom))
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    # Look for percentage pattern (e.g., "80%")
    percent_match = re.search(r"(\d+)%", text)
    if percent_match:
        try:
            return _normalize_score_value(float(percent_match.group(1)) / 100.0)
        except (TypeError, ValueError):
            pass

    # Default
    return 0.0


def _extract_passed(data: dict[str, Any], score: float, pass_threshold: float) -> bool:
    """Extract passed status from various locations."""
    # Direct passed field
    if "passed" in data:
        return bool(data["passed"])

    # is_correct field
    if "is_correct" in data:
        return bool(data["is_correct"])

    # Check in answer object
    if "answer" in data and isinstance(data["answer"], dict):
        if "passed" in data["answer"]:
            return bool(data["answer"]["passed"])
        if "is_correct" in data["answer"]:
            return bool(data["answer"]["is_correct"])

    # Calculate from score if not explicitly provided
    return score >= pass_threshold


def _extract_reasoning(data: dict[str, Any]) -> str:
    """Extract reasoning/explanation from various locations."""
    # Direct reasoning field
    if "reasoning" in data:
        return str(data["reasoning"])

    # explanation field
    if "explanation" in data:
        return str(data["explanation"])

    # analysis field
    if "analysis" in data:
        return str(data["analysis"])

    # rationale field
    if "rationale" in data:
        return str(data["rationale"])

    # Check in answer object
    if "answer" in data:
        answer = data["answer"]
        if isinstance(answer, dict):
            if "reasoning" in answer:
                return str(answer["reasoning"])
            if "explanation" in answer:
                return str(answer["explanation"])
            if "analysis" in answer:
                return str(answer["analysis"])
            if "overall_assessment" in answer and isinstance(answer["overall_assessment"], str):
                return answer["overall_assessment"]
        elif isinstance(answer, str):
            return answer

    # Try dimension explanations
    for key in ["factual_accuracy", "logical_soundness", "completeness"]:
        if key in data and isinstance(data[key], dict) and "explanation" in data[key]:
            return str(data[key]["explanation"])

    return "No reasoning provided"


def _extract_list(data: dict[str, Any], key: str) -> list[str]:
    """Extract a list from data, handling various formats."""
    # Check direct key first
    if key in data:
        value = data[key]
        if isinstance(value, list) and value:
            return [str(item) for item in value]
        elif isinstance(value, list):
            pass  # Empty list, check answer object
        elif isinstance(value, str):
            # Try to split by comma or newline
            items = [item.strip() for item in value.replace(",", "\n").split("\n") if item.strip()]
            if items:
                return items
            # Try JSON array format
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except (json.JSONDecodeError, TypeError):
                pass

    # Check in answer object
    if "answer" in data and isinstance(data["answer"], dict):
        if key in data["answer"]:
            value = data["answer"][key]
            if isinstance(value, list):
                return [str(item) for item in value]
            elif isinstance(value, str):
                items = [
                    item.strip() for item in value.replace(",", "\n").split("\n") if item.strip()
                ]
                if items:
                    return items

    return []


def _extract_breakdown(data: dict[str, Any]) -> dict[str, float] | None:
    """Extract breakdown scores from various formats."""
    if "breakdown" in data:
        bd = data["breakdown"]
        if isinstance(bd, dict):
            return {k: float(v) for k, v in bd.items() if isinstance(v, int | float)}
    if "answer" in data and isinstance(data["answer"], dict):
        if "breakdown" in data["answer"]:
            bd = data["answer"]["breakdown"]
            if isinstance(bd, dict):
                return {k: float(v) for k, v in bd.items() if isinstance(v, int | float)}
    # Build breakdown from individual dimension scores
    result = {}
    for key in DIMENSION_KEYS:
        if key in data:
            if isinstance(data[key], int | float):
                result[key] = _normalize_score_value(float(data[key]))
            elif isinstance(data[key], dict) and "score" in data[key]:
                try:
                    result[key] = _normalize_score_value(float(data[key]["score"]))
                except (TypeError, ValueError):
                    pass

    return result if result else None


def _create_failed_output(reason: str) -> NormalizedJudgeOutput:
    """Create a failed output with default values."""
    return NormalizedJudgeOutput(
        score=0.0,
        passed=False,
        reasoning=reason,
        issues=[reason],
        strengths=[],
    )


def _normalize_score_value(score: float) -> float:
    """Normalize a score value to 0.0-1.0 range.

    Handles scores that might be on different scales:
    - 0.0-1.0: return as-is
    - 1.0-10.0: divide by 10
    - 10.0-100.0: divide by 100
    """
    if math.isnan(score) or math.isinf(score):
        return 0.0
    if score < 0:
        return 0.0
    if score <= 1.0:
        return score
    if score <= 10.0:
        return score / 10.0
    if score <= 100.0:
        return score / 100.0
    return 1.0


__all__ = [
    "NormalizedJudgeOutput",
    "normalize_judge_output",
    "DIMENSION_KEYS",
]
