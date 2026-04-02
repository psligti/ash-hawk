# type-hygiene: skip-file  # test file — mock/factory types are intentionally loose
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ash_hawk.graders.emotion_config import EmotionGraderConfig
from ash_hawk.graders.emotion_scorer import StepEmotionScore
from ash_hawk.graders.emotional import EmotionalGrader
from ash_hawk.types import EvalTranscript, EvalTrial, GraderSpec


def _build_step_score(
    step_index: int,
    scores: dict[str, float],
    confidence: dict[str, float],
    reasoning: str,
) -> StepEmotionScore:
    return StepEmotionScore(
        step_index=step_index,
        event_type="trace",
        scores=scores,
        confidence=confidence,
        reasoning=reasoning,
    )


class TestEmotionGraderConfig:
    def test_default_dimensions(self) -> None:
        config = EmotionGraderConfig()
        assert len(config.dimensions) == 3


class TestEmotionalGrader:
    @pytest.mark.asyncio
    async def test_grade_empty_trace_events(self) -> None:
        grader = EmotionalGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(trace_events=[])
        spec = GraderSpec(grader_type="emotional")

        result = await grader.grade(trial, transcript, spec)

        assert result.score == 0.5
        assert result.passed is True
        assert result.details["data_quality"] == "empty"

    @pytest.mark.asyncio
    async def test_grade_llm_import_error(self) -> None:
        grader = EmotionalGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(trace_events=[{"event_type": "model_message"}])
        spec = GraderSpec(grader_type="emotional")

        with patch.object(grader, "_get_llm_client", side_effect=ImportError("missing")):
            result = await grader.grade(trial, transcript, spec)

        assert result.score == 0.5
        assert result.passed is True
        assert result.details["data_quality"] == "all_failed"
        assert result.error_message == "missing"

    @pytest.mark.asyncio
    async def test_grade_with_mocked_scorer(self) -> None:
        grader = EmotionalGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            trace_events=[
                {"event_type": "model_message", "data": {"content": "Hi"}},
                {"event_type": "tool_result", "data": {"result": "ok"}},
            ]
        )
        spec = GraderSpec(grader_type="emotional")

        step_scores = [
            _build_step_score(0, {"confidence": 0.2, "engagement": 0.3}, {"confidence": 0.5}, "ok"),
            _build_step_score(
                1, {"confidence": 0.1, "engagement": -0.1}, {"confidence": 0.4}, "ok"
            ),
        ]

        with (
            patch.object(
                getattr(grader, "_scorer"), "score_step", new=AsyncMock(side_effect=step_scores)
            ),
            patch.object(grader, "_get_llm_client", return_value=MagicMock()),
        ):
            result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.details["step_emotions"]
        assert result.details["emotion_trajectory"]["confidence"] == [0.2, 0.1]
        assert "dimension_summaries" in result.details

    @pytest.mark.asyncio
    async def test_grade_always_passed_true(self) -> None:
        grader = EmotionalGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(trace_events=[{"event_type": "model_message"}])
        spec = GraderSpec(grader_type="emotional")

        with (
            patch.object(
                getattr(grader, "_scorer"),
                "score_step",
                new=AsyncMock(
                    return_value=_build_step_score(0, {"confidence": -1.0}, {"confidence": 0.0}, "")
                ),
            ),
            patch.object(grader, "_get_llm_client", return_value=MagicMock()),
        ):
            result = await grader.grade(trial, transcript, spec)

        assert result.passed is True

    def test_compute_aggregate_score_normalizes_range(self) -> None:
        grader = EmotionalGrader()
        all_negative = [_build_step_score(0, {"confidence": -1.0}, {"confidence": 0.0}, "")]
        all_positive = [_build_step_score(0, {"confidence": 1.0}, {"confidence": 0.0}, "")]
        mixed = [
            _build_step_score(0, {"confidence": -1.0}, {"confidence": 0.0}, ""),
            _build_step_score(1, {"confidence": 1.0}, {"confidence": 0.0}, ""),
        ]
        compute_aggregate_score = getattr(grader, "_compute_aggregate_score")
        assert compute_aggregate_score(all_negative) == 0.0
        assert compute_aggregate_score(all_positive) == 1.0
        assert compute_aggregate_score(mixed) == 0.5

    def test_is_failure_score_detects_neutral(self) -> None:
        grader = EmotionalGrader()
        neutral = _build_step_score(0, {"confidence": 0.0}, {"confidence": 0.0}, "")
        is_failure_score = getattr(grader, "_is_failure_score")
        assert is_failure_score(neutral) is True

        with_reasoning = _build_step_score(0, {"confidence": 0.0}, {"confidence": 0.0}, "ok")
        assert is_failure_score(with_reasoning) is False

        with_scores = _build_step_score(0, {"confidence": 0.1}, {"confidence": 0.0}, "")
        assert is_failure_score(with_scores) is False

    def test_build_trajectory(self) -> None:
        grader = EmotionalGrader()
        step_scores = [
            _build_step_score(0, {"confidence": 0.2, "engagement": -0.1}, {"confidence": 0.3}, ""),
            _build_step_score(1, {"confidence": 0.4, "engagement": 0.2}, {"confidence": 0.5}, ""),
        ]
        build_trajectory = getattr(grader, "_build_trajectory")
        trajectory = build_trajectory(step_scores)
        assert trajectory["confidence"] == [0.2, 0.4]
        assert trajectory["engagement"] == [-0.1, 0.2]

    def test_build_dimension_summaries(self) -> None:
        grader = EmotionalGrader()
        step_scores = [
            _build_step_score(0, {"confidence": 0.2}, {"confidence": 0.4}, ""),
            _build_step_score(1, {"confidence": -0.2}, {"confidence": 0.6}, ""),
        ]
        build_dimension_summaries = getattr(grader, "_build_dimension_summaries")
        summaries = build_dimension_summaries(step_scores)
        summary = summaries["confidence"]
        assert summary["mean"] == 0.0
        assert summary["min"] == -0.2
        assert summary["max"] == 0.2
        assert summary["confidence_mean"] == 0.5
