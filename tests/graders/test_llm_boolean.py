"""Tests for ash_hawk.graders.llm_boolean module."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ash_hawk.graders.llm_boolean import (
    BooleanJudgeConfig,
    LLMBooleanJudgeGrader,
)
from ash_hawk.types import (
    EvalTranscript,
    EvalTrial,
    GraderResult,
    GraderSpec,
)


def make_transcript(
    agent_response: str | dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> EvalTranscript:
    return EvalTranscript(
        messages=messages or [],
        agent_response=agent_response,
    )


def make_trial() -> EvalTrial:
    return EvalTrial(
        id="test-trial-001",
        task_id="test-task-001",
    )


class TestBooleanJudgeConfig:
    def test_default_config(self) -> None:
        config = BooleanJudgeConfig()
        assert config.questions == []
        assert config.require_all is True
        assert config.context_max_chars == 8000
        assert config.judge_model is None
        assert config.judge_provider is None
        assert config.temperature == 0.0
        assert config.max_tokens == 256

    def test_custom_config(self) -> None:
        config = BooleanJudgeConfig(
            questions=["Is this AI-generated?", "Does this sound authentic?"],
            require_all=False,
            context_max_chars=5000,
            temperature=0.1,
        )
        assert len(config.questions) == 2
        assert config.require_all is False
        assert config.context_max_chars == 5000
        assert config.temperature == 0.1

    def test_config_from_dict(self) -> None:
        grader = LLMBooleanJudgeGrader(
            config={"questions": ["Test question?"], "require_all": False}
        )
        assert grader._config.questions == ["Test question?"]
        assert grader._config.require_all is False


class TestLLMBooleanJudgeGrader:
    def test_grader_name(self) -> None:
        grader = LLMBooleanJudgeGrader()
        assert grader.name == "llm_boolean"

    def test_extract_content_from_string_response(self) -> None:
        grader = LLMBooleanJudgeGrader()
        transcript = make_transcript(agent_response="This is the agent response.")
        trial = make_trial()

        content = grader._extract_content(transcript, trial)
        assert "This is the agent response." in content

    def test_extract_content_from_dict_response(self) -> None:
        grader = LLMBooleanJudgeGrader()
        transcript = make_transcript(agent_response={"content": "Dict response content"})
        trial = make_trial()

        content = grader._extract_content(transcript, trial)
        assert "Dict response content" in content

    def test_extract_content_from_messages(self) -> None:
        grader = LLMBooleanJudgeGrader()
        transcript = make_transcript(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        )
        trial = make_trial()

        content = grader._extract_content(transcript, trial)
        assert "[user]: Hello" in content
        assert "[assistant]: Hi there" in content

    def test_extract_content_truncation(self) -> None:
        grader = LLMBooleanJudgeGrader(config={"context_max_chars": 100})
        long_text = "x" * 200
        transcript = make_transcript(agent_response=long_text)
        trial = make_trial()

        content = grader._extract_content(transcript, trial)
        assert len(content) <= 120
        assert "[truncated]" in content

    def test_parse_boolean_response_true_variants(self) -> None:
        grader = LLMBooleanJudgeGrader()

        result = grader._parse_boolean_response("true\ntrue\ntrue", 3)
        assert result == [True, True, True]

        result = grader._parse_boolean_response("yes\n1\ny", 3)
        assert result == [True, True, True]

    def test_parse_boolean_response_false_variants(self) -> None:
        grader = LLMBooleanJudgeGrader()

        result = grader._parse_boolean_response("false\nfalse\nfalse", 3)
        assert result == [False, False, False]

        result = grader._parse_boolean_response("no\n0\nn", 3)
        assert result == [False, False, False]

    def test_parse_boolean_response_mixed(self) -> None:
        grader = LLMBooleanJudgeGrader()

        result = grader._parse_boolean_response("true\nfalse\ntrue", 3)
        assert result == [True, False, True]

    def test_parse_boolean_response_pads_missing(self) -> None:
        grader = LLMBooleanJudgeGrader()

        result = grader._parse_boolean_response("true", 3)
        assert result == [True, False, False]

    def test_parse_boolean_response_truncates_excess(self) -> None:
        grader = LLMBooleanJudgeGrader()

        result = grader._parse_boolean_response("true\ntrue\ntrue\ntrue", 2)
        assert result == [True, True]

    @pytest.mark.asyncio
    async def test_grade_no_questions_error(self) -> None:
        grader = LLMBooleanJudgeGrader()
        transcript = make_transcript(agent_response="Test content")
        trial = make_trial()
        spec = GraderSpec(grader_type="llm_boolean", config={})

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.0
        assert "No questions provided" in result.error_message

    @pytest.mark.asyncio
    async def test_grade_empty_content_error(self) -> None:
        grader = LLMBooleanJudgeGrader(config={"questions": ["Is this AI-generated?"]})
        transcript = make_transcript(agent_response=None, messages=[])
        trial = make_trial()
        spec = GraderSpec(grader_type="llm_boolean", config={})

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.0
        assert "No content found" in result.error_message

    @pytest.mark.asyncio
    async def test_grade_require_all_true(self) -> None:
        grader = LLMBooleanJudgeGrader(
            config={
                "questions": ["Question 1?", "Question 2?"],
                "require_all": True,
            }
        )

        mock_response = MagicMock()
        mock_response.text = "true\ntrue"

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=mock_response)
        grader._client = mock_client

        transcript = make_transcript(agent_response="Test content")
        trial = make_trial()
        spec = GraderSpec(grader_type="llm_boolean", config={})

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0
        assert result.details["true_count"] == 2

    @pytest.mark.asyncio
    async def test_grade_require_all_false_partial_fail(self) -> None:
        grader = LLMBooleanJudgeGrader(
            config={
                "questions": ["Question 1?", "Question 2?"],
                "require_all": True,
            }
        )

        mock_response = MagicMock()
        mock_response.text = "true\nfalse"

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=mock_response)
        grader._client = mock_client

        transcript = make_transcript(agent_response="Test content")
        trial = make_trial()
        spec = GraderSpec(grader_type="llm_boolean", config={})

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.5
        assert result.details["true_count"] == 1

    @pytest.mark.asyncio
    async def test_grade_require_any_true(self) -> None:
        grader = LLMBooleanJudgeGrader(
            config={
                "questions": ["Question 1?", "Question 2?"],
                "require_all": False,
            }
        )

        mock_response = MagicMock()
        mock_response.text = "false\ntrue"

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=mock_response)
        grader._client = mock_client

        transcript = make_transcript(agent_response="Test content")
        trial = make_trial()
        spec = GraderSpec(grader_type="llm_boolean", config={})

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_grade_spec_config_overrides_instance_config(self) -> None:
        grader = LLMBooleanJudgeGrader(config={"questions": ["Instance question?"]})

        mock_response = MagicMock()
        mock_response.text = "true"

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=mock_response)
        grader._client = mock_client

        transcript = make_transcript(agent_response="Test content")
        trial = make_trial()
        spec = GraderSpec(
            grader_type="llm_boolean",
            config={"questions": ["Spec question?"], "require_all": False},
        )

        result = await grader.grade(trial, transcript, spec)

        mock_client.complete.assert_called_once()
        assert result.details["questions"] == ["Spec question?"]


class TestLLMBooleanJudgeGraderIntegration:
    @pytest.mark.asyncio
    async def test_ai_slop_detection_scenario(self) -> None:
        grader = LLMBooleanJudgeGrader(
            config={
                "questions": [
                    "Does this contain generic AI phrases?",
                    "Is the content overly verbose?",
                    "Does it lack specific examples?",
                ],
                "require_all": False,
            }
        )

        mock_response = MagicMock()
        mock_response.text = "false\nfalse\ntrue"

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=mock_response)
        grader._client = mock_client

        transcript = make_transcript(
            agent_response="In conclusion, this framework represents a paradigm shift."
        )
        trial = make_trial()
        spec = GraderSpec(grader_type="llm_boolean", config={})

        result = await grader.grade(trial, transcript, spec)

        assert result.grader_type == "llm_boolean"
        assert "questions" in result.details
        assert "answers" in result.details
        assert result.details["total_count"] == 3

    @pytest.mark.asyncio
    async def test_empty_llm_response(self) -> None:
        grader = LLMBooleanJudgeGrader(config={"questions": ["Test question?"]})

        mock_response = MagicMock()
        mock_response.text = ""

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=mock_response)
        grader._client = mock_client

        transcript = make_transcript(agent_response="Test content")
        trial = make_trial()
        spec = GraderSpec(grader_type="llm_boolean", config={})

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert "Empty response" in result.error_message
