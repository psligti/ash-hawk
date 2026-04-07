# type-hygiene: skip-file
"""Tests for ash_hawk.graders.llm_judge module."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ash_hawk.graders.llm_judge import (
    JudgeAuditInfo,
    JudgeConfig,
    JudgeOutput,
    LLMJudgeGrader,
)
from ash_hawk.prompts import clear_cache, list_judge_prompts, load_judge_prompt
from ash_hawk.types import (
    EvalTranscript,
    EvalTrial,
    GraderResult,
    GraderSpec,
)


@pytest.fixture(autouse=True)
def clear_prompt_cache():
    clear_cache()
    yield
    clear_cache()


class TestJudgeConfig:
    def test_default_config(self):
        config = JudgeConfig()
        assert config.rubric == "correctness"
        assert config.pass_threshold == 0.7
        assert config.judge_model is None
        assert config.judge_provider is None
        assert config.temperature == 0.0
        assert config.max_tokens == 1024
        assert config.n_judges == 1
        assert config.consensus_mode == "mean"

    def test_custom_config(self):
        config = JudgeConfig(
            rubric="safety",
            pass_threshold=0.9,
            n_judges=3,
            consensus_mode="all_must_pass",
        )
        assert config.rubric == "safety"
        assert config.pass_threshold == 0.9
        assert config.n_judges == 3
        assert config.consensus_mode == "all_must_pass"

    def test_config_from_dict(self):
        grader = LLMJudgeGrader(config={"rubric": "quality", "temperature": 0.5})
        assert grader._config.rubric == "quality"
        assert grader._config.temperature == 0.5


class TestJudgeOutput:
    def test_valid_output(self):
        output = JudgeOutput(
            score=0.85,
            passed=True,
            reasoning="Good response",
            breakdown={"clarity": 0.9, "structure": 0.8},
            issues=["Minor issue"],
            strengths=["Clear explanation"],
        )
        assert output.score == 0.85
        assert output.passed is True
        assert output.reasoning == "Good response"

    def test_score_validation(self):
        with pytest.raises(Exception):
            JudgeOutput(score=1.5, passed=True, reasoning="test")

        with pytest.raises(Exception):
            JudgeOutput(score=-0.1, passed=True, reasoning="test")


class TestPromptLoading:
    def test_list_judge_prompts(self):
        prompts = list_judge_prompts()
        assert "correctness" in prompts
        assert "relevance" in prompts
        assert "safety" in prompts
        assert "quality" in prompts

    def test_load_correctness_prompt(self):
        info = load_judge_prompt("correctness")
        assert info.name == "correctness"
        assert info.version == "1.0.0"
        assert len(info.content) > 0
        assert len(info.content_hash) == 16

    def test_load_relevance_prompt(self):
        info = load_judge_prompt("relevance")
        assert info.name == "relevance"
        assert "{task_input}" in info.content

    def test_load_safety_prompt(self):
        info = load_judge_prompt("safety")
        assert info.name == "safety"
        assert "harm avoidance" in info.content.lower()

    def test_load_quality_prompt(self):
        info = load_judge_prompt("quality")
        assert info.name == "quality"
        assert "professionalism" in info.content.lower()

    def test_load_nonexistent_prompt(self):
        with pytest.raises(FileNotFoundError):
            load_judge_prompt("nonexistent")


class TestLLMJudgeGrader:
    def test_name(self):
        grader = LLMJudgeGrader()
        assert grader.name == "llm_judge"

    def test_init_with_config(self):
        config = JudgeConfig(rubric="safety")
        grader = LLMJudgeGrader(config=config)
        assert grader._config.rubric == "safety"

    def test_repr(self):
        grader = LLMJudgeGrader()
        assert "LLMJudgeGrader" in repr(grader)

    def test_format_transcript_context(self):
        grader = LLMJudgeGrader()
        transcript = EvalTranscript(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            tool_calls=[{"name": "read_file"}],
        )
        context = grader._format_transcript_context(transcript)
        assert "## Messages" in context
        assert "## Tool Calls" in context

    def test_format_transcript_context_handles_primitive_agent_response(self):
        grader = LLMJudgeGrader()
        transcript = EvalTranscript(
            messages=[
                {"role": "user", "content": "Return a number"},
                {"role": "assistant", "content": "4"},
            ]
        )
        primitive_response: Any = 4
        object.__setattr__(transcript, "agent_response", primitive_response)
        context = grader._format_transcript_context(transcript)
        assert "## Messages" in context
        assert "[assistant]: 4" in context

    def test_format_agent_response_from_transcript(self):
        grader = LLMJudgeGrader()
        transcript = EvalTranscript(agent_response="This is the agent's response.")
        response = grader._format_agent_response(transcript)
        assert response == "This is the agent's response."

    def test_format_agent_response_from_messages(self):
        grader = LLMJudgeGrader()
        transcript = EvalTranscript(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi from assistant!"},
            ]
        )
        response = grader._format_agent_response(transcript)
        assert response == "Hi from assistant!"

    def test_parse_judge_output_valid_json(self):
        grader = LLMJudgeGrader()
        raw = json.dumps(
            {
                "score": 0.85,
                "passed": True,
                "reasoning": "Good job",
                "issues": [],
                "strengths": ["Clear"],
            }
        )
        output = grader._parse_judge_output(raw)
        assert output.score == 0.85
        assert output.passed is True

    def test_parse_judge_output_markdown_wrapped(self):
        grader = LLMJudgeGrader()
        raw = """```json
{"score": 0.9, "passed": true, "reasoning": "Excellent", "issues": [], "strengths": []}
```"""
        output = grader._parse_judge_output(raw)
        assert output.score == 0.9

    def test_parse_judge_output_invalid_json(self):
        grader = LLMJudgeGrader()
        raw = "This is not JSON"
        with pytest.raises(ValueError, match="Failed to parse"):
            grader._parse_judge_output(raw)

    def test_parse_judge_output_with_prefixed_text_extracts_json_object(self):
        grader = LLMJudgeGrader()
        raw = (
            'Judge result: {"score": 0.8, "passed": true, "reasoning": "Good", '
            '"issues": [], "strengths": []}'
        )
        output = grader._parse_judge_output(raw)
        assert output.score == 0.8
        assert output.passed is True

    def test_parse_judge_output_empty_markdown_block_raises_clear_error(self):
        grader = LLMJudgeGrader()
        raw = "```json\n```"
        with pytest.raises(ValueError, match="empty JSON after extraction"):
            grader._parse_judge_output(raw)

    def test_parse_judge_output_with_nested_answer_payload(self):
        grader = LLMJudgeGrader()
        raw = json.dumps(
            {
                "answer": {
                    "overall_score": 4,
                    "explanation": "Solid response with minor gaps",
                }
            }
        )
        output = grader._parse_judge_output(raw)
        assert output.score == 0.75
        assert output.passed is True
        assert "Solid response" in output.reasoning

    def test_parse_judge_output_with_dimension_score_objects(self):
        grader = LLMJudgeGrader()
        raw = json.dumps(
            {
                "factual_accuracy": {"score": 0.8, "explanation": "Accurate"},
                "logical_soundness": {"score": 0.6, "explanation": "Mostly logical"},
                "completeness": {"score": 0.7, "explanation": "Complete enough"},
            }
        )
        output = grader._parse_judge_output(raw)
        assert output.score == pytest.approx(0.7, rel=0.001)
        assert output.passed is True
        assert output.breakdown is not None
        assert output.breakdown["factual_accuracy"] == 0.8

    def test_parse_judge_output_nested_passed_overrides_score(self):
        grader = LLMJudgeGrader()
        raw = json.dumps(
            {
                "answer": {
                    "overall_score": 5,
                    "passed": False,
                    "reasoning": "Critical issue present",
                }
            }
        )
        output = grader._parse_judge_output(raw)
        assert output.score == 1.0
        assert output.passed is False
        assert output.reasoning == "Critical issue present"

    def test_aggregate_consensus_single(self):
        grader = LLMJudgeGrader()
        output = JudgeOutput(score=0.8, passed=True, reasoning="Good")
        result = grader._aggregate_consensus([output])
        assert result.score == 0.8

    def test_aggregate_consensus_mean(self):
        config = JudgeConfig(consensus_mode="mean")
        grader = LLMJudgeGrader(config=config)
        outputs = [
            JudgeOutput(score=0.6, passed=True, reasoning="A"),
            JudgeOutput(score=0.8, passed=True, reasoning="B"),
        ]
        result = grader._aggregate_consensus(outputs)
        assert result.score == 0.7

    def test_aggregate_consensus_median(self):
        config = JudgeConfig(consensus_mode="median")
        grader = LLMJudgeGrader(config=config)
        outputs = [
            JudgeOutput(score=0.5, passed=True, reasoning="A"),
            JudgeOutput(score=0.7, passed=True, reasoning="B"),
            JudgeOutput(score=0.9, passed=True, reasoning="C"),
        ]
        result = grader._aggregate_consensus(outputs)
        assert result.score == 0.7

    def test_aggregate_consensus_min(self):
        config = JudgeConfig(consensus_mode="min")
        grader = LLMJudgeGrader(config=config)
        outputs = [
            JudgeOutput(score=0.5, passed=True, reasoning="A"),
            JudgeOutput(score=0.9, passed=True, reasoning="B"),
        ]
        result = grader._aggregate_consensus(outputs)
        assert result.score == 0.5

    def test_aggregate_consensus_all_must_pass(self):
        config = JudgeConfig(consensus_mode="all_must_pass", pass_threshold=0.6)
        grader = LLMJudgeGrader(config=config)
        outputs = [
            JudgeOutput(score=0.7, passed=True, reasoning="A"),
            JudgeOutput(score=0.8, passed=True, reasoning="B"),
        ]
        result = grader._aggregate_consensus(outputs)
        assert result.passed is True

        outputs[1] = JudgeOutput(score=0.5, passed=False, reasoning="B")
        result = grader._aggregate_consensus(outputs)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_grade_success(self):
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "score": 0.85,
                "passed": True,
                "reasoning": "Good response",
                "issues": [],
                "strengths": ["Clear"],
            }
        )

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=mock_response)

        config = JudgeConfig(rubric="correctness")
        grader = LLMJudgeGrader(config=config, client=mock_client)

        trial = EvalTrial(id="t1", task_id="task1", input_snapshot="Write a function")
        transcript = EvalTranscript(agent_response="Here is the function...")
        spec = GraderSpec(grader_type="llm_judge")

        result = await grader.grade(trial, transcript, spec)

        assert isinstance(result, GraderResult)
        assert result.grader_type == "llm_judge"
        assert result.score == 0.85
        assert result.passed is True
        assert "audit" in result.details

    @pytest.mark.asyncio
    async def test_grade_with_spec_config_override(self):
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "score": 0.9,
                "passed": True,
                "reasoning": "Excellent",
                "issues": [],
                "strengths": [],
            }
        )

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=mock_response)

        grader = LLMJudgeGrader(client=mock_client)

        trial = EvalTrial(id="t1", task_id="task1", input_snapshot="Test")
        transcript = EvalTranscript(agent_response="Response")
        spec = GraderSpec(
            grader_type="llm_judge",
            config={"rubric": "safety"},
        )

        await grader.grade(trial, transcript, spec)
        assert grader._config.rubric == "safety"

    @pytest.mark.asyncio
    async def test_grade_records_audit_info(self):
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "score": 0.8,
                "passed": True,
                "reasoning": "Good",
                "issues": [],
                "strengths": [],
            }
        )

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=mock_response)

        config = JudgeConfig(
            rubric="quality",
            temperature=0.5,
            n_judges=1,
        )
        grader = LLMJudgeGrader(config=config, client=mock_client)

        trial = EvalTrial(id="t1", task_id="task1", input_snapshot="Test")
        transcript = EvalTranscript(agent_response="Response")
        spec = GraderSpec(grader_type="llm_judge")

        result = await grader.grade(trial, transcript, spec)

        audit = result.details.get("audit", {})
        assert audit.get("prompt_name") == "quality"
        assert audit.get("prompt_version") == "1.0.0"
        assert len(audit.get("prompt_hash", "")) == 16
        assert "judge_params" in audit
        assert audit["judge_params"]["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_grade_handles_import_error(self):
        grader = LLMJudgeGrader()
        with patch(
            "dawn_kestrel.provider.llm_client.LLMClient",
            side_effect=ImportError("dawn-kestrel not found"),
        ):
            grader._client = None
            trial = EvalTrial(id="t1", task_id="task1")
            transcript = EvalTranscript()
            spec = GraderSpec(grader_type="llm_judge")

            result = await grader.grade(trial, transcript, spec)

            assert result.passed is False
            assert result.score == 0.0
            assert result.error_message is not None
            assert "dawn-kestrel" in result.error_message

    @pytest.mark.asyncio
    async def test_grade_handles_exception(self):
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(side_effect=Exception("API error"))

        grader = LLMJudgeGrader(client=mock_client)

        trial = EvalTrial(id="t1", task_id="task1", input_snapshot="Test")
        transcript = EvalTranscript(agent_response="Response")
        spec = GraderSpec(grader_type="llm_judge")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.0
        assert result.error_message is not None
        assert "API error" in result.error_message


class TestJudgeAuditInfo:
    def test_audit_info_creation(self):
        audit = JudgeAuditInfo(
            prompt_name="correctness",
            prompt_version="1.0.0",
            prompt_hash="abc123def456",
            judge_model="claude-sonnet-4-20250514",
            judge_provider="anthropic",
            judge_params={"temperature": 0.0},
        )
        assert audit.prompt_name == "correctness"
        assert audit.prompt_version == "1.0.0"
        assert audit.prompt_hash == "abc123def456"


class TestCustomPrompt:
    """Tests for inline custom_prompt functionality."""

    def test_custom_prompt_field_in_config(self):
        """Test that custom_prompt field exists and accepts string."""
        config = JudgeConfig(custom_prompt="Rate the response quality.")
        assert config.custom_prompt == "Rate the response quality."

    def test_custom_prompt_defaults_to_none(self):
        """Test that custom_prompt defaults to None."""
        config = JudgeConfig()
        assert config.custom_prompt is None

    def test_load_prompt_uses_custom_prompt(self):
        """Test that _load_prompt returns inline custom_prompt when provided."""
        custom = "Custom evaluation instructions here."
        grader = LLMJudgeGrader(config=JudgeConfig(custom_prompt=custom))
        prompt_info = grader._load_prompt()

        assert prompt_info.name == "custom_inline_rubric"
        assert custom in prompt_info.content
        assert "{task_input}" in prompt_info.content
        assert "{agent_response}" in prompt_info.content
        assert len(prompt_info.content_hash) == 16

    def test_load_prompt_custom_prompt_takes_precedence(self, tmp_path):
        """Test that custom_prompt takes precedence over custom_prompt_path."""
        # Create a custom prompt file
        prompt_file = tmp_path / "custom.txt"
        prompt_file.write_text("This is from the file path.")

        custom_inline = "This is the inline prompt."
        grader = LLMJudgeGrader(
            config=JudgeConfig(
                custom_prompt=custom_inline,
                custom_prompt_path=str(prompt_file),
            )
        )
        prompt_info = grader._load_prompt()

        # Inline should win
        assert prompt_info.name == "custom_inline_rubric"
        assert custom_inline in prompt_info.content
        assert "This is from the file path." not in prompt_info.content

    def test_load_prompt_uses_path_when_no_inline(self, tmp_path):
        """Test that custom_prompt_path works when custom_prompt is None."""
        prompt_file = tmp_path / "custom.txt"
        prompt_file.write_text("Custom from file.")

        grader = LLMJudgeGrader(config=JudgeConfig(custom_prompt_path=str(prompt_file)))
        prompt_info = grader._load_prompt()

        assert prompt_info.name == "custom"
        assert "Custom from file." in prompt_info.content
        assert "{task_input}" in prompt_info.content

    @pytest.mark.asyncio
    async def test_grade_with_inline_custom_prompt(self):
        """Test that grading works with inline custom_prompt from GraderSpec."""
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "score": 0.9,
                "passed": True,
                "reasoning": "Excellent",
                "issues": [],
                "strengths": [],
            }
        )

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=mock_response)

        grader = LLMJudgeGrader(client=mock_client)

        trial = EvalTrial(id="t1", task_id="task1", input_snapshot="Test")
        transcript = EvalTranscript(agent_response="Response")
        spec = GraderSpec(
            grader_type="llm_judge",
            config={"custom_prompt": "Evaluate empathy level."},
        )

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 0.9
        # Verify the custom prompt was loaded
        assert grader._prompt_info.name == "custom_inline_rubric"

    @pytest.mark.asyncio
    async def test_grade_custom_prompt_in_built_judge_prompt(self):
        """Test that custom_prompt content is used in the final judge prompt."""
        custom = "Rate from 0-1 how empathetic this response is."
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "score": 0.8,
                "passed": True,
                "reasoning": "Empathetic",
                "issues": [],
                "strengths": [],
            }
        )

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=mock_response)

        grader = LLMJudgeGrader(
            config=JudgeConfig(custom_prompt=custom),
            client=mock_client,
        )

        trial = EvalTrial(id="t1", task_id="task1", input_snapshot="Test")
        transcript = EvalTranscript(agent_response="I understand your concern.")
        spec = GraderSpec(grader_type="llm_judge")

        await grader.grade(trial, transcript, spec)

        # Verify the prompt sent to LLM contains our custom prompt
        call_args = mock_client.complete.call_args
        messages = call_args.kwargs["messages"]
        assert "empathetic" in messages[0]["content"].lower()
        assert "i understand your concern" in messages[0]["content"].lower()


class TestInlineRubricFallback:
    @pytest.mark.asyncio
    async def test_multiline_rubric_is_wrapped_with_context(self):
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "score": 0.75,
                "passed": True,
                "reasoning": "Looks good",
                "issues": [],
                "strengths": [],
            }
        )

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=mock_response)

        rubric = "Evaluate:\n1. Includes context\n2. Uses JSON output\n"
        grader = LLMJudgeGrader(client=mock_client)
        trial = EvalTrial(id="t1", task_id="task1", input_snapshot="My task")
        transcript = EvalTranscript(agent_response="My response")
        spec = GraderSpec(grader_type="llm_judge", config={"rubric": rubric})

        result = await grader.grade(trial, transcript, spec)
        assert result.passed is True
        assert result.score == 0.75

        call_args = mock_client.complete.call_args
        messages = call_args.kwargs["messages"]
        content = messages[0]["content"]
        assert "My task" in content
        assert "My response" in content
        assert "Evaluate:" in content


class TestNJudgeConsensus:
    @pytest.mark.asyncio
    async def test_multiple_judges(self):
        responses = [
            json.dumps(
                {"score": 0.6, "passed": True, "reasoning": "A", "issues": [], "strengths": []}
            ),
            json.dumps(
                {"score": 0.8, "passed": True, "reasoning": "B", "issues": [], "strengths": []}
            ),
            json.dumps(
                {"score": 0.7, "passed": True, "reasoning": "C", "issues": [], "strengths": []}
            ),
        ]
        call_count = 0

        def create_response():
            nonlocal call_count
            mock = MagicMock()
            mock.text = responses[call_count % len(responses)]
            call_count += 1
            return mock

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(side_effect=lambda **_: create_response())

        config = JudgeConfig(n_judges=3, consensus_mode="mean")
        grader = LLMJudgeGrader(config=config, client=mock_client)

        trial = EvalTrial(id="t1", task_id="task1", input_snapshot="Test")
        transcript = EvalTranscript(agent_response="Response")
        spec = GraderSpec(grader_type="llm_judge")

        result = await grader.grade(trial, transcript, spec)

        assert result.score == pytest.approx(0.7, rel=0.01)
        assert mock_client.complete.call_count == 3


class TestConfidenceField:
    """Tests for confidence field population when n_judges > 1."""

    @pytest.mark.asyncio
    async def test_single_judge_no_confidence(self):
        """Single judge should not populate confidence (None)."""
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {"score": 0.8, "passed": True, "reasoning": "Good", "issues": [], "strengths": []}
        )

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=mock_response)

        config = JudgeConfig(n_judges=1)
        grader = LLMJudgeGrader(config=config, client=mock_client)

        trial = EvalTrial(id="t1", task_id="task1", input_snapshot="Test")
        transcript = EvalTranscript(agent_response="Response")
        spec = GraderSpec(grader_type="llm_judge")

        result = await grader.grade(trial, transcript, spec)

        assert result.confidence is None

    @pytest.mark.asyncio
    async def test_multiple_judges_populates_confidence(self):
        """Multiple judges should populate confidence based on variance."""
        responses = [
            json.dumps(
                {"score": 0.7, "passed": True, "reasoning": "A", "issues": [], "strengths": []}
            ),
            json.dumps(
                {"score": 0.7, "passed": True, "reasoning": "B", "issues": [], "strengths": []}
            ),
            json.dumps(
                {"score": 0.7, "passed": True, "reasoning": "C", "issues": [], "strengths": []}
            ),
        ]
        call_count = 0

        def create_response():
            nonlocal call_count
            mock = MagicMock()
            mock.text = responses[call_count % len(responses)]
            call_count += 1
            return mock

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(side_effect=lambda **_: create_response())

        config = JudgeConfig(n_judges=3, consensus_mode="mean")
        grader = LLMJudgeGrader(config=config, client=mock_client)

        trial = EvalTrial(id="t1", task_id="task1", input_snapshot="Test")
        transcript = EvalTranscript(agent_response="Response")
        spec = GraderSpec(grader_type="llm_judge")

        result = await grader.grade(trial, transcript, spec)

        assert result.confidence is not None

    @pytest.mark.asyncio
    async def test_confidence_is_one_minus_variance(self):
        """Confidence should be 1.0 - variance."""
        responses = [
            json.dumps(
                {"score": 0.5, "passed": True, "reasoning": "A", "issues": [], "strengths": []}
            ),
            json.dumps(
                {"score": 0.7, "passed": True, "reasoning": "B", "issues": [], "strengths": []}
            ),
            json.dumps(
                {"score": 0.9, "passed": True, "reasoning": "C", "issues": [], "strengths": []}
            ),
        ]
        call_count = 0

        def create_response():
            nonlocal call_count
            mock = MagicMock()
            mock.text = responses[call_count % len(responses)]
            call_count += 1
            return mock

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(side_effect=lambda **_: create_response())

        config = JudgeConfig(n_judges=3, consensus_mode="mean")
        grader = LLMJudgeGrader(config=config, client=mock_client)

        trial = EvalTrial(id="t1", task_id="task1", input_snapshot="Test")
        transcript = EvalTranscript(agent_response="Response")
        spec = GraderSpec(grader_type="llm_judge")

        result = await grader.grade(trial, transcript, spec)

        # scores: [0.5, 0.7, 0.9], mean = 0.7
        # variance = ((0.5-0.7)^2 + (0.7-0.7)^2 + (0.9-0.7)^2) / 3
        #          = (0.04 + 0 + 0.04) / 3 = 0.08/3 ≈ 0.0267
        # confidence = 1 - 0.0267 ≈ 0.9733
        expected_variance = (0.04 + 0.0 + 0.04) / 3
        expected_confidence = 1.0 - expected_variance
        assert result.confidence == pytest.approx(expected_confidence, rel=0.01)

    @pytest.mark.asyncio
    async def test_perfect_agreement_confidence_is_one(self):
        """When all judges agree perfectly, confidence should be 1.0."""
        responses = [
            json.dumps(
                {"score": 0.8, "passed": True, "reasoning": "A", "issues": [], "strengths": []}
            ),
            json.dumps(
                {"score": 0.8, "passed": True, "reasoning": "B", "issues": [], "strengths": []}
            ),
            json.dumps(
                {"score": 0.8, "passed": True, "reasoning": "C", "issues": [], "strengths": []}
            ),
        ]
        call_count = 0

        def create_response():
            nonlocal call_count
            mock = MagicMock()
            mock.text = responses[call_count % len(responses)]
            call_count += 1
            return mock

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(side_effect=lambda **_: create_response())

        config = JudgeConfig(n_judges=3, consensus_mode="mean")
        grader = LLMJudgeGrader(config=config, client=mock_client)

        trial = EvalTrial(id="t1", task_id="task1", input_snapshot="Test")
        transcript = EvalTranscript(agent_response="Response")
        spec = GraderSpec(grader_type="llm_judge")

        result = await grader.grade(trial, transcript, spec)

        assert result.confidence == pytest.approx(1.0, rel=0.001)

    @pytest.mark.asyncio
    async def test_high_disagreement_low_confidence(self):
        """High disagreement (max variance) should give low confidence."""
        responses = [
            json.dumps(
                {"score": 0.0, "passed": False, "reasoning": "A", "issues": [], "strengths": []}
            ),
            json.dumps(
                {"score": 1.0, "passed": True, "reasoning": "B", "issues": [], "strengths": []}
            ),
        ]
        call_count = 0

        def create_response():
            nonlocal call_count
            mock = MagicMock()
            mock.text = responses[call_count % len(responses)]
            call_count += 1
            return mock

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(side_effect=lambda **_: create_response())

        config = JudgeConfig(n_judges=2, consensus_mode="mean")
        grader = LLMJudgeGrader(config=config, client=mock_client)

        trial = EvalTrial(id="t1", task_id="task1", input_snapshot="Test")
        transcript = EvalTranscript(agent_response="Response")
        spec = GraderSpec(grader_type="llm_judge")

        result = await grader.grade(trial, transcript, spec)

        # scores: [0.0, 1.0], mean = 0.5
        # variance = ((0-0.5)^2 + (1-0.5)^2) / 2 = (0.25 + 0.25) / 2 = 0.25
        # confidence = 1 - 0.25 = 0.75
        assert result.confidence == pytest.approx(0.75, rel=0.01)
