# type-hygiene: skip-file  # test file — mock/factory types are intentionally loose
import importlib
import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ash_hawk.graders.emotion_config import EmotionGraderConfig
from ash_hawk.graders.emotion_scorer import EmotionScorer, LLMEmotionResponse, StepEmotionScore
from ash_hawk.scenario.trace import (
    EVENT_TYPE_ARTIFACT,
    EVENT_TYPE_BUDGET,
    EVENT_TYPE_CANDIDATE_EVALUATED,
    EVENT_TYPE_DIFF,
    EVENT_TYPE_DIMENSION_SAMPLED,
    EVENT_TYPE_MODEL_MESSAGE,
    EVENT_TYPE_MUTATION_APPLIED,
    EVENT_TYPE_POLICY_DECISION,
    EVENT_TYPE_REJECTION,
    EVENT_TYPE_TODO,
    EVENT_TYPE_TOOL_CALL,
    EVENT_TYPE_TOOL_RESULT,
    EVENT_TYPE_VERIFICATION,
)


class TestStepEmotionScore:
    def test_model_creation(self) -> None:
        score = StepEmotionScore(
            step_index=1,
            event_type="model_message",
            scores={"confidence": 0.2},
            confidence={"confidence": 0.8},
            reasoning="steady",
        )
        assert score.step_index == 1
        assert score.event_type == "model_message"

    def test_extra_forbid(self) -> None:
        with pytest.raises(Exception):
            StepEmotionScore.model_validate(
                {
                    "step_index": 1,
                    "event_type": "model_message",
                    "scores": {"confidence": 0.2},
                    "confidence": {"confidence": 0.8},
                    "reasoning": "steady",
                    "extra_field": "nope",
                }
            )


class TestLLMEmotionResponse:
    def test_model_defaults(self) -> None:
        response = LLMEmotionResponse()
        assert response.scores == {}
        assert response.confidence == {}
        assert response.reasoning == ""

    def test_extra_forbid(self) -> None:
        with pytest.raises(Exception):
            LLMEmotionResponse.model_validate(
                {"scores": {}, "confidence": {}, "reasoning": "", "extra_field": True}
            )


class TestEmotionScorer:
    @pytest.mark.asyncio
    async def test_score_step_valid_json(self) -> None:
        config = EmotionGraderConfig()
        scorer = EmotionScorer(config)
        scores = {dimension.name: 0.4 for dimension in config.dimensions}
        confidence = {dimension.name: 0.6 for dimension in config.dimensions}
        response = MagicMock()
        response.text = json.dumps(
            {"scores": scores, "confidence": confidence, "reasoning": "steady"}
        )

        llm_client = MagicMock()
        llm_client.complete = AsyncMock(return_value=response)

        class DummyOptions:
            def __init__(
                self, temperature: float, max_tokens: int, response_format: dict[str, object]
            ) -> None:
                self.temperature = temperature
                self.max_tokens = max_tokens
                self.response_format = response_format

        module = SimpleNamespace(LLMRequestOptions=DummyOptions)

        step = {"event_type": EVENT_TYPE_MODEL_MESSAGE, "data": {"role": "assistant"}}
        with patch("ash_hawk.graders.emotion_scorer.importlib.import_module", return_value=module):
            result = await scorer.score_step(step, [], llm_client)

        assert result.scores == scores
        assert result.confidence == confidence
        assert result.reasoning == "steady"
        options = llm_client.complete.call_args.kwargs["options"]
        assert options.temperature == 0.0

    @pytest.mark.asyncio
    async def test_score_step_markdown_wrapped_json(self) -> None:
        config = EmotionGraderConfig()
        scorer = EmotionScorer(config)
        scores = {dimension.name: 0.1 for dimension in config.dimensions}
        response = MagicMock()
        response.text = "```json\n" + json.dumps({"scores": scores, "reasoning": "ok"}) + "\n```"

        llm_client = MagicMock()
        llm_client.complete = AsyncMock(return_value=response)

        class DummyOptions:
            def __init__(
                self, temperature: float, max_tokens: int, response_format: dict[str, object]
            ) -> None:
                self.temperature = temperature
                self.max_tokens = max_tokens
                self.response_format = response_format

        module = SimpleNamespace(LLMRequestOptions=DummyOptions)
        with patch("ash_hawk.graders.emotion_scorer.importlib.import_module", return_value=module):
            result = await scorer.score_step(
                {"event_type": EVENT_TYPE_MODEL_MESSAGE}, [], llm_client
            )

        assert result.scores == scores
        assert result.reasoning == "ok"

    @pytest.mark.asyncio
    async def test_score_step_fallback_after_retries(self) -> None:
        config = EmotionGraderConfig()
        scorer = EmotionScorer(config)
        llm_client = MagicMock()
        llm_client.complete = AsyncMock(side_effect=Exception("boom"))

        class DummyOptions:
            def __init__(
                self, temperature: float, max_tokens: int, response_format: dict[str, object]
            ) -> None:
                self.temperature = temperature
                self.max_tokens = max_tokens
                self.response_format = response_format

        module = SimpleNamespace(LLMRequestOptions=DummyOptions)
        real_import = importlib.import_module
        with (
            patch(
                "ash_hawk.graders.emotion_scorer.importlib.import_module",
                side_effect=lambda name: (
                    module if name == "dawn_kestrel.llm.client" else real_import(name)
                ),
            ),
            patch("asyncio.sleep", new_callable=AsyncMock) as sleep_mock,
        ):
            result = await scorer.score_step({"event_type": EVENT_TYPE_TOOL_RESULT}, [], llm_client)

        assert llm_client.complete.call_count == 3
        assert sleep_mock.call_count == 2
        assert all(value == 0.0 for value in result.scores.values())
        assert all(value == 0.0 for value in result.confidence.values())
        assert result.reasoning == ""

    def test_coerce_scores_clamps_values(self) -> None:
        config = EmotionGraderConfig()
        scorer = EmotionScorer(config)
        response = LLMEmotionResponse(
            scores={"confidence": 2.5, "engagement": -2.5, "effectiveness": 0.3},
            confidence={"confidence": 1.5, "engagement": -0.2, "effectiveness": 0.4},
        )
        coerce_scores = getattr(scorer, "_coerce_scores")
        scores, confidence = coerce_scores(response)
        assert scores["confidence"] == 1.0
        assert scores["engagement"] == -1.0
        assert confidence["confidence"] == 1.0
        assert confidence["engagement"] == 0.0

    def test_coerce_scores_missing_dimension_defaults_to_zero(self) -> None:
        config = EmotionGraderConfig()
        scorer = EmotionScorer(config)
        response = LLMEmotionResponse(scores={"confidence": 0.5}, confidence={})
        coerce_scores = getattr(scorer, "_coerce_scores")
        scores, confidence = coerce_scores(response)
        assert scores["engagement"] == 0.0
        assert confidence["engagement"] == 0.0

    def test_coerce_scores_non_numeric_defaults_to_zero(self) -> None:
        config = EmotionGraderConfig()
        scorer = EmotionScorer(config)
        bad_scores = cast(dict[str, float], {"confidence": "high"})
        bad_confidence = cast(dict[str, float], {"confidence": "low"})
        response = LLMEmotionResponse.model_construct(scores=bad_scores, confidence=bad_confidence)
        coerce_scores = getattr(scorer, "_coerce_scores")
        scores, confidence = coerce_scores(response)
        assert scores["confidence"] == 0.0
        assert confidence["confidence"] == 0.0

    def test_parse_llm_output_valid_json(self) -> None:
        config = EmotionGraderConfig()
        scorer = EmotionScorer(config)
        raw = json.dumps({"scores": {"confidence": 0.2}, "reasoning": "ok"})
        parse_llm_output = getattr(scorer, "_parse_llm_output")
        response = parse_llm_output(raw)
        assert response.scores["confidence"] == 0.2
        assert response.reasoning == "ok"

    def test_parse_llm_output_invalid_json_raises(self) -> None:
        config = EmotionGraderConfig()
        scorer = EmotionScorer(config)
        with pytest.raises(ValueError, match="No JSON object"):
            parse_llm_output = getattr(scorer, "_parse_llm_output")
            parse_llm_output("not json")

    @pytest.mark.parametrize(
        "event_type,data,expected",
        [
            (EVENT_TYPE_MODEL_MESSAGE, {"role": "user", "content": "Hi"}, "ModelMessage"),
            (EVENT_TYPE_TOOL_CALL, {"tool": "read"}, "ToolCall"),
            (EVENT_TYPE_TOOL_RESULT, {"tool": "read", "result": "ok"}, "ToolResult"),
            (EVENT_TYPE_VERIFICATION, {"pass": True, "message": "done"}, "Verification"),
            (EVENT_TYPE_TODO, {"content": "do"}, "Todo"),
            (EVENT_TYPE_DIFF, {"patch_text": "@@", "changed_files": 1, "added_lines": 2}, "Diff"),
            (EVENT_TYPE_ARTIFACT, {"artifact_key": "log"}, "Artifact"),
            (
                EVENT_TYPE_POLICY_DECISION,
                {"tool_name": "read", "policy": "allow"},
                "PolicyDecision",
            ),
            (EVENT_TYPE_REJECTION, {"tool_name": "read", "reason": "no"}, "Rejection"),
            (EVENT_TYPE_BUDGET, {"limit": 10}, "Budget"),
            (EVENT_TYPE_DIMENSION_SAMPLED, {"name": "confidence"}, "DimensionSampled"),
            (EVENT_TYPE_MUTATION_APPLIED, {"mutation": "x"}, "MutationApplied"),
            (EVENT_TYPE_CANDIDATE_EVALUATED, {"score": 0.2}, "CandidateEvaluated"),
            ("custom", {"value": 1}, "TraceEvent"),
        ],
    )
    def test_extract_step_text_event_types(
        self, event_type: str, data: dict[str, object], expected: str
    ) -> None:
        scorer = EmotionScorer(EmotionGraderConfig())
        extract_step_text = getattr(scorer, "_extract_step_text")
        text = extract_step_text({"event_type": event_type, "data": data})
        assert expected in text

    def test_build_scoring_prompt_includes_dimensions_and_context(self) -> None:
        config = EmotionGraderConfig(context_window=1)
        scorer = EmotionScorer(config)
        context = [
            {"event_type": EVENT_TYPE_MODEL_MESSAGE, "data": {"role": "user", "content": "Hi"}}
        ]
        step = {
            "event_type": EVENT_TYPE_MODEL_MESSAGE,
            "data": {"role": "assistant", "content": "Hello"},
        }
        build_scoring_prompt = getattr(scorer, "_build_scoring_prompt")
        prompt = build_scoring_prompt(step, context)
        for dimension in config.dimensions:
            assert dimension.name in prompt
        assert "ModelMessage" in prompt
        assert "Prior context" in prompt

    def test_clamp_score(self) -> None:
        clamp_score = getattr(EmotionScorer, "_clamp_score")
        assert clamp_score(2.0) == 1.0
        assert clamp_score(-2.0) == -1.0
        assert clamp_score(0.4) == 0.4
