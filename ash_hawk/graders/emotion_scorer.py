# type-hygiene: skip-file  # dynamic trace data payloads are intentionally Any
from __future__ import annotations

import asyncio
import importlib
import json
from typing import Any, cast

import pydantic as pd

from ash_hawk.graders.emotion_config import EmotionDimension, EmotionGraderConfig
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


class StepEmotionScore(pd.BaseModel):
    step_index: int
    event_type: str
    scores: dict[str, float]
    confidence: dict[str, float]
    reasoning: str

    model_config = pd.ConfigDict(extra="forbid")


class LLMEmotionResponse(pd.BaseModel):
    scores: dict[str, float] = pd.Field(default_factory=dict)
    confidence: dict[str, float] = pd.Field(default_factory=dict)
    reasoning: str = ""

    model_config = pd.ConfigDict(extra="forbid")


class EmotionScorer:
    def __init__(self, config: EmotionGraderConfig) -> None:
        self._config = config
        self._dimension_map = {dimension.name: dimension for dimension in config.dimensions}
        self._semaphore = asyncio.Semaphore(config.model_config_ref.max_concurrent)

    async def score_step(
        self,
        step: dict[str, Any],
        context: list[dict[str, Any]],
        llm_client: Any,
    ) -> StepEmotionScore:
        prompt = self._build_scoring_prompt(step, context)
        event_type = str(step.get("event_type", ""))
        step_index = int(step.get("_index", 0))

        for attempt in range(3):
            try:
                async with self._semaphore:
                    module = importlib.import_module("dawn_kestrel.provider.llm_client")
                    request_options = getattr(module, "LLMRequestOptions")
                    options = request_options(
                        temperature=self._config.model_config_ref.temperature,
                        max_tokens=512,
                        response_format={"type": "json_object"},
                    )
                    messages = [{"role": "user", "content": prompt}]
                    response = await llm_client.complete(messages=messages, options=options)
                parsed = self._parse_llm_output(response.text)
                scores, confidence = self._coerce_scores(parsed)
                return StepEmotionScore(
                    step_index=step_index,
                    event_type=event_type,
                    scores=scores,
                    confidence=confidence,
                    reasoning=parsed.reasoning,
                )
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(2**attempt)

        neutral_scores = {name: 0.0 for name in self._dimension_map}
        neutral_confidence = {name: 0.0 for name in self._dimension_map}
        return StepEmotionScore(
            step_index=step_index,
            event_type=event_type,
            scores=neutral_scores,
            confidence=neutral_confidence,
            reasoning="",
        )

    def _build_scoring_prompt(self, step: dict[str, Any], context: list[dict[str, Any]]) -> str:
        dimensions = [
            self._format_dimension(dimension) for dimension in self._dimension_map.values()
        ]
        dimensions_text = "\n".join(dimensions)

        window = self._config.context_window
        context_slice = context[-window:] if window > 0 else []
        context_lines = [self._extract_step_text(item) for item in context_slice]
        context_text = "\n".join(context_lines) if context_lines else "(no prior context)"

        step_text = self._extract_step_text(step)

        response_fields = [
            '"scores": {"dimension": float}',
            '"reasoning": "brief explanation"',
        ]
        if self._config.score_confidence:
            response_fields.insert(1, '"confidence": {"dimension": float}')

        schema = "{\n  " + ",\n  ".join(response_fields) + "\n}"

        return (
            "Score this step on each dimension from -1.0 (fully negative pole) to +1.0 "
            "(fully positive pole). 0.0 means neutral. Consider the context window of prior steps.\n\n"
            "Dimensions:\n"
            f"{dimensions_text}\n\n"
            "Prior context:\n"
            f"{context_text}\n\n"
            "Step to score:\n"
            f"{step_text}\n\n"
            "Return JSON: "
            f"{schema}"
        )

    def _format_dimension(self, dimension: EmotionDimension) -> str:
        anchor_text = ""
        if dimension.anchors:
            anchors = ", ".join(
                f"{label}: {value:+.2f}" for label, value in dimension.anchors.items()
            )
            anchor_text = f" Anchors: {anchors}."

        description = f" {dimension.description}" if dimension.description else ""
        return (
            f"- {dimension.name}: {dimension.negative_pole} (-1.0) to "
            f"{dimension.positive_pole} (+1.0).{description}{anchor_text}"
        )

    def _parse_llm_output(self, raw_output: str) -> LLMEmotionResponse:
        candidate = self._extract_json_candidate(raw_output)
        extracted = _extract_first_json_object(candidate)
        if extracted is None:
            raise ValueError("No JSON object found in LLM response")
        payload = json.loads(extracted)
        return LLMEmotionResponse.model_validate(payload)

    def _coerce_scores(
        self, response: LLMEmotionResponse
    ) -> tuple[dict[str, float], dict[str, float]]:
        scores: dict[str, float] = {}
        confidence: dict[str, float] = {}
        for name in self._dimension_map:
            value = response.scores.get(name)
            score_value = float(value) if isinstance(value, int | float) else 0.0
            scores[name] = self._clamp_score(score_value)

            conf_value = response.confidence.get(name)
            conf_score = float(conf_value) if isinstance(conf_value, int | float) else 0.0
            confidence[name] = max(0.0, min(1.0, conf_score))

        return scores, confidence

    def _extract_step_text(self, step: dict[str, Any]) -> str:
        event_type = str(step.get("event_type", ""))
        data_raw = step.get("data", {})
        data: dict[str, Any] = {}
        if isinstance(data_raw, dict):
            data_items = cast(dict[object, Any], data_raw)
            for raw_key, raw_value in data_items.items():
                key = str(raw_key)
                value: Any = raw_value
                data[key] = value

        if event_type == EVENT_TYPE_MODEL_MESSAGE:
            role_raw = data.get("role")
            content_raw = data.get("content")
            role = str(role_raw) if role_raw is not None else ""
            content = content_raw if content_raw is not None else ""
            return f"ModelMessage ({role}): {self._stringify(content)}"
        if event_type == EVENT_TYPE_TOOL_CALL:
            tool_raw = data.get("tool") or data.get("name") or data.get("tool_name")
            tool = str(tool_raw) if tool_raw is not None else ""
            tool_input = data.get("input") or data.get("arguments") or data.get("args")
            return f"ToolCall ({tool}): {self._stringify(tool_input)}"
        if event_type == EVENT_TYPE_TOOL_RESULT:
            tool_raw = data.get("tool") or data.get("tool_name") or data.get("name")
            tool = str(tool_raw) if tool_raw is not None else ""
            result = data.get("result")
            return f"ToolResult ({tool}): {self._stringify(result)}"
        if event_type == EVENT_TYPE_VERIFICATION:
            passed = data.get("pass")
            message = data.get("message")
            return f"Verification (pass={passed}): {self._stringify(message)}"
        if event_type == EVENT_TYPE_TODO:
            todo = data.get("content") or data.get("todo") or data.get("items")
            return f"Todo: {self._stringify(todo)}"
        if event_type == EVENT_TYPE_DIFF:
            patch = data.get("patch_text")
            changed = data.get("changed_files")
            added = data.get("added_lines")
            return f"Diff (files={changed}, added_lines={added}): {self._stringify(patch)}"
        if event_type == EVENT_TYPE_ARTIFACT:
            artifact = data.get("artifact_key") or data.get("artifact")
            return f"Artifact: {self._stringify(artifact)}"
        if event_type == EVENT_TYPE_POLICY_DECISION:
            tool_raw = data.get("tool_name")
            policy_raw = data.get("policy")
            rationale = data.get("reason") or data.get("rationale")
            tool = str(tool_raw) if tool_raw is not None else ""
            policy = str(policy_raw) if policy_raw is not None else ""
            return f"PolicyDecision ({policy or tool}): {self._stringify(rationale)}"
        if event_type == EVENT_TYPE_REJECTION:
            tool_raw = data.get("tool_name")
            reason = data.get("reason")
            tool = str(tool_raw) if tool_raw is not None else ""
            return f"Rejection ({tool}): {self._stringify(reason)}"
        if event_type == EVENT_TYPE_BUDGET:
            return f"Budget: {self._stringify(data)}"
        if event_type == EVENT_TYPE_DIMENSION_SAMPLED:
            return f"DimensionSampled: {self._stringify(data)}"
        if event_type == EVENT_TYPE_MUTATION_APPLIED:
            return f"MutationApplied: {self._stringify(data)}"
        if event_type == EVENT_TYPE_CANDIDATE_EVALUATED:
            return f"CandidateEvaluated: {self._stringify(data)}"

        return f"TraceEvent ({event_type}): {self._stringify(data)}"

    @staticmethod
    def _stringify(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)

    @staticmethod
    def _extract_json_candidate(raw_output: str) -> str:
        candidate = raw_output.strip()
        if "```json" in candidate:
            start = candidate.find("```json") + 7
            end = candidate.find("```", start)
            if end != -1:
                return candidate[start:end].strip()
        if "```" in candidate:
            start = candidate.find("```") + 3
            end = candidate.find("```", start)
            if end != -1:
                return candidate[start:end].strip()
        return candidate

    @staticmethod
    def _clamp_score(value: float) -> float:
        return max(-1.0, min(1.0, value))


def _extract_first_json_object(candidate: str) -> str | None:
    start = candidate.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for idx, char in enumerate(candidate[start:]):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return candidate[start : start + idx + 1]
    return None


__all__ = ["EmotionScorer", "StepEmotionScore", "LLMEmotionResponse"]
