from __future__ import annotations  # type-hygiene: skip-file

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ash_hawk.research.types import CauseCategory

logger = logging.getLogger(__name__)


def _empty_str_list() -> list[str]:
    return []


def _empty_cause_categories() -> list[CauseCategory]:
    return []


def _empty_hypotheses() -> list[CompetingHypothesis]:
    return []


@dataclass
class CompetingHypothesis:
    hypothesis_id: str
    cause_category: CauseCategory
    description: str
    confidence: float
    supporting_evidence: list[str] = field(default_factory=_empty_str_list)
    missing_evidence: list[str] = field(default_factory=_empty_str_list)
    expected_info_gain: float = 0.0


@dataclass
class DiagnosisReport:
    diagnosis_id: str
    run_id: str
    timestamp: str
    cause_categories: list[CauseCategory] = field(default_factory=_empty_cause_categories)
    hypotheses: list[CompetingHypothesis] = field(default_factory=_empty_hypotheses)
    primary_hypothesis: str | None = None
    uncertainty_level: float = 0.0
    missing_signals: list[str] = field(default_factory=_empty_str_list)
    recommended_action: str = "observe"


class DiagnosisEngine:
    def __init__(self, llm_client: Any | None = None) -> None:
        self._llm_client = llm_client

    async def diagnose(
        self,
        eval_results: dict[str, float],
        trace_events: list[dict[str, str | int | float | bool | list[str]]],
        scores: dict[str, float],
        experiment_log_path: Path | None = None,
        grader_details: dict[str, Any] | None = None,
        has_promotable_patterns: bool = False,
    ) -> DiagnosisReport:
        if not eval_results:
            return _fallback_diagnosis("unknown")

        if self._llm_client is None:
            return _fallback_diagnosis("unknown")

        prompt = _build_prompt(
            eval_results,
            trace_events,
            scores,
            experiment_log_path,
            grader_details,
        )
        response = await _call_llm(self._llm_client, prompt)
        if response is None:
            return _fallback_diagnosis("unknown")

        parsed = _parse_llm_response(response)
        if parsed is None:
            logger.warning("Malformed diagnosis JSON from LLM")
            return _fallback_diagnosis("unknown")

        hypotheses, missing_signals = parsed
        if not hypotheses:
            return _fallback_diagnosis("unknown")

        primary = max(hypotheses, key=lambda h: h.confidence)
        uncertainty = _calculate_uncertainty(hypotheses)
        recommended_action = _recommend_action(
            uncertainty, has_promotable_patterns=has_promotable_patterns
        )

        cause_categories = _unique_cause_categories(hypotheses)
        diagnosis_id = uuid.uuid4().hex
        return DiagnosisReport(
            diagnosis_id=diagnosis_id,
            run_id="unknown",
            timestamp=datetime.now(UTC).isoformat(),
            cause_categories=cause_categories,
            hypotheses=hypotheses,
            primary_hypothesis=primary.hypothesis_id,
            uncertainty_level=uncertainty,
            missing_signals=missing_signals,
            recommended_action=recommended_action,
        )


def _normalize_dict(raw: dict[object, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in raw.items():
        if isinstance(key, str):
            normalized[key] = value
    return normalized


async def _call_llm(client: Any, prompt: str) -> str | None:
    try:
        response: object = None

        if hasattr(client, "complete"):
            response = await client.complete(
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2},
            )
        elif hasattr(client, "chat"):
            response = await client.chat(prompt)
        else:
            logger.error("LLM client has no compatible method")
            return None

        if hasattr(response, "text"):
            text = getattr(response, "text", None)
            return str(text) if text is not None else None
        if hasattr(response, "content"):
            content = getattr(response, "content", None)
            return str(content) if content is not None else None
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            response_dict = _normalize_dict(cast(dict[object, object], response))
            raw = response_dict.get("content") or response_dict.get("text")
            return str(raw) if raw is not None else None
        return None
    except Exception as exc:
        logger.error(f"LLM call failed: {exc}")
        return None


def _build_prompt(
    eval_results: dict[str, float],
    trace_events: list[dict[str, str | int | float | bool | list[str]]],
    scores: dict[str, float],
    experiment_log_path: Path | None,
    grader_details: dict[str, Any] | None,
) -> str:
    eval_summary = _summarize_eval_results(eval_results)
    score_summary = _summarize_scores(scores)
    trace_summary = _summarize_trace_events(trace_events)
    log_path = f"Experiment log path: {experiment_log_path}" if experiment_log_path else ""
    grader_summary = _summarize_grader_details(grader_details)

    schema = (
        '{"hypotheses":[{"cause_category":"...","description":"...",'
        '"confidence":0.7,"supporting_evidence":["..."],'
        '"missing_evidence":["..."]}],"missing_signals":["..."]}'
    )
    prompt = (
        "You are diagnosing an agent evaluation run. "
        f"Return ONLY valid JSON with the schema: {schema}"
        "\n\nEvaluation results:\n"
        f"{eval_summary}\n\nScore summary:\n{score_summary}\n\nTrace events:\n"
        f"{trace_summary}\n"
        f"{grader_summary}"
        f"{log_path}"
    )

    return prompt[:8000]


def _summarize_eval_results(eval_results: dict[str, float]) -> str:
    if not eval_results:
        return "No evaluation results provided."
    lines = [f"- {name}: {score:.3f}" for name, score in list(eval_results.items())[:20]]
    return "\n".join(lines)


def _summarize_scores(scores: dict[str, float]) -> str:
    if not scores:
        return "No score summary available."
    lines = [f"- {name}: {score:.3f}" for name, score in list(scores.items())[:10]]
    return "\n".join(lines)


def _summarize_grader_details(grader_details: dict[str, Any] | None) -> str:
    if not grader_details:
        return ""

    lines: list[str] = []
    trajectory = grader_details.get("emotion_trajectory")
    if trajectory is None:
        trajectory = grader_details.get("step_emotions")
    if trajectory is not None:
        lines.append(f"- Emotion trajectory: {_format_grader_value(trajectory)}")

    inflection_points = grader_details.get("inflection_points")
    if inflection_points is not None:
        lines.append(f"- Inflection points: {_format_grader_value(inflection_points)}")

    if not lines:
        return ""

    return "\nGrader details:\n" + "\n".join(lines) + "\n"


def _format_grader_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int | float | bool):
        return str(value)
    try:
        return json.dumps(value)
    except TypeError:
        return str(value)


def _summarize_trace_events(
    trace_events: list[dict[str, str | int | float | bool | list[str]]],
) -> str:
    if not trace_events:
        return "No trace events provided."

    lines: list[str] = []
    for event in trace_events[:8]:
        label = _get_event_label(event)
        details = _format_event_details(event)
        lines.append(f"- {label}{details}")
    return "\n".join(lines)


def _get_event_label(event: dict[str, str | int | float | bool | list[str]]) -> str:
    for key in ("event_type", "type", "name"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    return "event"


def _format_event_details(event: dict[str, str | int | float | bool | list[str]]) -> str:
    detail_keys = ("tool", "status", "outcome", "error", "message")
    parts: list[str] = []
    for key in detail_keys:
        value = event.get(key)
        if isinstance(value, str | int | float | bool):
            parts.append(f"{key}={value}")
    return f" ({', '.join(parts)})" if parts else ""


def _parse_llm_response(
    response: str,
) -> tuple[list[CompetingHypothesis], list[str]] | None:
    json_text = _extract_json(response)
    if json_text is None:
        return None

    try:
        payload: object = json.loads(json_text)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    payload_dict = _normalize_dict(cast(dict[object, object], payload))
    hypotheses_payload = payload_dict.get("hypotheses")
    if not isinstance(hypotheses_payload, list):
        return None
    hypotheses_list = cast(list[object], hypotheses_payload)

    hypotheses: list[CompetingHypothesis] = []
    for item in hypotheses_list:
        if not isinstance(item, dict):
            continue
        item_dict = _normalize_dict(cast(dict[object, object], item))
        hypothesis = _parse_hypothesis(item_dict)
        if hypothesis is not None:
            hypotheses.append(hypothesis)

    missing_signals = _coerce_str_list(payload_dict.get("missing_signals"))
    return hypotheses, missing_signals


def _extract_json(response: str) -> str | None:
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    trimmed = response.strip()
    if trimmed.startswith("{") and trimmed.endswith("}"):
        return trimmed
    return None


def _parse_hypothesis(item: dict[str, object]) -> CompetingHypothesis | None:
    description_raw = item.get("description")
    if not isinstance(description_raw, str):
        return None
    description = description_raw.strip()
    if not description:
        return None

    category_raw = item.get("cause_category")
    cause_category = _parse_cause_category(category_raw)
    confidence = _clamp_confidence(item.get("confidence"))
    expected_info_gain = _clamp_confidence(item.get("expected_info_gain"))

    hypothesis_id = uuid.uuid4().hex
    return CompetingHypothesis(
        hypothesis_id=hypothesis_id,
        cause_category=cause_category,
        description=description,
        confidence=confidence,
        supporting_evidence=_coerce_str_list(item.get("supporting_evidence")),
        missing_evidence=_coerce_str_list(item.get("missing_evidence")),
        expected_info_gain=expected_info_gain,
    )


_CAUSE_ALIASES: dict[str, CauseCategory] = {
    "prompt": CauseCategory.PROMPT_QUALITY,
    "prompt_engineering": CauseCategory.PROMPT_QUALITY,
    "prompting": CauseCategory.PROMPT_QUALITY,
    "instruction_quality": CauseCategory.PROMPT_QUALITY,
    "tool_usage": CauseCategory.TOOL_MISUSE,
    "tool_error": CauseCategory.TOOL_MISUSE,
    "tool_selection": CauseCategory.TOOL_MISUSE,
    "incorrect_tool": CauseCategory.TOOL_MISUSE,
    "context_limit": CauseCategory.CONTEXT_OVERFLOW,
    "context_window": CauseCategory.CONTEXT_OVERFLOW,
    "token_limit": CauseCategory.CONTEXT_OVERFLOW,
    "context_length": CauseCategory.CONTEXT_OVERFLOW,
    "delegation_error": CauseCategory.DELEGATION_FAILURE,
    "delegation_issue": CauseCategory.DELEGATION_FAILURE,
    "sub_agent_failure": CauseCategory.DELEGATION_FAILURE,
    "agent_delegation": CauseCategory.DELEGATION_FAILURE,
    "orchestration_error": CauseCategory.ORCHESTRATION_BRANCH,
    "orchestration_issue": CauseCategory.ORCHESTRATION_BRANCH,
    "planning_failure": CauseCategory.ORCHESTRATION_BRANCH,
    "branching_error": CauseCategory.ORCHESTRATION_BRANCH,
    "timeout": CauseCategory.TIMEOUT_MISALLOCATION,
    "timeout_error": CauseCategory.TIMEOUT_MISALLOCATION,
    "time_management": CauseCategory.TIMEOUT_MISALLOCATION,
    "resource_allocation": CauseCategory.TIMEOUT_MISALLOCATION,
}


def _parse_cause_category(value: object) -> CauseCategory:
    if isinstance(value, CauseCategory):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        # Normalize spaces, hyphens, and mixed separators to underscores
        # so "prompt quality" / "prompt-quality" → "prompt_quality"
        normalized = re.sub(r"[\s\-]+", "_", normalized)

        for category in CauseCategory:
            if category.value == normalized:
                return category

        if normalized in _CAUSE_ALIASES:
            return _CAUSE_ALIASES[normalized]

        for alias, category in _CAUSE_ALIASES.items():
            if alias in normalized or normalized in alias:
                return category

        logger.debug("Unknown cause category from LLM: %r (normalized: %r)", value, normalized)
    return CauseCategory.UNKNOWN


def _clamp_confidence(value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return max(0.0, min(1.0, float(value)))
    return 0.0


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items = cast(list[object], value)
    return [str(item) for item in items if isinstance(item, str | int | float | bool)]


def _calculate_uncertainty(hypotheses: list[CompetingHypothesis]) -> float:
    if not hypotheses:
        return 1.0
    max_confidence = max(h.confidence for h in hypotheses)
    return max(0.0, min(1.0, 1.0 - max_confidence))


def _recommend_action(
    uncertainty_level: float,
    *,
    has_promotable_patterns: bool = False,
) -> str:
    if has_promotable_patterns and uncertainty_level < 0.2:
        return "promote"
    if uncertainty_level > 0.6:
        return "observe"
    if uncertainty_level > 0.3:
        return "experiment"
    return "fix"


def _unique_cause_categories(hypotheses: list[CompetingHypothesis]) -> list[CauseCategory]:
    seen: set[CauseCategory] = set()
    ordered: list[CauseCategory] = []
    for hypothesis in hypotheses:
        if hypothesis.cause_category not in seen:
            seen.add(hypothesis.cause_category)
            ordered.append(hypothesis.cause_category)
    return ordered


def _fallback_diagnosis(run_id: str) -> DiagnosisReport:
    diagnosis_id = uuid.uuid4().hex
    return DiagnosisReport(
        diagnosis_id=diagnosis_id,
        run_id=run_id,
        timestamp=datetime.now(UTC).isoformat(),
        cause_categories=[CauseCategory.UNKNOWN],
        hypotheses=[],
        primary_hypothesis=None,
        uncertainty_level=0.55,
        missing_signals=[],
        recommended_action="experiment",
    )


__all__ = ["CompetingHypothesis", "DiagnosisEngine", "DiagnosisReport"]
