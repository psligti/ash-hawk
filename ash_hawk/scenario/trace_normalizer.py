from __future__ import annotations

from typing import Any

from ash_hawk.scenario.trace import (
    DEFAULT_TRACE_TS,
    EVENT_TYPE_MODEL_MAP,
    ArtifactEvent,
    BudgetEvent,
    DiffEvent,
    ModelMessageEvent,
    PolicyDecisionEvent,
    RejectionEvent,
    TodoEvent,
    ToolCallEvent,
    ToolResultEvent,
    TraceEvent,
    VerificationEvent,
)
from ash_hawk.types import EvalTranscript

VOLATILE_KEYS = {
    "ts",
    "timestamp",
    "created_at",
    "started_at",
    "completed_at",
    "duration_seconds",
    "duration",
    "elapsed",
    "run_id",
    "trial_id",
}

PATH_KEYS = {
    "path",
    "file_path",
    "dir_path",
    "directory",
    "destination",
    "source",
    "temp_path",
    "tmp_path",
}

MESSAGE_TYPE_EVENT_MAP: dict[str, type[TraceEvent]] = {
    "verification": VerificationEvent,
    "todo": TodoEvent,
    "diff": DiffEvent,
    "artifact": ArtifactEvent,
    "policy_decision": PolicyDecisionEvent,
    "rejection": RejectionEvent,
    "budget": BudgetEvent,
}


def _is_temp_path(value: str) -> bool:
    lowered = value.lower()
    return "/tmp/" in lowered or "\\tmp\\" in lowered or "/var/folders/" in lowered


def _strip_volatile(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key in VOLATILE_KEYS:
                continue
            if key in PATH_KEYS and isinstance(item, str) and _is_temp_path(item):
                continue
            cleaned_item = _strip_volatile(item)
            cleaned[key] = cleaned_item
        return cleaned
    if isinstance(value, list):
        return [_strip_volatile(item) for item in value]
    return value


def _event_from_message(message: dict[str, Any]) -> TraceEvent:
    message_type = message.get("type")
    role = message.get("role")
    data = _strip_volatile(message)

    if isinstance(message_type, str) and message_type in MESSAGE_TYPE_EVENT_MAP:
        event_cls = MESSAGE_TYPE_EVENT_MAP[message_type]
        # Use create() factory method which handles defaults correctly
        return event_cls.create(DEFAULT_TRACE_TS, data)

    if role == "tool":
        return ToolResultEvent.create(DEFAULT_TRACE_TS, data)

    return ModelMessageEvent.create(DEFAULT_TRACE_TS, data)


def _tool_result_payload(tool_call: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("result", "output", "response"):
        if key in tool_call:
            return {"tool_name": tool_call.get("name"), "result": tool_call.get(key)}
    return None


def _event_from_trace_payload(payload: dict[str, Any]) -> TraceEvent | None:
    event_type = payload.get("event_type")
    if not isinstance(event_type, str):
        return None

    model = EVENT_TYPE_MODEL_MAP.get(event_type, TraceEvent)
    ts_value = payload.get("ts")
    data_value = payload.get("data")

    normalized_payload = dict(payload)
    if not isinstance(ts_value, str) or not ts_value:
        normalized_payload["ts"] = DEFAULT_TRACE_TS
    if not isinstance(data_value, dict):
        normalized_payload["data"] = {}

    try:
        return model.model_validate(normalized_payload)
    except Exception:
        return None


def normalize_eval_transcript(transcript: EvalTranscript) -> list[TraceEvent]:
    events: list[TraceEvent] = []

    for trace_event in transcript.trace_events:
        if not isinstance(trace_event, dict):
            continue
        parsed = _event_from_trace_payload(trace_event)
        if parsed is not None:
            events.append(parsed)

    for message in transcript.messages:
        if not isinstance(message, dict):
            continue
        events.append(_event_from_message(message))

    for tool_call in transcript.tool_calls:
        if not isinstance(tool_call, dict):
            continue
        events.append(ToolCallEvent.create(DEFAULT_TRACE_TS, _strip_volatile(tool_call)))
        result_payload = _tool_result_payload(tool_call)
        if result_payload is not None:
            events.append(ToolResultEvent.create(DEFAULT_TRACE_TS, _strip_volatile(result_payload)))

    return events


__all__ = [
    "normalize_eval_transcript",
]
