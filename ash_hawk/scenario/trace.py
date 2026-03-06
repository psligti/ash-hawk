from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

import pydantic as pd

EVENT_TYPE_MODEL_MESSAGE = "ModelMessageEvent"
EVENT_TYPE_TOOL_CALL = "ToolCallEvent"
EVENT_TYPE_TOOL_RESULT = "ToolResultEvent"
EVENT_TYPE_VERIFICATION = "VerificationEvent"
EVENT_TYPE_TODO = "TodoEvent"
EVENT_TYPE_DIFF = "DiffEvent"
EVENT_TYPE_ARTIFACT = "ArtifactEvent"
EVENT_TYPE_POLICY_DECISION = "PolicyDecisionEvent"
EVENT_TYPE_REJECTION = "RejectionEvent"
EVENT_TYPE_BUDGET = "BudgetEvent"

DEFAULT_TRACE_TS = "1970-01-01T00:00:00Z"


class TraceEvent(pd.BaseModel):
    schema_version: int = 1
    event_type: str
    ts: str
    data: dict[str, Any] = pd.Field(default_factory=dict)

    model_config = pd.ConfigDict(extra="forbid")


class ModelMessageEvent(TraceEvent):
    event_type: Literal["ModelMessageEvent"] = EVENT_TYPE_MODEL_MESSAGE

    @classmethod
    def create(cls, ts: str, data: dict[str, Any]) -> "ModelMessageEvent":
        return cls(ts=ts, data=data)


class ToolCallEvent(TraceEvent):
    event_type: Literal["ToolCallEvent"] = EVENT_TYPE_TOOL_CALL

    @classmethod
    def create(cls, ts: str, data: dict[str, Any]) -> "ToolCallEvent":
        return cls(ts=ts, data=data)


class ToolResultEvent(TraceEvent):
    event_type: Literal["ToolResultEvent"] = EVENT_TYPE_TOOL_RESULT

    @classmethod
    def create(cls, ts: str, data: dict[str, Any]) -> "ToolResultEvent":
        return cls(ts=ts, data=data)


class VerificationEvent(TraceEvent):
    event_type: Literal["VerificationEvent"] = EVENT_TYPE_VERIFICATION

    @classmethod
    def create(cls, ts: str, data: dict[str, Any]) -> "VerificationEvent":
        return cls(ts=ts, data=data)


class TodoEvent(TraceEvent):
    event_type: Literal["TodoEvent"] = EVENT_TYPE_TODO

    @classmethod
    def create(cls, ts: str, data: dict[str, Any]) -> "TodoEvent":
        return cls(ts=ts, data=data)


class DiffEvent(TraceEvent):
    event_type: Literal["DiffEvent"] = EVENT_TYPE_DIFF

    @classmethod
    def create(cls, ts: str, data: dict[str, Any]) -> "DiffEvent":
        return cls(ts=ts, data=data)


class ArtifactEvent(TraceEvent):
    event_type: Literal["ArtifactEvent"] = EVENT_TYPE_ARTIFACT

    @classmethod
    def create(cls, ts: str, data: dict[str, Any]) -> "ArtifactEvent":
        return cls(ts=ts, data=data)


class PolicyDecisionEvent(TraceEvent):
    event_type: Literal["PolicyDecisionEvent"] = EVENT_TYPE_POLICY_DECISION

    @classmethod
    def create(cls, ts: str, data: dict[str, Any]) -> "PolicyDecisionEvent":
        return cls(ts=ts, data=data)


class RejectionEvent(TraceEvent):
    event_type: Literal["RejectionEvent"] = EVENT_TYPE_REJECTION

    @classmethod
    def create(cls, ts: str, data: dict[str, Any]) -> "RejectionEvent":
        return cls(ts=ts, data=data)


class BudgetEvent(TraceEvent):
    event_type: Literal["BudgetEvent"] = EVENT_TYPE_BUDGET

    @classmethod
    def create(cls, ts: str, data: dict[str, Any]) -> "BudgetEvent":
        return cls(ts=ts, data=data)


EVENT_TYPE_MODEL_MAP: dict[str, type[TraceEvent]] = {
    EVENT_TYPE_MODEL_MESSAGE: ModelMessageEvent,
    EVENT_TYPE_TOOL_CALL: ToolCallEvent,
    EVENT_TYPE_TOOL_RESULT: ToolResultEvent,
    EVENT_TYPE_VERIFICATION: VerificationEvent,
    EVENT_TYPE_TODO: TodoEvent,
    EVENT_TYPE_DIFF: DiffEvent,
    EVENT_TYPE_ARTIFACT: ArtifactEvent,
    EVENT_TYPE_POLICY_DECISION: PolicyDecisionEvent,
    EVENT_TYPE_REJECTION: RejectionEvent,
    EVENT_TYPE_BUDGET: BudgetEvent,
}


def iter_trace_jsonl(path: str | Path) -> Iterator[TraceEvent]:
    trace_path = Path(path)
    if not trace_path.exists():
        return iter(())

    def _iterator() -> Iterator[TraceEvent]:
        with trace_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValueError("Trace JSONL line must be a JSON object")
                event_type = payload.get("event_type", "")
                model = EVENT_TYPE_MODEL_MAP.get(event_type, TraceEvent)
                yield model.model_validate(payload)

    return _iterator()


def write_trace_jsonl(path: str | Path, events: Iterable[TraceEvent]) -> None:
    trace_path = Path(path)
    with trace_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.model_dump()))
            handle.write("\n")


def append_trace_jsonl(path: str | Path, event: TraceEvent) -> None:
    trace_path = Path(path)
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event.model_dump()))
        handle.write("\n")


__all__ = [
    "DEFAULT_TRACE_TS",
    "TraceEvent",
    "ModelMessageEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "VerificationEvent",
    "TodoEvent",
    "DiffEvent",
    "ArtifactEvent",
    "PolicyDecisionEvent",
    "RejectionEvent",
    "BudgetEvent",
    "EVENT_TYPE_MODEL_MESSAGE",
    "EVENT_TYPE_TOOL_CALL",
    "EVENT_TYPE_TOOL_RESULT",
    "EVENT_TYPE_VERIFICATION",
    "EVENT_TYPE_TODO",
    "EVENT_TYPE_DIFF",
    "EVENT_TYPE_ARTIFACT",
    "EVENT_TYPE_POLICY_DECISION",
    "EVENT_TYPE_REJECTION",
    "EVENT_TYPE_BUDGET",
    "iter_trace_jsonl",
    "write_trace_jsonl",
    "append_trace_jsonl",
]
