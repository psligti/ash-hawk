from __future__ import annotations

from ash_hawk.scenario.trace import (
    DEFAULT_TRACE_TS,
    ModelMessageEvent,
    ToolCallEvent,
    ToolResultEvent,
    VerificationEvent,
    append_trace_jsonl,
    iter_trace_jsonl,
    write_trace_jsonl,
)
from ash_hawk.scenario.trace_normalizer import normalize_eval_transcript
from ash_hawk.types import EvalTranscript


def test_trace_jsonl_roundtrip(tmp_path) -> None:
    events = [
        ModelMessageEvent.create(DEFAULT_TRACE_TS, {"role": "user", "content": "hi"}),
        ToolCallEvent.create(DEFAULT_TRACE_TS, {"name": "read", "arguments": {"path": "/work"}}),
        ToolResultEvent.create(DEFAULT_TRACE_TS, {"tool_name": "read", "result": "ok"}),
    ]
    path = tmp_path / "trace.jsonl"
    write_trace_jsonl(path, events)

    loaded = list(iter_trace_jsonl(path))
    assert [event.model_dump() for event in loaded] == [event.model_dump() for event in events]


def test_trace_jsonl_append(tmp_path) -> None:
    first = ModelMessageEvent.create(DEFAULT_TRACE_TS, {"role": "user", "content": "one"})
    second = ModelMessageEvent.create(DEFAULT_TRACE_TS, {"role": "user", "content": "two"})
    path = tmp_path / "trace.jsonl"

    write_trace_jsonl(path, [first])
    append_trace_jsonl(path, second)

    loaded = list(iter_trace_jsonl(path))
    assert [event.model_dump() for event in loaded] == [
        first.model_dump(),
        second.model_dump(),
    ]


def test_normalize_eval_transcript_strips_volatile_fields() -> None:
    transcript = EvalTranscript(
        messages=[
            {"role": "user", "content": "hi", "timestamp": "2024-01-01T00:00:00Z"},
            {"role": "tool", "content": "ok", "tool_call_id": "abc", "ts": "2024-01-01"},
            {
                "role": "assistant",
                "content": "done",
                "type": "verification",
                "details": {"passed": True},
            },
        ],
        tool_calls=[
            {
                "name": "read",
                "arguments": {"path": "/tmp/file.txt", "content": "x"},
                "run_id": "run-123",
                "result": {"status": "ok", "temp_path": "/tmp/out.txt"},
            }
        ],
    )

    events = normalize_eval_transcript(transcript)

    assert any(isinstance(event, ToolCallEvent) for event in events)
    assert any(isinstance(event, ToolResultEvent) for event in events)
    assert any(isinstance(event, VerificationEvent) for event in events)

    message_event = next(event for event in events if isinstance(event, ModelMessageEvent))
    assert "timestamp" not in message_event.data

    tool_call_event = next(event for event in events if isinstance(event, ToolCallEvent))
    assert "run_id" not in tool_call_event.data
    assert "path" not in tool_call_event.data.get("arguments", {})

    tool_result_event = next(
        event
        for event in events
        if isinstance(event, ToolResultEvent) and "tool_name" in event.data
    )
    assert "temp_path" not in tool_result_event.data.get("result", {})
