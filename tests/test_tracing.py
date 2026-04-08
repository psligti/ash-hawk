"""Tests for ash_hawk.tracing — TraceContext, Span, JSONL output."""

from __future__ import annotations

import json
from pathlib import Path

from ash_hawk.tracing import Span, TraceContext


class TestSpan:
    """Test Span dataclass."""

    def test_creation_defaults(self) -> None:
        span = Span(name="test")
        assert span.name == "test"
        assert span.end_time is None
        assert span.status == "ok"
        assert span.attributes == {}
        assert span.events == []

    def test_creation_with_attrs(self) -> None:
        span = Span(name="tool_call", attributes={"tool": "read"})
        assert span.attributes == {"tool": "read"}

    def test_finish_sets_end_time_and_status(self) -> None:
        span = Span(name="test")
        span.finish("ok")
        assert span.end_time is not None
        assert span.status == "ok"

    def test_finish_with_error_status(self) -> None:
        span = Span(name="test")
        span.finish("error")
        assert span.status == "error"

    def test_double_finish_is_noop(self) -> None:
        span = Span(name="test")
        span.finish("ok")
        first_end = span.end_time
        assert first_end is not None
        span.finish("error")
        assert span.end_time == first_end
        assert span.status == "ok"  # unchanged

    def test_duration_seconds_before_finish(self) -> None:
        span = Span(name="test")
        duration = span.duration_seconds
        assert duration >= 0.0

    def test_duration_seconds_after_finish(self) -> None:
        span = Span(name="test")
        span.finish("ok")
        assert span.end_time is not None
        duration = span.duration_seconds
        assert duration >= 0.0

    def test_add_event(self) -> None:
        span = Span(name="test")
        span.add_event("started", task_id="t1")
        assert len(span.events) == 1
        assert span.events[0]["name"] == "started"
        assert span.events[0]["task_id"] == "t1"
        assert "ts" in span.events[0]

    def test_add_multiple_events(self) -> None:
        span = Span(name="test")
        span.add_event("first")
        span.add_event("second", key="val")
        assert len(span.events) == 2


class TestTraceContext:
    """Test TraceContext."""

    def test_init_stores_ids(self) -> None:
        ctx = TraceContext(trial_id="t1", run_id="r1")
        assert ctx.trial_id == "t1"
        assert ctx.run_id == "r1"

    def test_init_run_id_optional(self) -> None:
        ctx = TraceContext(trial_id="t1")
        assert ctx.run_id is None

    def test_empty_span_list(self) -> None:
        ctx = TraceContext(trial_id="t1")
        assert ctx.current_span is None
        d = ctx.to_dict()
        assert d["spans"] == []

    def test_start_span_returns_span(self) -> None:
        ctx = TraceContext(trial_id="t1")
        span = ctx.start_span("tool_call", tool="read")
        assert isinstance(span, Span)
        assert span.name == "tool_call"
        assert span.attributes == {"tool": "read"}

    def test_current_span(self) -> None:
        ctx = TraceContext(trial_id="t1")
        assert ctx.current_span is None
        span = ctx.start_span("outer")
        assert ctx.current_span is span

    def test_end_span(self) -> None:
        ctx = TraceContext(trial_id="t1")
        ctx.start_span("outer")
        span = ctx.end_span("ok")
        assert span is not None
        assert span.status == "ok"
        assert ctx.current_span is None

    def test_end_span_empty_stack_returns_none(self) -> None:
        ctx = TraceContext(trial_id="t1")
        result = ctx.end_span()
        assert result is None

    def test_nesting(self) -> None:
        ctx = TraceContext(trial_id="t1")
        outer = ctx.start_span("outer")
        inner = ctx.start_span("inner")
        assert ctx.current_span is inner

        ended = ctx.end_span("ok")
        assert ended is inner
        assert ctx.current_span is outer

        ended = ctx.end_span("ok")
        assert ended is outer
        assert ctx.current_span is None

    def test_to_dict(self) -> None:
        ctx = TraceContext(trial_id="t1", run_id="r1")
        ctx.start_span("first", key="val")
        ctx.end_span("ok")

        d = ctx.to_dict()
        assert d["trial_id"] == "t1"
        assert d["run_id"] == "r1"
        assert len(d["spans"]) == 1
        assert d["spans"][0]["name"] == "first"
        assert d["spans"][0]["status"] == "ok"
        assert d["spans"][0]["attributes"] == {"key": "val"}

    def test_to_dict_multiple_spans(self) -> None:
        ctx = TraceContext(trial_id="t1")
        ctx.start_span("a")
        ctx.end_span("ok")
        ctx.start_span("b")
        ctx.end_span("error")

        d = ctx.to_dict()
        assert len(d["spans"]) == 2
        assert d["spans"][0]["status"] == "ok"
        assert d["spans"][1]["status"] == "error"


class TestTraceContextWriteJsonl:
    """Test TraceContext.write_jsonl."""

    def test_writes_valid_jsonl(self, tmp_path: Path) -> None:
        ctx = TraceContext(trial_id="t1", run_id="r1")
        ctx.start_span("test")
        ctx.end_span("ok")

        out = tmp_path / "trace.jsonl"
        ctx.write_jsonl(out)

        assert out.exists()
        content = out.read_text().strip()
        data = json.loads(content)
        assert data["trial_id"] == "t1"
        assert data["run_id"] == "r1"
        assert len(data["spans"]) == 1

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        ctx = TraceContext(trial_id="t1")
        ctx.start_span("test")
        ctx.end_span()

        out = tmp_path / "deep" / "nested" / "trace.jsonl"
        ctx.write_jsonl(out)
        assert out.exists()

    def test_appends_to_existing(self, tmp_path: Path) -> None:
        out = tmp_path / "trace.jsonl"
        out.write_text('{"existing": true}\n')

        ctx = TraceContext(trial_id="t1")
        ctx.start_span("test")
        ctx.end_span()
        ctx.write_jsonl(out)

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_handles_unwritable_path(self, tmp_path: Path) -> None:
        ctx = TraceContext(trial_id="t1")
        ctx.start_span("test")
        ctx.end_span()

        # Use a path that will cause OSError
        bad_path = tmp_path / "readonly" / "dir"
        # Make the parent readonly so mkdir fails
        bad_path_parent = tmp_path / "readonly"
        bad_path_parent.mkdir()
        bad_path_parent.chmod(0o000)

        try:
            ctx.write_jsonl(bad_path)  # should not raise
        finally:
            bad_path_parent.chmod(0o755)  # restore for cleanup
