"""Tests for trace excerpting utilities."""

from pathlib import Path
from tempfile import TemporaryDirectory

from ash_hawk.scenario.reporting import excerpt_trace, locate_trace_jsonl
from ash_hawk.scenario.trace import (
    ToolCallEvent,
    ToolResultEvent,
    VerificationEvent,
    write_trace_jsonl,
)


def test_excerpt_trace_filters_by_event_type():
    """Test that excerpt_trace filters events by event type."""
    with TemporaryDirectory() as tmpdir:
        trace_path = Path(tmpdir) / "test.trace.jsonl"

        # Create 60 events: 30 ToolCallEvent, 30 ToolResultEvent
        events = []
        for i in range(30):
            events.append(
                ToolCallEvent.create(
                    ts=f"2024-01-01T00:{i:02d}:00Z",
                    data={"tool": f"tool_{i}", "args": {"arg": i}},
                )
            )
            events.append(
                ToolResultEvent.create(
                    ts=f"2024-01-01T00:{i:02d}:30Z",
                    data={"result": f"result_{i}"},
                )
            )

        write_trace_jsonl(trace_path, events)

        # Extract only ToolCallEvent
        excerpt = excerpt_trace(trace_path, match_event_type="ToolCallEvent")

        lines = excerpt.strip().split("\n")

        # Should have 20 lines (max_lines default) even though there are 30 matching events
        assert len(lines) <= 20

        # All lines should contain ToolCallEvent
        for line in lines:
            assert "ToolCallEvent" in line
            assert "ToolResultEvent" not in line


def test_excerpt_trace_filters_by_substring():
    """Test that excerpt_trace filters events by substring in data."""
    with TemporaryDirectory() as tmpdir:
        trace_path = Path(tmpdir) / "test.trace.jsonl"

        # Create events with specific substring in data
        events = []
        for i in range(50):
            # Every 3rd event has "special_marker" in data
            if i % 3 == 0:
                events.append(
                    ToolCallEvent.create(
                        ts=f"2024-01-01T00:{i:02d}:00Z",
                        data={"tool": "special_tool", "marker": "special_marker"},
                    )
                )
            else:
                events.append(
                    ToolCallEvent.create(
                        ts=f"2024-01-01T00:{i:02d}:00Z",
                        data={"tool": "normal_tool", "value": i},
                    )
                )

        write_trace_jsonl(trace_path, events)

        # Extract events containing "special_marker"
        excerpt = excerpt_trace(trace_path, match_substring="special_marker")

        lines = excerpt.strip().split("\n")

        # Should have matching events
        assert len(lines) > 0

        # All lines should contain the substring
        for line in lines:
            assert "special_marker" in line


def test_excerpt_trace_respects_max_lines():
    """Test that excerpt_trace limits output to max_lines."""
    with TemporaryDirectory() as tmpdir:
        trace_path = Path(tmpdir) / "test.trace.jsonl"

        # Create 100 matching events
        events = []
        for i in range(100):
            events.append(
                VerificationEvent.create(
                    ts=f"2024-01-01T00:{i:02d}:00Z",
                    data={"check": f"check_{i}"},
                )
            )

        write_trace_jsonl(trace_path, events)

        # Extract with max_lines=10
        excerpt = excerpt_trace(trace_path, match_event_type="VerificationEvent", max_lines=10)

        lines = excerpt.strip().split("\n")

        # Should have exactly 10 lines
        assert len(lines) == 10

        # Should have first 10 events
        assert "check_0" in lines[0]
        assert "check_9" in lines[9]


def test_excerpt_trace_no_matches():
    """Test that excerpt_trace returns empty string when no events match."""
    with TemporaryDirectory() as tmpdir:
        trace_path = Path(tmpdir) / "test.trace.jsonl"

        # Create events that won't match
        events = [
            ToolCallEvent.create(ts="2024-01-01T00:00:00Z", data={"tool": "test"}),
        ]

        write_trace_jsonl(trace_path, events)

        # Search for non-existent event type
        excerpt = excerpt_trace(trace_path, match_event_type="NonExistentEvent")

        assert excerpt == ""


def test_excerpt_trace_missing_file():
    """Test that excerpt_trace returns empty string for missing file."""
    with TemporaryDirectory() as tmpdir:
        trace_path = Path(tmpdir) / "nonexistent.trace.jsonl"

        excerpt = excerpt_trace(trace_path, match_event_type="ToolCallEvent")

        assert excerpt == ""


def test_excerpt_trace_no_filters():
    """Test that excerpt_trace returns all events when no filters specified."""
    with TemporaryDirectory() as tmpdir:
        trace_path = Path(tmpdir) / "test.trace.jsonl"

        # Create mixed events
        events = []
        for i in range(25):
            events.append(
                ToolCallEvent.create(
                    ts=f"2024-01-01T00:{i:02d}:00Z",
                    data={"call": i},
                )
            )
            events.append(
                ToolResultEvent.create(
                    ts=f"2024-01-01T00:{i:02d}:30Z",
                    data={"result": i},
                )
            )

        write_trace_jsonl(trace_path, events)

        # Extract all events (no filters)
        excerpt = excerpt_trace(trace_path)

        lines = excerpt.strip().split("\n")

        # Should have max_lines (20) even though there are 50 total events
        assert len(lines) == 20

        # Should include both event types
        has_tool_call = any("ToolCallEvent" in line for line in lines)
        has_tool_result = any("ToolResultEvent" in line for line in lines)
        assert has_tool_call
        assert has_tool_result


def test_excerpt_trace_format():
    """Test that excerpt_trace formats output correctly."""
    with TemporaryDirectory() as tmpdir:
        trace_path = Path(tmpdir) / "test.trace.jsonl"

        events = [
            ToolCallEvent.create(
                ts="2024-01-01T00:00:00Z",
                data={"tool": "test_tool", "args": {"key": "value"}},
            ),
        ]

        write_trace_jsonl(trace_path, events)

        excerpt = excerpt_trace(trace_path)

        # Should have format: [timestamp] event_type: data_summary
        assert "[2024-01-01T00:00:00Z]" in excerpt
        assert "ToolCallEvent" in excerpt
        assert ":" in excerpt
        assert "test_tool" in excerpt


def test_locate_trace_jsonl_exists():
    """Test that locate_trace_jsonl finds existing trace file."""
    with TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir)
        suite_id = "test_suite"
        run_id = "run_001"
        trial_id = "trial_001"

        # Create the expected directory structure
        trace_path = (
            storage_path / suite_id / "runs" / run_id / "trials" / f"{trial_id}.trace.jsonl"
        )
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("{}")

        result = locate_trace_jsonl(suite_id, run_id, trial_id, storage_path)

        assert result is not None
        assert result == trace_path


def test_locate_trace_jsonl_not_exists():
    """Test that locate_trace_jsonl returns None for missing file."""
    with TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir)

        result = locate_trace_jsonl(
            suite_id="test_suite",
            run_id="run_001",
            trial_id="trial_001",
            storage_path=storage_path,
        )

        assert result is None


def test_excerpt_trace_with_both_filters():
    """Test that excerpt_trace uses OR logic for filters (matches either)."""
    with TemporaryDirectory() as tmpdir:
        trace_path = Path(tmpdir) / "test.trace.jsonl"

        events = [
            # Matches event type only
            ToolCallEvent.create(ts="2024-01-01T00:00:00Z", data={"tool": "a"}),
            # Matches substring only
            ToolResultEvent.create(ts="2024-01-01T00:01:00Z", data={"marker": "special_marker"}),
            # Matches both
            ToolCallEvent.create(ts="2024-01-01T00:02:00Z", data={"marker": "special_marker"}),
            # Matches neither
            VerificationEvent.create(ts="2024-01-01T00:03:00Z", data={"check": "test"}),
        ]

        write_trace_jsonl(trace_path, events)

        # Filter by either event type OR substring
        excerpt = excerpt_trace(
            trace_path,
            match_event_type="ToolCallEvent",
            match_substring="special_marker",
        )

        lines = excerpt.strip().split("\n")

        # Should match 3 events (first 3)
        assert len(lines) == 3
        assert "ToolCallEvent" in lines[0]
        assert "ToolResultEvent" in lines[1]
        assert "ToolCallEvent" in lines[2]
