"""Trace-aware reporting utilities for scenario runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ash_hawk.scenario.trace import TraceEvent, iter_trace_jsonl


def excerpt_trace(
    path: Path,
    match_event_type: str | None = None,
    match_substring: str | None = None,
    max_lines: int = 20,
) -> str:
    """Extract a bounded excerpt from a trace.jsonl file.

    Args:
        path: Path to the trace.jsonl file
        match_event_type: Optional event type to filter for (e.g., "ToolCallEvent")
        match_substring: Optional substring to search for in event data
        max_lines: Maximum number of lines to return (default: 20)

    Returns:
        Formatted excerpt string with one line per event, or empty string if no matches
    """
    # Collect matching events using the iterator to avoid loading entire file
    matching_events: list[TraceEvent] = []

    for event in iter_trace_jsonl(path):
        # Check if this event matches our criteria
        matches = False

        if match_event_type is not None:
            if event.event_type == match_event_type:
                matches = True

        if match_substring is not None:
            # Search for substring in the event data (converted to string)
            data_str = json.dumps(event.data)
            if match_substring in data_str:
                matches = True

        # If no filters specified, include all events
        if match_event_type is None and match_substring is None:
            matches = True

        if matches:
            matching_events.append(event)

    # Return empty string if no matches
    if not matching_events:
        return ""

    # Format events, respecting max_lines limit
    lines: list[str] = []
    for event in matching_events[:max_lines]:
        # Create a brief summary of the data
        data_summary = _summarize_data(event.data)
        line = f"[{event.ts}] {event.event_type}: {data_summary}"
        lines.append(line)

    return "\n".join(lines)


def _summarize_data(data: dict[str, Any], max_len: int = 100) -> str:
    """Create a brief summary of event data for display.

    Args:
        data: Event data dictionary
        max_len: Maximum length of summary string

    Returns:
        Truncated JSON representation of data
    """
    try:
        summary = json.dumps(data, separators=(",", ":"))
        if len(summary) > max_len:
            return summary[: max_len - 3] + "..."
        return summary
    except (TypeError, ValueError):
        return "<non-serializable>"


def locate_trace_jsonl(
    suite_id: str,
    run_id: str,
    trial_id: str,
    storage_path: Path,
) -> Path | None:
    """Locate the trace.jsonl file for a specific trial.

    Args:
        suite_id: Suite identifier
        run_id: Run identifier
        trial_id: Trial identifier
        storage_path: Base storage path (FileStorage base_path)

    Returns:
        Path to the trace.jsonl file if it exists, None otherwise
    """
    trace_path = storage_path / suite_id / "runs" / run_id / "trials" / f"{trial_id}.trace.jsonl"

    if trace_path.exists():
        return trace_path

    return None


__all__ = [
    "excerpt_trace",
    "locate_trace_jsonl",
]
