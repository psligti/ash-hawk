from __future__ import annotations

from types import SimpleNamespace

import pytest

from ash_hawk.scenario.runner import ScenarioRunner


def test_error_trace_only_transcript_is_not_counted_as_empty() -> None:
    transcript = SimpleNamespace(
        messages=[],
        tool_calls=[],
        trace_events=[],
        agent_response=None,
        error_trace="adapter failed to initialize",
    )
    result = SimpleNamespace(aggregate_score=0.0, transcript=transcript)
    trial = SimpleNamespace(id="trial-1", task_id="task-1", result=result)
    summary = SimpleNamespace(trials=[trial])

    runner = ScenarioRunner(show_failure_patterns=False)
    detect_failures = getattr(runner, "_detect_and_surface_failures")
    failure_summary = detect_failures(summary)

    assert len(failure_summary["error_trace_trials"]) == 1
    assert failure_summary["empty_transcript_trials"] == []


def test_empty_transcript_detection_raises_hard_failure() -> None:
    transcript = SimpleNamespace(
        messages=[],
        tool_calls=[],
        trace_events=[],
        agent_response=None,
        error_trace=None,
    )
    result = SimpleNamespace(aggregate_score=0.0, transcript=transcript)
    trial = SimpleNamespace(id="trial-empty", task_id="task-empty", result=result)
    summary = SimpleNamespace(trials=[trial])

    runner = ScenarioRunner(show_failure_patterns=False)
    detect_failures = getattr(runner, "_detect_and_surface_failures")
    failure_summary = detect_failures(summary)

    with pytest.raises(ValueError, match="Detected empty transcripts; aborting scenario run"):
        runner._raise_on_critical_failures(failure_summary)
