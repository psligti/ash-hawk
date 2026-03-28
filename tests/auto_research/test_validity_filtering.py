"""Tests for transcript validity filtering in auto-research."""

import pytest

from ash_hawk.auto_research.cycle_runner import _filter_valid_transcripts
from ash_hawk.types import EvalTranscript


class TestFilterValidTranscripts:
    @pytest.mark.asyncio
    async def test_filters_empty_transcripts(self):
        transcripts = [
            EvalTranscript(),
            EvalTranscript(),
        ]

        valid, error_signals = await _filter_valid_transcripts(transcripts)

        assert len(valid) == 0
        assert len(error_signals) == 0

    @pytest.mark.asyncio
    async def test_keeps_transcripts_with_messages(self):
        transcripts = [
            EvalTranscript(messages=[{"role": "user", "content": "hello"}]),
            EvalTranscript(),
        ]

        valid, error_signals = await _filter_valid_transcripts(transcripts)

        assert len(valid) == 1
        assert len(error_signals) == 0

    @pytest.mark.asyncio
    async def test_keeps_transcripts_with_tool_calls(self):
        transcripts = [
            EvalTranscript(tool_calls=[{"name": "bash", "input": {}}]),
            EvalTranscript(),
        ]

        valid, error_signals = await _filter_valid_transcripts(transcripts)

        assert len(valid) == 1
        assert len(error_signals) == 0

    @pytest.mark.asyncio
    async def test_keeps_transcripts_with_trace_events(self):
        transcripts = [
            EvalTranscript(
                trace_events=[
                    {
                        "schema_version": 1,
                        "event_type": "ToolCallEvent",
                        "ts": "2026-01-01T00:00:00Z",
                        "data": {"tool": "bash"},
                    }
                ]
            ),
            EvalTranscript(),
        ]

        valid, error_signals = await _filter_valid_transcripts(transcripts)

        assert len(valid) == 1
        assert len(error_signals) == 0

    @pytest.mark.asyncio
    async def test_keeps_transcripts_with_agent_response(self):
        transcripts = [
            EvalTranscript(agent_response="Task completed."),
            EvalTranscript(),
        ]

        valid, error_signals = await _filter_valid_transcripts(transcripts)

        assert len(valid) == 1
        assert len(error_signals) == 0

    @pytest.mark.asyncio
    async def test_extracts_error_signals_from_crashed_transcripts(self):
        transcripts = [
            EvalTranscript(error_trace="Traceback (most recent call last):\nValueError: crash"),
        ]

        valid, error_signals = await _filter_valid_transcripts(transcripts)

        assert len(valid) == 0
        assert len(error_signals) == 1
        assert error_signals[0]["error_type"] == "crash"

    @pytest.mark.asyncio
    async def test_classifies_timeout_errors(self):
        transcripts = [
            EvalTranscript(error_trace="Error: Operation timed out after 300 seconds"),
        ]

        valid, error_signals = await _filter_valid_transcripts(transcripts)

        assert len(valid) == 0
        assert len(error_signals) == 1
        assert error_signals[0]["error_type"] == "timeout"

    @pytest.mark.asyncio
    async def test_classifies_rate_limit_errors(self):
        transcripts = [
            EvalTranscript(error_trace="Error: Rate limit exceeded"),
        ]

        valid, error_signals = await _filter_valid_transcripts(transcripts)

        assert len(valid) == 0
        assert len(error_signals) == 1
        assert error_signals[0]["error_type"] == "rate_limit"

    @pytest.mark.asyncio
    async def test_handles_mixed_transcripts(self):
        transcripts = [
            EvalTranscript(messages=[{"role": "user", "content": "hello"}]),
            EvalTranscript(error_trace="Error: crash"),
            EvalTranscript(tool_calls=[{"name": "bash", "input": {}}]),
            EvalTranscript(),
            EvalTranscript(agent_response="done"),
        ]

        valid, error_signals = await _filter_valid_transcripts(transcripts)

        assert len(valid) == 3
        assert len(error_signals) == 1

    @pytest.mark.asyncio
    async def test_handles_empty_list(self):
        transcripts: list[EvalTranscript] = []

        valid, error_signals = await _filter_valid_transcripts(transcripts)

        assert len(valid) == 0
        assert len(error_signals) == 0

    @pytest.mark.asyncio
    async def test_valid_transcript_with_error_trace_still_passes(self):
        transcripts = [
            EvalTranscript(
                messages=[{"role": "user", "content": "hello"}],
                error_trace="Minor warning that shouldn't fail",
            ),
        ]

        valid, error_signals = await _filter_valid_transcripts(transcripts)

        assert len(valid) == 1
        assert len(error_signals) == 0
