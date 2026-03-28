"""Tests for TranscriptValidityGrader."""

import pytest

from ash_hawk.graders.validity import TranscriptValidityGrader
from ash_hawk.types import (
    EvalOutcome,
    EvalStatus,
    EvalTranscript,
    EvalTrial,
    GraderSpec,
    TrialResult,
)


class TestTranscriptValidityGrader:
    def test_name(self):
        assert TranscriptValidityGrader().name == "transcript_validity"

    @pytest.mark.asyncio
    async def test_passes_with_messages(self):
        grader = TranscriptValidityGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            messages=[{"role": "user", "content": "hello"}],
        )
        spec = GraderSpec(grader_type="transcript_validity")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0
        assert result.details["has_messages"] is True
        assert result.details["message_count"] == 1

    @pytest.mark.asyncio
    async def test_passes_with_tool_calls(self):
        grader = TranscriptValidityGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            tool_calls=[{"name": "bash", "input": {"command": "ls"}}],
        )
        spec = GraderSpec(grader_type="transcript_validity")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0
        assert result.details["has_tool_calls"] is True
        assert result.details["tool_call_count"] == 1

    @pytest.mark.asyncio
    async def test_passes_with_trace_events(self):
        grader = TranscriptValidityGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "ToolCallEvent",
                    "ts": "2026-01-01T00:00:00Z",
                    "data": {"tool": "bash"},
                }
            ],
        )
        spec = GraderSpec(grader_type="transcript_validity")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0
        assert result.details["has_trace_events"] is True
        assert result.details["trace_event_count"] == 1

    @pytest.mark.asyncio
    async def test_passes_with_agent_response(self):
        grader = TranscriptValidityGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            agent_response="Task completed successfully.",
        )
        spec = GraderSpec(grader_type="transcript_validity")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0
        assert result.details["has_response"] is True

    @pytest.mark.asyncio
    async def test_fails_when_all_empty(self):
        grader = TranscriptValidityGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript()
        spec = GraderSpec(grader_type="transcript_validity")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.0
        assert result.details["has_messages"] is False
        assert result.details["has_tool_calls"] is False
        assert result.details["has_trace_events"] is False
        assert result.details["has_response"] is False

    @pytest.mark.asyncio
    async def test_extracts_error_signal_from_error_trace(self):
        grader = TranscriptValidityGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            error_trace="Traceback (most recent call last):\n  File 'test.py', line 1\nValueError: something went wrong",
        )
        spec = GraderSpec(grader_type="transcript_validity")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert "error_signal" in result.details
        assert result.details["error_type"] == "crash"
        assert result.details["error_signal"]["error_type"] == "crash"
        assert "ValueError" in result.details["error_signal"]["message"]

    @pytest.mark.asyncio
    async def test_classifies_rate_limit_error(self):
        grader = TranscriptValidityGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            error_trace="Error: Rate limit exceeded. Please retry after 60 seconds.",
        )
        spec = GraderSpec(grader_type="transcript_validity")

        result = await grader.grade(trial, transcript, spec)

        assert result.details["error_type"] == "rate_limit"

    @pytest.mark.asyncio
    async def test_classifies_timeout_error(self):
        grader = TranscriptValidityGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            error_trace="Error: Operation timed out after 300 seconds.",
        )
        spec = GraderSpec(grader_type="transcript_validity")

        result = await grader.grade(trial, transcript, spec)

        assert result.details["error_type"] == "timeout"

    @pytest.mark.asyncio
    async def test_classifies_validation_error(self):
        grader = TranscriptValidityGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            error_trace="pydantic.ValidationError: 2 validation errors for EvalTranscript",
        )
        spec = GraderSpec(grader_type="transcript_validity")

        result = await grader.grade(trial, transcript, spec)

        assert result.details["error_type"] == "validation"

    @pytest.mark.asyncio
    async def test_uses_trial_result_transcript_when_available(self):
        grader = TranscriptValidityGrader()
        trial = EvalTrial(
            id="t1",
            task_id="task1",
            result=TrialResult(
                trial_id="t1",
                outcome=EvalOutcome(status=EvalStatus.COMPLETED),
                transcript=EvalTranscript(
                    messages=[{"role": "user", "content": "from result"}],
                ),
            ),
        )
        transcript = EvalTranscript()
        spec = GraderSpec(grader_type="transcript_validity")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.details["message_count"] == 1

    @pytest.mark.asyncio
    async def test_truncates_long_stack_trace(self):
        grader = TranscriptValidityGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        long_trace = "Traceback (most recent call last):\n" + "\n".join(
            [f"  File 'line_{i}.py', line {i}" for i in range(100)]
        )
        transcript = EvalTranscript(error_trace=long_trace)
        spec = GraderSpec(grader_type="transcript_validity")

        result = await grader.grade(trial, transcript, spec)

        assert result.details["error_signal"]["stack_trace_excerpt"] is not None
        assert len(result.details["error_signal"]["stack_trace_excerpt"]) <= 503

    @pytest.mark.asyncio
    async def test_extracts_last_tool_name(self):
        grader = TranscriptValidityGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            tool_calls=[
                {"name": "bash", "input": {"command": "ls"}},
                {"name": "read", "input": {"filePath": "/tmp/test.txt"}},
            ],
            error_trace="Error: something failed",
        )
        spec = GraderSpec(grader_type="transcript_validity")

        result = await grader.grade(trial, transcript, spec)

        assert result.details["error_signal"]["tool_name"] == "read"

    @pytest.mark.asyncio
    async def test_no_error_signal_when_transcript_clean(self):
        grader = TranscriptValidityGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            messages=[{"role": "user", "content": "hello"}],
        )
        spec = GraderSpec(grader_type="transcript_validity")

        result = await grader.grade(trial, transcript, spec)

        assert "error_signal" not in result.details
        assert "error_type" not in result.details
