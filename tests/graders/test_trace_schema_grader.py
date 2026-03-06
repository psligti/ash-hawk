import pytest

from ash_hawk.graders.trace_assertions import TraceSchemaGrader
from ash_hawk.types import EvalTranscript, EvalTrial, GraderSpec


class TestTraceSchemaGrader:
    def test_name(self):
        assert TraceSchemaGrader().name == "trace_schema"

    @pytest.mark.asyncio
    async def test_grade_valid_events(self):
        grader = TraceSchemaGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "ToolCallEvent",
                    "ts": "2025-01-01T12:00:00Z",
                    "data": {"tool": "read"},
                },
                {
                    "schema_version": 1,
                    "event_type": "ToolResultEvent",
                    "ts": "2025-01-01T12:00:01+00:00",
                    "data": {"status": "ok"},
                },
            ]
        )
        spec = GraderSpec(grader_type="trace_schema")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0
        assert result.details["failed_events"] == []
        assert result.details["total_events"] == 2

    @pytest.mark.asyncio
    async def test_grade_invalid_event(self):
        grader = TraceSchemaGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            trace_events=[
                {
                    "event_type": "",
                    "ts": "not-a-timestamp",
                    "data": [],
                }
            ]
        )
        spec = GraderSpec(grader_type="trace_schema")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.0
        assert result.details["total_events"] == 1
        assert len(result.details["failed_events"]) == 1
        errors = result.details["failed_events"][0]["errors"]
        assert "Missing schema_version" in errors
        assert "event_type must be non-empty string" in errors
        assert "ts must be ISO timestamp string" in errors
        assert "data must be dict" in errors

    @pytest.mark.asyncio
    async def test_grade_empty_trace(self):
        grader = TraceSchemaGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(trace_events=[])
        spec = GraderSpec(grader_type="trace_schema")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0
        assert result.details["total_events"] == 0
