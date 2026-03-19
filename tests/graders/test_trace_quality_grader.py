import pytest

from ash_hawk.graders.trace_assertions import TraceQualityGrader
from ash_hawk.types import EvalTranscript, EvalTrial, GraderSpec


class TestTraceQualityGrader:
    def test_name(self):
        assert TraceQualityGrader().name == "trace_quality"

    @pytest.mark.asyncio
    async def test_perfect_score_at_target_without_rejections(self):
        grader = TraceQualityGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "ToolCallEvent",
                    "ts": "2026-01-01T00:00:00Z",
                    "data": {"tool": "read"},
                }
                for _ in range(5)
            ]
        )
        spec = GraderSpec(grader_type="trace_quality", config={"target_tool_calls": 5})

        result = await grader.grade(trial, transcript, spec)

        assert result.score == 1.0
        assert result.passed is True
        assert result.details["tool_call_count"] == 5
        assert result.details["rejection_count"] == 0

    @pytest.mark.asyncio
    async def test_penalizes_tool_call_distance(self):
        grader = TraceQualityGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "ToolCallEvent",
                    "ts": "2026-01-01T00:00:00Z",
                    "data": {"tool": "read"},
                }
                for _ in range(10)
            ]
        )
        spec = GraderSpec(
            grader_type="trace_quality",
            config={"target_tool_calls": 5, "tool_call_penalty": 0.1, "pass_threshold": 0.7},
        )

        result = await grader.grade(trial, transcript, spec)

        assert result.score == 0.5
        assert result.passed is False
        assert result.details["tool_call_distance"] == 5

    @pytest.mark.asyncio
    async def test_penalizes_rejections(self):
        grader = TraceQualityGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "ToolCallEvent",
                    "ts": "2026-01-01T00:00:00Z",
                    "data": {"tool": "read"},
                }
                for _ in range(5)
            ]
            + [
                {
                    "schema_version": 1,
                    "event_type": "RejectionEvent",
                    "ts": "2026-01-01T00:00:01Z",
                    "data": {"tool": "edit"},
                }
                for _ in range(2)
            ]
        )
        spec = GraderSpec(
            grader_type="trace_quality",
            config={"target_tool_calls": 5, "rejection_penalty": 0.15, "pass_threshold": 0.75},
        )

        result = await grader.grade(trial, transcript, spec)

        assert result.score == 0.7
        assert result.passed is False
        assert result.details["rejection_count"] == 2
