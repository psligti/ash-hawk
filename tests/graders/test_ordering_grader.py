import pytest

from ash_hawk.graders.trace_assertions import OrderingGrader
from ash_hawk.types import EvalTranscript, EvalTrial, GraderSpec


class TestOrderingGrader:
    def test_name(self):
        assert OrderingGrader().name == "ordering"

    @pytest.mark.asyncio
    async def test_grade_passes_when_in_order(self):
        grader = OrderingGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "ToolCallEvent",
                    "ts": "2025-01-01T12:00:00Z",
                    "data": {},
                },
                {
                    "schema_version": 1,
                    "event_type": "VerificationEvent",
                    "ts": "2025-01-01T12:00:01Z",
                    "data": {},
                },
            ]
        )
        spec = GraderSpec(
            grader_type="ordering",
            config={"ordering_rules": [{"before": "ToolCallEvent", "after": "VerificationEvent"}]},
        )

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0
        assert result.details["violations"] == []

    @pytest.mark.asyncio
    async def test_grade_fails_when_out_of_order(self):
        grader = OrderingGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "VerificationEvent",
                    "ts": "2025-01-01T12:00:00Z",
                    "data": {},
                },
                {
                    "schema_version": 1,
                    "event_type": "ToolCallEvent",
                    "ts": "2025-01-01T12:00:01Z",
                    "data": {},
                },
            ]
        )
        spec = GraderSpec(
            grader_type="ordering",
            config={"ordering_rules": [{"before": "ToolCallEvent", "after": "VerificationEvent"}]},
        )

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.0
        assert result.details["violations"]

    @pytest.mark.asyncio
    async def test_grade_fails_when_missing_event(self):
        grader = OrderingGrader()
        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "ToolCallEvent",
                    "ts": "2025-01-01T12:00:00Z",
                    "data": {},
                }
            ]
        )
        spec = GraderSpec(
            grader_type="ordering",
            config={"ordering_rules": [{"before": "ToolCallEvent", "after": "VerificationEvent"}]},
        )

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.0
        assert result.details["violations"]
