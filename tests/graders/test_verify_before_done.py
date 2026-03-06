import pytest

from ash_hawk.graders.trace_assertions import VerifyBeforeDoneGrader
from ash_hawk.types import EvalTranscript, EvalTrial, GraderSpec


class TestVerifyBeforeDoneGrader:
    def test_name(self):
        assert VerifyBeforeDoneGrader().name == "verify_before_done"

    @pytest.mark.asyncio
    async def test_grade_passes_with_verification(self):
        grader = VerifyBeforeDoneGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            agent_response="done",
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "VerificationEvent",
                    "ts": "2025-01-01T12:00:00Z",
                    "data": {"pass": True, "message": "ok"},
                }
            ],
        )
        spec = GraderSpec(grader_type="verify_before_done")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_grade_fails_without_verification(self):
        grader = VerifyBeforeDoneGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            agent_response="done",
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "ToolCallEvent",
                    "ts": "2025-01-01T12:00:00Z",
                    "data": {"tool": "read"},
                }
            ],
        )
        spec = GraderSpec(grader_type="verify_before_done")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_grade_passes_when_not_done(self):
        grader = VerifyBeforeDoneGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(agent_response=None, trace_events=[])
        spec = GraderSpec(grader_type="verify_before_done")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0
