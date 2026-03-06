import pytest

from ash_hawk.graders.trace_assertions import EvidenceRequiredGrader
from ash_hawk.types import EvalTranscript, EvalTrial, GraderSpec


class TestEvidenceRequiredGrader:
    def test_name(self):
        assert EvidenceRequiredGrader().name == "evidence_required"

    @pytest.mark.asyncio
    async def test_grade_passes_with_evidence_path(self):
        grader = EvidenceRequiredGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            agent_response="done",
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "TodoEvent",
                    "ts": "2025-01-01T12:00:00Z",
                    "data": {"completed": True, "evidence_path": "artifacts/run.txt"},
                }
            ],
        )
        spec = GraderSpec(grader_type="evidence_required")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_grade_fails_without_evidence_path(self):
        grader = EvidenceRequiredGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            agent_response="done",
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "TodoEvent",
                    "ts": "2025-01-01T12:00:00Z",
                    "data": {"completed": True},
                }
            ],
        )
        spec = GraderSpec(grader_type="evidence_required")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is False
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_grade_passes_without_todos(self):
        grader = EvidenceRequiredGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(agent_response="done", trace_events=[])
        spec = GraderSpec(grader_type="evidence_required")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_grade_passes_with_incomplete_todo(self):
        grader = EvidenceRequiredGrader()

        trial = EvalTrial(id="t1", task_id="task1")
        transcript = EvalTranscript(
            agent_response="done",
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "TodoEvent",
                    "ts": "2025-01-01T12:00:00Z",
                    "data": {"completed": False},
                }
            ],
        )
        spec = GraderSpec(grader_type="evidence_required")

        result = await grader.grade(trial, transcript, spec)

        assert result.passed is True
        assert result.score == 1.0
