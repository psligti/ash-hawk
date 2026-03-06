import json

from ash_hawk.scenario.trace import ModelMessageEvent, ToolCallEvent, TraceEvent
from ash_hawk.storage import FileStorage
from ash_hawk.types import (
    EvalOutcome,
    EvalStatus,
    EvalTranscript,
    EvalTrial,
    GraderResult,
    ToolSurfacePolicy,
    TrialEnvelope,
    TrialResult,
)


async def test_save_trial_writes_trace_jsonl(tmp_path):
    storage = FileStorage(tmp_path / ".ash-hawk")
    policy = ToolSurfacePolicy()
    envelope = TrialEnvelope(
        trial_id="trial-001",
        run_id="run-001",
        task_id="task-001",
        policy_snapshot=policy,
        created_at="2024-01-01T00:00:00Z",
    )
    trace_events = [
        ModelMessageEvent.create(
            "2024-01-01T00:00:00Z",
            {"role": "user", "content": "Hello"},
        ).model_dump(),
        ToolCallEvent.create(
            "2024-01-01T00:00:01Z",
            {"tool_name": "read"},
        ).model_dump(),
    ]
    trial = EvalTrial(
        id="trial-001",
        task_id="task-001",
        status=EvalStatus.COMPLETED,
        result=TrialResult(
            trial_id="trial-001",
            outcome=EvalOutcome.success(),
            transcript=EvalTranscript(trace_events=trace_events),
            grader_results=[
                GraderResult(grader_type="string_match", score=1.0, passed=True),
            ],
            aggregate_score=1.0,
            aggregate_passed=True,
        ),
    )

    await storage.save_trial("suite-001", "run-001", trial, envelope, policy)

    trace_path = (
        tmp_path
        / ".ash-hawk"
        / "suite-001"
        / "runs"
        / "run-001"
        / "trials"
        / "trial-001.trace.jsonl"
    )
    assert trace_path.exists()

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]
    expected = [TraceEvent.model_validate(event).model_dump() for event in trace_events]
    assert parsed == expected
