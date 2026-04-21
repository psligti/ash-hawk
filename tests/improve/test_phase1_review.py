from __future__ import annotations

from ash_hawk.improve.phase1_review import review_trial
from ash_hawk.types import (
    EvalOutcome,
    EvalStatus,
    EvalTranscript,
    EvalTrial,
    FailureMode,
    TrialResult,
)


def _make_trial(
    *,
    trial_id: str = "trial-1",
    scenario_path: str = "/tmp/delegation__example.scenario.yaml",
    aggregate_passed: bool = False,
    aggregate_score: float = 0.75,
    agent_response: str = "Done.",
    trace_events: list[dict[str, object]] | None = None,
    tool_calls: list[dict[str, object]] | None = None,
) -> EvalTrial:
    return EvalTrial(
        id=trial_id,
        task_id="task-1",
        status=EvalStatus.COMPLETED,
        attempt_number=1,
        input_snapshot={"scenario_path": scenario_path},
        result=TrialResult(
            trial_id=trial_id,
            outcome=(
                EvalOutcome.success()
                if aggregate_passed
                else EvalOutcome.failure(FailureMode.AGENT_ERROR, "failed")
            ),
            aggregate_passed=aggregate_passed,
            aggregate_score=aggregate_score,
            transcript=EvalTranscript(
                agent_response=agent_response,
                trace_events=trace_events or [],
                tool_calls=tool_calls or [],
            ),
        ),
    )


class TestPhase1Review:
    def test_review_flags_missing_verification_before_done(self) -> None:
        trial = _make_trial(
            agent_response="Done. I verified the fix.",
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "ToolCallEvent",
                    "ts": "2026-01-01T00:00:00Z",
                    "data": {"name": "read", "arguments": {}},
                }
            ],
        )

        review = review_trial(trial)

        assert review.verification_before_done is False
        assert review.claim_trace_alignment is False
        assert review.failure_bucket == "false_claim"
        assert "verification_claim_without_trace" in review.reasons

    def test_review_flags_false_delegation_claim_without_task_trace(self) -> None:
        trial = _make_trial(
            agent_response="I delegated the work to a specialist and completed it.",
            trace_events=[
                {
                    "schema_version": 1,
                    "event_type": "VerificationEvent",
                    "ts": "2026-01-01T00:00:00Z",
                    "data": {"pass": True, "message": "pytest ok"},
                }
            ],
        )

        review = review_trial(trial)

        assert review.verification_before_done is True
        assert review.claim_trace_alignment is False
        assert review.failure_bucket == "false_claim"
        assert "delegation_claim_without_trace" in review.reasons

    def test_review_flags_no_tool_use_before_missing_verification(self) -> None:
        trial = _make_trial(
            agent_response="Attempted update.",
            trace_events=[],
            tool_calls=[],
        )

        review = review_trial(trial)

        assert review.failure_bucket == "no_tool_use"
        assert review.suspicious is True
        assert "eval_failed" in review.reasons

    def test_review_flags_wrong_path_from_claim_signal(self) -> None:
        trial = _make_trial(
            agent_response="I updated the wrong path first, then stopped.",
            tool_calls=[{"tool": "edit", "arguments": {}}],
        )

        review = review_trial(trial)

        assert review.failure_bucket == "wrong_path"

    def test_review_flags_bad_search_for_high_search_volume(self) -> None:
        search_tools = ["read", "grep", "glob", "ls"] * 3
        tool_calls: list[dict[str, object]] = [
            {"tool": tool_name, "arguments": {}} for tool_name in search_tools
        ]
        trial = _make_trial(
            agent_response="I kept searching for the answer.",
            tool_calls=tool_calls,
        )

        review = review_trial(trial)

        assert review.failure_bucket == "bad_search"
        assert review.unnecessary_complexity is True

    def test_passing_trial_without_verification_is_not_suspicious_by_itself(self) -> None:
        trial = _make_trial(
            aggregate_passed=True,
            aggregate_score=1.0,
            agent_response="Updated src/config.py to return 30.",
            tool_calls=[{"tool": "edit", "arguments": {}}],
        )

        review = review_trial(trial)

        assert review.failure_bucket == "no_verification"
        assert review.suspicious is False
