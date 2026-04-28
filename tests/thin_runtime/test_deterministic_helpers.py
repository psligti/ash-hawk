from __future__ import annotations

from ash_hawk.thin_runtime.models import ToolCall
from ash_hawk.thin_runtime.tool_impl._evaluation_signals import collect_evaluation_signals
from ash_hawk.thin_runtime.tool_impl._workspace_targets import (
    preferred_workspace_target,
    rank_workspace_targets,
)
from ash_hawk.thin_runtime.tool_impl.aggregate_scores import run as aggregate_scores_run
from ash_hawk.thin_runtime.tool_impl.audit_claims import run as audit_claims_run
from ash_hawk.thin_runtime.tool_impl.verify_outcome import run as verify_outcome_run


def _tool_call(
    *,
    tool_name: str = "tool-under-test",
    evaluation: dict[str, object] | None = None,
    failure: dict[str, object] | None = None,
    audit: dict[str, object] | None = None,
) -> ToolCall:
    return ToolCall.model_validate(
        {
            "tool_name": tool_name,
            "goal_id": "goal-deterministic-helpers",
            "context": {
                "evaluation": evaluation or {},
                "failure": failure or {},
                "audit": audit or {},
            },
        }
    )


def test_collect_evaluation_signals_prefers_repeat_score_and_tracks_regression() -> None:
    signals = collect_evaluation_signals(
        _tool_call(
            evaluation={
                "baseline_summary": {"score": 0.9},
                "last_eval_summary": {"score": 0.88},
                "repeat_eval_summary": {"score": 0.84},
                "regressions": ["existing regression"],
                "verification": {"verified": False},
            },
            failure={
                "failure_family": "needs_improvement",
                "explanations": ["first issue", "second issue"],
            },
            audit={"run_result": {"aggregate_passed": False}},
        )
    )

    assert signals.baseline_score == 0.9
    assert signals.candidate_score == 0.84
    assert signals.candidate_label == "repeat evaluation"
    assert signals.score_regressed is True
    assert signals.failure_family == "needs_improvement"
    assert signals.failure_explanations == ("first issue", "second issue")
    assert signals.aggregate_passed is False
    assert signals.verification_verified is False
    assert signals.existing_regressions == ("existing regression",)


def test_aggregate_scores_averages_available_scores() -> None:
    result = aggregate_scores_run(
        _tool_call(
            tool_name="aggregate_scores",
            evaluation={
                "baseline_summary": {"score": 0.5},
                "last_eval_summary": {"score": 0.75},
                "repeat_eval_summary": {"score": 1.0},
            },
        )
    )

    assert result.success is True
    assert result.payload.evaluation_updates.aggregated_score == 0.75
    assert result.payload.audit_updates.run_summary["input_score_count"] == "3"


def test_aggregate_scores_fails_without_any_scores() -> None:
    result = aggregate_scores_run(_tool_call(tool_name="aggregate_scores"))

    assert result.success is False
    assert result.error == "missing_scores"
    assert result.payload.audit_updates.run_summary["status"] == "missing_scores"


def test_verify_outcome_reports_missing_evidence() -> None:
    result = verify_outcome_run(_tool_call(tool_name="verify_outcome"))

    assert result.success is True
    assert result.payload.evaluation_updates.verification.verified is False
    assert result.payload.evaluation_updates.verification.evidence_count == 0
    assert result.payload.audit_updates.run_summary["verified"] == "False"
    assert "no evaluation evidence is available" in result.payload.message


def test_audit_claims_reports_multiple_contradictions() -> None:
    result = audit_claims_run(
        _tool_call(
            tool_name="audit_claims",
            evaluation={
                "baseline_summary": {"score": 0.9},
                "repeat_eval_summary": {"score": 0.8},
                "verification": {"verified": False},
            },
            failure={"failure_family": "needs_improvement"},
            audit={
                "run_result": {
                    "message": "Scenario completed successfully",
                    "aggregate_passed": False,
                }
            },
        )
    )

    assert result.success is True
    assert result.payload.evaluation_updates.claim_audit.aligned is False
    assert result.payload.audit_updates.run_summary["contradiction_count"] == "4"
    assert "claims success while aggregate eval failed" in result.payload.message
    assert "failure family remains needs_improvement" in result.payload.message
    assert "verification step has not confirmed the outcome" in result.payload.message


def test_workspace_target_ranking_prefers_high_signal_and_demotes_docs() -> None:
    files = ["README.md", "src/runner.py", "agent.md", "notes.txt"]

    ranked = rank_workspace_targets(files)

    assert ranked == ["agent.md", "src/runner.py", "notes.txt", "README.md"]
    assert preferred_workspace_target(files) == "agent.md"
