from __future__ import annotations

from ash_hawk.thin_runtime import live_eval as live_eval_module
from ash_hawk.types import EvalOutcome, EvalTrial, GraderResult, TrialResult


def test_collect_failure_explanations_skips_inconclusive_provider_errors() -> None:
    trial = EvalTrial(id="trial-1", task_id="task-1")
    trial.result = TrialResult(
        trial_id="trial-1",
        outcome=EvalOutcome.success(),
        grader_results=[
            GraderResult(
                grader_type="llm_boolean",
                score=0.0,
                passed=False,
                needs_review=True,
                review_reason="transient_infrastructure_error",
                details={
                    "inconclusive": True,
                    "suppressed_error": "Provider operation failed after 3 retries",
                },
            )
        ],
        aggregate_passed=False,
    )

    explanations = getattr(live_eval_module, "_collect_failure_explanations")([trial])

    assert explanations == ["Scenario did not pass all graders"]


def test_has_only_inconclusive_failures_when_transient_error_is_only_failure() -> None:
    trial = EvalTrial(id="trial-1", task_id="task-1")
    trial.result = TrialResult(
        trial_id="trial-1",
        outcome=EvalOutcome.success(),
        grader_results=[
            GraderResult(
                grader_type="llm_boolean",
                score=0.0,
                passed=False,
                needs_review=True,
                review_reason="transient_infrastructure_error",
                details={"inconclusive": True},
            )
        ],
        aggregate_passed=False,
    )

    assert getattr(live_eval_module, "_has_only_inconclusive_failures")([trial.result]) is True
