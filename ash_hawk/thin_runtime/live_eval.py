from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import cast

from ash_hawk.scenario.runner import ScenarioRunner
from ash_hawk.scenario.tool_event_preview import tool_event_preview
from ash_hawk.thin_runtime.tool_types import (
    AuditRunResult,
    AuditToolContext,
    EvaluationToolContext,
    FailureBucketCount,
    FailureToolContext,
    ScoreSummary,
    ToolExecutionPayload,
    TraceRecord,
    TranscriptRecord,
)
from ash_hawk.types import EvalStatus

_OBSERVED_EVENT_CALLBACK: ContextVar[Callable[[dict[str, object]], None] | None] = ContextVar(
    "thin_runtime_observed_event_callback",
    default=None,
)


@contextmanager
def stream_observed_events(
    callback: Callable[[dict[str, object]], None],
) -> Iterator[None]:
    token = _OBSERVED_EVENT_CALLBACK.set(callback)
    try:
        yield
    finally:
        _OBSERVED_EVENT_CALLBACK.reset(token)


def emit_observed_event(payload: dict[str, object]) -> None:
    callback = _OBSERVED_EVENT_CALLBACK.get()
    if callback is not None:
        callback(payload)


def run_live_scenario_eval(
    tool_name: str,
    scenario_path: Path,
    *,
    summary_field: str = "baseline_summary",
    repetitions: int = 1,
) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    summaries = []
    for _ in range(repetitions):
        runner = ScenarioRunner()
        summaries.append(asyncio.run(runner.run_paths([str(scenario_path)])))
    trials = [summary.trials[0] for summary in summaries if summary.trials]
    if not trials:
        return (
            False,
            ToolExecutionPayload(),
            "No trials were produced by the scenario runner",
            ["no_trials"],
        )

    trial_results = [trial.result for trial in trials if trial.result is not None]
    if not trial_results:
        return False, ToolExecutionPayload(), "Trial result was missing", ["missing_trial_result"]

    traces: list[TraceRecord] = []
    transcripts: list[TranscriptRecord] = []
    for trial_result in trial_results:
        traces.extend(
            [
                TraceRecord(
                    tool=str(tool_call.get("name") or tool_call.get("tool") or ""),
                    success=(tool_call.get("error") is None),
                    error=(str(tool_call.get("error")) if tool_call.get("error") else None),
                    preview=tool_event_preview(
                        str(tool_call.get("name") or tool_call.get("tool") or ""),
                        tool_call.get("arguments") or tool_call.get("input") or {},
                        tool_call,
                    ),
                )
                for tool_call in trial_result.transcript.tool_calls
            ]
        )
        transcripts.extend(
            [
                TranscriptRecord(
                    speaker=str(message.get("role")),
                    type="message",
                    message=str(message.get("content")),
                    success=True,
                )
                for message in trial_result.transcript.messages
            ]
        )

    score = sum(result.aggregate_score for result in trial_results) / len(trial_results)
    passed = all(result.aggregate_passed for result in trial_results)
    outcome_success = all(result.outcome.status is EvalStatus.COMPLETED for result in trial_results)
    final_result = trial_results[-1]
    only_inconclusive_failures = _has_only_inconclusive_failures(trial_results)
    outcome_message = (
        None
        if final_result.outcome.error_message is not None
        else "Scenario completed successfully"
    )
    evaluation_updates = EvaluationToolContext()
    summary_status = "inconclusive" if only_inconclusive_failures and not passed else "completed"
    summary = ScoreSummary(score=score, status=summary_status, tool=tool_name)
    if summary_field == "baseline_summary":
        evaluation_updates.baseline_summary = summary
    elif summary_field == "last_eval_summary":
        evaluation_updates.last_eval_summary = summary
    elif summary_field == "repeat_eval_summary":
        evaluation_updates.repeat_eval_summary = summary
    elif summary_field == "targeted_validation_summary":
        evaluation_updates.targeted_validation_summary = summary
    elif summary_field == "integrity_summary":
        evaluation_updates.integrity_summary = summary

    payload = ToolExecutionPayload(
        evaluation_updates=evaluation_updates,
        failure_updates=(
            FailureToolContext(
                failure_buckets=[]
                if passed or only_inconclusive_failures
                else [FailureBucketCount(bucket="needs_improvement", count=1)],
                failure_family=None
                if passed or only_inconclusive_failures
                else "needs_improvement",
                explanations=[]
                if passed or only_inconclusive_failures
                else _collect_failure_explanations(trials),
            )
        ),
        audit_updates=AuditToolContext(
            validation_tools=[tool_name],
            run_result=AuditRunResult(
                run_id=summaries[-1].envelope.run_id,
                success=outcome_success,
                message=(
                    "Evaluation inconclusive due to transient grader failures"
                    if only_inconclusive_failures and not passed
                    else outcome_message
                ),
                error=(
                    "transient grader failures"
                    if only_inconclusive_failures and not passed
                    else final_result.outcome.error_message
                ),
                aggregate_score=score,
                aggregate_passed=None if only_inconclusive_failures and not passed else passed,
            ),
            events=traces,
            transcripts=transcripts,
        ),
    )
    if only_inconclusive_failures and not passed:
        return (
            False,
            payload,
            "Evaluation inconclusive due to transient grader failures",
            ["transient_grader_failures"],
        )
    return True, payload, "Executed live evaluation", []


def missing_live_eval_result(
    tool_name: str,
    *,
    reason: str = "scenario_path is required for live evaluation",
) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    payload = ToolExecutionPayload(
        evaluation_updates=EvaluationToolContext(
            baseline_summary=ScoreSummary(status="missing_scenario", tool=tool_name)
            if tool_name == "run_baseline_eval"
            else ScoreSummary(),
            targeted_validation_summary=ScoreSummary(status="missing_scenario", tool=tool_name)
            if tool_name == "run_targeted_validation"
            else ScoreSummary(),
            integrity_summary=ScoreSummary(status="missing_scenario", tool=tool_name)
            if tool_name == "run_integrity_validation"
            else ScoreSummary(),
            last_eval_summary=ScoreSummary(status="missing_scenario", tool=tool_name)
            if tool_name == "run_eval"
            else ScoreSummary(),
            repeat_eval_summary=ScoreSummary(status="missing_scenario", tool=tool_name)
            if tool_name == "run_eval_repeated"
            else ScoreSummary(),
        ),
        audit_updates=AuditToolContext(
            validation_tools=[tool_name],
            run_summary={"status": "missing_scenario", "reason": reason},
        ),
    )
    return False, payload, reason, ["missing_scenario_path"]


def _collect_failure_explanations(trials: Sequence[object]) -> list[str]:
    explanations: list[str] = []
    for trial in trials:
        trial_id = getattr(trial, "id", "unknown-trial")
        task_id = getattr(trial, "task_id", "unknown-task")
        result = getattr(trial, "result", None)
        if result is None or getattr(result, "aggregate_passed", False):
            continue
        grader_results = getattr(result, "grader_results", [])
        if isinstance(grader_results, list):
            for grader_result in grader_results:
                passed = getattr(grader_result, "passed", True)
                if passed:
                    continue
                if (
                    getattr(grader_result, "needs_review", False)
                    and getattr(grader_result, "review_reason", None)
                    == "transient_infrastructure_error"
                ):
                    continue
                grader_type = getattr(grader_result, "grader_type", "unknown_grader")
                detail_text = _grader_detail_text(getattr(grader_result, "details", {}))
                error_message = getattr(grader_result, "error_message", None)
                if error_message:
                    explanations.append(f"{task_id} ({trial_id}) {grader_type}: {error_message}")
                elif detail_text:
                    explanations.append(f"{task_id} ({trial_id}) {grader_type}: {detail_text}")
                else:
                    explanations.append(f"{task_id} ({trial_id}) {grader_type}: grader failed")
        outcome = getattr(result, "outcome", None)
        outcome_error = getattr(outcome, "error_message", None) if outcome is not None else None
        if outcome_error:
            explanations.append(f"{task_id} ({trial_id}) outcome: {outcome_error}")
    return explanations or ["Scenario did not pass all graders"]


def _has_only_inconclusive_failures(trial_results: Sequence[object]) -> bool:
    saw_inconclusive = False
    for result in trial_results:
        if getattr(result, "aggregate_passed", False):
            continue
        grader_results = getattr(result, "grader_results", [])
        if not isinstance(grader_results, list) or not grader_results:
            return False
        actionable_failure = False
        for grader_result in grader_results:
            if getattr(grader_result, "passed", True):
                continue
            if (
                getattr(grader_result, "needs_review", False)
                and getattr(grader_result, "review_reason", None)
                == "transient_infrastructure_error"
            ):
                saw_inconclusive = True
                continue
            actionable_failure = True
            break
        if actionable_failure:
            return False
    return saw_inconclusive


def _grader_detail_text(details: object) -> str:
    if not isinstance(details, dict):
        return ""
    if isinstance(details.get("missing_required"), list) and details["missing_required"]:
        return f"missing required files: {', '.join(str(item) for item in details['missing_required'][:4])}"
    if isinstance(details.get("questions"), list) and isinstance(details.get("answers"), list):
        failed_questions = [
            str(question)
            for question, answer in zip(details["questions"], details["answers"])
            if str(answer).strip().lower() == "false"
        ]
        if failed_questions:
            return f"failed checks: {'; '.join(failed_questions[:3])}"
    if isinstance(details.get("semantic_failures"), list) and details["semantic_failures"]:
        first_failure = details["semantic_failures"][0]
        if isinstance(first_failure, dict):
            path = first_failure.get("path")
            missing = first_failure.get("missing")
            if isinstance(missing, list) and missing:
                return f"{path}: missing {', '.join(str(item) for item in missing[:4])}"
            if first_failure.get("error"):
                return f"{path}: {first_failure['error']}"
    return ""
