"""Result aggregation utilities for ash-hawk evaluation harness.

This module provides functions for aggregating, grouping, filtering,
and analyzing trial results across tasks, graders, and time.

Key functions:
- aggregate_results: Combine trial results into suite-level metrics
- group_by_task: Group results by task ID
- group_by_grader: Group results by grader type
- group_by_time: Group results by time buckets
- filter_results: Filter results by various criteria
- calculate_statistics: Compute summary statistics
- detect_disagreements: Detect trials needing human review
"""

from __future__ import annotations

import pydantic as pd
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

from ash_hawk.types import (
    EvalStatus,
    EvalTrial,
    EvalRunSummary,
    GraderResult,
    RunEnvelope,
    SuiteMetrics,
    TokenUsage,
)


class DisagreementReport(pd.BaseModel):
    """Report of trials flagged for human review due to low-confidence judgments.

    Contains the list of flagged trial IDs and the reasons for each flag,
    along with the thresholds used for detection.
    """

    flagged_trial_ids: list[str] = pd.Field(
        default_factory=list,
        description="Trial IDs flagged for human review",
    )
    reasons: dict[str, str] = pd.Field(
        default_factory=dict,
        description="Map of trial_id to reason for flagging",
    )
    low_score_threshold: float = pd.Field(
        description="Threshold below which scores are considered low-confidence",
    )
    high_variance_threshold: float = pd.Field(
        description="Threshold above which variance is considered high disagreement",
    )

    model_config = pd.ConfigDict(extra="forbid")


def percentile(values: list[float], p: float) -> float | None:
    """Calculate the p-th percentile of a list of values.

    Args:
        values: List of numeric values.
        p: Percentile to calculate (0-100).

    Returns:
        The p-th percentile value, or None if values is empty.
    """
    if not values:
        return None

    sorted_values = sorted(values)
    n = len(sorted_values)

    k = (p / 100.0) * (n - 1)
    f = int(k)
    c = f + 1 if f + 1 < n else f

    if f == c:
        return sorted_values[f]

    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def aggregate_results(
    trials: list[EvalTrial],
    suite_id: str,
    run_id: str,
    envelope: RunEnvelope | None = None,
) -> SuiteMetrics:
    """Aggregate trial results into suite-level metrics.

    Calculates pass rate, mean score, token usage, cost, latency percentiles,
    and pass@k metrics from a list of trials.

    Args:
        trials: List of trials to aggregate.
        suite_id: ID of the evaluation suite.
        run_id: ID of the evaluation run.
        envelope: Optional run envelope for context.

    Returns:
        SuiteMetrics with aggregated statistics.
    """
    total_tasks = len(trials)
    completed_tasks = 0
    passed_tasks = 0
    failed_tasks = 0
    scores: list[float] = []
    latencies: list[float] = []
    total_tokens = TokenUsage()
    total_cost = 0.0
    total_duration = 0.0

    task_attempts: dict[str, list[bool]] = defaultdict(list)

    for trial in trials:
        task_attempts[trial.task_id].append(
            trial.result.aggregate_passed if trial.result else False
        )

        if trial.status != EvalStatus.COMPLETED:
            continue

        completed_tasks += 1

        if trial.result is None:
            failed_tasks += 1
            continue

        if trial.result.aggregate_passed:
            passed_tasks += 1
        else:
            failed_tasks += 1

        scores.append(trial.result.aggregate_score)

        transcript = trial.result.transcript
        total_tokens.input += transcript.token_usage.input
        total_tokens.output += transcript.token_usage.output
        total_tokens.reasoning += transcript.token_usage.reasoning
        total_tokens.cache_read += transcript.token_usage.cache_read
        total_tokens.cache_write += transcript.token_usage.cache_write

        total_cost += transcript.cost_usd
        total_duration += transcript.duration_seconds
        latencies.append(transcript.duration_seconds)

    pass_rate = passed_tasks / completed_tasks if completed_tasks > 0 else 0.0
    mean_score = sum(scores) / len(scores) if scores else 0.0

    pass_at_k: dict[int, float] = {}
    for k in [1, 2, 3, 5]:
        pass_at_k[k] = calculate_pass_at_k(task_attempts, k)

    return SuiteMetrics(
        suite_id=suite_id,
        run_id=run_id,
        total_tasks=total_tasks,
        completed_tasks=completed_tasks,
        passed_tasks=passed_tasks,
        failed_tasks=failed_tasks,
        pass_rate=pass_rate,
        mean_score=mean_score,
        total_tokens=total_tokens,
        total_cost_usd=total_cost,
        total_duration_seconds=total_duration,
        latency_p50_seconds=percentile(latencies, 50),
        latency_p95_seconds=percentile(latencies, 95),
        latency_p99_seconds=percentile(latencies, 99),
        pass_at_k=pass_at_k,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def calculate_pass_at_k(
    task_attempts: dict[str, list[bool]],
    k: int,
) -> float:
    """Calculate pass@k metric.

    pass@k measures the probability that at least one of k attempts
    passes for each task.

    Args:
        task_attempts: Dict mapping task_id to list of pass/fail results.
        k: Number of attempts to consider.

    Returns:
        pass@k rate (0.0-1.0).
    """
    if not task_attempts:
        return 0.0

    passed = 0
    total = 0

    for attempts in task_attempts.values():
        if len(attempts) < k:
            if any(attempts):
                passed += 1
        else:
            if any(attempts[:k]):
                passed += 1
        total += 1

    return passed / total if total > 0 else 0.0


def group_by_task(trials: list[EvalTrial]) -> dict[str, list[EvalTrial]]:
    """Group trials by task ID.

    Args:
        trials: List of trials to group.

    Returns:
        Dict mapping task_id to list of trials for that task.
    """
    grouped: dict[str, list[EvalTrial]] = defaultdict(list)
    for trial in trials:
        grouped[trial.task_id].append(trial)
    return dict(grouped)


def group_by_grader(trials: list[EvalTrial]) -> dict[str, list[GraderResult]]:
    """Group grader results by grader type across all trials.

    Args:
        trials: List of trials containing grader results.

    Returns:
        Dict mapping grader_type to list of GraderResults.
    """
    grouped: dict[str, list[GraderResult]] = defaultdict(list)
    for trial in trials:
        if trial.result:
            for grader_result in trial.result.grader_results:
                grouped[grader_result.grader_type].append(grader_result)
    return dict(grouped)


def group_by_time(
    trials: list[EvalTrial],
    bucket_seconds: float = 3600.0,
) -> dict[str, list[EvalTrial]]:
    """Group trials by time bucket.

    Groups trials based on their completion time into fixed-size buckets.

    Args:
        trials: List of trials to group.
        bucket_seconds: Size of each time bucket in seconds (default: 1 hour).

    Returns:
        Dict mapping bucket start time (ISO format) to list of trials.
    """
    grouped: dict[str, list[EvalTrial]] = defaultdict(list)

    for trial in trials:
        completed_at = None
        if trial.envelope and trial.envelope.completed_at:
            completed_at = trial.envelope.completed_at
        elif trial.result and trial.result.outcome.completed_at:
            completed_at = trial.result.outcome.completed_at

        if completed_at:
            try:
                dt = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
                timestamp = dt.timestamp()
                bucket_start = int(timestamp // bucket_seconds) * bucket_seconds
                bucket_key = datetime.fromtimestamp(bucket_start, tz=timezone.utc).isoformat()
                grouped[bucket_key].append(trial)
            except (ValueError, TypeError):
                continue

    return dict(grouped)


def filter_results(
    trials: list[EvalTrial],
    *,
    status: EvalStatus | None = None,
    passed: bool | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
    task_ids: list[str] | None = None,
    tags: list[str] | None = None,
    custom_filter: Callable[[EvalTrial], bool] | None = None,
) -> list[EvalTrial]:
    """Filter trials by various criteria.

    Args:
        trials: List of trials to filter.
        status: Filter by trial status.
        passed: Filter by pass/fail status (requires result to be present).
        min_score: Minimum aggregate score (inclusive).
        max_score: Maximum aggregate score (inclusive).
        task_ids: List of task IDs to include.
        tags: List of tags (trial must have at least one matching tag).
        custom_filter: Custom filter function that takes a trial and returns bool.

    Returns:
        Filtered list of trials.
    """
    result = trials

    if status is not None:
        result = [t for t in result if t.status == status]

    if passed is not None:
        result = [t for t in result if t.result is not None and t.result.aggregate_passed == passed]

    if min_score is not None:
        result = [
            t for t in result if t.result is not None and t.result.aggregate_score >= min_score
        ]

    if max_score is not None:
        result = [
            t for t in result if t.result is not None and t.result.aggregate_score <= max_score
        ]

    if task_ids is not None:
        task_id_set = set(task_ids)
        result = [t for t in result if t.task_id in task_id_set]

    if tags is not None:
        tag_set = set(tags)
        result = [t for t in result if bool(set(t.task_tags) & tag_set)]

    if custom_filter is not None:
        result = [t for t in result if custom_filter(t)]

    return result


def slice_results(
    trials: list[EvalTrial],
    *,
    offset: int = 0,
    limit: int | None = None,
) -> list[EvalTrial]:
    """Slice results with offset and limit.

    Args:
        trials: List of trials to slice.
        offset: Number of trials to skip.
        limit: Maximum number of trials to return.

    Returns:
        Sliced list of trials.
    """
    if limit is None:
        return trials[offset:]
    return trials[offset : offset + limit]


def calculate_statistics(trials: list[EvalTrial]) -> dict[str, Any]:
    """Calculate summary statistics for a list of trials.

    Args:
        trials: List of trials to analyze.

    Returns:
        Dict with various statistics:
        - count: Total number of trials
        - completed: Number of completed trials
        - passed: Number of passed trials
        - failed: Number of failed trials
        - pass_rate: Overall pass rate
        - score_mean: Mean aggregate score
        - score_std: Standard deviation of scores
        - score_min: Minimum score
        - score_max: Maximum score
        - latency_mean: Mean latency in seconds
        - latency_p50: Median latency
        - latency_p95: 95th percentile latency
        - latency_p99: 99th percentile latency
        - total_tokens: Total token usage
        - total_cost: Total cost in USD
    """
    if not trials:
        return {
            "count": 0,
            "completed": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "score_mean": 0.0,
            "score_std": 0.0,
            "score_min": 0.0,
            "score_max": 0.0,
            "latency_mean": 0.0,
            "latency_p50": None,
            "latency_p95": None,
            "latency_p99": None,
            "total_tokens": {"input": 0, "output": 0, "reasoning": 0},
            "total_cost": 0.0,
        }

    scores: list[float] = []
    latencies: list[float] = []
    completed = 0
    passed = 0
    failed = 0
    total_tokens = TokenUsage()
    total_cost = 0.0

    for trial in trials:
        if trial.status == EvalStatus.COMPLETED and trial.result:
            completed += 1
            if trial.result.aggregate_passed:
                passed += 1
            else:
                failed += 1

            scores.append(trial.result.aggregate_score)
            latencies.append(trial.result.transcript.duration_seconds)

            total_tokens.input += trial.result.transcript.token_usage.input
            total_tokens.output += trial.result.transcript.token_usage.output
            total_tokens.reasoning += trial.result.transcript.token_usage.reasoning
            total_cost += trial.result.transcript.cost_usd

    pass_rate = passed / completed if completed > 0 else 0.0

    mean_score = sum(scores) / len(scores) if scores else 0.0

    if len(scores) > 1:
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_score = variance**0.5
    else:
        std_score = 0.0

    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "count": len(trials),
        "completed": completed,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "score_mean": mean_score,
        "score_std": std_score,
        "score_min": min(scores) if scores else 0.0,
        "score_max": max(scores) if scores else 0.0,
        "latency_mean": mean_latency,
        "latency_p50": percentile(latencies, 50),
        "latency_p95": percentile(latencies, 95),
        "latency_p99": percentile(latencies, 99),
        "total_tokens": {
            "input": total_tokens.input,
            "output": total_tokens.output,
            "reasoning": total_tokens.reasoning,
        },
        "total_cost": total_cost,
    }


def grader_summary(trials: list[EvalTrial]) -> dict[str, dict[str, Any]]:
    """Generate per-grader summary statistics.

    Args:
        trials: List of trials with grader results.

    Returns:
        Dict mapping grader_type to statistics:
        - count: Number of results
        - pass_count: Number of passed results
        - fail_count: Number of failed results
        - pass_rate: Pass rate
        - score_mean: Mean score
        - score_min: Minimum score
        - score_max: Maximum score
    """
    grouped = group_by_grader(trials)

    summary: dict[str, dict[str, Any]] = {}

    for grader_type, results in grouped.items():
        if not results:
            continue

        scores = [r.score for r in results]
        passed = sum(1 for r in results if r.passed)

        summary[grader_type] = {
            "count": len(results),
            "pass_count": passed,
            "fail_count": len(results) - passed,
            "pass_rate": passed / len(results),
            "score_mean": sum(scores) / len(scores),
            "score_min": min(scores),
            "score_max": max(scores),
        }

    return summary


def create_run_summary(
    envelope: RunEnvelope,
    trials: list[EvalTrial],
) -> EvalRunSummary:
    """Create a complete run summary from envelope and trials.

    Args:
        envelope: Run envelope with reproducibility metadata.
        trials: List of all trials in the run.

    Returns:
        EvalRunSummary with envelope, metrics, and trials.
    """
    metrics = aggregate_results(
        trials=trials,
        suite_id=envelope.suite_id,
        run_id=envelope.run_id,
        envelope=envelope,
    )

    return EvalRunSummary(
        envelope=envelope,
        metrics=metrics,
        trials=trials,
    )


def _compute_variance(scores: list[float]) -> float:
    """Compute population variance of a list of scores."""
    if not scores or len(scores) < 2:
        return 0.0
    mean = sum(scores) / len(scores)
    return sum((s - mean) ** 2 for s in scores) / len(scores)


def detect_disagreements(
    trials: list[EvalTrial],
    low_score_threshold: float = 0.7,
    high_variance_threshold: float = 0.2,
) -> DisagreementReport:
    """Detect trials with low-confidence judgments that need human review.

    A trial is flagged if:
    - aggregate_score < low_score_threshold, OR
    - variance > high_variance_threshold (when multiple judges present)

    Variance is computed from the scores in grader_results.

    Args:
        trials: List of trials to analyze.
        low_score_threshold: Score below which a trial is flagged (default 0.7).
        high_variance_threshold: Variance above which a trial is flagged (default 0.2).

    Returns:
        DisagreementReport with flagged trial IDs and reasons.
    """
    flagged_trial_ids: list[str] = []
    reasons: dict[str, str] = {}

    for trial in trials:
        if trial.result is None:
            continue

        trial_id = trial.id
        aggregate_score = trial.result.aggregate_score
        grader_scores = [gr.score for gr in trial.result.grader_results]

        is_flagged = False

        if aggregate_score < low_score_threshold:
            is_flagged = True
            reasons[trial_id] = (
                f"Low aggregate score: {aggregate_score:.2f} < {low_score_threshold}"
            )

        if len(grader_scores) > 1:
            variance = _compute_variance(grader_scores)
            if variance > high_variance_threshold:
                if is_flagged:
                    reasons[trial_id] += (
                        f"; High variance: {variance:.3f} > {high_variance_threshold}"
                    )
                else:
                    is_flagged = True
                    reasons[trial_id] = (
                        f"High variance between judges: {variance:.3f} > {high_variance_threshold}"
                    )

        if is_flagged:
            flagged_trial_ids.append(trial_id)

    return DisagreementReport(
        flagged_trial_ids=flagged_trial_ids,
        reasons=reasons,
        low_score_threshold=low_score_threshold,
        high_variance_threshold=high_variance_threshold,
    )


__all__ = [
    "aggregate_results",
    "calculate_pass_at_k",
    "calculate_statistics",
    "create_run_summary",
    "detect_disagreements",
    "DisagreementReport",
    "filter_results",
    "grader_summary",
    "group_by_grader",
    "group_by_task",
    "group_by_time",
    "percentile",
    "slice_results",
]
