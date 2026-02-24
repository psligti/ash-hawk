"""Tests for ash_hawk.graders.aggregation module."""

import pytest

from ash_hawk.graders import (
    aggregate_results,
    calculate_pass_at_k,
    calculate_statistics,
    create_run_summary,
    detect_disagreements,
    filter_results,
    grader_summary,
    group_by_grader,
    group_by_task,
    group_by_time,
    percentile,
    slice_results,
)
from ash_hawk.graders.aggregation import DisagreementReport
from ash_hawk.types import (
    EvalOutcome,
    EvalStatus,
    EvalTranscript,
    EvalTrial,
    GraderResult,
    RunEnvelope,
    TokenUsage,
    ToolSurfacePolicy,
    TrialResult,
)


def make_trial(
    trial_id: str,
    task_id: str,
    status: EvalStatus = EvalStatus.COMPLETED,
    passed: bool = True,
    score: float = 1.0,
    latency: float = 1.0,
    grader_results: list[GraderResult] | None = None,
    completed_at: str | None = None,
    task_tags: list[str] | None = None,
) -> EvalTrial:
    """Create a test trial."""
    result = None
    if status == EvalStatus.COMPLETED:
        result = TrialResult(
            trial_id=trial_id,
            outcome=EvalOutcome(
                status=status,
                completed_at=completed_at or "2024-01-01T12:00:00+00:00",
            ),
            transcript=EvalTranscript(
                duration_seconds=latency,
                token_usage=TokenUsage(input=100, output=50),
                cost_usd=0.01,
            ),
            grader_results=grader_results
            or [GraderResult(grader_type="test", score=score, passed=passed)],
            aggregate_score=score,
            aggregate_passed=passed,
        )

    return EvalTrial(
        id=trial_id,
        task_id=task_id,
        status=status,
        result=result,
        task_tags=task_tags or [],
    )


def make_envelope(run_id: str = "run-1", suite_id: str = "suite-1") -> RunEnvelope:
    """Create a test run envelope."""
    return RunEnvelope(
        run_id=run_id,
        suite_id=suite_id,
        suite_hash="abc123",
        harness_version="1.0.0",
        agent_name="test-agent",
        provider="test",
        model="test-model",
        tool_policy_hash="def456",
        python_version="3.12",
        os_info="linux",
        created_at="2024-01-01T00:00:00+00:00",
    )


class TestPercentile:
    """Test percentile function."""

    def test_empty_list(self):
        assert percentile([], 50) is None

    def test_single_value(self):
        assert percentile([5.0], 50) == 5.0
        assert percentile([5.0], 0) == 5.0
        assert percentile([5.0], 100) == 5.0

    def test_two_values(self):
        assert percentile([0.0, 10.0], 50) == 5.0
        assert percentile([0.0, 10.0], 0) == 0.0
        assert percentile([0.0, 10.0], 100) == 10.0

    def test_multiple_values(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert percentile(values, 50) == 3.0
        assert percentile(values, 0) == 1.0
        assert percentile(values, 100) == 5.0


class TestAggregateResults:
    """Test aggregate_results function."""

    def test_empty_trials(self):
        metrics = aggregate_results([], "suite-1", "run-1")

        assert metrics.suite_id == "suite-1"
        assert metrics.run_id == "run-1"
        assert metrics.total_tasks == 0
        assert metrics.completed_tasks == 0
        assert metrics.pass_rate == 0.0
        assert metrics.mean_score == 0.0

    def test_single_passed_trial(self):
        trials = [make_trial("t1", "task1", passed=True, score=0.9)]

        metrics = aggregate_results(trials, "suite-1", "run-1")

        assert metrics.total_tasks == 1
        assert metrics.completed_tasks == 1
        assert metrics.passed_tasks == 1
        assert metrics.failed_tasks == 0
        assert metrics.pass_rate == 1.0
        assert metrics.mean_score == 0.9

    def test_single_failed_trial(self):
        trials = [make_trial("t1", "task1", passed=False, score=0.3)]

        metrics = aggregate_results(trials, "suite-1", "run-1")

        assert metrics.total_tasks == 1
        assert metrics.completed_tasks == 1
        assert metrics.passed_tasks == 0
        assert metrics.failed_tasks == 1
        assert metrics.pass_rate == 0.0
        assert metrics.mean_score == 0.3

    def test_mixed_trials(self):
        trials = [
            make_trial("t1", "task1", passed=True, score=1.0),
            make_trial("t2", "task2", passed=False, score=0.5),
            make_trial("t3", "task3", passed=True, score=0.8),
        ]

        metrics = aggregate_results(trials, "suite-1", "run-1")

        assert metrics.total_tasks == 3
        assert metrics.completed_tasks == 3
        assert metrics.passed_tasks == 2
        assert metrics.failed_tasks == 1
        assert metrics.pass_rate == pytest.approx(2 / 3)
        assert metrics.mean_score == pytest.approx((1.0 + 0.5 + 0.8) / 3)

    def test_pending_trials_not_counted(self):
        trials = [
            make_trial("t1", "task1", status=EvalStatus.PENDING),
            make_trial("t2", "task2", passed=True),
        ]

        metrics = aggregate_results(trials, "suite-1", "run-1")

        assert metrics.total_tasks == 2
        assert metrics.completed_tasks == 1
        assert metrics.passed_tasks == 1

    def test_token_aggregation(self):
        trials = [
            make_trial("t1", "task1", latency=2.0),
            make_trial("t2", "task2", latency=3.0),
        ]

        metrics = aggregate_results(trials, "suite-1", "run-1")

        assert metrics.total_tokens.input == 200
        assert metrics.total_tokens.output == 100
        assert metrics.total_cost_usd == 0.02
        assert metrics.total_duration_seconds == 5.0

    def test_latency_percentiles(self):
        trials = [make_trial(f"t{i}", f"task{i}", latency=float(i)) for i in range(1, 101)]

        metrics = aggregate_results(trials, "suite-1", "run-1")

        assert metrics.latency_p50_seconds is not None
        assert metrics.latency_p95_seconds is not None
        assert metrics.latency_p99_seconds is not None
        assert metrics.latency_p50_seconds < metrics.latency_p95_seconds
        assert metrics.latency_p95_seconds < metrics.latency_p99_seconds


class TestCalculatePassAtK:
    """Test calculate_pass_at_k function."""

    def test_empty(self):
        assert calculate_pass_at_k({}, 1) == 0.0

    def test_all_pass_at_k1(self):
        task_attempts = {"task1": [True], "task2": [True]}
        assert calculate_pass_at_k(task_attempts, 1) == 1.0

    def test_none_pass_at_k1(self):
        task_attempts = {"task1": [False], "task2": [False]}
        assert calculate_pass_at_k(task_attempts, 1) == 0.0

    def test_half_pass_at_k1(self):
        task_attempts = {"task1": [True], "task2": [False]}
        assert calculate_pass_at_k(task_attempts, 1) == 0.5

    def test_pass_at_k2(self):
        task_attempts = {
            "task1": [False, True],
            "task2": [True, False],
            "task3": [False, False],
        }
        assert calculate_pass_at_k(task_attempts, 2) == pytest.approx(2 / 3)

    def test_insufficient_attempts(self):
        task_attempts = {"task1": [True]}
        assert calculate_pass_at_k(task_attempts, 2) == 1.0


class TestGroupByTask:
    """Test group_by_task function."""

    def test_empty(self):
        assert group_by_task([]) == {}

    def test_single_trial(self):
        trials = [make_trial("t1", "task1")]
        result = group_by_task(trials)

        assert result == {"task1": trials}

    def test_multiple_trials_same_task(self):
        trials = [
            make_trial("t1", "task1"),
            make_trial("t2", "task1"),
        ]
        result = group_by_task(trials)

        assert len(result) == 1
        assert len(result["task1"]) == 2

    def test_multiple_trials_different_tasks(self):
        trials = [
            make_trial("t1", "task1"),
            make_trial("t2", "task2"),
            make_trial("t3", "task1"),
        ]
        result = group_by_task(trials)

        assert len(result) == 2
        assert len(result["task1"]) == 2
        assert len(result["task2"]) == 1


class TestGroupByGrader:
    """Test group_by_grader function."""

    def test_empty(self):
        assert group_by_grader([]) == {}

    def test_single_grader(self):
        grader_result = GraderResult(grader_type="string_match", score=1.0, passed=True)
        trials = [make_trial("t1", "task1", grader_results=[grader_result])]

        result = group_by_grader(trials)

        assert "string_match" in result
        assert len(result["string_match"]) == 1

    def test_multiple_graders(self):
        grader_results = [
            GraderResult(grader_type="string_match", score=1.0, passed=True),
            GraderResult(grader_type="test_runner", score=0.5, passed=False),
        ]
        trials = [make_trial("t1", "task1", grader_results=grader_results)]

        result = group_by_grader(trials)

        assert len(result) == 2
        assert "string_match" in result
        assert "test_runner" in result

    def test_trials_without_results_skipped(self):
        trials = [
            make_trial("t1", "task1", status=EvalStatus.PENDING),
        ]

        result = group_by_grader(trials)

        assert result == {}


class TestGroupByTime:
    """Test group_by_time function."""

    def test_empty(self):
        assert group_by_time([]) == {}

    def test_groups_by_hour_bucket(self):
        trials = [
            make_trial("t1", "task1", completed_at="2024-01-01T10:30:00+00:00"),
            make_trial("t2", "task2", completed_at="2024-01-01T10:45:00+00:00"),
            make_trial("t3", "task3", completed_at="2024-01-01T11:15:00+00:00"),
        ]

        result = group_by_time(trials, bucket_seconds=3600)

        assert len(result) == 2

    def test_custom_bucket_size(self):
        trials = [
            make_trial("t1", "task1", completed_at="2024-01-01T10:00:00+00:00"),
            make_trial("t2", "task2", completed_at="2024-01-01T10:30:00+00:00"),
        ]

        result = group_by_time(trials, bucket_seconds=1800)

        assert len(result) == 2


class TestFilterResults:
    """Test filter_results function."""

    def test_empty(self):
        assert filter_results([]) == []

    def test_filter_by_status(self):
        trials = [
            make_trial("t1", "task1", status=EvalStatus.PENDING),
            make_trial("t2", "task2", status=EvalStatus.COMPLETED),
        ]

        result = filter_results(trials, status=EvalStatus.COMPLETED)

        assert len(result) == 1
        assert result[0].id == "t2"

    def test_filter_by_passed(self):
        trials = [
            make_trial("t1", "task1", passed=True),
            make_trial("t2", "task2", passed=False),
        ]

        result = filter_results(trials, passed=True)

        assert len(result) == 1
        assert result[0].id == "t1"

    def test_filter_by_min_score(self):
        trials = [
            make_trial("t1", "task1", score=0.5),
            make_trial("t2", "task2", score=0.8),
        ]

        result = filter_results(trials, min_score=0.6)

        assert len(result) == 1
        assert result[0].id == "t2"

    def test_filter_by_max_score(self):
        trials = [
            make_trial("t1", "task1", score=0.5),
            make_trial("t2", "task2", score=0.8),
        ]

        result = filter_results(trials, max_score=0.6)

        assert len(result) == 1
        assert result[0].id == "t1"

    def test_filter_by_task_ids(self):
        trials = [
            make_trial("t1", "task1"),
            make_trial("t2", "task2"),
            make_trial("t3", "task3"),
        ]

        result = filter_results(trials, task_ids=["task1", "task3"])

        assert len(result) == 2

    def test_filter_with_custom_filter(self):
        trials = [
            make_trial("t1", "task1", latency=5.0),
            make_trial("t2", "task2", latency=15.0),
        ]

        result = filter_results(
            trials,
            custom_filter=lambda t: t.result.transcript.duration_seconds > 10.0,
        )

        assert len(result) == 1
        assert result[0].id == "t2"

    def test_filter_by_tags_single_match(self):
        """Filter trials by a single tag - include trials with matching tag."""
        trials = [
            make_trial("t1", "task1", task_tags=["python", "coding"]),
            make_trial("t2", "task2", task_tags=["javascript", "coding"]),
            make_trial("t3", "task3", task_tags=["golang"]),
        ]

        result = filter_results(trials, tags=["python"])

        assert len(result) == 1
        assert result[0].id == "t1"

    def test_filter_by_tags_multiple_match(self):
        """Filter trials by tag - include any trial with at least one matching tag."""
        trials = [
            make_trial("t1", "task1", task_tags=["python", "coding"]),
            make_trial("t2", "task2", task_tags=["javascript", "coding"]),
            make_trial("t3", "task3", task_tags=["golang", "backend"]),
        ]

        result = filter_results(trials, tags=["python", "coding"])

        assert len(result) == 2
        assert result[0].id == "t1"
        assert result[1].id == "t2"

    def test_filter_by_tags_no_match(self):
        """Filter by tags with no matches returns empty list."""
        trials = [
            make_trial("t1", "task1", task_tags=["python", "coding"]),
            make_trial("t2", "task2", task_tags=["javascript"]),
        ]

        result = filter_results(trials, tags=["rust"])

        assert result == []

    def test_filter_by_tags_none_returns_all(self):
        """When tags is None, no filtering is applied."""
        trials = [
            make_trial("t1", "task1", task_tags=["python"]),
            make_trial("t2", "task2", task_tags=[]),
        ]

        result = filter_results(trials, tags=None)

        assert len(result) == 2

    def test_filter_by_tags_empty_trial_tags(self):
        """Trials with empty task_tags are excluded when filtering by tags."""
        trials = [
            make_trial("t1", "task1", task_tags=["python"]),
            make_trial("t2", "task2", task_tags=[]),
        ]

        result = filter_results(trials, tags=["python"])

        assert len(result) == 1
        assert result[0].id == "t1"

    def test_filter_by_tags_combined_with_other_filters(self):
        """Tag filtering works in combination with other filters."""
        trials = [
            make_trial("t1", "task1", passed=True, task_tags=["python"]),
            make_trial("t2", "task2", passed=False, task_tags=["python"]),
            make_trial("t3", "task3", passed=True, task_tags=["golang"]),
        ]

        # Filter by both tags AND passed status
        result = filter_results(trials, tags=["python"], passed=True)

        assert len(result) == 1
        assert result[0].id == "t1"


class TestSliceResults:
    """Test slice_results function."""

    def test_empty(self):
        assert slice_results([]) == []

    def test_no_offset_no_limit(self):
        trials = [make_trial("t1", "task1")]
        assert slice_results(trials) == trials

    def test_with_offset(self):
        trials = [make_trial(f"t{i}", f"task{i}") for i in range(5)]

        result = slice_results(trials, offset=2)

        assert len(result) == 3
        assert result[0].id == "t2"

    def test_with_limit(self):
        trials = [make_trial(f"t{i}", f"task{i}") for i in range(5)]

        result = slice_results(trials, limit=3)

        assert len(result) == 3

    def test_with_offset_and_limit(self):
        trials = [make_trial(f"t{i}", f"task{i}") for i in range(10)]

        result = slice_results(trials, offset=3, limit=4)

        assert len(result) == 4
        assert result[0].id == "t3"
        assert result[-1].id == "t6"


class TestCalculateStatistics:
    """Test calculate_statistics function."""

    def test_empty(self):
        stats = calculate_statistics([])

        assert stats["count"] == 0
        assert stats["completed"] == 0
        assert stats["pass_rate"] == 0.0

    def test_single_trial(self):
        trials = [make_trial("t1", "task1", passed=True, score=0.9, latency=2.5)]

        stats = calculate_statistics(trials)

        assert stats["count"] == 1
        assert stats["completed"] == 1
        assert stats["passed"] == 1
        assert stats["pass_rate"] == 1.0
        assert stats["score_mean"] == 0.9
        assert stats["score_min"] == 0.9
        assert stats["score_max"] == 0.9
        assert stats["latency_mean"] == 2.5

    def test_multiple_trials(self):
        trials = [
            make_trial("t1", "task1", passed=True, score=1.0),
            make_trial("t2", "task2", passed=False, score=0.5),
            make_trial("t3", "task3", passed=True, score=0.8),
        ]

        stats = calculate_statistics(trials)

        assert stats["count"] == 3
        assert stats["completed"] == 3
        assert stats["passed"] == 2
        assert stats["failed"] == 1
        assert stats["pass_rate"] == pytest.approx(2 / 3)
        assert stats["score_mean"] == pytest.approx((1.0 + 0.5 + 0.8) / 3)

    def test_score_std(self):
        trials = [
            make_trial("t1", "task1", score=1.0),
            make_trial("t2", "task2", score=0.0),
        ]

        stats = calculate_statistics(trials)

        assert stats["score_std"] == pytest.approx(0.5)


class TestGraderSummary:
    """Test grader_summary function."""

    def test_empty(self):
        assert grader_summary([]) == {}

    def test_single_grader_type(self):
        grader_results = [
            GraderResult(grader_type="string_match", score=1.0, passed=True),
        ]
        trials = [make_trial("t1", "task1", grader_results=grader_results)]

        summary = grader_summary(trials)

        assert "string_match" in summary
        assert summary["string_match"]["count"] == 1
        assert summary["string_match"]["pass_count"] == 1
        assert summary["string_match"]["pass_rate"] == 1.0
        assert summary["string_match"]["score_mean"] == 1.0

    def test_multiple_grader_types(self):
        grader_results = [
            GraderResult(grader_type="string_match", score=1.0, passed=True),
            GraderResult(grader_type="test_runner", score=0.5, passed=False),
        ]
        trials = [make_trial("t1", "task1", grader_results=grader_results)]

        summary = grader_summary(trials)

        assert len(summary) == 2
        assert "string_match" in summary
        assert "test_runner" in summary


class TestCreateRunSummary:
    """Test create_run_summary function."""

    def test_creates_summary(self):
        envelope = make_envelope()
        trials = [
            make_trial("t1", "task1", passed=True),
            make_trial("t2", "task2", passed=False),
        ]

        summary = create_run_summary(envelope, trials)

        assert summary.envelope.run_id == "run-1"
        assert summary.metrics.total_tasks == 2
        assert len(summary.trials) == 2

    def test_empty_trials(self):
        envelope = make_envelope()

        summary = create_run_summary(envelope, [])

        assert summary.metrics.total_tasks == 0
        assert summary.metrics.completed_tasks == 0


class TestDetectDisagreements:
    """Test detect_disagreements function."""

    def test_empty_trials(self):
        """Empty trials returns empty disagreement report."""
        report = detect_disagreements([])

        assert report.flagged_trial_ids == []
        assert report.reasons == {}
        assert report.low_score_threshold == 0.7
        assert report.high_variance_threshold == 0.2

    def test_no_disagreements_high_confidence(self):
        """Trials with high scores and low variance are not flagged."""
        grader_results = [
            GraderResult(grader_type="judge_a", score=0.95, passed=True),
        ]
        trials = [
            make_trial("t1", "task1", score=0.95, passed=True, grader_results=grader_results),
        ]

        report = detect_disagreements(trials)

        assert report.flagged_trial_ids == []
        assert report.reasons == {}

    def test_low_aggregate_score_flagged(self):
        """Trials with aggregate score below threshold are flagged."""
        grader_results = [
            GraderResult(grader_type="judge_a", score=0.5, passed=False),
        ]
        trials = [
            make_trial("t1", "task1", score=0.5, passed=False, grader_results=grader_results),
        ]

        report = detect_disagreements(trials)

        assert "t1" in report.flagged_trial_ids
        assert "t1" in report.reasons
        assert "aggregate score" in report.reasons["t1"].lower()

    def test_high_variance_flagged(self):
        """Trials with high variance between multiple judges are flagged."""
        # Scores: 1.0 and 0.3, variance should be ~0.1225
        # But to exceed 0.2 threshold, use 1.0 and 0.0: variance = 0.25
        grader_results = [
            GraderResult(grader_type="judge_a", score=1.0, passed=True),
            GraderResult(grader_type="judge_b", score=0.0, passed=False),
        ]
        trials = [
            make_trial(
                "t1",
                "task1",
                score=0.5,  # average of 1.0 and 0.0
                passed=False,
                grader_results=grader_results,
            ),
        ]

        report = detect_disagreements(trials)

        assert "t1" in report.flagged_trial_ids
        assert "t1" in report.reasons
        assert "variance" in report.reasons["t1"].lower()

    def test_custom_thresholds(self):
        """Custom thresholds can be passed to override defaults."""
        grader_results = [
            GraderResult(grader_type="judge_a", score=0.65, passed=False),
        ]
        trials = [
            make_trial("t1", "task1", score=0.65, passed=False, grader_results=grader_results),
        ]

        # With default threshold (0.7), should be flagged
        report_default = detect_disagreements(trials)
        assert "t1" in report_default.flagged_trial_ids

        # With lower threshold (0.6), should NOT be flagged
        report_custom = detect_disagreements(trials, low_score_threshold=0.6)
        assert "t1" not in report_custom.flagged_trial_ids
        assert report_custom.low_score_threshold == 0.6

    def test_trials_without_result_skipped(self):
        """Trials without results are skipped."""
        trials = [
            make_trial("t1", "task1", status=EvalStatus.PENDING),
        ]

        report = detect_disagreements(trials)

        assert report.flagged_trial_ids == []

    def test_disagreement_report_dataclass(self):
        """DisagreementReport has expected fields."""
        report = DisagreementReport(
            flagged_trial_ids=["t1"],
            reasons={"t1": "low score"},
            low_score_threshold=0.7,
            high_variance_threshold=0.2,
        )

        assert report.flagged_trial_ids == ["t1"]
        assert report.reasons == {"t1": "low score"}
        assert report.low_score_threshold == 0.7
        assert report.high_variance_threshold == 0.2

    def test_variance_single_judge_not_flagged(self):
        """Single judge can't have variance - never flagged for variance."""
        grader_results = [
            GraderResult(grader_type="judge_a", score=0.8, passed=True),
        ]
        trials = [
            make_trial("t1", "task1", score=0.8, passed=True, grader_results=grader_results),
        ]

        report = detect_disagreements(trials)

        # Score 0.8 > 0.7, single judge so no variance check
        assert "t1" not in report.flagged_trial_ids
