"""Tests for ash_hawk.metrics.statistics module."""

import math

import pytest

from ash_hawk.metrics import (
    ConfidenceInterval,
    SignificanceResult,
    calculate_cost_metrics,
    calculate_latency_metrics,
    calculate_pass_at_k,
    calculate_pass_at_k_from_trials,
    calculate_pass_caret_k,
    calculate_pass_caret_k_from_trials,
    calculate_suite_metrics_detailed,
    calculate_task_metrics,
    calculate_token_metrics,
    chi_square_test,
    clopper_pearson_confidence_interval,
    compare_graders,
    compare_runs_significance,
    fisher_exact_test,
    mean,
    normal_confidence_interval,
    percentile,
    std,
    to_suite_metrics,
    two_proportion_z_test,
    wilson_confidence_interval,
)
from ash_hawk.types import (
    EvalOutcome,
    EvalStatus,
    EvalTranscript,
    EvalTrial,
    GraderResult,
    TokenUsage,
    TrialResult,
)


def make_trial(
    trial_id: str,
    task_id: str,
    status: EvalStatus = EvalStatus.COMPLETED,
    passed: bool = True,
    score: float = 1.0,
    latency: float = 1.0,
    tokens_input: int = 100,
    tokens_output: int = 50,
    cost_usd: float = 0.01,
    grader_results: list[GraderResult] | None = None,
) -> EvalTrial:
    result = None
    if status == EvalStatus.COMPLETED:
        result = TrialResult(
            trial_id=trial_id,
            outcome=EvalOutcome(status=status),
            transcript=EvalTranscript(
                duration_seconds=latency,
                token_usage=TokenUsage(
                    input=tokens_input,
                    output=tokens_output,
                ),
                cost_usd=cost_usd,
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
    )


class TestPercentile:
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

    def test_interpolation(self):
        values = [0.0, 1.0, 2.0, 3.0, 4.0]
        p25 = percentile(values, 25)
        assert p25 is not None
        assert 0.5 < p25 < 1.5


class TestMean:
    def test_empty_list(self):
        assert mean([]) == 0.0

    def test_single_value(self):
        assert mean([5.0]) == 5.0

    def test_multiple_values(self):
        assert mean([1.0, 2.0, 3.0, 4.0, 5.0]) == 3.0

    def test_negative_values(self):
        assert mean([-1.0, 1.0]) == 0.0


class TestStd:
    def test_empty_list(self):
        assert std([]) == 0.0

    def test_single_value(self):
        assert std([5.0]) == 0.0

    def test_two_values(self):
        result = std([0.0, 2.0], ddof=1)
        assert result == pytest.approx(math.sqrt(2))

    def test_population_std(self):
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        result = std(values, ddof=0)
        assert result == pytest.approx(2.0)


class TestCalculatePassAtK:
    def test_zero_total(self):
        assert calculate_pass_at_k(0, 0, 1) == 0.0

    def test_all_correct(self):
        assert calculate_pass_at_k(10, 10, 1) == 1.0
        assert calculate_pass_at_k(10, 10, 5) == 1.0

    def test_none_correct(self):
        assert calculate_pass_at_k(0, 10, 1) == 0.0
        assert calculate_pass_at_k(0, 10, 5) == 0.0

    def test_half_correct_k1(self):
        result = calculate_pass_at_k(5, 10, 1)
        assert result == 0.5

    def test_pass_at_2_formula(self):
        result = calculate_pass_at_k(1, 3, 2)
        expected = 1.0 - (2 * 1) / (3 * 2)
        assert result == pytest.approx(expected)

    def test_k_greater_than_total(self):
        result = calculate_pass_at_k(5, 10, 20)
        assert result > 0.0

    def test_edge_cases(self):
        assert calculate_pass_at_k(1, 1, 1) == 1.0
        assert calculate_pass_at_k(0, 1, 1) == 0.0


class TestCalculatePassAtKFromTrials:
    def test_empty_trials(self):
        assert calculate_pass_at_k_from_trials([], 1) == 0.0

    def test_all_pass(self):
        trials = [
            make_trial("t1", "task1", passed=True),
            make_trial("t2", "task2", passed=True),
        ]
        assert calculate_pass_at_k_from_trials(trials, 1) == 1.0

    def test_half_pass(self):
        trials = [
            make_trial("t1", "task1", passed=True),
            make_trial("t2", "task2", passed=False),
        ]
        result = calculate_pass_at_k_from_trials(trials, 1)
        assert result == 0.5

    def test_multiple_attempts_same_task(self):
        trials = [
            make_trial("t1", "task1", passed=False),
            make_trial("t2", "task1", passed=True),
        ]
        result = calculate_pass_at_k_from_trials(trials, 2)
        assert result > 0.0


class TestCalculatePassCaretK:
    def test_zero_total(self):
        assert calculate_pass_caret_k(0, 0, 1) == 0.0

    def test_all_correct(self):
        assert calculate_pass_caret_k(10, 10, 3) == 1.0

    def test_none_correct(self):
        assert calculate_pass_caret_k(0, 10, 3) == 0.0

    def test_half_correct_k2(self):
        result = calculate_pass_caret_k(5, 10, 2)
        expected = (5 / 10) * (4 / 9)
        assert result == pytest.approx(expected)

    def test_insufficient_correct(self):
        result = calculate_pass_caret_k(2, 10, 5)
        assert result == 0.0

    def test_k_zero(self):
        assert calculate_pass_caret_k(5, 10, 0) == 1.0


class TestCalculatePassCaretKFromTrials:
    def test_empty_trials(self):
        assert calculate_pass_caret_k_from_trials([], 1) == 0.0

    def test_all_pass(self):
        trials = [
            make_trial("t1", "task1", passed=True),
            make_trial("t2", "task1", passed=True),
        ]
        assert calculate_pass_caret_k_from_trials(trials, 2) == 1.0

    def test_one_fails(self):
        trials = [
            make_trial("t1", "task1", passed=True),
            make_trial("t2", "task1", passed=False),
        ]
        result = calculate_pass_caret_k_from_trials(trials, 2)
        assert result == 0.0


class TestWilsonConfidenceInterval:
    def test_zero_total(self):
        ci = wilson_confidence_interval(0, 0)
        assert ci.lower == 0.0
        assert ci.upper == 1.0
        assert ci.method == "wilson"

    def test_all_success(self):
        ci = wilson_confidence_interval(100, 100)
        assert ci.lower > 0.9
        assert ci.upper == 1.0

    def test_no_success(self):
        ci = wilson_confidence_interval(0, 100)
        assert ci.lower == pytest.approx(0.0, abs=1e-10)
        assert ci.upper < 0.1

    def test_half_success(self):
        ci = wilson_confidence_interval(50, 100)
        assert 0.35 < ci.lower < 0.45
        assert 0.55 < ci.upper < 0.65

    def test_small_sample(self):
        ci = wilson_confidence_interval(1, 2)
        assert ci.lower < 0.5
        assert ci.upper > 0.5

    def test_custom_confidence_level(self):
        ci_95 = wilson_confidence_interval(50, 100, 0.95)
        ci_99 = wilson_confidence_interval(50, 100, 0.99)
        assert ci_99.width > ci_95.width


class TestNormalConfidenceInterval:
    def test_zero_total(self):
        ci = normal_confidence_interval(0, 0)
        assert ci.lower == 0.0
        assert ci.upper == 1.0
        assert ci.method == "normal"

    def test_half_success(self):
        ci = normal_confidence_interval(50, 100)
        assert 0.35 < ci.lower < 0.45
        assert 0.55 < ci.upper < 0.65


class TestClopperPearsonConfidenceInterval:
    def test_zero_total(self):
        ci = clopper_pearson_confidence_interval(0, 0)
        assert ci.lower == 0.0
        assert ci.upper == 1.0
        assert ci.method == "clopper_pearson"

    def test_all_success(self):
        ci = clopper_pearson_confidence_interval(100, 100)
        assert ci.lower > 0.9
        assert ci.upper == 1.0

    def test_no_success(self):
        ci = clopper_pearson_confidence_interval(0, 100)
        assert ci.lower == 0.0
        assert ci.upper < 0.1


class TestTwoProportionZTest:
    def test_identical_proportions(self):
        result = two_proportion_z_test(50, 100, 50, 100)
        assert result.statistic == 0.0
        assert result.p_value == 1.0
        assert not result.significant

    def test_significantly_different(self):
        result = two_proportion_z_test(90, 100, 10, 100)
        assert abs(result.statistic) > 5
        assert result.p_value < 0.001
        assert result.significant

    def test_zero_totals(self):
        result = two_proportion_z_test(0, 0, 0, 0)
        assert result.p_value == 1.0
        assert not result.significant

    def test_effect_size(self):
        result = two_proportion_z_test(90, 100, 10, 100)
        assert result.effect_size is not None
        assert result.effect_size > 0.5


class TestChiSquareTest:
    def test_perfect_independence(self):
        observed = [[50, 50], [50, 50]]
        result = chi_square_test(observed)
        assert result.statistic == 0.0
        assert result.p_value == 1.0
        assert not result.significant

    def test_perfect_association(self):
        observed = [[100, 0], [0, 100]]
        result = chi_square_test(observed)
        assert result.statistic > 50
        assert result.p_value < 0.001
        assert result.significant

    def test_empty_table(self):
        observed = [[0, 0], [0, 0]]
        result = chi_square_test(observed)
        assert result.p_value == 1.0

    def test_effect_size(self):
        observed = [[100, 0], [0, 100]]
        result = chi_square_test(observed)
        assert result.effect_size is not None


class TestCompareRunsSignificance:
    def test_identical_runs(self):
        run1 = [make_trial(f"t{i}", f"task{i}", passed=True) for i in range(10)]
        run2 = [make_trial(f"t{i}", f"task{i}", passed=True) for i in range(10)]
        result = compare_runs_significance(run1, run2)
        assert not result.significant

    def test_different_runs(self):
        run1 = [make_trial(f"t{i}", f"task{i}", passed=True) for i in range(100)]
        run2 = [make_trial(f"t{i}", f"task{i}", passed=False) for i in range(100)]
        result = compare_runs_significance(run1, run2)
        assert result.significant


class TestCalculateLatencyMetrics:
    def test_empty_trials(self):
        metrics = calculate_latency_metrics([])
        assert metrics.mean_seconds == 0.0
        assert metrics.p50_seconds is None

    def test_single_trial(self):
        trials = [make_trial("t1", "task1", latency=2.5)]
        metrics = calculate_latency_metrics(trials)
        assert metrics.mean_seconds == 2.5
        assert metrics.min_seconds == 2.5
        assert metrics.max_seconds == 2.5

    def test_multiple_trials(self):
        trials = [
            make_trial("t1", "task1", latency=1.0),
            make_trial("t2", "task2", latency=2.0),
            make_trial("t3", "task3", latency=3.0),
        ]
        metrics = calculate_latency_metrics(trials)
        assert metrics.mean_seconds == 2.0
        assert metrics.min_seconds == 1.0
        assert metrics.max_seconds == 3.0
        assert metrics.p50_seconds == 2.0

    def test_percentiles(self):
        trials = [make_trial(f"t{i}", f"task{i}", latency=float(i)) for i in range(1, 101)]
        metrics = calculate_latency_metrics(trials)
        assert metrics.p50_seconds is not None
        assert metrics.p95_seconds is not None
        assert metrics.p99_seconds is not None
        assert metrics.p50_seconds < metrics.p95_seconds
        assert metrics.p95_seconds < metrics.p99_seconds

    def test_pending_trials_excluded(self):
        trials = [
            make_trial("t1", "task1", latency=1.0),
            make_trial("t2", "task2", status=EvalStatus.PENDING),
        ]
        metrics = calculate_latency_metrics(trials)
        assert metrics.mean_seconds == 1.0


class TestCalculateTokenMetrics:
    def test_empty_trials(self):
        metrics = calculate_token_metrics([])
        assert metrics.total_input == 0
        assert metrics.total_output == 0

    def test_single_trial(self):
        trials = [make_trial("t1", "task1", tokens_input=200, tokens_output=100)]
        metrics = calculate_token_metrics(trials)
        assert metrics.total_input == 200
        assert metrics.total_output == 100
        assert metrics.mean_input_per_trial == 200.0
        assert metrics.mean_output_per_trial == 100.0

    def test_multiple_trials(self):
        trials = [
            make_trial("t1", "task1", tokens_input=100, tokens_output=50),
            make_trial("t2", "task2", tokens_input=200, tokens_output=100),
        ]
        metrics = calculate_token_metrics(trials)
        assert metrics.total_input == 300
        assert metrics.total_output == 150
        assert metrics.mean_input_per_trial == 150.0


class TestCalculateCostMetrics:
    def test_empty_trials(self):
        metrics = calculate_cost_metrics([])
        assert metrics.total_usd == 0.0

    def test_single_trial(self):
        trials = [make_trial("t1", "task1", cost_usd=0.05)]
        metrics = calculate_cost_metrics(trials)
        assert metrics.total_usd == 0.05
        assert metrics.mean_usd_per_trial == 0.05

    def test_multiple_trials(self):
        trials = [
            make_trial("t1", "task1", cost_usd=0.01),
            make_trial("t2", "task2", cost_usd=0.02),
            make_trial("t3", "task3", cost_usd=0.03),
        ]
        metrics = calculate_cost_metrics(trials)
        assert metrics.total_usd == 0.06
        assert metrics.mean_usd_per_trial == pytest.approx(0.02)
        assert metrics.min_usd_per_trial == 0.01
        assert metrics.max_usd_per_trial == 0.03


class TestCalculateTaskMetrics:
    def test_empty_trials(self):
        metrics = calculate_task_metrics([])
        assert metrics == {}

    def test_single_task(self):
        trials = [
            make_trial("t1", "task1", passed=True),
            make_trial("t2", "task1", passed=False),
        ]
        metrics = calculate_task_metrics(trials)
        assert "task1" in metrics
        assert metrics["task1"].total_attempts == 2
        assert metrics["task1"].successful_attempts == 1
        assert metrics["task1"].pass_rate == 0.5

    def test_multiple_tasks(self):
        trials = [
            make_trial("t1", "task1", passed=True),
            make_trial("t2", "task2", passed=False),
        ]
        metrics = calculate_task_metrics(trials)
        assert len(metrics) == 2
        assert metrics["task1"].pass_rate == 1.0
        assert metrics["task2"].pass_rate == 0.0

    def test_pass_at_k_included(self):
        trials = [make_trial(f"t{i}", "task1", passed=(i % 2 == 0)) for i in range(10)]
        metrics = calculate_task_metrics(trials, k_values=[1, 2, 3])
        assert 1 in metrics["task1"].pass_at_k
        assert 2 in metrics["task1"].pass_at_k
        assert 3 in metrics["task1"].pass_at_k

    def test_confidence_interval_included(self):
        trials = [make_trial(f"t{i}", "task1", passed=True) for i in range(10)]
        metrics = calculate_task_metrics(trials)
        assert metrics["task1"].confidence_interval is not None


class TestCalculateSuiteMetricsDetailed:
    def test_empty_trials(self):
        metrics = calculate_suite_metrics_detailed([], "suite-1", "run-1")
        assert metrics.suite_id == "suite-1"
        assert metrics.run_id == "run-1"
        assert metrics.total_trials == 0

    def test_single_trial(self):
        trials = [make_trial("t1", "task1", passed=True, score=0.9)]
        metrics = calculate_suite_metrics_detailed(trials, "suite-1", "run-1")
        assert metrics.total_trials == 1
        assert metrics.passed_trials == 1
        assert metrics.pass_rate == 1.0
        assert metrics.mean_score == 0.9

    def test_mixed_trials(self):
        trials = [
            make_trial("t1", "task1", passed=True, score=1.0),
            make_trial("t2", "task2", passed=False, score=0.5),
            make_trial("t3", "task3", passed=True, score=0.8),
        ]
        metrics = calculate_suite_metrics_detailed(trials, "suite-1", "run-1")
        assert metrics.total_trials == 3
        assert metrics.passed_trials == 2
        assert metrics.failed_trials == 1
        assert metrics.pass_rate == pytest.approx(2 / 3)

    def test_pass_at_k_included(self):
        trials = [make_trial(f"t{i}", f"task{i}", passed=(i % 2 == 0)) for i in range(10)]
        metrics = calculate_suite_metrics_detailed(trials, "suite-1", "run-1")
        assert 1 in metrics.pass_at_k
        assert 2 in metrics.pass_at_k

    def test_latency_metrics_included(self):
        trials = [make_trial(f"t{i}", f"task{i}", latency=float(i + 1)) for i in range(5)]
        metrics = calculate_suite_metrics_detailed(trials, "suite-1", "run-1")
        assert metrics.latency.mean_seconds == 3.0
        assert metrics.latency.p50_seconds is not None

    def test_token_metrics_included(self):
        trials = [
            make_trial(f"t{i}", f"task{i}", tokens_input=100, tokens_output=50) for i in range(3)
        ]
        metrics = calculate_suite_metrics_detailed(trials, "suite-1", "run-1")
        assert metrics.tokens.total_input == 300
        assert metrics.tokens.total_output == 150

    def test_cost_metrics_included(self):
        trials = [make_trial(f"t{i}", f"task{i}", cost_usd=0.01) for i in range(5)]
        metrics = calculate_suite_metrics_detailed(trials, "suite-1", "run-1")
        assert metrics.cost.total_usd == 0.05

    def test_task_metrics_included(self):
        trials = [
            make_trial("t1", "task1", passed=True),
            make_trial("t2", "task2", passed=False),
        ]
        metrics = calculate_suite_metrics_detailed(trials, "suite-1", "run-1")
        assert "task1" in metrics.task_metrics
        assert "task2" in metrics.task_metrics

    def test_grader_metrics_included(self):
        grader_results = [
            GraderResult(grader_type="string_match", score=1.0, passed=True),
        ]
        trials = [make_trial("t1", "task1", grader_results=grader_results)]
        metrics = calculate_suite_metrics_detailed(trials, "suite-1", "run-1")
        assert "string_match" in metrics.grader_metrics


class TestToSuiteMetrics:
    def test_conversion(self):
        trials = [
            make_trial("t1", "task1", passed=True, score=1.0, latency=2.0),
            make_trial("t2", "task2", passed=False, score=0.5, latency=3.0),
        ]
        detailed = calculate_suite_metrics_detailed(trials, "suite-1", "run-1")
        simple = to_suite_metrics(detailed)

        assert simple.suite_id == "suite-1"
        assert simple.run_id == "run-1"
        assert simple.total_tasks == 2
        assert simple.completed_tasks == 2
        assert simple.pass_rate == 0.5
        assert simple.mean_score == 0.75
        assert simple.latency_p50_seconds is not None


class TestConfidenceIntervalProperties:
    def test_width(self):
        ci = ConfidenceInterval(lower=0.4, upper=0.6, confidence_level=0.95)
        assert ci.width == pytest.approx(0.2)

    def test_midpoint(self):
        ci = ConfidenceInterval(lower=0.4, upper=0.6, confidence_level=0.95)
        assert ci.midpoint == 0.5


class TestSignificanceResultProperties:
    def test_default_values(self):
        result = SignificanceResult(
            statistic=1.5,
            p_value=0.134,
            significant=False,
        )
        assert result.alpha == 0.05
        assert result.test_type == "z_test"
        assert result.effect_size is None


class TestFisherExactTest:
    def test_identical_proportions(self):
        table = [[50, 50], [50, 50]]
        result = fisher_exact_test(table)
        assert result.p_value == pytest.approx(1.0, abs=0.01)
        assert not result.significant
        assert result.test_type == "fisher_exact"

    def test_significantly_different(self):
        table = [[10, 0], [0, 10]]
        result = fisher_exact_test(table)
        assert result.p_value < 0.001
        assert result.significant

    def test_empty_table(self):
        table = [[0, 0], [0, 0]]
        result = fisher_exact_test(table)
        assert result.p_value == 1.0
        assert not result.significant

    def test_small_sample(self):
        table = [[3, 1], [1, 3]]
        result = fisher_exact_test(table)
        assert 0.0 < result.p_value <= 1.0
        assert result.effect_size is not None

    def test_one_tailed_extreme(self):
        table = [[8, 2], [1, 9]]
        result = fisher_exact_test(table)
        assert result.p_value < 0.05

    def test_invalid_table_shape(self):
        result = fisher_exact_test([[1, 2, 3], [4, 5, 6]])
        assert result.p_value == 1.0
        assert not result.significant

    def test_edge_case_all_pass_vs_all_fail(self):
        table = [[10, 0], [0, 10]]
        result = fisher_exact_test(table)
        assert result.p_value < 0.001
        assert result.significant


class TestCompareGraders:
    def test_identical_results(self):
        grader_a = [GraderResult(grader_type="a", score=1.0, passed=True) for _ in range(10)]
        grader_b = [GraderResult(grader_type="b", score=1.0, passed=True) for _ in range(10)]
        result = compare_graders(grader_a, grader_b)
        assert not result.significant

    def test_significantly_different(self):
        grader_a = [GraderResult(grader_type="a", score=1.0, passed=True) for _ in range(20)]
        grader_b = [GraderResult(grader_type="b", score=0.0, passed=False) for _ in range(20)]
        result = compare_graders(grader_a, grader_b)
        assert result.significant
        assert result.p_value < 0.001

    def test_empty_results(self):
        result = compare_graders([], [])
        assert result.p_value == 1.0
        assert not result.significant

    def test_uses_chi_square_for_large_samples(self):
        grader_a = [
            GraderResult(grader_type="a", score=1.0, passed=(i % 2 == 0)) for i in range(60)
        ]
        grader_b = [
            GraderResult(grader_type="b", score=1.0, passed=(i % 3 == 0)) for i in range(60)
        ]
        result = compare_graders(grader_a, grader_b, use_chi_square_threshold=100)
        assert result.test_type == "chi_square_test"

    def test_uses_fisher_for_small_samples(self):
        grader_a = [GraderResult(grader_type="a", score=1.0, passed=True) for _ in range(5)]
        grader_b = [GraderResult(grader_type="b", score=0.0, passed=False) for _ in range(5)]
        result = compare_graders(grader_a, grader_b, use_chi_square_threshold=100)
        assert result.test_type == "fisher_exact"

    def test_mixed_results(self):
        grader_a = [
            GraderResult(grader_type="a", score=1.0, passed=True),
            GraderResult(grader_type="a", score=1.0, passed=True),
            GraderResult(grader_type="a", score=0.0, passed=False),
        ]
        grader_b = [
            GraderResult(grader_type="b", score=0.0, passed=False),
            GraderResult(grader_type="b", score=0.0, passed=False),
            GraderResult(grader_type="b", score=1.0, passed=True),
        ]
        result = compare_graders(grader_a, grader_b)
        assert 0.0 < result.p_value <= 1.0
