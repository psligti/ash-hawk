"""Tests for ash_hawk.metrics.statistics."""

from __future__ import annotations

import math

import pydantic as pd
import pytest

from ash_hawk.metrics.statistics import (
    ConfidenceInterval,
    CostMetrics,
    LatencyMetrics,
    SignificanceResult,
    TaskMetrics,
    TokenMetrics,
    calculate_cost_metrics,
    calculate_latency_metrics,
    calculate_pass_at_k,
    calculate_pass_at_k_from_trials,
    calculate_task_metrics,
    calculate_token_metrics,
    mean,
    percentile,
    std,
    wilson_confidence_interval,
)
from ash_hawk.types import (
    EvalOutcome,
    EvalStatus,
    EvalTranscript,
    EvalTrial,
    TokenUsage,
    TrialResult,
)

# =============================================================================
# HELPERS
# =============================================================================


def _make_trial(
    trial_id: str = "t1",
    task_id: str = "task-1",
    duration_seconds: float = 1.0,
    cost_usd: float = 0.01,
    token_input: int = 100,
    token_output: int = 50,
    token_reasoning: int = 0,
    token_cache_read: int = 0,
    token_cache_write: int = 0,
    passed: bool = True,
    score: float = 1.0,
) -> EvalTrial:
    """Build an EvalTrial with a populated TrialResult."""
    return EvalTrial(
        id=trial_id,
        task_id=task_id,
        result=TrialResult(
            trial_id=trial_id,
            outcome=EvalOutcome(status=EvalStatus.COMPLETED),
            transcript=EvalTranscript(
                duration_seconds=duration_seconds,
                cost_usd=cost_usd,
                token_usage=TokenUsage(
                    input=token_input,
                    output=token_output,
                    reasoning=token_reasoning,
                    cache_read=token_cache_read,
                    cache_write=token_cache_write,
                ),
            ),
            aggregate_passed=passed,
            aggregate_score=score,
        ),
    )


def _make_trial_no_result(
    trial_id: str = "t-no-result",
    task_id: str = "task-1",
) -> EvalTrial:
    """Build an EvalTrial with no result."""
    return EvalTrial(id=trial_id, task_id=task_id)


# =============================================================================
# PYDANTIC MODEL TESTS
# =============================================================================


class TestConfidenceInterval:
    """Test ConfidenceInterval model."""

    def test_basic_construction(self) -> None:
        ci = ConfidenceInterval(lower=0.3, upper=0.7)
        assert ci.lower == 0.3
        assert ci.upper == 0.7
        assert ci.confidence_level == 0.95
        assert ci.method == "wilson"

    def test_custom_defaults(self) -> None:
        ci = ConfidenceInterval(lower=0.0, upper=1.0, confidence_level=0.99, method="agresti-coull")
        assert ci.confidence_level == 0.99
        assert ci.method == "agresti-coull"

    def test_extra_forbid(self) -> None:
        with pytest.raises(pd.ValidationError):
            ConfidenceInterval(lower=0.0, upper=1.0, extra_field="nope")  # type: ignore[call-arg]


class TestSignificanceResult:
    """Test SignificanceResult model."""

    def test_basic_construction(self) -> None:
        sr = SignificanceResult(statistic=2.5, p_value=0.01, significant=True)
        assert sr.statistic == 2.5
        assert sr.p_value == 0.01
        assert sr.significant is True
        assert sr.alpha == 0.05
        assert sr.test_type == "z_test"

    def test_custom_defaults(self) -> None:
        sr = SignificanceResult(
            statistic=1.0, p_value=0.10, significant=False, alpha=0.10, test_type="t_test"
        )
        assert sr.alpha == 0.10
        assert sr.test_type == "t_test"

    def test_extra_forbid(self) -> None:
        with pytest.raises(pd.ValidationError):
            SignificanceResult(
                statistic=0.0,
                p_value=0.5,
                significant=False,
                bad="field",  # type: ignore[call-arg]
            )


class TestLatencyMetrics:
    """Test LatencyMetrics model."""

    def test_defaults(self) -> None:
        lm = LatencyMetrics()
        assert lm.min_seconds == 0.0
        assert lm.max_seconds == 0.0
        assert lm.mean_seconds == 0.0
        assert lm.median_seconds is None
        assert lm.p50_seconds is None
        assert lm.p90_seconds is None
        assert lm.p95_seconds is None
        assert lm.p99_seconds is None
        assert lm.std_seconds == 0.0

    def test_ge_zero_constraint(self) -> None:
        with pytest.raises(pd.ValidationError):
            LatencyMetrics(min_seconds=-1.0)

    def test_extra_forbid(self) -> None:
        with pytest.raises(pd.ValidationError):
            LatencyMetrics(bogus=1.0)  # type: ignore[call-arg]


class TestTokenMetrics:
    """Test TokenMetrics model."""

    def test_defaults(self) -> None:
        tm = TokenMetrics()
        assert tm.total_input == 0
        assert tm.total_output == 0
        assert tm.total_reasoning == 0
        assert tm.total_cache_read == 0
        assert tm.total_cache_write == 0
        assert tm.mean_input_per_trial == 0.0
        assert tm.mean_output_per_trial == 0.0

    def test_ge_zero_constraint(self) -> None:
        with pytest.raises(pd.ValidationError):
            TokenMetrics(total_input=-1)

    def test_extra_forbid(self) -> None:
        with pytest.raises(pd.ValidationError):
            TokenMetrics(extra=5)  # type: ignore[call-arg]


class TestCostMetrics:
    """Test CostMetrics model."""

    def test_defaults(self) -> None:
        cm = CostMetrics()
        assert cm.total_usd == 0.0
        assert cm.mean_usd_per_trial == 0.0
        assert cm.min_usd_per_trial == 0.0
        assert cm.max_usd_per_trial == 0.0

    def test_ge_zero_constraint(self) -> None:
        with pytest.raises(pd.ValidationError):
            CostMetrics(total_usd=-0.01)

    def test_extra_forbid(self) -> None:
        with pytest.raises(pd.ValidationError):
            CostMetrics(foo="bar")  # type: ignore[call-arg]


class TestTaskMetrics:
    """Test TaskMetrics model."""

    def test_defaults(self) -> None:
        tm = TaskMetrics(task_id="task-1")
        assert tm.task_id == "task-1"
        assert tm.total_attempts == 0
        assert tm.successful_attempts == 0
        assert tm.pass_rate == 0.0
        assert tm.pass_at_k == {}
        assert tm.confidence_interval is None
        assert tm.mean_score == 0.0
        assert isinstance(tm.latency, LatencyMetrics)
        assert isinstance(tm.tokens, TokenMetrics)
        assert isinstance(tm.cost, CostMetrics)

    def test_pass_rate_bounds(self) -> None:
        with pytest.raises(pd.ValidationError):
            TaskMetrics(task_id="t", pass_rate=1.5)

    def test_mean_score_bounds(self) -> None:
        with pytest.raises(pd.ValidationError):
            TaskMetrics(task_id="t", mean_score=-0.1)

    def test_extra_forbid(self) -> None:
        with pytest.raises(pd.ValidationError):
            TaskMetrics(task_id="t", nope=True)  # type: ignore[call-arg]

    def test_nested_confidence_interval(self) -> None:
        ci = ConfidenceInterval(lower=0.2, upper=0.8)
        tm = TaskMetrics(task_id="t", confidence_interval=ci)
        assert tm.confidence_interval is not None
        assert tm.confidence_interval.lower == 0.2


# =============================================================================
# PURE FUNCTION TESTS
# =============================================================================


class TestPercentile:
    """Test percentile function."""

    def test_empty_list(self) -> None:
        assert percentile([], 50) is None

    def test_single_value(self) -> None:
        assert percentile([5.0], 50) == 5.0
        assert percentile([5.0], 0) == 5.0
        assert percentile([5.0], 100) == 5.0

    def test_p0_returns_min(self) -> None:
        assert percentile([1.0, 2.0, 3.0], 0) == 1.0

    def test_p100_returns_max(self) -> None:
        assert percentile([1.0, 2.0, 3.0], 100) == 3.0

    def test_p50_median_odd(self) -> None:
        result = percentile([1.0, 2.0, 3.0], 50)
        assert result == 2.0

    def test_p50_median_even(self) -> None:
        result = percentile([1.0, 2.0, 3.0, 4.0], 50)
        assert result is not None
        assert result == pytest.approx(2.5)

    def test_interpolation(self) -> None:
        result = percentile([10.0, 20.0, 30.0, 40.0], 25)
        assert result is not None
        assert result == pytest.approx(17.5)

    def test_unsorted_input(self) -> None:
        result = percentile([3.0, 1.0, 2.0], 50)
        assert result == 2.0

    def test_all_same_values(self) -> None:
        result = percentile([5.0, 5.0, 5.0], 90)
        assert result == 5.0


class TestMean:
    """Test mean function."""

    def test_empty_list(self) -> None:
        assert mean([]) == 0.0

    def test_single_value(self) -> None:
        assert mean([7.0]) == 7.0

    def test_multiple_values(self) -> None:
        assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_negative_values(self) -> None:
        assert mean([-1.0, 1.0]) == pytest.approx(0.0)


class TestStd:
    """Test std function."""

    def test_empty_list(self) -> None:
        assert std([]) == 0.0

    def test_single_value(self) -> None:
        assert std([5.0]) == 0.0

    def test_population_std_ddof_0(self) -> None:
        result = std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert result == pytest.approx(2.0, abs=0.01)

    def test_sample_std_ddof_1(self) -> None:
        result = std([2.0, 4.0], ddof=1)
        assert result == pytest.approx(math.sqrt(2.0))

    def test_all_same_values(self) -> None:
        assert std([3.0, 3.0, 3.0]) == 0.0

    def test_two_values_ddof_0(self) -> None:
        result = std([0.0, 10.0], ddof=0)
        assert result == pytest.approx(5.0)


# =============================================================================
# PASS@K TESTS
# =============================================================================


class TestCalculatePassAtK:
    """Test calculate_pass_at_k."""

    def test_total_zero(self) -> None:
        assert calculate_pass_at_k(0, 0, 1) == 0.0

    def test_correct_zero(self) -> None:
        assert calculate_pass_at_k(0, 10, 5) == 0.0

    def test_correct_equals_total(self) -> None:
        assert calculate_pass_at_k(10, 10, 5) == 1.0

    def test_correct_greater_than_total(self) -> None:
        assert calculate_pass_at_k(15, 10, 5) == 1.0

    def test_k_equals_1(self) -> None:
        result = calculate_pass_at_k(3, 10, 1)
        assert result == pytest.approx(0.3)

    def test_k_greater_than_total_caps(self) -> None:
        # k=20, total=10, correct=5 -> k capped to 10
        result = calculate_pass_at_k(5, 10, 20)
        assert result == 1.0  # all 10 sampled, 5 correct, guaranteed at least one

    def test_pass_at_k_basic(self) -> None:
        # 5 correct out of 10, k=3
        result = calculate_pass_at_k(5, 10, 3)
        assert 0.0 < result < 1.0

    def test_n_minus_correct_lt_k(self) -> None:
        # n=10, correct=8, k=5 -> n-correct=2 < k=5 -> returns 1.0
        result = calculate_pass_at_k(8, 10, 5)
        assert result == 1.0

    def test_k_equals_total(self) -> None:
        # k=total means we sample all, guaranteed pass if correct>0
        result = calculate_pass_at_k(1, 10, 10)
        assert result == 1.0

    def test_monotonic_with_k(self) -> None:
        """pass@k should increase with k."""
        results = [calculate_pass_at_k(3, 10, k) for k in [1, 2, 3, 5]]
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1]


class TestCalculatePassAtKFromTrials:
    """Test calculate_pass_at_k_from_trials."""

    def test_empty_trials(self) -> None:
        assert calculate_pass_at_k_from_trials([], 1) == 0.0

    def test_single_task_all_pass(self) -> None:
        trials = [_make_trial(trial_id=f"t{i}", passed=True) for i in range(5)]
        result = calculate_pass_at_k_from_trials(trials, 1)
        assert result == 1.0

    def test_single_task_none_pass(self) -> None:
        trials = [_make_trial(trial_id=f"t{i}", passed=False) for i in range(5)]
        result = calculate_pass_at_k_from_trials(trials, 1)
        assert result == 0.0

    def test_single_task_partial_pass(self) -> None:
        trials = [
            _make_trial(trial_id="t1", passed=True),
            _make_trial(trial_id="t2", passed=False),
        ]
        result = calculate_pass_at_k_from_trials(trials, 1)
        assert result == pytest.approx(0.5)

    def test_multiple_tasks_averaged(self) -> None:
        trials = [
            # task-a: 2/2 pass
            _make_trial(trial_id="t1", task_id="task-a", passed=True),
            _make_trial(trial_id="t2", task_id="task-a", passed=True),
            # task-b: 0/2 pass
            _make_trial(trial_id="t3", task_id="task-b", passed=False),
            _make_trial(trial_id="t4", task_id="task-b", passed=False),
        ]
        result = calculate_pass_at_k_from_trials(trials, 1)
        # task-a pass@1=1.0, task-b pass@1=0.0, mean=0.5
        assert result == pytest.approx(0.5)

    def test_trials_without_results_not_counted_as_pass(self) -> None:
        trials = [
            _make_trial(trial_id="t1", passed=True),
            _make_trial_no_result(trial_id="t2"),
        ]
        result = calculate_pass_at_k_from_trials(trials, 1)
        # 1 correct out of 2 total
        assert result == pytest.approx(0.5)


# =============================================================================
# WILSON CONFIDENCE INTERVAL TESTS
# =============================================================================


class TestWilsonConfidenceInterval:
    """Test wilson_confidence_interval."""

    def test_total_zero(self) -> None:
        ci = wilson_confidence_interval(0, 0)
        assert ci.lower == 0.0
        assert ci.upper == 1.0
        assert ci.method == "wilson"
        assert ci.confidence_level == 0.95

    def test_all_successes(self) -> None:
        ci = wilson_confidence_interval(100, 100)
        assert ci.lower > 0.9
        assert ci.upper == 1.0  # clamped

    def test_all_failures(self) -> None:
        ci = wilson_confidence_interval(0, 100)
        assert ci.lower == pytest.approx(0.0, abs=1e-10)
        assert ci.upper < 0.1

    def test_half_and_half(self) -> None:
        ci = wilson_confidence_interval(50, 100)
        assert ci.lower < 0.5
        assert ci.upper > 0.5
        assert ci.lower > 0.0
        assert ci.upper < 1.0

    def test_custom_confidence_level(self) -> None:
        ci_95 = wilson_confidence_interval(50, 100, confidence_level=0.95)
        ci_99 = wilson_confidence_interval(50, 100, confidence_level=0.99)
        # 99% CI should be wider than 95% CI
        assert (ci_99.upper - ci_99.lower) > (ci_95.upper - ci_95.lower)

    def test_small_sample(self) -> None:
        ci = wilson_confidence_interval(1, 2)
        assert ci.lower >= 0.0
        assert ci.upper <= 1.0


# =============================================================================
# TRIAL-BASED FUNCTION TESTS
# =============================================================================


class TestCalculateLatencyMetrics:
    """Test calculate_latency_metrics."""

    def test_empty_trials(self) -> None:
        result = calculate_latency_metrics([])
        assert result.min_seconds == 0.0
        assert result.max_seconds == 0.0
        assert result.mean_seconds == 0.0
        assert result.median_seconds is None

    def test_trials_without_results(self) -> None:
        trials = [_make_trial_no_result()]
        result = calculate_latency_metrics(trials)
        assert result == LatencyMetrics()

    def test_single_trial(self) -> None:
        trials = [_make_trial(duration_seconds=5.0)]
        result = calculate_latency_metrics(trials)
        assert result.min_seconds == 5.0
        assert result.max_seconds == 5.0
        assert result.mean_seconds == 5.0
        assert result.median_seconds == 5.0
        assert result.p50_seconds == 5.0
        assert result.p90_seconds == 5.0
        assert result.p95_seconds == 5.0
        assert result.p99_seconds == 5.0
        assert result.std_seconds == 0.0

    def test_multiple_trials(self) -> None:
        trials = [
            _make_trial(trial_id="t1", duration_seconds=1.0),
            _make_trial(trial_id="t2", duration_seconds=2.0),
            _make_trial(trial_id="t3", duration_seconds=3.0),
        ]
        result = calculate_latency_metrics(trials)
        assert result.min_seconds == 1.0
        assert result.max_seconds == 3.0
        assert result.mean_seconds == pytest.approx(2.0)
        assert result.median_seconds == 2.0
        assert result.std_seconds > 0.0

    def test_mixed_result_and_no_result(self) -> None:
        trials = [
            _make_trial(trial_id="t1", duration_seconds=10.0),
            _make_trial_no_result(trial_id="t2"),
        ]
        result = calculate_latency_metrics(trials)
        assert result.min_seconds == 10.0
        assert result.max_seconds == 10.0
        assert result.mean_seconds == 10.0


class TestCalculateTokenMetrics:
    """Test calculate_token_metrics."""

    def test_empty_trials(self) -> None:
        result = calculate_token_metrics([])
        assert result == TokenMetrics()

    def test_trials_without_results(self) -> None:
        trials = [_make_trial_no_result()]
        result = calculate_token_metrics(trials)
        assert result.total_input == 0

    def test_single_trial(self) -> None:
        trials = [
            _make_trial(
                token_input=100,
                token_output=50,
                token_reasoning=10,
                token_cache_read=5,
                token_cache_write=3,
            )
        ]
        result = calculate_token_metrics(trials)
        assert result.total_input == 100
        assert result.total_output == 50
        assert result.total_reasoning == 10
        assert result.total_cache_read == 5
        assert result.total_cache_write == 3
        assert result.mean_input_per_trial == 100.0
        assert result.mean_output_per_trial == 50.0

    def test_multiple_trials(self) -> None:
        trials = [
            _make_trial(trial_id="t1", token_input=100, token_output=50),
            _make_trial(trial_id="t2", token_input=200, token_output=100),
        ]
        result = calculate_token_metrics(trials)
        assert result.total_input == 300
        assert result.total_output == 150
        assert result.mean_input_per_trial == pytest.approx(150.0)
        assert result.mean_output_per_trial == pytest.approx(75.0)

    def test_mixed_result_and_no_result(self) -> None:
        trials = [
            _make_trial(trial_id="t1", token_input=200, token_output=80),
            _make_trial_no_result(trial_id="t2"),
        ]
        result = calculate_token_metrics(trials)
        assert result.total_input == 200
        assert result.mean_input_per_trial == 200.0


class TestCalculateCostMetrics:
    """Test calculate_cost_metrics."""

    def test_empty_trials(self) -> None:
        result = calculate_cost_metrics([])
        assert result == CostMetrics()

    def test_trials_without_results(self) -> None:
        trials = [_make_trial_no_result()]
        result = calculate_cost_metrics(trials)
        assert result.total_usd == 0.0

    def test_single_trial(self) -> None:
        trials = [_make_trial(cost_usd=0.05)]
        result = calculate_cost_metrics(trials)
        assert result.total_usd == pytest.approx(0.05)
        assert result.mean_usd_per_trial == pytest.approx(0.05)
        assert result.min_usd_per_trial == pytest.approx(0.05)
        assert result.max_usd_per_trial == pytest.approx(0.05)

    def test_multiple_trials(self) -> None:
        trials = [
            _make_trial(trial_id="t1", cost_usd=0.01),
            _make_trial(trial_id="t2", cost_usd=0.03),
            _make_trial(trial_id="t3", cost_usd=0.05),
        ]
        result = calculate_cost_metrics(trials)
        assert result.total_usd == pytest.approx(0.09)
        assert result.mean_usd_per_trial == pytest.approx(0.03)
        assert result.min_usd_per_trial == pytest.approx(0.01)
        assert result.max_usd_per_trial == pytest.approx(0.05)

    def test_mixed_result_and_no_result(self) -> None:
        trials = [
            _make_trial(trial_id="t1", cost_usd=0.10),
            _make_trial_no_result(trial_id="t2"),
        ]
        result = calculate_cost_metrics(trials)
        assert result.total_usd == pytest.approx(0.10)
        assert result.mean_usd_per_trial == pytest.approx(0.10)


# =============================================================================
# CALCULATE_TASK_METRICS TESTS
# =============================================================================


class TestCalculateTaskMetrics:
    """Test calculate_task_metrics."""

    def test_empty_trials(self) -> None:
        result = calculate_task_metrics([])
        assert result == {}

    def test_single_task_all_pass(self) -> None:
        trials = [
            _make_trial(trial_id="t1", task_id="task-a", passed=True, score=1.0),
            _make_trial(trial_id="t2", task_id="task-a", passed=True, score=0.8),
        ]
        result = calculate_task_metrics(trials)
        assert "task-a" in result
        tm = result["task-a"]
        assert tm.total_attempts == 2
        assert tm.successful_attempts == 2
        assert tm.pass_rate == 1.0
        assert tm.mean_score == pytest.approx(0.9)

    def test_single_task_partial_pass(self) -> None:
        trials = [
            _make_trial(trial_id="t1", task_id="task-a", passed=True, score=1.0),
            _make_trial(trial_id="t2", task_id="task-a", passed=False, score=0.0),
        ]
        result = calculate_task_metrics(trials)
        tm = result["task-a"]
        assert tm.total_attempts == 2
        assert tm.successful_attempts == 1
        assert tm.pass_rate == pytest.approx(0.5)
        assert tm.mean_score == pytest.approx(0.5)

    def test_multiple_tasks(self) -> None:
        trials = [
            _make_trial(trial_id="t1", task_id="task-a", passed=True),
            _make_trial(trial_id="t2", task_id="task-b", passed=False, score=0.0),
        ]
        result = calculate_task_metrics(trials)
        assert len(result) == 2
        assert "task-a" in result
        assert "task-b" in result

    def test_default_k_values(self) -> None:
        trials = [
            _make_trial(trial_id=f"t{i}", task_id="task-a", passed=(i % 2 == 0)) for i in range(10)
        ]
        result = calculate_task_metrics(trials)
        tm = result["task-a"]
        assert set(tm.pass_at_k.keys()) == {1, 2, 3, 5}

    def test_custom_k_values(self) -> None:
        trials = [_make_trial(trial_id="t1", task_id="task-a")]
        result = calculate_task_metrics(trials, k_values=[1, 10])
        tm = result["task-a"]
        assert set(tm.pass_at_k.keys()) == {1, 10}

    def test_confidence_interval_populated(self) -> None:
        trials = [
            _make_trial(trial_id=f"t{i}", task_id="task-a", passed=(i < 5)) for i in range(10)
        ]
        result = calculate_task_metrics(trials)
        tm = result["task-a"]
        assert tm.confidence_interval is not None
        assert tm.confidence_interval.method == "wilson"
        assert tm.confidence_interval.lower < tm.pass_rate
        assert tm.confidence_interval.upper > tm.pass_rate

    def test_latency_tokens_cost_populated(self) -> None:
        trials = [
            _make_trial(
                trial_id="t1",
                task_id="task-a",
                duration_seconds=2.0,
                cost_usd=0.05,
                token_input=200,
                token_output=100,
            ),
        ]
        result = calculate_task_metrics(trials)
        tm = result["task-a"]
        assert tm.latency.mean_seconds == 2.0
        assert tm.tokens.total_input == 200
        assert tm.cost.total_usd == pytest.approx(0.05)

    def test_custom_confidence_level(self) -> None:
        trials = [
            _make_trial(trial_id=f"t{i}", task_id="task-a", passed=(i < 3)) for i in range(10)
        ]
        result = calculate_task_metrics(trials, confidence_level=0.99)
        tm = result["task-a"]
        assert tm.confidence_interval is not None
        assert tm.confidence_interval.confidence_level == 0.99

    def test_trials_without_results_counted_as_attempts(self) -> None:
        trials = [
            _make_trial(trial_id="t1", task_id="task-a", passed=True),
            _make_trial_no_result(trial_id="t2", task_id="task-a"),
        ]
        result = calculate_task_metrics(trials)
        tm = result["task-a"]
        assert tm.total_attempts == 2
        assert tm.successful_attempts == 1
        assert tm.pass_rate == pytest.approx(0.5)
