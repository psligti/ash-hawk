"""Statistical metrics for ash-hawk evaluation harness.

This module provides statistical calculations for evaluating AI agent
performance following Anthropic's eval best practices.

Key metrics:
- pass@k: Probability that at least 1 of k attempts passes
- pass^k: Probability that all k attempts pass
- Confidence intervals (Wilson score, normal approximation)
- Significance testing (z-test, chi-square)
- Latency percentiles (p50, p95, p99)
- Token usage and cost metrics

References:
- Chen et al. (2021): "Evaluating Large Language Models Trained on Code"
- Brown et al. (2020): "Language Models are Few-Shot Learners"
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import pydantic as pd

from ash_hawk.types import (
    EvalStatus,
    EvalTrial,
    GraderResult,
    SuiteMetrics,
    TokenUsage,
)


# =============================================================================
# DATA CLASSES FOR METRICS
# =============================================================================


class ConfidenceInterval(pd.BaseModel):
    """Confidence interval for a metric."""

    lower: float = pd.Field(
        description="Lower bound of the confidence interval",
    )
    upper: float = pd.Field(
        description="Upper bound of the confidence interval",
    )
    confidence_level: float = pd.Field(
        default=0.95,
        description="Confidence level (e.g., 0.95 for 95% CI)",
    )
    method: str = pd.Field(
        default="wilson",
        description="Method used to calculate the CI (wilson, normal, clopper_pearson)",
    )

    @pd.computed_field  # type: ignore[prop-decorator]
    @property
    def width(self) -> float:
        """Width of the confidence interval."""
        return self.upper - self.lower

    @pd.computed_field  # type: ignore[prop-decorator]
    @property
    def midpoint(self) -> float:
        """Midpoint of the confidence interval."""
        return (self.lower + self.upper) / 2

    model_config = pd.ConfigDict(extra="forbid")


class SignificanceResult(pd.BaseModel):
    """Result of a statistical significance test."""

    statistic: float = pd.Field(
        description="Test statistic value",
    )
    p_value: float = pd.Field(
        description="P-value of the test",
    )
    significant: bool = pd.Field(
        description="Whether the result is statistically significant",
    )
    alpha: float = pd.Field(
        default=0.05,
        description="Significance level used",
    )
    test_type: str = pd.Field(
        default="z_test",
        description="Type of significance test performed",
    )
    effect_size: float | None = pd.Field(
        default=None,
        description="Effect size (e.g., Cohen's h for proportions)",
    )

    model_config = pd.ConfigDict(extra="forbid")


class LatencyMetrics(pd.BaseModel):
    """Latency metrics for trials."""

    min_seconds: float = pd.Field(
        default=0.0,
        ge=0.0,
        description="Minimum latency",
    )
    max_seconds: float = pd.Field(
        default=0.0,
        ge=0.0,
        description="Maximum latency",
    )
    mean_seconds: float = pd.Field(
        default=0.0,
        ge=0.0,
        description="Mean latency",
    )
    median_seconds: float | None = pd.Field(
        default=None,
        description="Median (p50) latency",
    )
    p50_seconds: float | None = pd.Field(
        default=None,
        description="50th percentile latency",
    )
    p90_seconds: float | None = pd.Field(
        default=None,
        description="90th percentile latency",
    )
    p95_seconds: float | None = pd.Field(
        default=None,
        description="95th percentile latency",
    )
    p99_seconds: float | None = pd.Field(
        default=None,
        description="99th percentile latency",
    )
    std_seconds: float = pd.Field(
        default=0.0,
        ge=0.0,
        description="Standard deviation of latency",
    )

    model_config = pd.ConfigDict(extra="forbid")


class TokenMetrics(pd.BaseModel):
    """Token usage metrics for trials."""

    total_input: int = pd.Field(
        default=0,
        ge=0,
        description="Total input tokens",
    )
    total_output: int = pd.Field(
        default=0,
        ge=0,
        description="Total output tokens",
    )
    total_reasoning: int = pd.Field(
        default=0,
        ge=0,
        description="Total reasoning tokens",
    )
    total_cache_read: int = pd.Field(
        default=0,
        ge=0,
        description="Total cache read tokens",
    )
    total_cache_write: int = pd.Field(
        default=0,
        ge=0,
        description="Total cache write tokens",
    )
    mean_input_per_trial: float = pd.Field(
        default=0.0,
        ge=0.0,
        description="Mean input tokens per trial",
    )
    mean_output_per_trial: float = pd.Field(
        default=0.0,
        ge=0.0,
        description="Mean output tokens per trial",
    )

    @pd.computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.total_input + self.total_output + self.total_reasoning

    model_config = pd.ConfigDict(extra="forbid")


class CostMetrics(pd.BaseModel):
    """Cost metrics for trials."""

    total_usd: float = pd.Field(
        default=0.0,
        ge=0.0,
        description="Total cost in USD",
    )
    mean_usd_per_trial: float = pd.Field(
        default=0.0,
        ge=0.0,
        description="Mean cost per trial in USD",
    )
    min_usd_per_trial: float = pd.Field(
        default=0.0,
        ge=0.0,
        description="Minimum cost per trial in USD",
    )
    max_usd_per_trial: float = pd.Field(
        default=0.0,
        ge=0.0,
        description="Maximum cost per trial in USD",
    )

    model_config = pd.ConfigDict(extra="forbid")


class TaskMetrics(pd.BaseModel):
    """Metrics for a single task across multiple attempts."""

    task_id: str = pd.Field(
        description="Task identifier",
    )
    total_attempts: int = pd.Field(
        default=0,
        ge=0,
        description="Total number of attempts",
    )
    successful_attempts: int = pd.Field(
        default=0,
        ge=0,
        description="Number of successful attempts",
    )
    pass_rate: float = pd.Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Pass rate (successful / total)",
    )
    pass_at_k: dict[int, float] = pd.Field(
        default_factory=dict,
        description="pass@k metrics for various k values",
    )
    pass_caret_k: dict[int, float] = pd.Field(
        default_factory=dict,
        description="pass^k metrics for various k values",
    )
    confidence_interval: ConfidenceInterval | None = pd.Field(
        default=None,
        description="Confidence interval for pass rate",
    )
    mean_score: float = pd.Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Mean aggregate score across attempts",
    )
    latency: LatencyMetrics = pd.Field(
        default_factory=LatencyMetrics,
        description="Latency metrics",
    )
    tokens: TokenMetrics = pd.Field(
        default_factory=TokenMetrics,
        description="Token usage metrics",
    )
    cost: CostMetrics = pd.Field(
        default_factory=CostMetrics,
        description="Cost metrics",
    )

    model_config = pd.ConfigDict(extra="forbid")


class SuiteMetricsDetailed(pd.BaseModel):
    """Detailed suite-level metrics with full statistical analysis."""

    suite_id: str = pd.Field(
        description="Suite identifier",
    )
    run_id: str = pd.Field(
        description="Run identifier",
    )
    total_tasks: int = pd.Field(
        default=0,
        ge=0,
        description="Total number of unique tasks",
    )
    total_trials: int = pd.Field(
        default=0,
        ge=0,
        description="Total number of trials",
    )
    completed_trials: int = pd.Field(
        default=0,
        ge=0,
        description="Number of completed trials",
    )
    passed_trials: int = pd.Field(
        default=0,
        ge=0,
        description="Number of passed trials",
    )
    failed_trials: int = pd.Field(
        default=0,
        ge=0,
        description="Number of failed trials",
    )
    pass_rate: float = pd.Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall pass rate",
    )
    pass_rate_ci: ConfidenceInterval | None = pd.Field(
        default=None,
        description="Confidence interval for overall pass rate",
    )
    pass_at_k: dict[int, float] = pd.Field(
        default_factory=dict,
        description="pass@k metrics for various k values",
    )
    pass_caret_k: dict[int, float] = pd.Field(
        default_factory=dict,
        description="pass^k metrics for various k values",
    )
    mean_score: float = pd.Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Mean aggregate score",
    )
    score_std: float = pd.Field(
        default=0.0,
        ge=0.0,
        description="Standard deviation of scores",
    )
    latency: LatencyMetrics = pd.Field(
        default_factory=LatencyMetrics,
        description="Latency metrics",
    )
    tokens: TokenMetrics = pd.Field(
        default_factory=TokenMetrics,
        description="Token usage metrics",
    )
    cost: CostMetrics = pd.Field(
        default_factory=CostMetrics,
        description="Cost metrics",
    )
    task_metrics: dict[str, TaskMetrics] = pd.Field(
        default_factory=dict,
        description="Per-task metrics",
    )
    grader_metrics: dict[str, dict[str, Any]] = pd.Field(
        default_factory=dict,
        description="Per-grader metrics",
    )
    created_at: str = pd.Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="When metrics were computed",
    )

    model_config = pd.ConfigDict(extra="forbid")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def percentile(values: list[float], p: float) -> float | None:
    """Calculate the p-th percentile of a list of values.

    Uses linear interpolation between adjacent values.

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


def mean(values: list[float]) -> float:
    """Calculate the arithmetic mean of a list of values.

    Args:
        values: List of numeric values.

    Returns:
        The arithmetic mean, or 0.0 if empty.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def std(values: list[float], ddof: int = 0) -> float:
    """Calculate the standard deviation of a list of values.

    Args:
        values: List of numeric values.
        ddof: Delta degrees of freedom (0 for population, 1 for sample).

    Returns:
        The standard deviation, or 0.0 if empty.
    """
    if len(values) < 2:
        return 0.0

    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - ddof)
    return math.sqrt(variance)


# =============================================================================
# PASS@K CALCULATIONS
# =============================================================================


def calculate_pass_at_k(
    correct: int,
    total: int,
    k: int,
) -> float:
    """Calculate pass@k using the combinatorial formula.

    pass@k measures the probability that at least one of k attempts
    passes. Uses the formula from Chen et al. (2021):

        pass@k = 1 - C(n-c, k) / C(n, k)

    where:
        n = total number of attempts
        c = number of correct attempts
        k = number of attempts we're willing to try

    Args:
        correct: Number of correct/passing attempts (c).
        total: Total number of attempts (n).
        k: Number of attempts to consider.

    Returns:
        pass@k probability (0.0-1.0).
    """
    if total == 0:
        return 0.0
    if k > total:
        k = total
    if correct >= total:
        return 1.0
    if correct == 0:
        return 0.0

    # If k == 1, use simple proportion
    if k == 1:
        return correct / total

    # Use combinatorial formula: 1 - C(n-c, k) / C(n, k)
    # For numerical stability, compute log of combinations
    n = total
    c = correct

    # C(n-c, k) / C(n, k) = product of (n-c-k+1+i)/(n-k+1+i) for i in 0..k-1
    # Simplify to avoid large numbers
    if n - c < k:
        return 1.0  # Can't choose k from n-c if n-c < k

    # Compute using log for numerical stability
    log_ratio = 0.0
    for i in range(k):
        numerator = n - c - k + 1 + i
        denominator = n - k + 1 + i
        if numerator > 0 and denominator > 0:
            log_ratio += math.log(numerator) - math.log(denominator)

    ratio = math.exp(log_ratio)
    return max(0.0, min(1.0, 1.0 - ratio))


def calculate_pass_at_k_from_trials(
    trials: list[EvalTrial],
    k: int,
) -> float:
    """Calculate pass@k from a list of trials.

    Groups trials by task and calculates pass@k across all tasks.

    Args:
        trials: List of evaluation trials.
        k: Number of attempts to consider.

    Returns:
        pass@k probability (0.0-1.0).
    """
    if not trials:
        return 0.0

    # Group trials by task
    task_trials: dict[str, list[EvalTrial]] = defaultdict(list)
    for trial in trials:
        task_trials[trial.task_id].append(trial)

    # Calculate pass@k for each task
    task_pass_at_k: list[float] = []
    for task_id, task_trial_list in task_trials.items():
        correct = sum(
            1 for t in task_trial_list if t.result is not None and t.result.aggregate_passed
        )
        total = len(task_trial_list)
        if total > 0:
            task_pass_at_k.append(calculate_pass_at_k(correct, total, k))

    return mean(task_pass_at_k) if task_pass_at_k else 0.0


# =============================================================================
# PASS^k CALCULATIONS
# =============================================================================


def calculate_pass_caret_k(
    correct: int,
    total: int,
    k: int,
) -> float:
    """Calculate pass^k (all k pass probability).

    pass^k measures the probability that all k attempts pass.
    This is equivalent to the product of individual pass rates.

    Formula:
        pass^k = (c/n)^k when using sampling with replacement
        pass^k = C(c, k) / C(n, k) when using sampling without replacement

    We use sampling without replacement for exact calculation.

    Args:
        correct: Number of correct/passing attempts (c).
        total: Total number of attempts (n).
        k: Number of attempts that must all pass.

    Returns:
        pass^k probability (0.0-1.0).
    """
    if total == 0:
        return 0.0
    if k > total:
        k = total
    if k == 0:
        return 1.0
    if correct < k:
        return 0.0
    if correct >= total:
        return 1.0

    # C(c, k) / C(n, k)
    # = (c/n) * ((c-1)/(n-1)) * ... * ((c-k+1)/(n-k+1))
    result = 1.0
    for i in range(k):
        numerator = correct - i
        denominator = total - i
        if denominator == 0:
            return 0.0
        result *= numerator / denominator

    return max(0.0, min(1.0, result))


def calculate_pass_caret_k_from_trials(
    trials: list[EvalTrial],
    k: int,
) -> float:
    """Calculate pass^k from a list of trials.

    Groups trials by task and calculates pass^k across all tasks.

    Args:
        trials: List of evaluation trials.
        k: Number of attempts that must all pass.

    Returns:
        pass^k probability (0.0-1.0).
    """
    if not trials:
        return 0.0

    # Group trials by task
    task_trials: dict[str, list[EvalTrial]] = defaultdict(list)
    for trial in trials:
        task_trials[trial.task_id].append(trial)

    # Calculate pass^k for each task
    task_pass_caret_k: list[float] = []
    for task_id, task_trial_list in task_trials.items():
        correct = sum(
            1 for t in task_trial_list if t.result is not None and t.result.aggregate_passed
        )
        total = len(task_trial_list)
        if total > 0:
            task_pass_caret_k.append(calculate_pass_caret_k(correct, total, k))

    return mean(task_pass_caret_k) if task_pass_caret_k else 0.0


# =============================================================================
# CONFIDENCE INTERVALS
# =============================================================================


def wilson_confidence_interval(
    successes: int,
    total: int,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """Calculate Wilson score confidence interval for a proportion.

    The Wilson score interval is better than the normal approximation
    for proportions near 0 or 1, and for small sample sizes.

    Formula:
        (p + z^2/2n ± z*sqrt(p*(1-p)/n + z^2/4n^2)) / (1 + z^2/n)

    Args:
        successes: Number of successes.
        total: Total number of trials.
        confidence_level: Confidence level (default 0.95 for 95% CI).

    Returns:
        ConfidenceInterval with lower and upper bounds.
    """
    if total == 0:
        return ConfidenceInterval(
            lower=0.0,
            upper=1.0,
            confidence_level=confidence_level,
            method="wilson",
        )

    p = successes / total

    # Z-score for the confidence level
    alpha = 1 - confidence_level
    z = _z_score(1 - alpha / 2)

    z2 = z * z
    n = total

    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator

    return ConfidenceInterval(
        lower=max(0.0, center - margin),
        upper=min(1.0, center + margin),
        confidence_level=confidence_level,
        method="wilson",
    )


def normal_confidence_interval(
    successes: int,
    total: int,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """Calculate normal approximation confidence interval for a proportion.

    Uses the standard normal approximation:
        p ± z * sqrt(p*(1-p)/n)

    Note: Wilson score interval is generally preferred.

    Args:
        successes: Number of successes.
        total: Total number of trials.
        confidence_level: Confidence level (default 0.95 for 95% CI).

    Returns:
        ConfidenceInterval with lower and upper bounds.
    """
    if total == 0:
        return ConfidenceInterval(
            lower=0.0,
            upper=1.0,
            confidence_level=confidence_level,
            method="normal",
        )

    p = successes / total
    alpha = 1 - confidence_level
    z = _z_score(1 - alpha / 2)

    margin = z * math.sqrt(p * (1 - p) / total)

    return ConfidenceInterval(
        lower=max(0.0, p - margin),
        upper=min(1.0, p + margin),
        confidence_level=confidence_level,
        method="normal",
    )


def clopper_pearson_confidence_interval(
    successes: int,
    total: int,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """Calculate Clopper-Pearson (exact) confidence interval.

    Uses the binomial distribution for an exact (conservative) interval.

    Args:
        successes: Number of successes.
        total: Total number of trials.
        confidence_level: Confidence level (default 0.95 for 95% CI).

    Returns:
        ConfidenceInterval with lower and upper bounds.
    """
    if total == 0:
        return ConfidenceInterval(
            lower=0.0,
            upper=1.0,
            confidence_level=confidence_level,
            method="clopper_pearson",
        )

    if successes == 0:
        return ConfidenceInterval(
            lower=0.0,
            upper=_beta_ppf(1 - confidence_level / 2, 1, total),
            confidence_level=confidence_level,
            method="clopper_pearson",
        )

    if successes == total:
        return ConfidenceInterval(
            lower=_beta_ppf(confidence_level / 2, total, 1),
            upper=1.0,
            confidence_level=confidence_level,
            method="clopper_pearson",
        )

    alpha = 1 - confidence_level
    lower = _beta_ppf(alpha / 2, successes, total - successes + 1)
    upper = _beta_ppf(1 - alpha / 2, successes + 1, total - successes)

    return ConfidenceInterval(
        lower=max(0.0, lower),
        upper=min(1.0, upper),
        confidence_level=confidence_level,
        method="clopper_pearson",
    )


def _z_score(p: float) -> float:
    """Approximate z-score for a given cumulative probability.

    Uses lookup table for common values and Wichura's algorithm for others.

    Args:
        p: Cumulative probability (0 < p < 1).

    Returns:
        Approximate z-score.
    """
    if p <= 0 or p >= 1:
        return 0.0

    Z_SCORE_TABLE = {
        0.80: 0.8416212335729143,
        0.85: 1.0364333894937896,
        0.90: 1.2815515655446004,
        0.91: 1.3407550336900525,
        0.92: 1.4050715603096322,
        0.93: 1.4757910281791508,
        0.94: 1.5547735945968535,
        0.95: 1.6448536269514722,
        0.96: 1.7506860712521694,
        0.97: 1.8807936081512509,
        0.975: 1.959963984540054,
        0.98: 2.0537489106318225,
        0.985: 2.170090377601209,
        0.99: 2.3263478740408408,
        0.995: 2.5758293035489004,
        0.9975: 2.807033768432971,
        0.999: 3.090232306167813,
        0.9995: 3.2905267314919255,
    }

    if p in Z_SCORE_TABLE:
        return Z_SCORE_TABLE[p]

    if p < 0.5:
        return -_z_score(1 - p)

    if p in Z_SCORE_TABLE:
        return Z_SCORE_TABLE[p]

    closest = min(Z_SCORE_TABLE.keys(), key=lambda k: abs(k - p))
    z_closest = Z_SCORE_TABLE[closest]
    delta_p = p - closest

    pdf_at_closest = (1.0 / math.sqrt(2 * math.pi)) * math.exp(-(z_closest**2) / 2)
    z_approx = z_closest + delta_p / pdf_at_closest

    return z_approx


def _beta_ppf(p: float, a: float, b: float) -> float:
    """Approximate beta distribution percent point function (inverse CDF).

    Uses Newton-Raphson iteration with a normal approximation starting point.

    Args:
        p: Cumulative probability (0 < p < 1).
        a: First shape parameter.
        b: Second shape parameter.

    Returns:
        Approximate quantile.
    """
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0

    # Starting guess using normal approximation
    mean = a / (a + b)
    var = (a * b) / ((a + b) ** 2 * (a + b + 1))
    std_dev = math.sqrt(var) if var > 0 else 0.1

    x = mean + std_dev * _z_score(p)
    x = max(1e-10, min(1 - 1e-10, x))

    # Newton-Raphson iteration
    for _ in range(20):
        # Beta CDF approximation using incomplete beta function
        cdf = _regularized_incomplete_beta(x, a, b)
        if abs(cdf - p) < 1e-10:
            break

        # PDF (derivative of CDF)
        pdf = (x ** (a - 1)) * ((1 - x) ** (b - 1)) / _beta_function(a, b)
        if pdf < 1e-15:
            break

        # Newton-Raphson update
        x_new = x - (cdf - p) / pdf
        if x_new == x:
            break
        x = max(1e-10, min(1 - 1e-10, x_new))

    return x


def _beta_function(a: float, b: float) -> float:
    """Calculate the beta function B(a, b) using gamma function.

    Args:
        a: First parameter.
        b: Second parameter.

    Returns:
        Beta function value.
    """
    return math.exp(_log_gamma(a) + _log_gamma(b) - _log_gamma(a + b))


def _log_gamma(x: float) -> float:
    """Calculate log(gamma(x)) using Stirling's approximation.

    Args:
        x: Input value.

    Returns:
        Natural logarithm of gamma(x).
    """
    if x <= 0:
        return float("inf")

    # Lanczos approximation coefficients
    g = 7
    c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]

    if x < 0.5:
        return math.log(math.pi / math.sin(math.pi * x)) - _log_gamma(1 - x)

    x -= 1
    a = c[0]
    t = x + g + 0.5
    for i in range(1, len(c)):
        a += c[i] / (x + i)

    return 0.5 * math.log(2 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Calculate regularized incomplete beta function I_x(a, b).

    Uses continued fraction expansion.

    Args:
        x: Upper limit of integration (0 <= x <= 1).
        a: First shape parameter.
        b: Second shape parameter.

    Returns:
        Regularized incomplete beta function value.
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use symmetry if x > (a+1)/(a+b+2)
    if x > (a + 1) / (a + b + 2):
        return 1 - _regularized_incomplete_beta(1 - x, b, a)

    # Continued fraction expansion (Lentz's algorithm)
    front: float = (x**a) * ((1 - x) ** b) / _beta_function(a, b) / a

    f: float = 1.0
    c: float = 1.0
    d: float = 1e30  # Initialize to large value to avoid division by zero issues

    for m in range(1, 200):
        # Even step
        m2 = 2 * m
        aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2))

        d = 1 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1 / d

        c = 1 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30

        f *= c * d

        # Odd step
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))

        d = 1 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1 / d

        c = 1 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30

        delta = c * d
        f *= delta

        if abs(delta - 1) < 1e-10:
            break

    return front * (f - 1)


# =============================================================================
# SIGNIFICANCE TESTING
# =============================================================================


def two_proportion_z_test(
    successes1: int,
    total1: int,
    successes2: int,
    total2: int,
    alpha: float = 0.05,
) -> SignificanceResult:
    """Perform two-proportion z-test.

    Tests whether two proportions are significantly different.

    Args:
        successes1: Number of successes in group 1.
        total1: Total trials in group 1.
        successes2: Number of successes in group 2.
        total2: Total trials in group 2.
        alpha: Significance level (default 0.05).

    Returns:
        SignificanceResult with test statistic and p-value.
    """
    if total1 == 0 or total2 == 0:
        return SignificanceResult(
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            test_type="two_proportion_z_test",
        )

    p1 = successes1 / total1
    p2 = successes2 / total2

    # Pooled proportion
    p_pooled = (successes1 + successes2) / (total1 + total2)

    if p_pooled == 0 or p_pooled == 1:
        return SignificanceResult(
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            test_type="two_proportion_z_test",
        )

    # Standard error
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1 / total1 + 1 / total2))

    if se == 0:
        return SignificanceResult(
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            test_type="two_proportion_z_test",
        )

    # Z-statistic
    z = (p1 - p2) / se

    # Two-tailed p-value using normal CDF approximation
    p_value = 2 * (1 - _normal_cdf(abs(z)))

    # Cohen's h effect size
    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))
    effect_size = abs(phi1 - phi2)

    return SignificanceResult(
        statistic=z,
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
        test_type="two_proportion_z_test",
        effect_size=effect_size,
    )


def chi_square_test(
    observed: list[list[int]],
    alpha: float = 0.05,
) -> SignificanceResult:
    """Perform chi-square test of independence.

    Tests whether there is a significant association between
    categorical variables.

    Args:
        observed: Contingency table as 2D list of observed frequencies.
        alpha: Significance level (default 0.05).

    Returns:
        SignificanceResult with test statistic and p-value.
    """
    if not observed or not observed[0]:
        return SignificanceResult(
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            test_type="chi_square_test",
        )

    rows = len(observed)
    cols = len(observed[0])

    # Calculate row and column totals
    row_totals = [sum(row) for row in observed]
    col_totals = [sum(observed[i][j] for i in range(rows)) for j in range(cols)]
    grand_total = sum(row_totals)

    if grand_total == 0:
        return SignificanceResult(
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            test_type="chi_square_test",
        )

    # Calculate expected frequencies and chi-square statistic
    chi_sq = 0.0
    for i in range(rows):
        for j in range(cols):
            expected = (row_totals[i] * col_totals[j]) / grand_total
            if expected > 0:
                chi_sq += (observed[i][j] - expected) ** 2 / expected

    # Degrees of freedom
    df = (rows - 1) * (cols - 1)

    if df <= 0:
        return SignificanceResult(
            statistic=chi_sq,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            test_type="chi_square_test",
        )

    # P-value from chi-square distribution
    p_value = 1 - _chi_square_cdf(chi_sq, df)

    # Cramer's V effect size
    min_dim = min(rows - 1, cols - 1)
    cramers_v = math.sqrt(chi_sq / (grand_total * min_dim)) if min_dim > 0 else 0.0

    return SignificanceResult(
        statistic=chi_sq,
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
        test_type="chi_square_test",
        effect_size=cramers_v,
    )


def compare_graders(
    grader_a_results: list[GraderResult],
    grader_b_results: list[GraderResult],
    alpha: float = 0.05,
    use_chi_square_threshold: int = 100,
) -> SignificanceResult:
    """Compare pass rates between two graders for statistical significance.

    Builds a 2x2 contingency table and uses Fisher's exact test for small
    samples or chi-squared test for large samples.

    Args:
        grader_a_results: GraderResult list from grader A.
        grader_b_results: GraderResult list from grader B.
        alpha: Significance level (default 0.05).
        use_chi_square_threshold: Use chi-squared if total samples >= this
            value (default 100, as Fisher's exact test becomes slow for large n).

    Returns:
        SignificanceResult with test statistic and p-value.
    """
    a_passes = sum(1 for r in grader_a_results if r.passed)
    a_fails = len(grader_a_results) - a_passes
    b_passes = sum(1 for r in grader_b_results if r.passed)
    b_fails = len(grader_b_results) - b_passes

    n = len(grader_a_results) + len(grader_b_results)

    if n == 0:
        return SignificanceResult(
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            test_type="grader_comparison",
        )

    table = [[a_passes, a_fails], [b_passes, b_fails]]

    if n < use_chi_square_threshold:
        return fisher_exact_test(table, alpha)
    else:
        return chi_square_test(table, alpha)


def compare_runs_significance(
    run1_trials: list[EvalTrial],
    run2_trials: list[EvalTrial],
    alpha: float = 0.05,
) -> SignificanceResult:
    """Compare two evaluation runs for statistical significance.

    Performs a two-proportion z-test on the pass rates of two runs.

    Args:
        run1_trials: Trials from the first run.
        run2_trials: Trials from the second run.
        alpha: Significance level (default 0.05).

    Returns:
        SignificanceResult with test statistic and p-value.
    """

    def count_passes(trials: list[EvalTrial]) -> tuple[int, int]:
        successes = 0
        total = 0
        for trial in trials:
            if trial.status == EvalStatus.COMPLETED and trial.result:
                total += 1
                if trial.result.aggregate_passed:
                    successes += 1
        return successes, total

    s1, n1 = count_passes(run1_trials)
    s2, n2 = count_passes(run2_trials)

    return two_proportion_z_test(s1, n1, s2, n2, alpha)


def fisher_exact_test(
    table: list[list[int]],
    alpha: float = 0.05,
) -> SignificanceResult:
    """Perform Fisher's exact test for a 2x2 contingency table.

    Fisher's exact test computes the exact probability of observing the
    given table (or more extreme) assuming the null hypothesis of independence.

    For small sample sizes, this is preferred over chi-square test.

    Args:
        table: 2x2 contingency table as [[a, b], [c, d]] where:
            - a: successes in group 1
            - b: failures in group 1
            - c: successes in group 2
            - d: failures in group 2
        alpha: Significance level (default 0.05).

    Returns:
        SignificanceResult with test statistic and p-value.
    """
    if not table or len(table) != 2 or len(table[0]) != 2 or len(table[1]) != 2:
        return SignificanceResult(
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            test_type="fisher_exact",
        )

    a, b = table[0][0], table[0][1]
    c, d = table[1][0], table[1][1]

    # Handle edge case of all zeros
    n = a + b + c + d
    if n == 0:
        return SignificanceResult(
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            test_type="fisher_exact",
        )

    # Calculate probability of observed table
    def hypergeometric_p(a_val: int, b_val: int, c_val: int, d_val: int) -> float:
        """Calculate hypergeometric probability for a 2x2 table."""
        # Use log factorial for numerical stability
        row1_total = a_val + b_val
        row2_total = c_val + d_val
        col1_total = a_val + c_val
        col2_total = b_val + d_val
        total = row1_total + row2_total

        if total == 0:
            return 1.0

        # log P = log(C(row1_total, a_val)) + log(C(row2_total, c_val)) - log(C(total, col1_total))
        # C(n,k) = n! / (k! * (n-k)!)
        log_p = (
            _log_factorial(row1_total)
            - _log_factorial(a_val)
            - _log_factorial(row1_total - a_val)
            + _log_factorial(row2_total)
            - _log_factorial(c_val)
            - _log_factorial(row2_total - c_val)
            - _log_factorial(total)
            + _log_factorial(col1_total)
            + _log_factorial(col2_total)
        )
        return math.exp(log_p)

    # Calculate probability of observed table
    p_observed = hypergeometric_p(a, b, c, d)

    # Find all possible tables with same marginals
    row1_total = a + b
    row2_total = c + d
    col1_total = a + c
    col2_total = b + d

    # Two-tailed test: sum probabilities of tables at least as extreme
    # "More extreme" means probability <= p_observed
    p_value = 0.0

    # Iterate through all possible values of a (the top-left cell)
    min_a = max(0, col1_total - row2_total)
    max_a = min(row1_total, col1_total)

    for a_val in range(min_a, max_a + 1):
        b_val = row1_total - a_val
        c_val = col1_total - a_val
        d_val = row2_total - c_val

        if b_val < 0 or c_val < 0 or d_val < 0:
            continue

        p_table = hypergeometric_p(a_val, b_val, c_val, d_val)

        # For two-tailed test, include tables with probability <= p_observed
        if p_table <= p_observed + 1e-10:  # Small epsilon for floating point comparison
            p_value += p_table

    # Effect size: odds ratio
    # OR = (a*d) / (b*c), with Haldane-Anscombe correction for zeros
    a_adj = a + 0.5
    b_adj = b + 0.5
    c_adj = c + 0.5
    d_adj = d + 0.5
    odds_ratio = (a_adj * d_adj) / (b_adj * c_adj)

    return SignificanceResult(
        statistic=odds_ratio,
        p_value=min(p_value, 1.0),  # Clamp to 1.0
        significant=p_value < alpha,
        alpha=alpha,
        test_type="fisher_exact",
        effect_size=odds_ratio,
    )


def _log_factorial(n: int) -> float:
    """Calculate log(n!) using log gamma.

    Args:
        n: Non-negative integer.

    Returns:
        Natural logarithm of n!.
    """
    if n < 0:
        return float("inf")
    if n <= 1:
        return 0.0
    return _log_gamma(n + 1)


def _normal_cdf(x: float) -> float:
    """Calculate the standard normal CDF using error function approximation.

    Args:
        x: Input value.

    Returns:
        P(X <= x) for standard normal distribution.
    """
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _chi_square_cdf(x: float, df: int) -> float:
    """Calculate chi-square CDF using incomplete gamma function.

    Args:
        x: Chi-square statistic.
        df: Degrees of freedom.

    Returns:
        P(X <= x) for chi-square distribution.
    """
    if x <= 0:
        return 0.0
    return _regularized_incomplete_gamma(df / 2, x / 2)


def _regularized_incomplete_gamma(a: float, x: float) -> float:
    """Calculate regularized incomplete gamma function P(a, x).

    Uses series expansion for small x, continued fraction for large x.

    Args:
        a: Shape parameter.
        x: Upper limit of integration.

    Returns:
        Regularized incomplete gamma value.
    """
    if x < 0 or a <= 0:
        return 0.0
    if x == 0:
        return 0.0

    if x < a + 1:
        # Series expansion
        ap = a
        sum_val = 1 / a
        delta = sum_val
        for _ in range(200):
            ap += 1
            delta *= x / ap
            sum_val += delta
            if abs(delta) < abs(sum_val) * 1e-10:
                break
        return sum_val * math.exp(-x + a * math.log(x) - _log_gamma(a))
    else:
        # Continued fraction
        b = x + 1 - a
        c = 1e30
        d = 1 / b
        h = d
        for i in range(1, 200):
            an = -i * (i - a)
            b += 2
            d = an * d + b
            if abs(d) < 1e-30:
                d = 1e-30
            c = b + an / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1 / d
            delta = d * c
            h *= delta
            if abs(delta - 1) < 1e-10:
                break
        return 1 - math.exp(-x + a * math.log(x) - _log_gamma(a)) * h


# =============================================================================
# LATENCY METRICS
# =============================================================================


def calculate_latency_metrics(
    trials: list[EvalTrial],
) -> LatencyMetrics:
    """Calculate latency metrics from trials.

    Args:
        trials: List of evaluation trials.

    Returns:
        LatencyMetrics with min, max, mean, percentiles, and std.
    """
    latencies: list[float] = []

    for trial in trials:
        if trial.result is not None:
            latencies.append(trial.result.transcript.duration_seconds)

    if not latencies:
        return LatencyMetrics()

    return LatencyMetrics(
        min_seconds=min(latencies),
        max_seconds=max(latencies),
        mean_seconds=mean(latencies),
        median_seconds=percentile(latencies, 50),
        p50_seconds=percentile(latencies, 50),
        p90_seconds=percentile(latencies, 90),
        p95_seconds=percentile(latencies, 95),
        p99_seconds=percentile(latencies, 99),
        std_seconds=std(latencies, ddof=1) if len(latencies) > 1 else 0.0,
    )


# =============================================================================
# TOKEN AND COST METRICS
# =============================================================================


def calculate_token_metrics(
    trials: list[EvalTrial],
) -> TokenMetrics:
    """Calculate token usage metrics from trials.

    Args:
        trials: List of evaluation trials.

    Returns:
        TokenMetrics with total and per-trial token usage.
    """
    total_input = 0
    total_output = 0
    total_reasoning = 0
    total_cache_read = 0
    total_cache_write = 0
    trial_count = 0

    for trial in trials:
        if trial.result is not None:
            tokens = trial.result.transcript.token_usage
            total_input += tokens.input
            total_output += tokens.output
            total_reasoning += tokens.reasoning
            total_cache_read += tokens.cache_read
            total_cache_write += tokens.cache_write
            trial_count += 1

    return TokenMetrics(
        total_input=total_input,
        total_output=total_output,
        total_reasoning=total_reasoning,
        total_cache_read=total_cache_read,
        total_cache_write=total_cache_write,
        mean_input_per_trial=total_input / trial_count if trial_count > 0 else 0.0,
        mean_output_per_trial=total_output / trial_count if trial_count > 0 else 0.0,
    )


def calculate_cost_metrics(
    trials: list[EvalTrial],
) -> CostMetrics:
    """Calculate cost metrics from trials.

    Args:
        trials: List of evaluation trials.

    Returns:
        CostMetrics with total and per-trial costs.
    """
    costs: list[float] = []

    for trial in trials:
        if trial.result is not None:
            costs.append(trial.result.transcript.cost_usd)

    if not costs:
        return CostMetrics()

    return CostMetrics(
        total_usd=sum(costs),
        mean_usd_per_trial=mean(costs),
        min_usd_per_trial=min(costs),
        max_usd_per_trial=max(costs),
    )


# =============================================================================
# COMPREHENSIVE METRICS CALCULATION
# =============================================================================


def calculate_task_metrics(
    trials: list[EvalTrial],
    k_values: list[int] | None = None,
    confidence_level: float = 0.95,
) -> dict[str, TaskMetrics]:
    """Calculate per-task metrics.

    Args:
        trials: List of evaluation trials.
        k_values: List of k values for pass@k/pass^k (default: [1, 2, 3, 5]).
        confidence_level: Confidence level for intervals.

    Returns:
        Dict mapping task_id to TaskMetrics.
    """
    if k_values is None:
        k_values = [1, 2, 3, 5]

    # Group trials by task
    task_trials: dict[str, list[EvalTrial]] = defaultdict(list)
    for trial in trials:
        task_trials[trial.task_id].append(trial)

    result: dict[str, TaskMetrics] = {}

    for task_id, task_trial_list in task_trials.items():
        total_attempts = len(task_trial_list)
        successful = sum(
            1 for t in task_trial_list if t.result is not None and t.result.aggregate_passed
        )
        pass_rate = successful / total_attempts if total_attempts > 0 else 0.0

        # Calculate pass@k and pass^k
        pass_at_k: dict[int, float] = {}
        pass_caret_k: dict[int, float] = {}
        for k in k_values:
            pass_at_k[k] = calculate_pass_at_k(successful, total_attempts, k)
            pass_caret_k[k] = calculate_pass_caret_k(successful, total_attempts, k)

        # Calculate confidence interval
        ci = wilson_confidence_interval(successful, total_attempts, confidence_level)

        # Calculate mean score
        scores = [t.result.aggregate_score for t in task_trial_list if t.result is not None]
        mean_score_val = mean(scores) if scores else 0.0

        # Calculate latency, token, and cost metrics
        latency = calculate_latency_metrics(task_trial_list)
        tokens = calculate_token_metrics(task_trial_list)
        cost = calculate_cost_metrics(task_trial_list)

        result[task_id] = TaskMetrics(
            task_id=task_id,
            total_attempts=total_attempts,
            successful_attempts=successful,
            pass_rate=pass_rate,
            pass_at_k=pass_at_k,
            pass_caret_k=pass_caret_k,
            confidence_interval=ci,
            mean_score=mean_score_val,
            latency=latency,
            tokens=tokens,
            cost=cost,
        )

    return result


def calculate_suite_metrics_detailed(
    trials: list[EvalTrial],
    suite_id: str,
    run_id: str,
    k_values: list[int] | None = None,
    confidence_level: float = 0.95,
) -> SuiteMetricsDetailed:
    """Calculate comprehensive suite-level metrics.

    Args:
        trials: List of evaluation trials.
        suite_id: Suite identifier.
        run_id: Run identifier.
        k_values: List of k values for pass@k/pass^k (default: [1, 2, 3, 5]).
        confidence_level: Confidence level for intervals.

    Returns:
        SuiteMetricsDetailed with comprehensive statistical analysis.
    """
    if k_values is None:
        k_values = [1, 2, 3, 5]

    total_trials = len(trials)
    unique_tasks = set(t.task_id for t in trials)

    completed_trials = 0
    passed_trials = 0
    failed_trials = 0
    scores: list[float] = []

    for trial in trials:
        if trial.status == EvalStatus.COMPLETED and trial.result:
            completed_trials += 1
            if trial.result.aggregate_passed:
                passed_trials += 1
            else:
                failed_trials += 1
            scores.append(trial.result.aggregate_score)

    pass_rate = passed_trials / completed_trials if completed_trials > 0 else 0.0
    mean_score_val = mean(scores)
    score_std_val = std(scores, ddof=1) if len(scores) > 1 else 0.0

    # Calculate pass@k and pass^k
    pass_at_k: dict[int, float] = {}
    pass_caret_k: dict[int, float] = {}
    for k in k_values:
        pass_at_k[k] = calculate_pass_at_k_from_trials(trials, k)
        pass_caret_k[k] = calculate_pass_caret_k_from_trials(trials, k)

    # Confidence interval for overall pass rate
    pass_rate_ci = wilson_confidence_interval(passed_trials, completed_trials, confidence_level)

    # Per-task metrics
    task_metrics = calculate_task_metrics(trials, k_values, confidence_level)

    # Grader-level metrics
    grader_metrics = _calculate_grader_metrics(trials)

    # Latency, token, and cost metrics
    latency = calculate_latency_metrics(trials)
    tokens = calculate_token_metrics(trials)
    cost = calculate_cost_metrics(trials)

    return SuiteMetricsDetailed(
        suite_id=suite_id,
        run_id=run_id,
        total_tasks=len(unique_tasks),
        total_trials=total_trials,
        completed_trials=completed_trials,
        passed_trials=passed_trials,
        failed_trials=failed_trials,
        pass_rate=pass_rate,
        pass_rate_ci=pass_rate_ci,
        pass_at_k=pass_at_k,
        pass_caret_k=pass_caret_k,
        mean_score=mean_score_val,
        score_std=score_std_val,
        latency=latency,
        tokens=tokens,
        cost=cost,
        task_metrics=task_metrics,
        grader_metrics=grader_metrics,
    )


def _calculate_grader_metrics(
    trials: list[EvalTrial],
) -> dict[str, dict[str, Any]]:
    """Calculate per-grader metrics.

    Args:
        trials: List of evaluation trials.

    Returns:
        Dict mapping grader_type to metrics dict.
    """
    grader_results: dict[str, list[GraderResult]] = defaultdict(list)

    for trial in trials:
        if trial.result:
            for grader_result in trial.result.grader_results:
                grader_results[grader_result.grader_type].append(grader_result)

    metrics: dict[str, dict[str, Any]] = {}

    for grader_type, results in grader_results.items():
        if not results:
            continue

        passed = sum(1 for r in results if r.passed)
        scores = [r.score for r in results]

        metrics[grader_type] = {
            "count": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": passed / len(results),
            "mean_score": mean(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "score_std": std(scores, ddof=1) if len(scores) > 1 else 0.0,
        }

    return metrics


# =============================================================================
# CONVERSION FUNCTIONS
# =============================================================================


def to_suite_metrics(
    detailed: SuiteMetricsDetailed,
) -> SuiteMetrics:
    """Convert SuiteMetricsDetailed to SuiteMetrics for backward compatibility.

    Args:
        detailed: Detailed suite metrics.

    Returns:
        Simplified SuiteMetrics.
    """
    return SuiteMetrics(
        suite_id=detailed.suite_id,
        run_id=detailed.run_id,
        total_tasks=detailed.total_tasks,
        completed_tasks=detailed.completed_trials,
        passed_tasks=detailed.passed_trials,
        failed_tasks=detailed.failed_trials,
        pass_rate=detailed.pass_rate,
        mean_score=detailed.mean_score,
        total_tokens=TokenUsage(
            input=detailed.tokens.total_input,
            output=detailed.tokens.total_output,
            reasoning=detailed.tokens.total_reasoning,
            cache_read=detailed.tokens.total_cache_read,
            cache_write=detailed.tokens.total_cache_write,
        ),
        total_cost_usd=detailed.cost.total_usd,
        total_duration_seconds=detailed.latency.mean_seconds * detailed.completed_trials,
        latency_p50_seconds=detailed.latency.p50_seconds,
        latency_p95_seconds=detailed.latency.p95_seconds,
        latency_p99_seconds=detailed.latency.p99_seconds,
        pass_at_k=detailed.pass_at_k,
        created_at=detailed.created_at,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    "ConfidenceInterval",
    "SignificanceResult",
    "LatencyMetrics",
    "TokenMetrics",
    "CostMetrics",
    "TaskMetrics",
    "SuiteMetricsDetailed",
    # Pass@k and pass^k
    "calculate_pass_at_k",
    "calculate_pass_at_k_from_trials",
    "calculate_pass_caret_k",
    "calculate_pass_caret_k_from_trials",
    # Confidence intervals
    "wilson_confidence_interval",
    "normal_confidence_interval",
    "clopper_pearson_confidence_interval",
    # Significance testing
    "two_proportion_z_test",
    "chi_square_test",
    "fisher_exact_test",
    "compare_graders",
    "compare_runs_significance",
    # Latency, token, cost
    "calculate_latency_metrics",
    "calculate_token_metrics",
    "calculate_cost_metrics",
    # Comprehensive metrics
    "calculate_task_metrics",
    "calculate_suite_metrics_detailed",
    # Conversion
    "to_suite_metrics",
    # Helper functions
    "percentile",
    "mean",
    "std",
]
