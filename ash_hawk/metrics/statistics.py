"""Statistical metrics for ash-hawk evaluation harness."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import pydantic as pd

from ash_hawk.types import EvalStatus, EvalTrial, GraderResult, TokenUsage


class ConfidenceInterval(pd.BaseModel):
    lower: float = pd.Field(description="Lower bound")
    upper: float = pd.Field(description="Upper bound")
    confidence_level: float = pd.Field(default=0.95, description="Confidence level")
    method: str = pd.Field(default="wilson", description="CI method")
    model_config = pd.ConfigDict(extra="forbid")


class SignificanceResult(pd.BaseModel):
    statistic: float = pd.Field(description="Test statistic value")
    p_value: float = pd.Field(description="P-value")
    significant: bool = pd.Field(description="Whether statistically significant")
    alpha: float = pd.Field(default=0.05, description="Significance level")
    test_type: str = pd.Field(default="z_test", description="Test type")
    model_config = pd.ConfigDict(extra="forbid")


class LatencyMetrics(pd.BaseModel):
    min_seconds: float = pd.Field(default=0.0, ge=0.0)
    max_seconds: float = pd.Field(default=0.0, ge=0.0)
    mean_seconds: float = pd.Field(default=0.0, ge=0.0)
    median_seconds: float | None = pd.Field(default=None)
    p50_seconds: float | None = pd.Field(default=None)
    p90_seconds: float | None = pd.Field(default=None)
    p95_seconds: float | None = pd.Field(default=None)
    p99_seconds: float | None = pd.Field(default=None)
    std_seconds: float = pd.Field(default=0.0, ge=0.0)
    model_config = pd.ConfigDict(extra="forbid")


class TokenMetrics(pd.BaseModel):
    total_input: int = pd.Field(default=0, ge=0)
    total_output: int = pd.Field(default=0, ge=0)
    total_reasoning: int = pd.Field(default=0, ge=0)
    total_cache_read: int = pd.Field(default=0, ge=0)
    total_cache_write: int = pd.Field(default=0, ge=0)
    mean_input_per_trial: float = pd.Field(default=0.0, ge=0.0)
    mean_output_per_trial: float = pd.Field(default=0.0, ge=0.0)
    model_config = pd.ConfigDict(extra="forbid")


class CostMetrics(pd.BaseModel):
    total_usd: float = pd.Field(default=0.0, ge=0.0)
    mean_usd_per_trial: float = pd.Field(default=0.0, ge=0.0)
    min_usd_per_trial: float = pd.Field(default=0.0, ge=0.0)
    max_usd_per_trial: float = pd.Field(default=0.0, ge=0.0)
    model_config = pd.ConfigDict(extra="forbid")


class TaskMetrics(pd.BaseModel):
    task_id: str = pd.Field(description="Task identifier")
    total_attempts: int = pd.Field(default=0, ge=0)
    successful_attempts: int = pd.Field(default=0, ge=0)
    pass_rate: float = pd.Field(default=0.0, ge=0.0, le=1.0)
    pass_at_k: dict[int, float] = pd.Field(default_factory=dict)
    confidence_interval: ConfidenceInterval | None = pd.Field(default=None)
    mean_score: float = pd.Field(default=0.0, ge=0.0, le=1.0)
    latency: LatencyMetrics = pd.Field(default_factory=LatencyMetrics)
    tokens: TokenMetrics = pd.Field(default_factory=TokenMetrics)
    cost: CostMetrics = pd.Field(default_factory=CostMetrics)
    model_config = pd.ConfigDict(extra="forbid")


def percentile(values: list[float], p: float) -> float | None:
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
    if not values:
        return 0.0
    return sum(values) / len(values)


def std(values: list[float], ddof: int = 0) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - ddof)
    return math.sqrt(variance)


def calculate_pass_at_k(correct: int, total: int, k: int) -> float:
    if total == 0:
        return 0.0
    if k > total:
        k = total
    if correct >= total:
        return 1.0
    if correct == 0:
        return 0.0
    if k == 1:
        return correct / total
    n = total
    if n - correct < k:
        return 1.0
    log_ratio = 0.0
    for i in range(k):
        numerator = n - correct - k + 1 + i
        denominator = n - k + 1 + i
        if numerator > 0 and denominator > 0:
            log_ratio += math.log(numerator) - math.log(denominator)
    ratio = math.exp(log_ratio)
    return max(0.0, min(1.0, 1.0 - ratio))


def calculate_pass_at_k_from_trials(trials: list[EvalTrial], k: int) -> float:
    if not trials:
        return 0.0
    task_trials: dict[str, list[EvalTrial]] = defaultdict(list)
    for trial in trials:
        task_trials[trial.task_id].append(trial)
    task_pass_at_k: list[float] = []
    for task_id, task_trial_list in task_trials.items():
        correct = sum(
            1 for t in task_trial_list if t.result is not None and t.result.aggregate_passed
        )
        total = len(task_trial_list)
        if total > 0:
            task_pass_at_k.append(calculate_pass_at_k(correct, total, k))
    return mean(task_pass_at_k) if task_pass_at_k else 0.0


def _z_score(p: float) -> float:
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
    closest = min(Z_SCORE_TABLE.keys(), key=lambda k: abs(k - p))
    z_closest = Z_SCORE_TABLE[closest]
    delta_p = p - closest
    pdf_at_closest = (1.0 / math.sqrt(2 * math.pi)) * math.exp(-(z_closest**2) / 2)
    return z_closest + delta_p / pdf_at_closest


def wilson_confidence_interval(
    successes: int,
    total: int,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    if total == 0:
        return ConfidenceInterval(
            lower=0.0, upper=1.0, confidence_level=confidence_level, method="wilson"
        )
    p = successes / total
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


def calculate_latency_metrics(trials: list[EvalTrial]) -> LatencyMetrics:
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


def calculate_token_metrics(trials: list[EvalTrial]) -> TokenMetrics:
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


def calculate_cost_metrics(trials: list[EvalTrial]) -> CostMetrics:
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


def calculate_task_metrics(
    trials: list[EvalTrial],
    k_values: list[int] | None = None,
    confidence_level: float = 0.95,
) -> dict[str, TaskMetrics]:
    if k_values is None:
        k_values = [1, 2, 3, 5]
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
        pass_at_k: dict[int, float] = {}
        for k in k_values:
            pass_at_k[k] = calculate_pass_at_k(successful, total_attempts, k)
        ci = wilson_confidence_interval(successful, total_attempts, confidence_level)
        scores = [t.result.aggregate_score for t in task_trial_list if t.result is not None]
        mean_score_val = mean(scores) if scores else 0.0
        latency = calculate_latency_metrics(task_trial_list)
        tokens = calculate_token_metrics(task_trial_list)
        cost = calculate_cost_metrics(task_trial_list)
        result[task_id] = TaskMetrics(
            task_id=task_id,
            total_attempts=total_attempts,
            successful_attempts=successful,
            pass_rate=pass_rate,
            pass_at_k=pass_at_k,
            confidence_interval=ci,
            mean_score=mean_score_val,
            latency=latency,
            tokens=tokens,
            cost=cost,
        )
    return result


__all__ = [
    "ConfidenceInterval",
    "SignificanceResult",
    "LatencyMetrics",
    "TokenMetrics",
    "CostMetrics",
    "TaskMetrics",
    "percentile",
    "mean",
    "std",
    "calculate_pass_at_k",
    "calculate_pass_at_k_from_trials",
    "wilson_confidence_interval",
    "calculate_latency_metrics",
    "calculate_token_metrics",
    "calculate_cost_metrics",
    "calculate_task_metrics",
]
