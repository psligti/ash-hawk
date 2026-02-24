"""Ash Hawk Metrics Module.

This module provides statistical calculations for evaluating AI agent
performance, including:
- pass@k and pass^k metrics
- Confidence intervals
- Significance testing
- Latency percentiles
- Token and cost tracking
"""

from ash_hawk.metrics.statistics import (
    ConfidenceInterval,
    CostMetrics,
    LatencyMetrics,
    SignificanceResult,
    SuiteMetricsDetailed,
    TaskMetrics,
    TokenMetrics,
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
