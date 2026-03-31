"""Grader infrastructure for ash-hawk evaluation harness.

This module provides the base grader abstract class and a registry
for managing and discovering graders.
"""

from ash_hawk.graders.aggregation import (
    DisagreementReport,
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
from ash_hawk.graders.base import Grader, PassThroughGrader
from ash_hawk.graders.judge_normalizer import (
    NormalizedJudgeOutput,
    normalize_judge_output,
)
from ash_hawk.graders.llm_boolean import LLMBooleanJudgeGrader
from ash_hawk.graders.llm_boolean_specialized import BooleanJudgeGrader, create_boolean_graders
from ash_hawk.graders.prompt_stack_optimizer import (
    PromptStackOptimizerConfig,
    PromptStackOptimizerGrader,
)
from ash_hawk.graders.registry import (
    ENTRY_POINT_GROUP,
    GraderRegistry,
    get_default_registry,
)
from ash_hawk.graders.rubric_guard import (
    enforce_rubric_based_evaluation,
    task_has_rubric_based_evaluation,
    validate_suite_has_rubric_based_evaluation,
    validate_task_has_rubric,
)
from ash_hawk.graders.score_normalizer import (
    compute_weighted_score,
    normalize_grader_scores,
    normalize_score,
    score_to_grade,
)

__all__ = [
    "Grader",
    "PassThroughGrader",
    "GraderRegistry",
    "get_default_registry",
    "ENTRY_POINT_GROUP",
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
    "NormalizedJudgeOutput",
    "normalize_judge_output",
    "compute_weighted_score",
    "normalize_grader_scores",
    "normalize_score",
    "score_to_grade",
    "enforce_rubric_based_evaluation",
    "task_has_rubric_based_evaluation",
    "validate_suite_has_rubric_based_evaluation",
    "validate_task_has_rubric",
    "PromptStackOptimizerGrader",
    "PromptStackOptimizerConfig",
    "LLMBooleanJudgeGrader",
    "BooleanJudgeGrader",
    "create_boolean_graders",
]
