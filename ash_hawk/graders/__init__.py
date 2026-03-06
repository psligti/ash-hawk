"""Grader infrastructure for ash-hawk evaluation harness.

This module provides the base grader abstract class and a registry
for managing and discovering graders.

Key components:
- Grader: Abstract base class that all graders must implement
- GraderRegistry: Registry for managing grader instances
- PassThroughGrader: Simple always-passing grader for testing

Graders evaluate trial results and return structured GraderResult objects
with scores (0.0-1.0) and pass/fail status.
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
from ash_hawk.graders.registry import (
    ENTRY_POINT_GROUP,
    GraderRegistry,
    get_default_registry,
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
]
