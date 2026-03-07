"""Ash Hawk - Reporting module for evaluation harness.

This module provides various export formats for evaluation results:
- JSONExporter: Machine-readable JSON/JSONL output
- HTMLReporter: Human-readable HTML reports with Chart.js

All reporters include full RunEnvelope for reproducibility.
"""

from ash_hawk.reporting.gap_scorecard import GapScorecardGenerator, load_scorecard
from ash_hawk.reporting.html import (
    HTMLReporter,
    Theme,
    generate_html_report,
    generate_task_html,
    generate_trial_html,
)
from ash_hawk.reporting.json_export import (
    EXPORT_SCHEMA,
    JSON_SCHEMA_VERSION,
    JSONExporter,
    JSONSchemaValidator,
    export_json,
    export_jsonl,
    validate_export,
)
from ash_hawk.reporting.scorecard_types import (
    AgentDepth,
    GapDiff,
    GapScorecard,
    Requirement,
    RequirementCoverage,
)

__all__ = [
    "EXPORT_SCHEMA",
    "HTMLReporter",
    "JSONExporter",
    "JSONSchemaValidator",
    "JSON_SCHEMA_VERSION",
    "Theme",
    "export_json",
    "export_jsonl",
    "generate_html_report",
    "generate_task_html",
    "generate_trial_html",
    "validate_export",
    # Gap scorecard
    "AgentDepth",
    "GapDiff",
    "GapScorecard",
    "GapScorecardGenerator",
    "Requirement",
    "RequirementCoverage",
    "load_scorecard",
]
