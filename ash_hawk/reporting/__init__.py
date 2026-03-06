"""Ash Hawk - Reporting module for evaluation harness.

This module provides various export formats for evaluation results:
- JSONExporter: Machine-readable JSON/JSONL output
- HTMLReporter: Human-readable HTML reports with Chart.js

All reporters include full RunEnvelope for reproducibility.
"""

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
]
