"""JSON/JSONL export functionality for ash-hawk evaluation results.

Provides machine-readable export of evaluation runs including:
- Full RunEnvelope for reproducibility
- All trial transcripts and grader results
- Aggregate metrics
- Schema validation support
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timezone
from io import StringIO
from typing import Any

from pydantic import ValidationError

from ash_hawk.types import (
    EvalRunSummary,
    EvalSuite,
    EvalTrial,
    RunEnvelope,
    SuiteMetrics,
)

JSON_SCHEMA_VERSION = "1.0.0"

EXPORT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://ash-hawk.dev/schemas/eval-export-v1.json",
    "title": "AshHawkEvalExport",
    "description": "Schema for Ash-Hawk evaluation run export",
    "type": "object",
    "required": ["schema_version", "exported_at", "envelope", "metrics", "trials"],
    "properties": {
        "schema_version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
        "exported_at": {"type": "string", "format": "date-time"},
        "envelope": {"type": "object"},
        "metrics": {"type": "object"},
        "trials": {"type": "array"},
        "suite": {"type": "object"},
    },
}


COMPUTED_FIELDS_TO_REMOVE: dict[str, list[str]] = {
    "TokenUsage": ["total"],
    "EvalSuite": ["task_count"],
}


def _strip_computed_fields(data: dict[str, Any], model_name: str) -> dict[str, Any]:
    fields_to_remove = COMPUTED_FIELDS_TO_REMOVE.get(model_name, [])
    if not fields_to_remove:
        return data
    return {k: v for k, v in data.items() if k not in fields_to_remove}


def _strip_nested_computed(obj: dict[str, Any]) -> dict[str, Any]:
    result = dict(obj)
    if "token_usage" in result and isinstance(result["token_usage"], dict):
        result["token_usage"] = _strip_computed_fields(result["token_usage"], "TokenUsage")
    if "total_tokens" in result and isinstance(result["total_tokens"], dict):
        result["total_tokens"] = _strip_computed_fields(result["total_tokens"], "TokenUsage")
    if "transcript" in result and isinstance(result["transcript"], dict):
        transcript = dict(result["transcript"])
        if "token_usage" in transcript and isinstance(transcript["token_usage"], dict):
            transcript["token_usage"] = _strip_computed_fields(
                transcript["token_usage"], "TokenUsage"
            )
        result["transcript"] = transcript
    if "result" in result and isinstance(result["result"], dict):
        trial_result = _strip_nested_computed(result["result"])
        result["result"] = trial_result
    return result


class JSONSchemaValidator:
    def __init__(self) -> None:
        self._errors: list[str] = []

    def validate(self, data: dict[str, Any]) -> bool:
        self._errors = []

        required_fields = ["schema_version", "exported_at", "envelope", "metrics", "trials"]
        for field in required_fields:
            if field not in data:
                self._errors.append(f"Missing required field: {field}")

        if "schema_version" in data:
            version = data["schema_version"]
            if not isinstance(version, str):
                self._errors.append("schema_version must be a string")
            elif not version.count(".") >= 2:
                self._errors.append("schema_version must be in semver format (X.Y.Z)")

        if "exported_at" in data:
            exported_at = data["exported_at"]
            if not isinstance(exported_at, str):
                self._errors.append("exported_at must be an ISO timestamp string")
            else:
                try:
                    datetime.fromisoformat(exported_at.replace("Z", "+00:00"))
                except ValueError:
                    self._errors.append("exported_at must be a valid ISO timestamp")

        if "envelope" in data:
            try:
                RunEnvelope.model_validate(data["envelope"])
            except ValidationError as e:
                self._errors.append(f"Invalid envelope: {e}")

        if "metrics" in data:
            try:
                cleaned_metrics = _strip_nested_computed(data["metrics"])
                SuiteMetrics.model_validate(cleaned_metrics)
            except ValidationError as e:
                self._errors.append(f"Invalid metrics: {e}")

        if "trials" in data:
            if not isinstance(data["trials"], list):
                self._errors.append("trials must be a list")
            else:
                for i, trial_data in enumerate(data["trials"]):
                    try:
                        cleaned_trial = _strip_nested_computed(trial_data)
                        EvalTrial.model_validate(cleaned_trial)
                    except ValidationError as e:
                        self._errors.append(f"Invalid trial at index {i}: {e}")

        return len(self._errors) == 0

    @property
    def errors(self) -> list[str]:
        return self._errors.copy()


class JSONExporter:
    """Exports evaluation results to JSON/JSONL format.

    Provides methods for exporting complete run summaries or individual
    components in machine-readable JSON format.

    Example:
        exporter = JSONExporter()
        json_str = exporter.export_run_summary(summary)
        with open("results.json", "w") as f:
            f.write(json_str)
    """

    def __init__(
        self,
        *,
        indent: int | None = 2,
        include_schema: bool = True,
    ) -> None:
        """Initialize the exporter.

        Args:
            indent: JSON indentation level. None for compact output.
            include_schema: Whether to include $schema in output.
        """
        self._indent = indent
        self._include_schema = include_schema

    def export_run_summary(
        self,
        summary: EvalRunSummary,
        *,
        suite: EvalSuite | None = None,
    ) -> str:
        """Export a complete run summary to JSON.

        Args:
            summary: The run summary to export.
            suite: Optional suite definition to include.

        Returns:
            JSON string of the exported data.
        """
        data = self._build_export_data(
            envelope=summary.envelope,
            metrics=summary.metrics,
            trials=summary.trials,
            suite=suite,
        )
        return json.dumps(data, indent=self._indent, default=self._json_serializer)

    def export_components(
        self,
        envelope: RunEnvelope,
        metrics: SuiteMetrics,
        trials: list[EvalTrial],
        *,
        suite: EvalSuite | None = None,
    ) -> str:
        """Export individual components to JSON.

        Args:
            envelope: The run envelope.
            metrics: The suite metrics.
            trials: List of trials.
            suite: Optional suite definition.

        Returns:
            JSON string of the exported data.
        """
        data = self._build_export_data(
            envelope=envelope,
            metrics=metrics,
            trials=trials,
            suite=suite,
        )
        return json.dumps(data, indent=self._indent, default=self._json_serializer)

    def export_envelope(self, envelope: RunEnvelope) -> str:
        """Export just the run envelope.

        Args:
            envelope: The run envelope to export.

        Returns:
            JSON string of the envelope.
        """
        return json.dumps(
            envelope.model_dump(),
            indent=self._indent,
            default=self._json_serializer,
        )

    def export_metrics(self, metrics: SuiteMetrics) -> str:
        """Export just the metrics.

        Args:
            metrics: The metrics to export.

        Returns:
            JSON string of the metrics.
        """
        return json.dumps(
            metrics.model_dump(),
            indent=self._indent,
            default=self._json_serializer,
        )

    def export_trials(self, trials: list[EvalTrial]) -> str:
        """Export a list of trials.

        Args:
            trials: The trials to export.

        Returns:
            JSON string of the trials.
        """
        return json.dumps(
            [t.model_dump() for t in trials],
            indent=self._indent,
            default=self._json_serializer,
        )

    def export_trial_jsonl(self, trials: list[EvalTrial]) -> str:
        """Export trials in JSONL format for streaming.

        Each trial is on a separate line as a complete JSON object.
        This is useful for large result sets that need to be streamed.

        Args:
            trials: The trials to export.

        Returns:
            JSONL string (newline-delimited JSON).
        """
        lines = []
        for trial in trials:
            line = json.dumps(trial.model_dump(), default=self._json_serializer)
            lines.append(line)
        return "\n".join(lines)

    def export_run_summary_jsonl(
        self,
        summary: EvalRunSummary,
        *,
        suite: EvalSuite | None = None,
    ) -> str:
        """Export run summary in JSONL format.

        First line: header with envelope and metrics
        Subsequent lines: one trial per line

        Args:
            summary: The run summary to export.
            suite: Optional suite definition.

        Returns:
            JSONL string with header + trials.
        """
        lines = []

        header = {
            "type": "header",
            "schema_version": JSON_SCHEMA_VERSION,
            "exported_at": datetime.now(UTC).isoformat(),
            "envelope": summary.envelope.model_dump(),
            "metrics": summary.metrics.model_dump(),
        }
        if suite:
            header["suite"] = suite.model_dump()
        lines.append(json.dumps(header, default=self._json_serializer))

        for trial in summary.trials:
            line = {
                "type": "trial",
                "trial": trial.model_dump(),
            }
            lines.append(json.dumps(line, default=self._json_serializer))

        return "\n".join(lines)

    def stream_trials_jsonl(
        self,
        trials: list[EvalTrial],
        buffer: StringIO | None = None,
    ) -> StringIO:
        """Stream trials to a StringIO buffer in JSONL format.

        Args:
            trials: The trials to stream.
            buffer: Optional existing buffer to append to.

        Returns:
            StringIO buffer containing JSONL data.
        """
        if buffer is None:
            buffer = StringIO()

        for trial in trials:
            buffer.write(json.dumps(trial.model_dump(), default=self._json_serializer))
            buffer.write("\n")

        return buffer

    def _build_export_data(
        self,
        envelope: RunEnvelope,
        metrics: SuiteMetrics,
        trials: list[EvalTrial],
        suite: EvalSuite | None = None,
    ) -> dict[str, Any]:
        """Build the export data dictionary.

        Args:
            envelope: The run envelope.
            metrics: The suite metrics.
            trials: List of trials.
            suite: Optional suite definition.

        Returns:
            Dictionary ready for JSON serialization.
        """
        data: dict[str, Any] = {
            "schema_version": JSON_SCHEMA_VERSION,
            "exported_at": datetime.now(UTC).isoformat(),
            "envelope": envelope.model_dump(),
            "metrics": metrics.model_dump(),
            "trials": [t.model_dump() for t in trials],
        }

        if self._include_schema:
            data["$schema"] = EXPORT_SCHEMA["$id"]

        if suite:
            data["suite"] = suite.model_dump()

        return data

    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """Custom JSON serializer for non-standard types.

        Args:
            obj: Object to serialize.

        Returns:
            Serializable representation.
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def export_json(
    summary: EvalRunSummary,
    *,
    suite: EvalSuite | None = None,
    indent: int | None = 2,
) -> str:
    """Convenience function to export a run summary to JSON.

    Args:
        summary: The run summary to export.
        suite: Optional suite definition to include.
        indent: JSON indentation level.

    Returns:
        JSON string.
    """
    exporter = JSONExporter(indent=indent)
    return exporter.export_run_summary(summary, suite=suite)


def export_jsonl(
    summary: EvalRunSummary,
    *,
    suite: EvalSuite | None = None,
) -> str:
    """Convenience function to export a run summary to JSONL.

    Args:
        summary: The run summary to export.
        suite: Optional suite definition to include.

    Returns:
        JSONL string.
    """
    exporter = JSONExporter()
    return exporter.export_run_summary_jsonl(summary, suite=suite)


def validate_export(data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate export data against the schema.

    Args:
        data: The data dictionary to validate.

    Returns:
        Tuple of (is_valid, list_of_errors).
    """
    validator = JSONSchemaValidator()
    is_valid = validator.validate(data)
    return is_valid, validator.errors


__all__ = [
    "EXPORT_SCHEMA",
    "JSONSchemaValidator",
    "JSONExporter",
    "JSON_SCHEMA_VERSION",
    "export_json",
    "export_jsonl",
    "validate_export",
]
