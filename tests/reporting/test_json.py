"""Tests for ash_hawk.reporting.json_export module."""

import json
from io import StringIO

import pytest

from ash_hawk.reporting.json_export import (
    EXPORT_SCHEMA,
    JSON_SCHEMA_VERSION,
    JSONExporter,
    JSONSchemaValidator,
    export_json,
    export_jsonl,
    validate_export,
)
from ash_hawk.types import (
    EvalOutcome,
    EvalRunSummary,
    EvalStatus,
    EvalSuite,
    EvalTask,
    EvalTranscript,
    EvalTrial,
    GraderResult,
    RunEnvelope,
    SuiteMetrics,
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
    grader_results: list[GraderResult] | None = None,
) -> EvalTrial:
    result = None
    if status == EvalStatus.COMPLETED:
        result = TrialResult(
            trial_id=trial_id,
            outcome=EvalOutcome(
                status=status,
                completed_at="2024-01-01T12:00:00+00:00",
            ),
            transcript=EvalTranscript(
                duration_seconds=latency,
                token_usage=TokenUsage(input=100, output=50),
                cost_usd=0.01,
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


def make_envelope(run_id: str = "run-1", suite_id: str = "suite-1") -> RunEnvelope:
    return RunEnvelope(
        run_id=run_id,
        suite_id=suite_id,
        suite_hash="abc123",
        harness_version="1.0.0",
        agent_name="test-agent",
        provider="test",
        model="test-model",
        tool_policy_hash="def456",
        python_version="3.12",
        os_info="linux",
        created_at="2024-01-01T00:00:00+00:00",
    )


def make_metrics(
    suite_id: str = "suite-1",
    run_id: str = "run-1",
    total_tasks: int = 1,
) -> SuiteMetrics:
    return SuiteMetrics(
        suite_id=suite_id,
        run_id=run_id,
        total_tasks=total_tasks,
        completed_tasks=total_tasks,
        passed_tasks=total_tasks,
        pass_rate=1.0,
        mean_score=1.0,
        created_at="2024-01-01T12:00:00+00:00",
    )


def make_run_summary(
    trials: list[EvalTrial] | None = None,
    envelope: RunEnvelope | None = None,
) -> EvalRunSummary:
    if envelope is None:
        envelope = make_envelope()
    if trials is None:
        trials = [make_trial("t1", "task1")]

    metrics = SuiteMetrics(
        suite_id=envelope.suite_id,
        run_id=envelope.run_id,
        total_tasks=len(trials),
        completed_tasks=len(trials),
        passed_tasks=sum(1 for t in trials if t.result and t.result.aggregate_passed),
        pass_rate=1.0 if trials else 0.0,
        mean_score=1.0,
        created_at="2024-01-01T12:00:00+00:00",
    )

    return EvalRunSummary(
        envelope=envelope,
        metrics=metrics,
        trials=trials,
    )


def make_suite(suite_id: str = "suite-1") -> EvalSuite:
    return EvalSuite(
        id=suite_id,
        name="Test Suite",
        tasks=[
            EvalTask(id="task1", input="test input"),
        ],
    )


class TestJSONSchemaValidator:
    def test_validate_valid_data(self):
        envelope = make_envelope()
        metrics = make_metrics()
        trial = make_trial("t1", "task1")

        data = {
            "schema_version": "1.0.0",
            "exported_at": "2024-01-01T12:00:00+00:00",
            "envelope": envelope.model_dump(),
            "metrics": metrics.model_dump(),
            "trials": [trial.model_dump()],
        }

        validator = JSONSchemaValidator()
        assert validator.validate(data)
        assert len(validator.errors) == 0

    def test_validate_missing_required_fields(self):
        data = {"schema_version": "1.0.0"}

        validator = JSONSchemaValidator()
        assert not validator.validate(data)
        assert "Missing required field: exported_at" in validator.errors
        assert "Missing required field: envelope" in validator.errors
        assert "Missing required field: metrics" in validator.errors
        assert "Missing required field: trials" in validator.errors

    def test_validate_invalid_schema_version(self):
        data = {
            "schema_version": "not-semver",
            "exported_at": "2024-01-01T12:00:00+00:00",
            "envelope": make_envelope().model_dump(),
            "metrics": make_metrics().model_dump(),
            "trials": [],
        }

        validator = JSONSchemaValidator()
        assert not validator.validate(data)
        assert any("semver" in e for e in validator.errors)

    def test_validate_invalid_timestamp(self):
        data = {
            "schema_version": "1.0.0",
            "exported_at": "not-a-timestamp",
            "envelope": make_envelope().model_dump(),
            "metrics": make_metrics().model_dump(),
            "trials": [],
        }

        validator = JSONSchemaValidator()
        assert not validator.validate(data)
        assert any("ISO timestamp" in e for e in validator.errors)

    def test_validate_invalid_envelope(self):
        data = {
            "schema_version": "1.0.0",
            "exported_at": "2024-01-01T12:00:00+00:00",
            "envelope": {"invalid": "data"},
            "metrics": make_metrics().model_dump(),
            "trials": [],
        }

        validator = JSONSchemaValidator()
        assert not validator.validate(data)
        assert any("Invalid envelope" in e for e in validator.errors)

    def test_validate_trials_not_list(self):
        data = {
            "schema_version": "1.0.0",
            "exported_at": "2024-01-01T12:00:00+00:00",
            "envelope": make_envelope().model_dump(),
            "metrics": make_metrics().model_dump(),
            "trials": "not a list",
        }

        validator = JSONSchemaValidator()
        assert not validator.validate(data)
        assert any("trials must be a list" in e for e in validator.errors)

    def test_validate_invalid_trial(self):
        data = {
            "schema_version": "1.0.0",
            "exported_at": "2024-01-01T12:00:00+00:00",
            "envelope": make_envelope().model_dump(),
            "metrics": make_metrics().model_dump(),
            "trials": [{"invalid": "trial"}],
        }

        validator = JSONSchemaValidator()
        assert not validator.validate(data)
        assert any("Invalid trial at index 0" in e for e in validator.errors)


class TestJSONExporter:
    def test_export_run_summary_produces_valid_json(self):
        summary = make_run_summary()
        exporter = JSONExporter()

        json_str = exporter.export_run_summary(summary)
        data = json.loads(json_str)

        assert "schema_version" in data
        assert "exported_at" in data
        assert "envelope" in data
        assert "metrics" in data
        assert "trials" in data

    def test_export_run_summary_includes_envelope(self):
        envelope = make_envelope(run_id="my-run", suite_id="my-suite")
        summary = make_run_summary(envelope=envelope)
        exporter = JSONExporter()

        json_str = exporter.export_run_summary(summary)
        data = json.loads(json_str)

        assert data["envelope"]["run_id"] == "my-run"
        assert data["envelope"]["suite_id"] == "my-suite"
        assert data["envelope"]["suite_hash"] == "abc123"
        assert data["envelope"]["harness_version"] == "1.0.0"
        assert data["envelope"]["agent_name"] == "test-agent"

    def test_export_run_summary_includes_metrics(self):
        summary = make_run_summary()
        exporter = JSONExporter()

        json_str = exporter.export_run_summary(summary)
        data = json.loads(json_str)

        assert data["metrics"]["suite_id"] == "suite-1"
        assert data["metrics"]["run_id"] == "run-1"
        assert data["metrics"]["total_tasks"] == 1

    def test_export_run_summary_includes_trials(self):
        trials = [
            make_trial("t1", "task1", passed=True, score=0.9),
            make_trial("t2", "task2", passed=False, score=0.3),
        ]
        summary = make_run_summary(trials=trials)
        exporter = JSONExporter()

        json_str = exporter.export_run_summary(summary)
        data = json.loads(json_str)

        assert len(data["trials"]) == 2
        assert data["trials"][0]["id"] == "t1"
        assert data["trials"][1]["id"] == "t2"

    def test_export_run_summary_includes_suite(self):
        summary = make_run_summary()
        suite = make_suite()
        exporter = JSONExporter()

        json_str = exporter.export_run_summary(summary, suite=suite)
        data = json.loads(json_str)

        assert "suite" in data
        assert data["suite"]["id"] == "suite-1"
        assert data["suite"]["name"] == "Test Suite"

    def test_export_with_indentation(self):
        summary = make_run_summary()
        exporter = JSONExporter(indent=4)

        json_str = exporter.export_run_summary(summary)

        assert "    " in json_str

    def test_export_compact(self):
        summary = make_run_summary()
        exporter = JSONExporter(indent=None)

        json_str = exporter.export_run_summary(summary)

        assert "\n" not in json_str

    def test_export_includes_schema_by_default(self):
        summary = make_run_summary()
        exporter = JSONExporter()

        json_str = exporter.export_run_summary(summary)
        data = json.loads(json_str)

        assert "$schema" in data

    def test_export_without_schema(self):
        summary = make_run_summary()
        exporter = JSONExporter(include_schema=False)

        json_str = exporter.export_run_summary(summary)
        data = json.loads(json_str)

        assert "$schema" not in data

    def test_export_components(self):
        envelope = make_envelope()
        metrics = make_metrics()
        trials = [make_trial("t1", "task1")]

        exporter = JSONExporter()
        json_str = exporter.export_components(
            envelope=envelope,
            metrics=metrics,
            trials=trials,
        )

        data = json.loads(json_str)
        assert data["envelope"]["run_id"] == "run-1"
        assert data["metrics"]["suite_id"] == "suite-1"
        assert len(data["trials"]) == 1

    def test_export_envelope_only(self):
        envelope = make_envelope()
        exporter = JSONExporter()

        json_str = exporter.export_envelope(envelope)
        data = json.loads(json_str)

        assert data["run_id"] == "run-1"
        assert data["suite_id"] == "suite-1"

    def test_export_metrics_only(self):
        metrics = make_metrics()
        exporter = JSONExporter()

        json_str = exporter.export_metrics(metrics)
        data = json.loads(json_str)

        assert data["suite_id"] == "suite-1"

    def test_export_trials_only(self):
        trials = [make_trial("t1", "task1"), make_trial("t2", "task2")]
        exporter = JSONExporter()

        json_str = exporter.export_trials(trials)
        data = json.loads(json_str)

        assert len(data) == 2
        assert data[0]["id"] == "t1"
        assert data[1]["id"] == "t2"


class TestJSONLExport:
    def test_export_trial_jsonl(self):
        trials = [make_trial("t1", "task1"), make_trial("t2", "task2")]
        exporter = JSONExporter()

        jsonl_str = exporter.export_trial_jsonl(trials)
        lines = jsonl_str.strip().split("\n")

        assert len(lines) == 2
        data1 = json.loads(lines[0])
        data2 = json.loads(lines[1])
        assert data1["id"] == "t1"
        assert data2["id"] == "t2"

    def test_export_run_summary_jsonl(self):
        trials = [make_trial("t1", "task1"), make_trial("t2", "task2")]
        summary = make_run_summary(trials=trials)
        exporter = JSONExporter()

        jsonl_str = exporter.export_run_summary_jsonl(summary)
        lines = jsonl_str.strip().split("\n")

        assert len(lines) == 3

        header = json.loads(lines[0])
        assert header["type"] == "header"
        assert "envelope" in header
        assert "metrics" in header

        trial1 = json.loads(lines[1])
        assert trial1["type"] == "trial"
        assert trial1["trial"]["id"] == "t1"

        trial2 = json.loads(lines[2])
        assert trial2["type"] == "trial"
        assert trial2["trial"]["id"] == "t2"

    def test_export_run_summary_jsonl_includes_suite(self):
        summary = make_run_summary()
        suite = make_suite()
        exporter = JSONExporter()

        jsonl_str = exporter.export_run_summary_jsonl(summary, suite=suite)
        lines = jsonl_str.strip().split("\n")

        header = json.loads(lines[0])
        assert "suite" in header
        assert header["suite"]["id"] == "suite-1"

    def test_stream_trials_jsonl(self):
        trials = [make_trial("t1", "task1"), make_trial("t2", "task2")]
        exporter = JSONExporter()

        buffer = exporter.stream_trials_jsonl(trials)
        content = buffer.getvalue()
        lines = content.strip().split("\n")

        assert len(lines) == 2
        data1 = json.loads(lines[0])
        assert data1["id"] == "t1"

    def test_stream_trials_jsonl_append_to_buffer(self):
        trials1 = [make_trial("t1", "task1")]
        trials2 = [make_trial("t2", "task2")]
        exporter = JSONExporter()

        buffer = exporter.stream_trials_jsonl(trials1)
        buffer = exporter.stream_trials_jsonl(trials2, buffer=buffer)

        lines = buffer.getvalue().strip().split("\n")
        assert len(lines) == 2


class TestConvenienceFunctions:
    def test_export_json(self):
        summary = make_run_summary()

        json_str = export_json(summary)
        data = json.loads(json_str)

        assert data["envelope"]["run_id"] == "run-1"
        assert len(data["trials"]) == 1

    def test_export_json_with_suite(self):
        summary = make_run_summary()
        suite = make_suite()

        json_str = export_json(summary, suite=suite)
        data = json.loads(json_str)

        assert "suite" in data

    def test_export_json_with_indent(self):
        summary = make_run_summary()

        json_str = export_json(summary, indent=None)

        assert "\n" not in json_str

    def test_export_jsonl(self):
        summary = make_run_summary()

        jsonl_str = export_jsonl(summary)
        lines = jsonl_str.strip().split("\n")

        assert len(lines) == 2
        header = json.loads(lines[0])
        assert header["type"] == "header"

    def test_validate_export(self):
        envelope = make_envelope()
        metrics = make_metrics()
        trial = make_trial("t1", "task1")

        data = {
            "schema_version": "1.0.0",
            "exported_at": "2024-01-01T12:00:00+00:00",
            "envelope": envelope.model_dump(),
            "metrics": metrics.model_dump(),
            "trials": [trial.model_dump()],
        }

        is_valid, errors = validate_export(data)

        assert is_valid
        assert len(errors) == 0

    def test_validate_export_invalid(self):
        data = {
            "schema_version": "1.0.0",
            "exported_at": "2024-01-01T12:00:00+00:00",
            "envelope": {"invalid": "data"},
            "metrics": {"invalid": "data"},
            "trials": [],
        }

        is_valid, errors = validate_export(data)

        assert not is_valid
        assert len(errors) > 0


class TestSchemaConstants:
    def test_schema_version_format(self):
        parts = JSON_SCHEMA_VERSION.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()

    def test_export_schema_structure(self):
        assert "$schema" in EXPORT_SCHEMA
        assert "$id" in EXPORT_SCHEMA
        assert "title" in EXPORT_SCHEMA
        assert "required" in EXPORT_SCHEMA
        assert "properties" in EXPORT_SCHEMA


class TestFullExportRoundtrip:
    def test_json_roundtrip_preserves_data(self):
        trials = [
            make_trial("t1", "task1", passed=True, score=0.9, latency=2.5),
            make_trial("t2", "task2", passed=False, score=0.3, latency=1.5),
        ]
        summary = make_run_summary(trials=trials)
        suite = make_suite()

        exporter = JSONExporter()
        json_str = exporter.export_run_summary(summary, suite=suite)
        data = json.loads(json_str)

        is_valid, errors = validate_export(data)
        assert is_valid, f"Validation failed: {errors}"

        assert data["envelope"]["run_id"] == summary.envelope.run_id
        assert data["metrics"]["total_tasks"] == summary.metrics.total_tasks
        assert len(data["trials"]) == len(summary.trials)
        assert data["suite"]["id"] == suite.id

    def test_jsonl_roundtrip_can_parse_all_lines(self):
        trials = [
            make_trial("t1", "task1"),
            make_trial("t2", "task2"),
            make_trial("t3", "task3"),
        ]
        summary = make_run_summary(trials=trials)

        exporter = JSONExporter()
        jsonl_str = exporter.export_run_summary_jsonl(summary)
        lines = jsonl_str.strip().split("\n")

        for line in lines:
            data = json.loads(line)
            assert "type" in data
            assert data["type"] in ("header", "trial")
