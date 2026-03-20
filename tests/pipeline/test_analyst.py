"""Tests for AnalystRole."""

from __future__ import annotations

import pytest

from ash_hawk.contracts import ReviewFinding, ReviewMetrics
from ash_hawk.pipeline.analyst import AnalystInput, AnalystOutput, AnalystRole
from tests.pipeline.conftest import MockRunArtifact, MockToolCall


@pytest.fixture
def analyst() -> AnalystRole:
    return AnalystRole()


@pytest.fixture
def analyst_input(mock_run_artifact: MockRunArtifact) -> AnalystInput:
    input_data = AnalystInput()
    input_data.artifact = mock_run_artifact
    input_data.focus_areas = []
    return input_data


class TestAnalystInput:
    def test_init_defaults(self):
        input_data = AnalystInput()
        assert input_data.artifact is None
        assert input_data.focus_areas is None


class TestAnalystOutput:
    def test_init_defaults(self):
        output = AnalystOutput()
        assert output.findings == []
        assert output.metrics.score == 0.0
        assert output.tool_efficiency == {}
        assert output.failure_patterns == []
        assert output.risk_areas == []


class TestAnalystRoleAnalyze:
    def test_analyze_returns_output(self, analyst: AnalystRole, analyst_input: AnalystInput):
        output = analyst.analyze(analyst_input)
        assert isinstance(output, AnalystOutput)

    def test_analyze_with_null_artifact(self, analyst: AnalystRole):
        input_data = AnalystInput()
        input_data.artifact = None
        output = analyst.analyze(input_data)
        assert output.findings == []
        assert output.metrics.score == 0.0

    def test_analyze_calculates_metrics(self, analyst: AnalystRole, analyst_input: AnalystInput):
        output = analyst.analyze(analyst_input)
        assert isinstance(output.metrics, ReviewMetrics)

    def test_analyze_calculates_tool_efficiency(
        self, analyst: AnalystRole, analyst_input: AnalystInput
    ):
        output = analyst.analyze(analyst_input)
        assert "unique_tool_count" in output.tool_efficiency
        assert "total_tool_count" in output.tool_efficiency

    def test_analyze_identifies_failure_patterns(
        self, analyst: AnalystRole, analyst_input: AnalystInput
    ):
        output = analyst.analyze(analyst_input)
        assert isinstance(output.failure_patterns, list)

    def test_analyze_identifies_risk_areas(self, analyst: AnalystRole, analyst_input: AnalystInput):
        output = analyst.analyze(analyst_input)
        assert isinstance(output.risk_areas, list)


class TestAnalystRoleWithSuccessRun:
    def test_success_run_no_failure_patterns(
        self, analyst: AnalystRole, mock_run_artifact: MockRunArtifact
    ):
        input_data = AnalystInput()
        input_data.artifact = mock_run_artifact
        output = analyst.analyze(input_data)
        assert output.failure_patterns == []

    def test_success_run_positive_metrics(
        self, analyst: AnalystRole, mock_run_artifact: MockRunArtifact
    ):
        input_data = AnalystInput()
        input_data.artifact = mock_run_artifact
        output = analyst.analyze(input_data)
        assert output.metrics.score > 0


class TestAnalystRoleWithFailedRun:
    def test_failed_run_detects_failure(
        self, analyst: AnalystRole, mock_failed_run_artifact: MockRunArtifact
    ):
        input_data = AnalystInput()
        input_data.artifact = mock_failed_run_artifact
        output = analyst.analyze(input_data)
        assert len(output.failure_patterns) > 0

    def test_failed_run_detects_timeout(
        self, analyst: AnalystRole, mock_failed_run_artifact: MockRunArtifact
    ):
        input_data = AnalystInput()
        input_data.artifact = mock_failed_run_artifact
        output = analyst.analyze(input_data)
        timeout_patterns = [
            p for p in output.failure_patterns if "timed out" in p.lower() or "timeout" in p.lower()
        ]
        assert len(timeout_patterns) > 0

    def test_failed_run_generates_findings(
        self, analyst: AnalystRole, mock_failed_run_artifact: MockRunArtifact
    ):
        input_data = AnalystInput()
        input_data.artifact = mock_failed_run_artifact
        output = analyst.analyze(input_data)
        assert len(output.findings) > 0


class TestAnalystRoleWithRiskAreas:
    def test_detects_file_modification_risk(self, analyst: AnalystRole):
        artifact = MockRunArtifact(
            run_id="run-risk-001",
            outcome="success",
            tool_calls=[
                MockToolCall(tool_name="write", outcome="success"),
                MockToolCall(tool_name="edit", outcome="success"),
            ],
        )
        input_data = AnalystInput()
        input_data.artifact = artifact
        output = analyst.analyze(input_data)
        assert "file_modification" in output.risk_areas

    def test_detects_command_execution_risk(self, analyst: AnalystRole):
        artifact = MockRunArtifact(
            run_id="run-risk-002",
            outcome="success",
            tool_calls=[
                MockToolCall(tool_name="bash", outcome="success"),
            ],
        )
        input_data = AnalystInput()
        input_data.artifact = artifact
        output = analyst.analyze(input_data)
        assert "command_execution" in output.risk_areas

    def test_no_risk_for_safe_tools(self, analyst: AnalystRole):
        artifact = MockRunArtifact(
            run_id="run-safe-001",
            outcome="success",
            tool_calls=[
                MockToolCall(tool_name="read", outcome="success"),
            ],
        )
        input_data = AnalystInput()
        input_data.artifact = artifact
        output = analyst.analyze(input_data)
        assert output.risk_areas == []


class TestAnalystRoleToolEfficiency:
    def test_calculates_redundancy_ratio(self, analyst: AnalystRole):
        artifact = MockRunArtifact(
            run_id="run-redundant-001",
            outcome="success",
            tool_calls=[
                MockToolCall(tool_name="read", outcome="success"),
                MockToolCall(tool_name="read", outcome="success"),
                MockToolCall(tool_name="read", outcome="success"),
            ],
        )
        input_data = AnalystInput()
        input_data.artifact = artifact
        output = analyst.analyze(input_data)
        assert output.tool_efficiency["redundancy_ratio"] == 2 / 3

    def test_empty_tool_calls_returns_empty_efficiency(self, analyst: AnalystRole):
        artifact = MockRunArtifact(
            run_id="run-empty-001",
            outcome="success",
            tool_calls=[],
        )
        input_data = AnalystInput()
        input_data.artifact = artifact
        output = analyst.analyze(input_data)
        assert output.tool_efficiency == {}


class TestAnalystRoleMetrics:
    def test_success_rate_calculation(self, analyst: AnalystRole):
        artifact = MockRunArtifact(
            run_id="run-mixed-001",
            outcome="success",
            tool_calls=[
                MockToolCall(tool_name="read", outcome="success"),
                MockToolCall(tool_name="write", outcome="success"),
                MockToolCall(tool_name="bash", outcome="failure"),
            ],
        )
        input_data = AnalystInput()
        input_data.artifact = artifact
        output = analyst.analyze(input_data)
        assert output.metrics.score == 2 / 3

    def test_quality_score_success(self, analyst: AnalystRole):
        artifact = MockRunArtifact(
            run_id="run-quality-001",
            outcome="success",
            tool_calls=[MockToolCall(tool_name="read", outcome="success")],
        )
        input_data = AnalystInput()
        input_data.artifact = artifact
        output = analyst.analyze(input_data)
        assert output.metrics.quality_score == 1.0

    def test_quality_score_failure(self, analyst: AnalystRole):
        artifact = MockRunArtifact(
            run_id="run-quality-002",
            outcome="failure",
            tool_calls=[MockToolCall(tool_name="read", outcome="success")],
        )
        input_data = AnalystInput()
        input_data.artifact = artifact
        output = analyst.analyze(input_data)
        assert output.metrics.quality_score == 0.0


class TestAnalystRolePermissionErrors:
    def test_detects_permission_denied(self, analyst: AnalystRole):
        artifact = MockRunArtifact(
            run_id="run-perm-001",
            outcome="failure",
            tool_calls=[
                MockToolCall(
                    tool_name="delete",
                    outcome="failure",
                    error_message="Permission denied for this operation",
                ),
            ],
        )
        input_data = AnalystInput()
        input_data.artifact = artifact
        output = analyst.analyze(input_data)
        permission_patterns = [p for p in output.failure_patterns if "permission" in p.lower()]
        assert len(permission_patterns) > 0
