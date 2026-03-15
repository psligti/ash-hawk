from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from ash_hawk.adapters.artifact_adapter import ArtifactAdapter
from ash_hawk.contracts import RunArtifact, ToolCallRecord
from ash_hawk.storage import FileStorage
from ash_hawk.types import EvalRunSummary, RunEnvelope, SuiteMetrics


@pytest.fixture
def mock_storage() -> Any:
    storage = AsyncMock(spec=FileStorage)
    storage.list_suites = AsyncMock(return_value=["test-suite"])
    return storage


@pytest.fixture
def sample_run_envelope() -> RunEnvelope:
    return RunEnvelope(
        run_id="run-test-001",
        suite_id="test-suite",
        suite_hash="abc123",
        harness_version="0.1.0",
        agent_name="test-agent",
        provider="zai",
        model="glm-4-plus",
        tool_policy_hash="def456",
        python_version="3.11",
        os_info="darwin",
        created_at=datetime.now(UTC).isoformat(),
    )


@pytest.fixture
def sample_run_summary(sample_run_envelope: RunEnvelope) -> EvalRunSummary:
    return EvalRunSummary(
        envelope=sample_run_envelope,
        metrics=SuiteMetrics(
            suite_id="test-suite",
            run_id="run-test-001",
            total_tasks=1,
            completed_tasks=1,
            passed_tasks=1,
            failed_tasks=0,
            pass_rate=1.0,
            created_at=datetime.now(UTC).isoformat(),
        ),
        trials=[],
    )


class TestArtifactAdapter:
    def test_create_artifact_from_summary(
        self,
        mock_storage: Any,
        sample_run_envelope: RunEnvelope,
        sample_run_summary: EvalRunSummary,
    ) -> None:
        adapter = ArtifactAdapter(mock_storage)
        artifact = adapter.create_artifact_from_summary(sample_run_summary, sample_run_envelope)

        assert artifact.run_id == "run-test-001"
        assert artifact.suite_id == "test-suite"
        assert artifact.agent_name == "test-agent"
        assert artifact.outcome == "success"

    def test_create_artifact_from_summary_with_failures(
        self, mock_storage: Any, sample_run_envelope: RunEnvelope
    ) -> None:
        summary = EvalRunSummary(
            envelope=sample_run_envelope,
            metrics=SuiteMetrics(
                suite_id="test-suite",
                run_id="run-test-001",
                total_tasks=2,
                completed_tasks=2,
                passed_tasks=1,
                failed_tasks=1,
                pass_rate=0.5,
                created_at=datetime.now(UTC).isoformat(),
            ),
            trials=[],
        )

        adapter = ArtifactAdapter(mock_storage)
        artifact = adapter.create_artifact_from_summary(summary, sample_run_envelope)

        assert artifact.outcome == "failure"

    @pytest.mark.asyncio
    async def test_load_run_artifact_not_found(self, mock_storage: Any) -> None:
        mock_storage.load_run_envelope = AsyncMock(return_value=None)

        adapter = ArtifactAdapter(mock_storage)
        artifact = await adapter.load_run_artifact("nonexistent-run")

        assert artifact is None

    @pytest.mark.asyncio
    async def test_load_run_artifact_from_suite(
        self,
        mock_storage: Any,
        sample_run_envelope: RunEnvelope,
        sample_run_summary: EvalRunSummary,
    ) -> None:
        mock_storage.load_run_envelope = AsyncMock(return_value=sample_run_envelope)
        mock_storage.load_summary = AsyncMock(return_value=sample_run_summary)

        adapter = ArtifactAdapter(mock_storage)
        artifact = await adapter.load_run_artifact_from_suite("test-suite", "run-test-001")

        assert artifact is not None
        assert artifact.run_id == "run-test-001"


class TestToolCallRecord:
    def test_tool_call_record_creation(self) -> None:
        record = ToolCallRecord(
            tool_name="read",
            outcome="success",
            duration_ms=100,
            input_args={"filePath": "/test.py"},
            output="file contents",
        )

        assert record.tool_name == "read"
        assert record.outcome == "success"
        assert record.duration_ms == 100


class TestRunArtifact:
    def test_is_successful(self) -> None:
        success_artifact = RunArtifact(
            run_id="run-001",
            outcome="success",
            tool_calls=[ToolCallRecord(tool_name="read", outcome="success")],
        )
        failed_artifact = RunArtifact(
            run_id="run-002",
            outcome="failure",
        )

        assert success_artifact.is_successful() is True
        assert failed_artifact.is_successful() is False

    def test_get_tool_success_rate(self) -> None:
        artifact = RunArtifact(
            run_id="run-001",
            tool_calls=[
                ToolCallRecord(tool_name="read", outcome="success"),
                ToolCallRecord(tool_name="write", outcome="success"),
                ToolCallRecord(tool_name="bash", outcome="failure"),
            ],
        )

        assert artifact.get_tool_success_rate() == 2 / 3

    def test_get_tool_success_rate_empty(self) -> None:
        artifact = RunArtifact(run_id="run-001", tool_calls=[])
        assert artifact.get_tool_success_rate() == 0.0

    def test_get_total_tokens(self) -> None:
        artifact = RunArtifact(
            run_id="run-001",
            token_usage={"input": 100, "output": 50, "total": 150},
        )

        assert artifact.get_total_tokens() == 150
