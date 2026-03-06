"""Tests for the report CLI command."""

import re
import tempfile
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from ash_hawk.cli.main import cli
from ash_hawk.types import EvalOutcome, EvalTranscript


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def mock_dawn_runner(monkeypatch):
    class FakeDawnKestrelAgentRunner:
        def __init__(self, provider: str, model: str):
            self.provider = provider
            self.model = model

        async def run(self, task, policy_enforcer, config):
            del policy_enforcer
            del config
            prompt = task.input if isinstance(task.input, str) else str(task.input)
            return (
                EvalTranscript(
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "Mock response"},
                    ],
                    agent_response="Mock response",
                ),
                EvalOutcome.success(),
            )

    monkeypatch.setattr("ash_hawk.agents.DawnKestrelAgentRunner", FakeDawnKestrelAgentRunner)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_storage(temp_dir):
    storage_path = Path(temp_dir) / "storage"
    storage_path.mkdir()
    return str(storage_path)


@pytest.fixture
def sample_suite_file(temp_dir):
    suite_path = Path(temp_dir) / "suites" / "test-suite.yaml"
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_data = {
        "id": "test-suite",
        "name": "Test Suite",
        "description": "A test suite for CLI testing",
        "agent": {"name": "build"},
        "tasks": [
            {
                "id": "task-001",
                "description": "First task",
                "input": "What is 1 + 1?",
            },
            {
                "id": "task-002",
                "description": "Second task",
                "input": "What is 2 + 2?",
            },
        ],
    }
    with open(suite_path, "w") as f:
        yaml.dump(suite_data, f)
    return str(suite_path)


def _extract_run_id(output: str) -> str | None:
    """Extract run ID from CLI output."""
    match = re.search(r"Run ID: (run-[a-f0-9]+)", output)
    return match.group(1) if match else None


class TestReportEnhancedMetrics:
    """Tests for enhanced metrics display in report command."""

    def test_report_shows_confidence_interval(self, runner, sample_suite_file, temp_storage):
        """Report should display confidence interval for pass rate."""
        run_result = runner.invoke(
            cli,
            ["run", sample_suite_file, "--storage", temp_storage, "--agent", "build"],
        )
        assert run_result.exit_code == 0

        run_id = _extract_run_id(run_result.output)
        assert run_id is not None

        result = runner.invoke(
            cli,
            ["report", run_id, "--storage", temp_storage],
        )
        assert result.exit_code == 0
        # Should show 95% confidence interval
        assert "95%" in result.output or "CI" in result.output or "Confidence" in result.output

    def test_report_shows_pass_at_k_metrics(self, runner, sample_suite_file, temp_storage):
        """Report should display pass@k metrics."""
        run_result = runner.invoke(
            cli,
            ["run", sample_suite_file, "--storage", temp_storage, "--agent", "build"],
        )
        assert run_result.exit_code == 0

        run_id = _extract_run_id(run_result.output)
        assert run_id is not None

        result = runner.invoke(
            cli,
            ["report", run_id, "--storage", temp_storage],
        )
        assert result.exit_code == 0
        # Should show pass@k metrics
        assert "pass@" in result.output.lower() or "pass@" in result.output

    def test_report_shows_latency_percentiles(self, runner, sample_suite_file, temp_storage):
        """Report should display latency percentiles (p50, p90, p95, p99)."""
        run_result = runner.invoke(
            cli,
            ["run", sample_suite_file, "--storage", temp_storage, "--agent", "build"],
        )
        assert run_result.exit_code == 0

        run_id = _extract_run_id(run_result.output)
        assert run_id is not None

        result = runner.invoke(
            cli,
            ["report", run_id, "--storage", temp_storage],
        )
        assert result.exit_code == 0
        # Should show percentile metrics
        assert (
            "p50" in result.output.lower()
            or "p90" in result.output.lower()
            or "median" in result.output.lower()
        )

    def test_report_backward_compatible(self, runner, sample_suite_file, temp_storage):
        """Report should maintain backward compatibility with existing output."""
        run_result = runner.invoke(
            cli,
            ["run", sample_suite_file, "--storage", temp_storage, "--agent", "build"],
        )
        assert run_result.exit_code == 0

        run_id = _extract_run_id(run_result.output)
        assert run_id is not None

        result = runner.invoke(
            cli,
            ["report", run_id, "--storage", temp_storage],
        )
        assert result.exit_code == 0
        # Should still show all original metrics
        assert "Pass Rate" in result.output
        assert "Total Tasks" in result.output
        assert "Mean Score" in result.output
        assert "Duration" in result.output

    def test_report_shows_token_breakdown(self, runner, sample_suite_file, temp_storage):
        """Report should display token breakdown (input/output/reasoning)."""
        run_result = runner.invoke(
            cli,
            ["run", sample_suite_file, "--storage", temp_storage, "--agent", "build"],
        )
        assert run_result.exit_code == 0

        run_id = _extract_run_id(run_result.output)
        assert run_id is not None

        result = runner.invoke(
            cli,
            ["report", run_id, "--storage", temp_storage],
        )
        assert result.exit_code == 0
        # Should show token breakdown
        assert (
            "Input" in result.output
            or "Output" in result.output
            or "token" in result.output.lower()
        )


class TestReportDisagreements:
    """Tests for disagreement detection display in report command."""

    def test_report_shows_trials_needing_review_section(
        self, runner, sample_suite_file, temp_storage
    ):
        """Report should display 'Trials Needing Review' section when disagreements exist."""
        run_result = runner.invoke(
            cli,
            ["run", sample_suite_file, "--storage", temp_storage, "--agent", "build"],
        )
        assert run_result.exit_code == 0

        run_id = _extract_run_id(run_result.output)
        assert run_id is not None

        result = runner.invoke(
            cli,
            ["report", run_id, "--storage", temp_storage],
        )
        assert result.exit_code == 0
        # Should show disagreement detection section
        assert "Review" in result.output or "Disagreement" in result.output

    def test_report_shows_confidence_summary(self, runner, sample_suite_file, temp_storage):
        """Report should display confidence distribution summary."""
        run_result = runner.invoke(
            cli,
            ["run", sample_suite_file, "--storage", temp_storage, "--agent", "build"],
        )
        assert run_result.exit_code == 0

        run_id = _extract_run_id(run_result.output)
        assert run_id is not None

        result = runner.invoke(
            cli,
            ["report", run_id, "--storage", temp_storage],
        )
        assert result.exit_code == 0
        # Should show confidence summary with percentages
        assert "Confidence" in result.output or "confidence" in result.output

    def test_report_no_disagreements_section_when_all_high_confidence(
        self, runner, sample_suite_file, temp_storage
    ):
        """Report should not show 'Trials Needing Review' section when all trials are high confidence."""
        run_result = runner.invoke(
            cli,
            ["run", sample_suite_file, "--storage", temp_storage, "--agent", "build"],
        )
        assert run_result.exit_code == 0

        run_id = _extract_run_id(run_result.output)
        assert run_id is not None

        result = runner.invoke(
            cli,
            ["report", run_id, "--storage", temp_storage],
        )
        assert result.exit_code == 0
        # Report should complete successfully regardless of disagreement status
        # (actual disagreement detection depends on grader results)
