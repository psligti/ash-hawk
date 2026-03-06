"""Tests for ash_hawk.reporting.html module."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ash_hawk.reporting.html import (
    HTMLReporter,
    generate_html_report,
    generate_task_html,
    generate_trial_html,
)
from ash_hawk.types import (
    EvalOutcome,
    EvalRunSummary,
    EvalStatus,
    EvalTask,
    EvalTranscript,
    EvalTrial,
    GraderResult,
    GraderSpec,
    RunEnvelope,
    SuiteMetrics,
    TokenUsage,
    TrialResult,
)


def create_sample_envelope() -> RunEnvelope:
    return RunEnvelope(
        run_id="run-001",
        suite_id="suite-001",
        suite_hash="abc123def456",
        harness_version="0.1.0",
        git_commit="1234567890abcdef",
        agent_name="test-agent",
        agent_version="1.0.0",
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        model_params={"temperature": 0.7},
        seed=42,
        tool_policy_hash="policy-hash-123",
        python_version="3.12.0",
        os_info="macOS 14.0",
        config_snapshot={"parallelism": 4},
        created_at="2024-01-15T10:30:00Z",
    )


def create_sample_trial(
    trial_id: str,
    task_id: str,
    passed: bool,
    score: float,
    duration: float = 10.0,
) -> EvalTrial:
    return EvalTrial(
        id=trial_id,
        task_id=task_id,
        status=EvalStatus.COMPLETED,
        attempt_number=1,
        result=TrialResult(
            trial_id=trial_id,
            outcome=EvalOutcome.success(),
            transcript=EvalTranscript(
                messages=[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
                tool_calls=[],
                token_usage=TokenUsage(input=100, output=50, reasoning=25),
                cost_usd=0.005,
                duration_seconds=duration,
            ),
            grader_results=[
                GraderResult(
                    grader_type="string_match",
                    score=score,
                    passed=passed,
                    details={"match_type": "exact"},
                )
            ],
            aggregate_score=score,
            aggregate_passed=passed,
        ),
    )


def create_sample_summary() -> EvalRunSummary:
    envelope = create_sample_envelope()

    trials = [
        create_sample_trial("trial-001", "task-001", True, 1.0, 12.5),
        create_sample_trial("trial-002", "task-002", True, 0.85, 15.0),
        create_sample_trial("trial-003", "task-003", False, 0.3, 8.0),
        create_sample_trial("trial-004", "task-004", True, 0.95, 20.0),
    ]

    metrics = SuiteMetrics(
        suite_id="suite-001",
        run_id="run-001",
        total_tasks=4,
        completed_tasks=4,
        passed_tasks=3,
        failed_tasks=1,
        pass_rate=0.75,
        mean_score=0.775,
        total_tokens=TokenUsage(input=400, output=200, reasoning=100),
        total_cost_usd=0.02,
        total_duration_seconds=55.5,
        latency_p50_seconds=13.75,
        latency_p95_seconds=19.5,
        latency_p99_seconds=20.0,
        pass_at_k={1: 0.75, 2: 0.85, 3: 0.9},
        created_at="2024-01-15T10:45:00Z",
    )

    return EvalRunSummary(
        envelope=envelope,
        metrics=metrics,
        trials=trials,
    )


class TestHTMLReporter:
    def test_initialization(self) -> None:
        reporter = HTMLReporter()
        assert reporter.theme == "auto"
        assert reporter.include_charts is True

        reporter_dark = HTMLReporter(theme="dark")
        assert reporter_dark.theme == "dark"

        reporter_no_charts = HTMLReporter(include_charts=False)
        assert reporter_no_charts.include_charts is False

    def test_generate_suite_report(self) -> None:
        reporter = HTMLReporter(theme="light")
        summary = create_sample_summary()

        html = reporter.generate_suite_report(summary)

        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "Ash Hawk" in html
        assert summary.envelope.run_id in html
        assert summary.envelope.suite_id in html
        assert "75.0%" in html

        assert "chart.js" in html.lower()

    def test_generate_suite_report_without_charts(self) -> None:
        reporter = HTMLReporter(include_charts=False)
        summary = create_sample_summary()

        html = reporter.generate_suite_report(summary)

        assert "Chart.js" not in html
        assert "<canvas" not in html

    def test_generate_suite_report_to_file(self) -> None:
        reporter = HTMLReporter()
        summary = create_sample_summary()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            temp_path = Path(f.name)

        try:
            html = reporter.generate_suite_report(summary, output_path=temp_path)

            assert temp_path.exists()
            file_content = temp_path.read_text()
            assert file_content == html
            assert "<!DOCTYPE html>" in file_content
        finally:
            temp_path.unlink()

    def test_generate_task_report(self) -> None:
        reporter = HTMLReporter()

        task = EvalTask(
            id="task-001",
            description="Test task description",
            input="Write a hello world program",
            grader_specs=[
                GraderSpec(grader_type="string_match", config={"expected": "hello world"})
            ],
        )

        trials = [
            create_sample_trial("trial-001", "task-001", True, 1.0),
            create_sample_trial("trial-002", "task-001", False, 0.5),
        ]

        html = reporter.generate_task_report(task, trials)

        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert task.id in html
        assert "Trial Comparison" in html or "Trials" in html

    def test_generate_trial_report(self) -> None:
        reporter = HTMLReporter()

        trial = create_sample_trial("trial-001", "task-001", True, 1.0)
        task = EvalTask(
            id="task-001",
            description="Test task",
            input="Test input",
            grader_specs=[],
        )

        html = reporter.generate_trial_report(trial, task=task)

        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert trial.id in html
        assert "Token Usage" in html or "Grader Results" in html

    def test_generate_from_summary(self) -> None:
        reporter = HTMLReporter()
        summary = create_sample_summary()

        html = reporter.generate_from_summary(summary)

        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html

    def test_format_duration(self) -> None:
        assert HTMLReporter._format_duration(None) == "N/A"
        assert HTMLReporter._format_duration(0.5) == "0.5s"
        assert HTMLReporter._format_duration(30.0) == "30.0s"
        assert HTMLReporter._format_duration(90.0) == "1m 30s"
        assert HTMLReporter._format_duration(3661.0) == "1h 1m"

    def test_format_tokens(self) -> None:
        assert HTMLReporter._format_tokens(None) == "N/A"
        assert HTMLReporter._format_tokens(500) == "500"
        assert HTMLReporter._format_tokens(1500) == "1.5K"
        assert HTMLReporter._format_tokens(1500000) == "1.5M"

    def test_format_cost(self) -> None:
        assert HTMLReporter._format_cost(None) == "N/A"
        assert HTMLReporter._format_cost(0.005) == "$0.0050"
        assert HTMLReporter._format_cost(1.23) == "$1.23"
        assert HTMLReporter._format_cost(12.50) == "$12.50"

    def test_format_percent(self) -> None:
        assert HTMLReporter._format_percent(None) == "N/A"
        assert HTMLReporter._format_percent(0.75) == "75.0%"
        assert HTMLReporter._format_percent(1.0) == "100.0%"
        assert HTMLReporter._format_percent(0.0) == "0.0%"

    def test_theme_in_html(self) -> None:
        summary = create_sample_summary()

        light_reporter = HTMLReporter(theme="light")
        html_light = light_reporter.generate_suite_report(summary)
        assert 'data-theme="light"' in html_light

        dark_reporter = HTMLReporter(theme="dark")
        html_dark = dark_reporter.generate_suite_report(summary)
        assert 'data-theme="dark"' in html_dark


class TestConvenienceFunctions:
    def test_generate_html_report(self) -> None:
        summary = create_sample_summary()

        html = generate_html_report(summary, theme="dark")

        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert 'data-theme="dark"' in html

    def test_generate_html_report_to_file(self) -> None:
        summary = create_sample_summary()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            temp_path = Path(f.name)

        try:
            html = generate_html_report(summary, output_path=temp_path)

            assert temp_path.exists()
            assert temp_path.read_text() == html
        finally:
            temp_path.unlink()

    def test_generate_task_html(self) -> None:
        task = EvalTask(
            id="task-001",
            description="Test task",
            input="Test input",
            grader_specs=[],
        )
        trials = [create_sample_trial("trial-001", "task-001", True, 1.0)]

        html = generate_task_html(task, trials)

        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert task.id in html

    def test_generate_trial_html(self) -> None:
        trial = create_sample_trial("trial-001", "task-001", True, 1.0)

        html = generate_trial_html(trial)

        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert trial.id in html


class TestHTMLStructure:
    def test_suite_report_contains_envelope(self) -> None:
        reporter = HTMLReporter()
        summary = create_sample_summary()

        html = reporter.generate_suite_report(summary)

        assert "Run Envelope" in html or "Reproducibility" in html
        assert summary.envelope.run_id in html
        assert summary.envelope.model in html
        assert summary.envelope.provider in html

    def test_suite_report_contains_metrics(self) -> None:
        reporter = HTMLReporter()
        summary = create_sample_summary()

        html = reporter.generate_suite_report(summary)

        assert "Pass Rate" in html
        assert "Mean Score" in html
        assert "Total Cost" in html
        assert "Duration" in html

    def test_suite_report_contains_task_table(self) -> None:
        reporter = HTMLReporter()
        summary = create_sample_summary()

        html = reporter.generate_suite_report(summary)

        assert "Task Results" in html or "task" in html.lower()
        for trial in summary.trials:
            assert trial.task_id in html

    def test_trial_report_contains_transcript(self) -> None:
        reporter = HTMLReporter()
        trial = create_sample_trial("trial-001", "task-001", True, 1.0)

        html = reporter.generate_trial_report(trial)

        assert "Message Transcript" in html or "transcript" in html.lower() or "Token Usage" in html

    def test_dark_light_theme_variables(self) -> None:
        reporter = HTMLReporter()
        summary = create_sample_summary()

        html = reporter.generate_suite_report(summary)

        assert ":root" in html
        assert "[data-theme=" in html
        assert "--bg-primary" in html
        assert "--accent" in html
