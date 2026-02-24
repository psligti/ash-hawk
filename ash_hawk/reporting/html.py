"""HTML report generation using Jinja2 templates and Chart.js.

This module provides HTML report generation for evaluation results with
beautiful visualizations including charts via Chart.js CDN.

Key features:
- Suite overview with pass rate charts
- Per-task breakdown with detailed metrics
- Full transcript view for individual trials
- Dark/light theme support
- RunEnvelope included for reproducibility
- Single HTML file export (all assets embedded or CDN-linked)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ash_hawk.graders.aggregation import (
    aggregate_results,
    calculate_statistics,
    grader_summary,
    group_by_task,
)
from ash_hawk.types import (
    EvalRunSummary,
    EvalStatus,
    EvalSuite,
    EvalTask,
    EvalTrial,
    RunEnvelope,
    SuiteMetrics,
    TrialResult,
)

# Theme type
Theme = Literal["light", "dark", "auto"]

# Get templates directory
_TEMPLATES_DIR = Path(__file__).parent.parent / "templates" / "html"

# Create Jinja2 environment
_jinja_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)


class HTMLReporter:
    """Generate HTML reports for evaluation runs.

    This class generates beautiful, self-contained HTML reports with
    interactive Chart.js charts and full reproducibility metadata.

    Attributes:
        theme: Color theme for the report (light, dark, auto).
        include_charts: Whether to include Chart.js charts.
        chartjs_cdn: URL for Chart.js CDN.
    """

    CHARTJS_CDN = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"

    def __init__(
        self,
        theme: Theme = "auto",
        include_charts: bool = True,
        custom_templates_dir: Path | str | None = None,
    ) -> None:
        """Initialize HTML reporter.

        Args:
            theme: Color theme for reports (light, dark, auto).
            include_charts: Whether to include Chart.js charts.
            custom_templates_dir: Optional custom templates directory.
        """
        self.theme = theme
        self.include_charts = include_charts
        self.chartjs_cdn = self.CHARTJS_CDN

        # Use custom templates if provided
        if custom_templates_dir:
            self._env = Environment(
                loader=FileSystemLoader(str(custom_templates_dir)),
                autoescape=select_autoescape(["html", "xml"]),
            )
        else:
            self._env = _jinja_env

    def generate_suite_report(
        self,
        summary: EvalRunSummary,
        output_path: Path | str | None = None,
    ) -> str:
        """Generate a comprehensive suite report.

        Creates a single HTML file with:
        - Suite overview with pass rate donut chart
        - Task-by-task breakdown table
        - Per-task detail sections with bar charts
        - RunEnvelope for reproducibility
        - Interactive theme toggle

        Args:
            summary: Complete run summary with envelope, metrics, and trials.
            output_path: Optional path to write the HTML file.

        Returns:
            Generated HTML string.
        """
        # Prepare data for templates
        trials_by_task = group_by_task(summary.trials)
        task_summaries = self._prepare_task_summaries(summary.trials, trials_by_task)
        grader_stats = grader_summary(summary.trials)

        # Chart data
        pass_fail_data = {
            "passed": summary.metrics.passed_tasks,
            "failed": summary.metrics.failed_tasks,
            "pending": summary.metrics.total_tasks - summary.metrics.completed_tasks,
        }

        task_scores = [
            {
                "task_id": task_id,
                "score": trials[0].result.aggregate_score if trials and trials[0].result else 0,
                "passed": trials[0].result.aggregate_passed
                if trials and trials[0].result
                else False,
            }
            for task_id, trials in trials_by_task.items()
        ]

        latency_data = {
            "p50": summary.metrics.latency_p50_seconds,
            "p95": summary.metrics.latency_p95_seconds,
            "p99": summary.metrics.latency_p99_seconds,
        }

        token_data = {
            "input": summary.metrics.total_tokens.input,
            "output": summary.metrics.total_tokens.output,
            "reasoning": summary.metrics.total_tokens.reasoning,
        }

        template = self._env.get_template("suite_report.html")
        html = template.render(
            # Core data
            envelope=summary.envelope,
            metrics=summary.metrics,
            trials=summary.trials,
            trials_by_task=trials_by_task,
            task_summaries=task_summaries,
            grader_stats=grader_stats,
            # Chart data (as JSON for JS)
            pass_fail_data=json.dumps(pass_fail_data),
            task_scores=json.dumps(task_scores),
            latency_data=json.dumps(latency_data),
            token_data=json.dumps(token_data),
            pass_at_k=json.dumps(summary.metrics.pass_at_k),
            # Settings
            theme=self.theme,
            include_charts=self.include_charts,
            chartjs_cdn=self.chartjs_cdn,
            # Helpers
            now=datetime.now(timezone.utc).isoformat(),
            format_duration=self._format_duration,
            format_tokens=self._format_tokens,
            format_cost=self._format_cost,
            format_percent=self._format_percent,
        )

        if output_path:
            Path(output_path).write_text(html, encoding="utf-8")

        return html

    def generate_task_report(
        self,
        task: EvalTask,
        trials: list[EvalTrial],
        envelope: RunEnvelope | None = None,
        output_path: Path | str | None = None,
    ) -> str:
        """Generate a detailed report for a single task.

        Creates an HTML file with:
        - Task description and metadata
        - All trials for the task
        - Trial comparison charts
        - Grader breakdown

        Args:
            task: The task to report on.
            trials: All trials for this task.
            envelope: Optional run envelope for context.
            output_path: Optional path to write the HTML file.

        Returns:
            Generated HTML string.
        """
        # Calculate task-level statistics
        stats = calculate_statistics(trials)
        grader_stats = grader_summary(trials)

        # Trial comparison data
        trial_scores = [
            {
                "trial_id": t.id,
                "attempt": t.attempt_number,
                "score": t.result.aggregate_score if t.result else 0,
                "passed": t.result.aggregate_passed if t.result else False,
                "duration": t.result.transcript.duration_seconds if t.result else 0,
            }
            for t in trials
        ]

        template = self._env.get_template("task_detail.html")
        html = template.render(
            task=task,
            trials=trials,
            envelope=envelope,
            stats=stats,
            grader_stats=grader_stats,
            trial_scores=json.dumps(trial_scores),
            theme=self.theme,
            include_charts=self.include_charts,
            chartjs_cdn=self.chartjs_cdn,
            now=datetime.now(timezone.utc).isoformat(),
            format_duration=self._format_duration,
            format_tokens=self._format_tokens,
            format_cost=self._format_cost,
            format_percent=self._format_percent,
        )

        if output_path:
            Path(output_path).write_text(html, encoding="utf-8")

        return html

    def generate_trial_report(
        self,
        trial: EvalTrial,
        task: EvalTask | None = None,
        envelope: RunEnvelope | None = None,
        output_path: Path | str | None = None,
    ) -> str:
        """Generate a detailed report for a single trial.

        Creates an HTML file with:
        - Full transcript view
        - Message timeline
        - Tool call history
        - Token usage breakdown
        - Grader results with details

        Args:
            trial: The trial to report on.
            task: Optional parent task for context.
            envelope: Optional run envelope for context.
            output_path: Optional path to write the HTML file.

        Returns:
            Generated HTML string.
        """
        template = self._env.get_template("trial_detail.html")
        html = template.render(
            trial=trial,
            task=task,
            envelope=envelope,
            theme=self.theme,
            include_charts=self.include_charts,
            chartjs_cdn=self.chartjs_cdn,
            now=datetime.now(timezone.utc).isoformat(),
            format_duration=self._format_duration,
            format_tokens=self._format_tokens,
            format_cost=self._format_cost,
            format_percent=self._format_percent,
            json_dump=json.dumps,
        )

        if output_path:
            Path(output_path).write_text(html, encoding="utf-8")

        return html

    def generate_from_summary(
        self,
        summary: EvalRunSummary,
        output_path: Path | str | None = None,
    ) -> str:
        """Generate HTML report from an EvalRunSummary.

        This is an alias for generate_suite_report for API consistency.

        Args:
            summary: Complete run summary.
            output_path: Optional path to write the HTML file.

        Returns:
            Generated HTML string.
        """
        return self.generate_suite_report(summary, output_path)

    def _prepare_task_summaries(
        self,
        trials: list[EvalTrial],
        trials_by_task: dict[str, list[EvalTrial]],
    ) -> list[dict[str, Any]]:
        """Prepare per-task summary data for the report.

        Args:
            trials: All trials.
            trials_by_task: Trials grouped by task ID.

        Returns:
            List of task summary dicts.
        """
        summaries = []

        for task_id, task_trials in trials_by_task.items():
            completed = [t for t in task_trials if t.status == EvalStatus.COMPLETED]
            passed = [t for t in completed if t.result and t.result.aggregate_passed]

            scores = [t.result.aggregate_score for t in completed if t.result]

            latencies = [t.result.transcript.duration_seconds for t in completed if t.result]

            avg_score = sum(scores) / len(scores) if scores else 0.0
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

            summaries.append(
                {
                    "task_id": task_id,
                    "total_trials": len(task_trials),
                    "completed_trials": len(completed),
                    "passed_trials": len(passed),
                    "pass_rate": len(passed) / len(completed) if completed else 0.0,
                    "avg_score": avg_score,
                    "avg_latency_seconds": avg_latency,
                }
            )

        return summaries

    @staticmethod
    def _format_duration(seconds: float | None) -> str:
        """Format duration in human-readable form.

        Args:
            seconds: Duration in seconds.

        Returns:
            Formatted string like "2m 30s" or "45.2s".
        """
        if seconds is None:
            return "N/A"

        if seconds < 60:
            return f"{seconds:.1f}s"

        minutes = int(seconds // 60)
        secs = int(seconds % 60)

        if minutes < 60:
            return f"{minutes}m {secs}s"

        hours = minutes // 60
        mins = minutes % 60
        return f"{hours}h {mins}m"

    @staticmethod
    def _format_tokens(tokens: int | None) -> str:
        """Format token count in human-readable form.

        Args:
            tokens: Token count.

        Returns:
            Formatted string like "1.2K" or "15K".
        """
        if tokens is None:
            return "N/A"

        if tokens < 1000:
            return str(tokens)

        if tokens < 1_000_000:
            return f"{tokens / 1000:.1f}K"

        return f"{tokens / 1_000_000:.1f}M"

    @staticmethod
    def _format_cost(cost_usd: float | None) -> str:
        """Format cost in USD.

        Args:
            cost_usd: Cost in USD.

        Returns:
            Formatted string like "$0.05" or "$12.34".
        """
        if cost_usd is None:
            return "N/A"

        if cost_usd < 0.01:
            return f"${cost_usd:.4f}"

        return f"${cost_usd:.2f}"

    @staticmethod
    def _format_percent(value: float | None) -> str:
        """Format value as percentage.

        Args:
            value: Value between 0 and 1.

        Returns:
            Formatted string like "85%" or "N/A".
        """
        if value is None:
            return "N/A"

        return f"{value * 100:.1f}%"


def generate_html_report(
    summary: EvalRunSummary,
    output_path: Path | str | None = None,
    theme: Theme = "auto",
) -> str:
    """Convenience function to generate HTML report.

    Args:
        summary: Complete run summary.
        output_path: Optional path to write the HTML file.
        theme: Color theme (light, dark, auto).

    Returns:
        Generated HTML string.
    """
    reporter = HTMLReporter(theme=theme)
    return reporter.generate_suite_report(summary, output_path)


def generate_task_html(
    task: EvalTask,
    trials: list[EvalTrial],
    envelope: RunEnvelope | None = None,
    output_path: Path | str | None = None,
    theme: Theme = "auto",
) -> str:
    """Convenience function to generate task HTML report.

    Args:
        task: The task to report on.
        trials: All trials for this task.
        envelope: Optional run envelope.
        output_path: Optional path to write the HTML file.
        theme: Color theme (light, dark, auto).

    Returns:
        Generated HTML string.
    """
    reporter = HTMLReporter(theme=theme)
    return reporter.generate_task_report(task, trials, envelope, output_path)


def generate_trial_html(
    trial: EvalTrial,
    task: EvalTask | None = None,
    envelope: RunEnvelope | None = None,
    output_path: Path | str | None = None,
    theme: Theme = "auto",
) -> str:
    """Convenience function to generate trial HTML report.

    Args:
        trial: The trial to report on.
        task: Optional parent task.
        envelope: Optional run envelope.
        output_path: Optional path to write the HTML file.
        theme: Color theme (light, dark, auto).

    Returns:
        Generated HTML string.
    """
    reporter = HTMLReporter(theme=theme)
    return reporter.generate_trial_report(trial, task, envelope, output_path)


__all__ = [
    "HTMLReporter",
    "Theme",
    "generate_html_report",
    "generate_task_html",
    "generate_trial_html",
]
