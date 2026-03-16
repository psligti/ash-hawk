from __future__ import annotations

import json
from xml.sax.saxutils import escape

from rich.console import Console
from rich.table import Table

from ash_hawk.types import FastEvalSuiteResult


def render_fast_eval_table(console: Console, result: FastEvalSuiteResult) -> None:
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Fast Eval", style="cyan")
    table.add_column("Status")
    table.add_column("Score", justify="right")
    table.add_column("Grader")
    table.add_column("Duration", justify="right")

    for eval_result in result.results:
        status = "[green]PASS[/green]" if eval_result.passed else "[red]FAIL[/red]"
        table.add_row(
            eval_result.eval_id,
            status,
            f"{eval_result.score:.2f}",
            eval_result.grader_type,
            f"{eval_result.duration_seconds:.2f}s",
        )

    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_column("Metric", style="dim")
    summary.add_column("Value", justify="right")
    summary.add_row("Total", str(result.total_evals))
    summary.add_row("Passed", str(result.passed_evals))
    summary.add_row("Failed", str(result.failed_evals))
    summary.add_row("Pass Rate", f"{result.pass_rate:.1%}")
    summary.add_row("Mean Score", f"{result.mean_score:.2f}")
    summary.add_row("Duration", f"{result.total_duration_seconds:.2f}s")
    summary.add_row("Total Tokens", f"{result.total_tokens.total:,}")
    summary.add_row("Total Cost", f"${result.total_cost_usd:.4f}")

    console.print(table)
    console.print()
    console.print(summary)


def fast_eval_result_to_json(result: FastEvalSuiteResult) -> str:
    return json.dumps(result.model_dump(), indent=2)


def fast_eval_result_to_junit_xml(result: FastEvalSuiteResult) -> str:
    failures = sum(1 for r in result.results if not r.passed)
    tests = len(result.results)
    duration = result.total_duration_seconds

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<testsuite name="{escape(result.suite_id)}" tests="{tests}" '
            f'failures="{failures}" errors="0" time="{duration:.6f}">'
        ),
    ]

    for eval_result in result.results:
        case_name = escape(eval_result.eval_id)
        case_time = f"{eval_result.duration_seconds:.6f}"
        lines.append(f'  <testcase classname="fast_eval" name="{case_name}" time="{case_time}">')

        if not eval_result.passed:
            message = escape(eval_result.error_message or "fast eval failed")
            details = escape(json.dumps(eval_result.details))
            lines.append(f'    <failure message="{message}">{details}</failure>')

        lines.append("  </testcase>")

    lines.append("</testsuite>")
    return "\n".join(lines)


__all__ = [
    "render_fast_eval_table",
    "fast_eval_result_to_json",
    "fast_eval_result_to_junit_xml",
]
