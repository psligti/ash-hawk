import asyncio
from collections import defaultdict
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ash_hawk.config import get_config
from ash_hawk.graders.aggregation import detect_disagreements
from ash_hawk.metrics.statistics import (
    calculate_suite_metrics_detailed,
    compare_graders,
    wilson_confidence_interval,
)
from ash_hawk.storage import FileStorage
from ash_hawk.types import EvalStatus, GraderResult

console = Console()


def format_pass_rate_with_ci(pass_rate: float, successes: int, total: int) -> str:
    if total == 0:
        return f"{pass_rate:.1%}"
    ci = wilson_confidence_interval(successes, total, confidence_level=0.95)
    return f"{pass_rate:.1%} [{ci.lower:.0%}-{ci.upper:.0%}]"


@click.command()
@click.argument("run-id")
@click.option(
    "--suite",
    "-s",
    type=str,
    default=None,
    help="Suite ID (auto-detected if only one suite exists)",
)
@click.option(
    "--storage",
    type=click.Path(),
    default=None,
    help="Storage path (default from config)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed trial information",
)
def report(run_id: str, suite: str | None, storage: str | None, verbose: bool) -> None:
    _show_report(run_id, suite, storage, verbose)


def _show_report(
    run_id: str,
    suite_id: str | None,
    storage_path: str | None,
    verbose: bool,
) -> None:
    asyncio.run(_show_report_async(run_id, suite_id, storage_path, verbose))


async def _show_report_async(
    run_id: str,
    suite_id: str | None,
    storage_path: str | None,
    verbose: bool,
) -> None:
    config = get_config()
    effective_storage_path = storage_path or str(config.storage_path_resolved())

    storage = FileStorage(base_path=effective_storage_path)

    if suite_id is None:
        suite_ids = await storage.list_suites()
        if len(suite_ids) == 1:
            suite_id = suite_ids[0]
        elif len(suite_ids) == 0:
            console.print("[red]Error:[/red] No suites found in storage")
            raise SystemExit(1)
        else:
            console.print(f"[red]Error:[/red] Multiple suites found. Specify --suite")
            console.print(f"[dim]Available suites: {', '.join(suite_ids)}[/dim]")
            raise SystemExit(1)

    summary = await storage.load_summary(suite_id, run_id)
    if summary is None:
        console.print(f"[red]Error:[/red] Run {run_id} not found in suite {suite_id}")
        raise SystemExit(1)

    envelope = summary.envelope
    metrics = summary.metrics
    detailed_metrics = calculate_suite_metrics_detailed(
        trials=summary.trials,
        suite_id=envelope.suite_id,
        run_id=run_id,
    )

    header = Text()
    header.append("Run Report: ", style="bold")
    header.append(run_id, style="cyan")
    console.print(Panel(header, expand=False))

    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Key", style="dim")
    info_table.add_column("Value")

    info_table.add_row("Suite", f"{envelope.suite_id}")
    info_table.add_row("Agent", f"{envelope.agent_name}")
    info_table.add_row("Model", f"{envelope.provider}/{envelope.model}")
    info_table.add_row("Created", envelope.created_at[:19] if envelope.created_at else "N/A")
    info_table.add_row("Harness", f"v{envelope.harness_version}")

    console.print(info_table)
    console.print()

    results_table = Table(
        title="Summary",
        show_header=True,
        header_style="bold",
        box=None,
        padding=(0, 2),
    )
    results_table.add_column("Metric")
    results_table.add_column("Value", justify="right")

    pass_rate = metrics.pass_rate
    pass_color = "green" if pass_rate >= 0.8 else "yellow" if pass_rate >= 0.5 else "red"

    results_table.add_row("Total Tasks", str(metrics.total_tasks))
    results_table.add_row("Completed", str(metrics.completed_tasks))
    results_table.add_row("Passed", f"[{pass_color}]{metrics.passed_tasks}[/{pass_color}]")
    results_table.add_row("Failed", str(metrics.failed_tasks))

    if detailed_metrics.pass_rate_ci:
        ci = detailed_metrics.pass_rate_ci
        results_table.add_row(
            "Pass Rate (95% CI)",
            f"[{pass_color}]{pass_rate:.1%}[/{pass_color}] [{ci.lower:.0%}-{ci.upper:.0%}]",
        )
    else:
        results_table.add_row("Pass Rate", f"[{pass_color}]{pass_rate:.1%}[/{pass_color}]")

    results_table.add_row("Mean Score", f"{metrics.mean_score:.2f}")
    results_table.add_row("Duration", f"{metrics.total_duration_seconds:.2f}s")

    if metrics.latency_p50_seconds:
        results_table.add_row("Latency (p50)", f"{metrics.latency_p50_seconds:.2f}s")
    if metrics.latency_p95_seconds:
        results_table.add_row("Latency (p95)", f"{metrics.latency_p95_seconds:.2f}s")

    results_table.add_row("Total Tokens", f"{metrics.total_tokens.total:,}")
    results_table.add_row("Total Cost", f"${metrics.total_cost_usd:.4f}")

    console.print(results_table)

    if detailed_metrics.pass_at_k:
        console.print()
        pass_at_k_table = Table(
            title="Pass@k Metrics",
            show_header=True,
            header_style="bold",
            box=None,
            padding=(0, 2),
        )
        pass_at_k_table.add_column("Metric")
        pass_at_k_table.add_column("Value", justify="right")
        for k, value in sorted(detailed_metrics.pass_at_k.items()):
            pass_at_k_table.add_row(f"pass@{k}", f"{value:.1%}")
        console.print(pass_at_k_table)

    grader_results_by_type: dict[str, list[GraderResult]] = defaultdict(list)
    for trial in summary.trials:
        if trial.result:
            for gr in trial.result.grader_results:
                grader_results_by_type[gr.grader_type].append(gr)

    if len(grader_results_by_type) >= 2:
        console.print()
        graders_table = Table(
            title="Grader Comparison",
            show_header=True,
            header_style="bold",
            box=None,
            padding=(0, 2),
        )
        graders_table.add_column("Grader")
        graders_table.add_column("Pass Rate", justify="right")
        graders_table.add_column("n", justify="right")
        graders_table.add_column("Significance", justify="right")

        grader_types = sorted(grader_results_by_type.keys())
        first_grader_type = grader_types[0]
        first_results = grader_results_by_type[first_grader_type]

        for grader_type in grader_types:
            results = grader_results_by_type[grader_type]
            passes = sum(1 for r in results if r.passed)
            n = len(results)
            pass_rate = passes / n if n > 0 else 0.0

            if grader_type == first_grader_type:
                significance_str = "(baseline)"
            else:
                sig_result = compare_graders(first_results, results)
                if sig_result.significant:
                    significance_str = f"[yellow]p={sig_result.p_value:.3f}[/yellow]"
                else:
                    significance_str = f"p={sig_result.p_value:.3f}"

            graders_table.add_row(
                grader_type,
                f"{pass_rate:.1%}",
                str(n),
                significance_str,
            )

        console.print(graders_table)

    if detailed_metrics.tokens.total_input > 0 or detailed_metrics.tokens.total_output > 0:
        console.print()
        tokens_table = Table(
            title="Token Usage",
            show_header=True,
            header_style="bold",
            box=None,
            padding=(0, 2),
        )
        tokens_table.add_column("Type")
        tokens_table.add_column("Count", justify="right")
        tokens_table.add_row("Input Tokens", f"{detailed_metrics.tokens.total_input:,}")
        tokens_table.add_row("Output Tokens", f"{detailed_metrics.tokens.total_output:,}")
        if detailed_metrics.tokens.total_reasoning > 0:
            tokens_table.add_row("Reasoning Tokens", f"{detailed_metrics.tokens.total_reasoning:,}")
        tokens_table.add_row("Total", f"{detailed_metrics.tokens.total_tokens:,}")
        console.print(tokens_table)

    disagreement_report = detect_disagreements(summary.trials)

    if disagreement_report.flagged_trial_ids:
        console.print()
        review_lines = []
        for trial_id in disagreement_report.flagged_trial_ids:
            reason = disagreement_report.reasons.get(trial_id, "Unknown reason")
            review_lines.append(f"[yellow]{trial_id}[/yellow]: {reason}")

        review_panel = Panel(
            "\n".join(review_lines),
            title="Trials Needing Review",
            border_style="yellow",
        )
        console.print(review_panel)

    total_trials = len(summary.trials)
    flagged_count = len(disagreement_report.flagged_trial_ids)
    if total_trials > 0:
        console.print()
        high_conf_count = total_trials - flagged_count
        high_conf_pct = high_conf_count / total_trials * 100
        review_pct = flagged_count / total_trials * 100
        console.print(
            f"[bold]Confidence Summary:[/bold] "
            f"{high_conf_pct:.0f}% high confidence, "
            f"{review_pct:.0f}% need review"
        )

    if verbose and summary.trials:
        console.print()
        trials_table = Table(
            title="Trial Details",
            show_header=True,
            header_style="bold cyan",
        )
        trials_table.add_column("Task ID", style="green")
        trials_table.add_column("Status")
        trials_table.add_column("Score", justify="right")
        trials_table.add_column("Passed", justify="center")
        trials_table.add_column("Duration", justify="right")

        for trial in summary.trials:
            status_str = trial.status.value
            if trial.status == EvalStatus.COMPLETED:
                status_display = f"[green]{status_str}[/green]"
            elif trial.status == EvalStatus.ERROR:
                status_display = f"[red]{status_str}[/red]"
            else:
                status_display = f"[yellow]{status_str}[/yellow]"

            score = trial.result.aggregate_score if trial.result else 0.0
            passed = trial.result.aggregate_passed if trial.result else False
            passed_display = "[green]Yes[/green]" if passed else "[red]No[/red]"

            duration = (
                trial.result.transcript.duration_seconds
                if trial.result and trial.result.transcript
                else 0.0
            )

            trials_table.add_row(
                trial.task_id,
                status_display,
                f"{score:.2f}",
                passed_display,
                f"{duration:.2f}s",
            )

        console.print(trials_table)

    console.print()
    console.print(f"[dim]Location: {effective_storage_path}/{suite_id}/runs/{run_id}/[/dim]")
