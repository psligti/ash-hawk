"""Experiment comparison CLI command."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from ash_hawk.experiments.registry import ExperimentRegistry

console = Console()


@click.command("compare")
@click.option("--exp-a", required=True, help="Baseline experiment ID")
@click.option("--exp-b", help="Treatment experiment B ID")
@click.option("--exp-c", help="Treatment experiment C ID (optional)")
@click.option("--output", "-o", type=click.Path(), help="Output file for report")
@click.option("--format", type=click.Choice(["table", "markdown"]), default="table")
def compare_cmd(
    exp_a: str,
    exp_b: str | None,
    exp_c: str | None,
    output: Path | None,
    format: str,
) -> None:
    registry = ExperimentRegistry()
    exp_ids = [exp_a]
    if exp_b:
        exp_ids.append(exp_b)
    if exp_c:
        exp_ids.append(exp_c)

    experiments = []
    for exp_id in exp_ids:
        exp = registry.get(exp_id)
        if not exp:
            console.print(f"[red]Experiment '{exp_id}' not found[/red]")
            raise SystemExit(1)
        experiments.append(exp)

    baseline = experiments[0]
    treatments = experiments[1:] if len(experiments) > 1 else []

    results: dict[str, Any] = {
        "baseline": {
            "experiment_id": baseline.experiment_id,
            "strategy": str(baseline.strategy) if baseline.strategy else None,
            "target_agent": baseline.target_agent,
            "trial_count": baseline.trial_count,
            "lesson_count": baseline.lesson_count,
            "status": baseline.status,
        },
        "comparisons": [],
    }

    for treatment in treatments:
        results["comparisons"].append(
            {
                "treatment_id": treatment.experiment_id,
                "strategy": str(treatment.strategy) if treatment.strategy else None,
                "trial_count_delta": treatment.trial_count - baseline.trial_count,
                "lesson_count_delta": treatment.lesson_count - baseline.lesson_count,
                "status": treatment.status,
            }
        )

    if format == "table":
        _render_table(results, console)
    elif format == "markdown":
        markdown = _render_markdown(results)
        if output:
            output.write_text(markdown)
            console.print(f"[green]Report saved to {output}[/green]")
        else:
            console.print(markdown)


def _render_table(results: dict[str, Any], console: Console) -> None:
    baseline = results["baseline"]

    console.print("\n[bold]Experiment Comparison Report[/bold]\n")

    baseline_table = Table(title="Baseline Experiment")
    baseline_table.add_column("Property", style="cyan")
    baseline_table.add_column("Value", style="green")
    baseline_table.add_row("Experiment ID", baseline["experiment_id"])
    baseline_table.add_row("Strategy", baseline.get("strategy") or "N/A")
    baseline_table.add_row("Target Agent", baseline.get("target_agent") or "N/A")
    baseline_table.add_row("Trial Count", str(baseline["trial_count"]))
    baseline_table.add_row("Lesson Count", str(baseline["lesson_count"]))
    baseline_table.add_row("Status", baseline["status"])
    console.print(baseline_table)

    if results["comparisons"]:
        comparison_table = Table(title="Treatment Comparisons")
        comparison_table.add_column("Treatment ID", style="cyan")
        comparison_table.add_column("Strategy", style="yellow")
        comparison_table.add_column("Trial Delta", style="magenta")
        comparison_table.add_column("Lesson Delta", style="magenta")
        comparison_table.add_column("Status", style="green")

        for comp in results["comparisons"]:
            trial_delta = (
                f"+{comp['trial_count_delta']}"
                if comp["trial_count_delta"] >= 0
                else str(comp["trial_count_delta"])
            )
            lesson_delta = (
                f"+{comp['lesson_count_delta']}"
                if comp["lesson_count_delta"] >= 0
                else str(comp["lesson_count_delta"])
            )
            comparison_table.add_row(
                comp["treatment_id"],
                comp.get("strategy") or "N/A",
                trial_delta,
                lesson_delta,
                comp["status"],
            )
        console.print(comparison_table)


def _render_markdown(results: dict[str, Any]) -> str:
    baseline = results["baseline"]
    lines = [
        "# Experiment Comparison Report",
        "",
        "## Baseline Experiment",
        "",
        f"- **Experiment ID**: {baseline['experiment_id']}",
        f"- **Strategy**: {baseline.get('strategy') or 'N/A'}",
        f"- **Target Agent**: {baseline.get('target_agent') or 'N/A'}",
        f"- **Trial Count**: {baseline['trial_count']}",
        f"- **Lesson Count**: {baseline['lesson_count']}",
        f"- **Status**: {baseline['status']}",
        "",
    ]

    if results["comparisons"]:
        lines.append("## Treatment Comparisons")
        lines.append("")
        lines.append("| Treatment ID | Strategy | Trial Delta | Lesson Delta | Status |")
        lines.append("|-------------|----------|-------------|--------------|--------|")
        for comp in results["comparisons"]:
            trial_delta = (
                f"+{comp['trial_count_delta']}"
                if comp["trial_count_delta"] >= 0
                else str(comp["trial_count_delta"])
            )
            lesson_delta = (
                f"+{comp['lesson_count_delta']}"
                if comp["lesson_count_delta"] >= 0
                else str(comp["lesson_count_delta"])
            )
            lines.append(
                f"| {comp['treatment_id']} | {comp.get('strategy') or 'N/A'} | {trial_delta} | {lesson_delta} | {comp['status']} |"
            )

    return "\n".join(lines)
