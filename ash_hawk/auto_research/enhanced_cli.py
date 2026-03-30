"""Enhanced CLI commands for auto-research improvement cycle."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from ash_hawk.auto_research.enhanced_cycle_runner import run_enhanced_cycle
from ash_hawk.auto_research.types import (
    DEFAULT_LEVER_SPACE,
    EnhancedCycleConfig,
    EnhancedCycleResult,
)
from ash_hawk.config import get_config

console = Console()


def _serialize_result(result: EnhancedCycleResult) -> dict[str, Any]:
    """Serialize result to JSON-compatible dict."""
    data: dict[str, Any] = {
        "agent_name": result.agent_name,
        "status": result.status.value if hasattr(result, "status") else "unknown",
        "overall_improvement": result.overall_improvement,
        "total_iterations": result.total_iterations,
        "total_promoted": result.total_promoted,
        "converged": result.converged,
        "convergence_reason": result.convergence_reason,
        "total_duration_seconds": result.total_duration_seconds,
        "started_at": result.started_at.isoformat() if result.started_at else None,
        "completed_at": result.completed_at.isoformat() if result.completed_at else None,
        "error_message": result.error_message,
        "target_results": {},
        "promoted_lessons": [],
        "cleanup": None,
    }

    if result.cleanup_result:
        data["cleanup"] = {
            "cleaned_skills": result.cleanup_result.cleaned_skills,
            "kept_skills": result.cleanup_result.kept_skills,
            "errors": result.cleanup_result.errors,
            "duration_seconds": result.cleanup_result.duration_seconds,
        }

    return data


def _print_result(result: EnhancedCycleResult) -> None:
    """Print formatted result to console."""
    console.print()
    console.rule("[bold]Enhanced Cycle Results[/bold]")

    table = Table(title="Target Results")
    table.add_column("Target", style="cyan")
    table.add_column("Type", style="dim")
    table.add_column("Initial", justify="right")
    table.add_column("Final", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Iterations", justify="right")
    table.add_column("Status", style="green")

    for name, cycle_result in result.target_results.items():
        table.add_row(
            name,
            cycle_result.target_type.value if cycle_result.target_type else "unknown",
            f"{cycle_result.initial_score:.3f}",
            f"{cycle_result.final_score:.3f}",
            f"{cycle_result.improvement_delta:+.3f}",
            str(cycle_result.total_iterations),
            cycle_result.status.value,
        )

    console.print(table)

    if result.promoted_lessons:
        console.print()
        promo_table = Table(title="Promoted Lessons")
        promo_table.add_column("Target", style="cyan")
        promo_table.add_column("Improvement", style="dim")
        promo_table.add_column("Delta", justify="right")
        promo_table.add_column("Status", style="green")

        for lesson in result.promoted_lessons:
            promo_table.add_row(
                lesson.target_name,
                lesson.improvement_text[:50] + "..."
                if len(lesson.improvement_text) > 50
                else lesson.improvement_text,
                f"{lesson.score_delta:+.3f}",
                lesson.promotion_status.value,
            )

        console.print(promo_table)

    console.print()
    console.print(f"[bold]Overall Improvement:[/bold] {result.overall_improvement:+.3f}")
    console.print(f"[bold]Total Iterations:[/bold] {result.total_iterations}")
    console.print(f"[bold]Converged:[/bold] {result.converged}")
    console.print(f"[bold]Duration:[/bold] {result.total_duration_seconds:.1f}s")

    if result.cleanup_result:
        console.print()
        cleanup_table = Table(title="Skill Cleanup")
        cleanup_table.add_column("Cleaned", style="red")
        cleanup_table.add_column("Kept", style="green")
        cleanup_table.add_column("Errors", style="yellow")
        cleanup_table.add_row(
            str(len(result.cleanup_result.cleaned_skills)),
            str(len(result.cleanup_result.kept_skills)),
            str(len(result.cleanup_result.errors)),
        )
        console.print(cleanup_table)


@click.command(name="enhanced-run")
@click.option(
    "--scenario",
    "-s",
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Scenario file to evaluate (can specify multiple)",
)
@click.option(
    "--iterations",
    "-i",
    default=50,
    type=int,
    help="Iterations per target (default: 50)",
)
@click.option(
    "--threshold",
    default=0.02,
    type=float,
    help="Minimum improvement to keep change (default: 0.02)",
)
@click.option(
    "--parallel-targets",
    default=4,
    type=int,
    help="Max concurrent target cycles (default: 4)",
)
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    help="Project root directory",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output JSON file for results",
)
@click.option(
    "--multi-target/--single-target",
    default=True,
    help="Run multi-target improvement (default: enabled)",
)
@click.option(
    "--intent-analysis/--no-intent-analysis",
    default=True,
    help="Analyze intent from baseline transcripts (default: enabled)",
)
@click.option(
    "--knowledge-promotion/--no-knowledge-promotion",
    default=True,
    help="Promote validated lessons to knowledge base (default: enabled)",
)
@click.option(
    "--note-lark/--no-note-lark",
    default=True,
    help="Enable note-lark integration (default: enabled)",
)
@click.option(
    "--lever-search/--no-lever-search",
    default=False,
    help="Explore lever configuration space (default: disabled)",
)
@click.option(
    "--cleanup/--no-cleanup",
    default=True,
    help="Clean up low-value skills after cycle (default: enabled)",
)
@click.option(
    "--project-name",
    default="ash-hawk",
    help="Project name for knowledge promotion",
)
def enhanced_run(
    scenario: tuple[Path, ...],
    iterations: int,
    threshold: float,
    parallel_targets: int,
    project: Path | None,
    output: Path | None,
    multi_target: bool,
    intent_analysis: bool,
    knowledge_promotion: bool,
    note_lark: bool,
    lever_search: bool,
    cleanup: bool,
    project_name: str,
) -> None:
    """Run enhanced auto-research improvement cycle.

    Examples:
        ash-hawk auto-research enhanced-run -s evals/scenarios/*.yaml
        ash-hawk auto-research enhanced-run -s scenarios/*.yaml --iterations 100
        ash-hawk auto-research enhanced-run -s scenarios/*.yaml --no-knowledge-promotion
    """
    config = get_config()

    scenarios = list(scenario)
    if not scenarios:
        console.print("[red]No scenarios specified. Use -s to provide scenario files.[/red]")
        raise SystemExit(1)

    project_root = project or Path.cwd()

    enhanced_config = EnhancedCycleConfig(
        enable_multi_target=multi_target,
        max_parallel_targets=parallel_targets,
        enable_lever_search=lever_search,
        lever_space=dict(DEFAULT_LEVER_SPACE) if lever_search else None,
        enable_intent_analysis=intent_analysis,
        enable_knowledge_promotion=knowledge_promotion,
        enable_skill_cleanup=cleanup,
        note_lark_enabled=note_lark,
        iterations_per_target=iterations,
        improvement_threshold=threshold,
        project_name=project_name,
    )

    async def _run() -> EnhancedCycleResult:
        return await run_enhanced_cycle(
            scenarios=scenarios,
            config=enhanced_config,
            project_root=project_root,
        )

    result = asyncio.run(_run())

    _print_result(result)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        data = _serialize_result(result)
        output.write_text(json.dumps(data, indent=2))
        console.print(f"\n[cyan]Results saved to: {output}[/cyan]")


__all__ = ["enhanced_run"]
