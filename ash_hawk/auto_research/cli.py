"""CLI for auto-research improvement cycle."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console

from ash_hawk.auto_research.cycle_runner import CycleRunner
from ash_hawk.auto_research.discovery import discover_repo_config, filter_targets_by_type
from ash_hawk.auto_research.types import ImprovementType

console = Console()


@click.group(name="auto-research")
def auto_research() -> None:
    """Auto-research improvement cycle commands."""
    pass


@auto_research.command(name="run")
@click.option(
    "--scenarios",
    "-s",
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Scenario files/directories (auto-discovered if not specified)",
)
@click.option(
    "--target",
    "-t",
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Improvement targets (auto-discovered if not specified)",
)
@click.option(
    "--target-type",
    multiple=True,
    type=click.Choice([t.value for t in ImprovementType], case_sensitive=False),
    help="Filter improvement targets by type (skills, tools, policies, agents). Can specify multiple.",
)
@click.option(
    "--iterations",
    "-i",
    default=100,
    type=int,
    help="Maximum iterations (default: 100)",
)
@click.option(
    "--improvement-threshold",
    default=0.02,
    type=float,
    help="Minimum improvement to keep a change (default: 0.02)",
)
@click.option(
    "--promotion-threshold",
    default=3,
    type=int,
    help="Consecutive improvements before promoting (default: 3)",
)
def run(
    scenarios: tuple[Path, ...],
    target: tuple[Path, ...],
    target_type: tuple[str, ...],
    iterations: int,
    improvement_threshold: float,
    promotion_threshold: int,
) -> None:
    """Run auto-research improvement cycle.

    Auto-discovers all configuration from the repository.
    Auto-generates experiment ID for tracking.

    Examples:
        ash-hawk auto-research run
        ash-hawk auto-research run --target skills/delegation.md
        ash-hawk auto-research run --target-type skills
        ash-hawk auto-research run --target-type skills --target-type tools
    """
    repo_config = discover_repo_config()

    console.print("\n[cyan]Auto-Research Improvement Cycle[/cyan]\n")
    console.print("[dim]Experiment:[/dim] (auto-generated)")
    console.print(f"[dim]Agent:[/dim] {repo_config.agent_name or 'unknown'}")

    scenarios_discovered = len(repo_config.scenarios)
    targets_discovered = len(repo_config.improvement_targets)
    console.print(f"[dim]Scenarios:[/dim] {scenarios_discovered} discovered")
    console.print(f"[dim]Targets:[/dim] {targets_discovered} discovered")

    effective_scenarios = list(scenarios) if scenarios else repo_config.scenarios
    effective_targets = list(target) if target else repo_config.improvement_targets

    if target_type:
        type_filters = [ImprovementType(t.lower()) for t in target_type]
        effective_targets = filter_targets_by_type(effective_targets, type_filters)
        console.print(
            f"[dim]Filtered to:[/dim] {len(effective_targets)} {', '.join(target_type)} targets"
        )

    if not effective_scenarios:
        console.print("[red]No scenarios found. Run from repo root.[/red]")
        return

    if not effective_targets:
        console.print("[red]No improvement targets found.[/red]")
        return

    async def _run_cycle() -> None:
        runner = CycleRunner(
            repo_config=repo_config,
            targets=effective_targets,
            scenarios=effective_scenarios,
            max_iterations=iterations,
            improvement_threshold=improvement_threshold,
            promotion_threshold=promotion_threshold,
        )

        result = await runner.run()

        console.print()
        console.rule("[bold]Cycle Complete[/bold]")
        console.print(f"[bold]Status:[/bold] {result.status.value}")
        console.print(f"[bold]Iterations:[/bold] {result.total_iterations}")
        if result.improvement_delta > 0:
            console.print(f"[bold]Improvement:[/bold] {result.improvement_delta:+.3f}")
        else:
            console.print("[bold]No improvement[/bold]")

    asyncio.run(_run_cycle())


@auto_research.command(name="list")
@click.option(
    "--agent",
    "-a",
    default=None,
    help="Filter by agent",
)
def list_experiments(agent: str | None) -> None:
    """List experiments and results."""
    console.print("[dim]Listing experiments... (not yet implemented)[/dim]")


__all__ = ["auto_research", "run", "list_experiments"]
