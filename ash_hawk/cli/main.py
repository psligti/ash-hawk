# type-hygiene: skip-file
from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import click
from rich.console import Console
from rich.text import Text

from ash_hawk import __version__
from ash_hawk.config import get_config
from ash_hawk.context import setup_eval_logging

console = Console()


@contextmanager
def _suppress_console_logs() -> Iterator[None]:
    root_logger = logging.getLogger()
    original_levels = [(handler, handler.level) for handler in root_logger.handlers]
    try:
        for handler, _ in original_levels:
            handler.setLevel(logging.CRITICAL + 1)
        yield
    finally:
        for handler, level in original_levels:
            handler.setLevel(level)


def _print_banner() -> None:
    banner = Text()
    banner.append("ash-hawk", style="bold cyan")
    banner.append(" v" + __version__, style="dim")
    console.print(banner)


@click.group(invoke_without_command=True)
@click.option("--version", "-v", is_flag=True, help="Show version and exit")
@click.pass_context
def cli(ctx: click.Context, version: bool) -> None:
    level_str = get_config().log_level
    level = getattr(logging, level_str, logging.INFO)
    setup_eval_logging(level)

    if version:
        console.print(f"ash-hawk {__version__}")
        return
    if ctx.invoked_subcommand is None:
        _print_banner()
        console.print(ctx.get_help())


from ash_hawk.agents.agent_resolver import AgentResolutionError, resolve_agent_path
from ash_hawk.cli.run import run
from ash_hawk.cli.thin import thin

cli.add_command(run)
cli.add_command(thin)


@click.command(help="Run iterative improvement cycle on an eval suite or directory")
@click.argument("suite_path")
@click.option("--agent", default="build", help="Agent name to evaluate")
@click.option("--target", default=1.0, type=float, help="Target pass rate (0.0-1.0)")
@click.option("--max-iterations", default=5, type=int, help="Maximum improvement iterations")
@click.option(
    "--threshold",
    default=0.02,
    type=float,
    help="Min selected-score delta to keep a change",
)
@click.option(
    "--eval-repeats", default=1, type=int, help="Eval runs per iteration (baseline + hypothesis)"
)
@click.option(
    "--integrity-repeats",
    default=3,
    type=int,
    help="Higher-confidence validation repeats before keeping a mutation",
)
@click.option("--max-reverts", default=5, type=int, help="Max reverts before stopping")
@click.option(
    "--output-dir", default=None, type=click.Path(), help="Directory for output artifacts"
)
def improve(
    suite_path: str,
    agent: str,
    target: float,
    max_iterations: int,
    threshold: float,
    eval_repeats: int,
    integrity_repeats: int,
    max_reverts: int,
    output_dir: str | None,
) -> None:
    from ash_hawk.improve.loop import improve as _improve
    from ash_hawk.improve.stop_condition import StopConditionConfig

    try:
        resolution = resolve_agent_path(agent, Path("."))
    except AgentResolutionError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1)

    stop_config = StopConditionConfig(
        max_reverts=max_reverts,
    )

    console.rule("[bold]Improve Run[/bold]")
    console.print(f"[bold]Suite:[/bold] {suite_path}")
    console.print(f"[bold]Agent:[/bold] {resolution.name}")
    console.print(f"[bold]Agent source:[/bold] {resolution.path}")
    console.print(
        f"[bold]Validation:[/bold] baseline x{eval_repeats}, integrity x{integrity_repeats}, keep threshold {threshold:+.2%} on selected score"
    )
    console.print(f"[bold]Iteration cap:[/bold] {max_iterations}")
    console.print("[bold]Mutation mode:[/bold] disposable git worktree per hypothesis")
    if output_dir:
        console.print(f"[bold]Artifacts:[/bold] {Path(output_dir)}")
    console.print(
        "[dim]Console shows process steps only. Logger output is muted for this run.[/dim]"
    )
    console.print()

    with _suppress_console_logs():
        result = asyncio.run(
            _improve(
                suite_path=suite_path,
                agent_name=resolution.name,
                agent_path=resolution.path,
                target=target,
                max_iterations=max_iterations,
                score_threshold=threshold,
                eval_repeats=eval_repeats,
                integrity_repeats=integrity_repeats,
                stop_config=stop_config,
                output_dir=Path(output_dir) if output_dir else None,
                console=console,
            )
        )

    if not result.convergence_achieved and result.final_pass_rate < target:
        raise SystemExit(1)


cli.add_command(improve)
