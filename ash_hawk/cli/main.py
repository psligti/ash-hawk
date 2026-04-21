# type-hygiene: skip-file
from __future__ import annotations

import asyncio
import glob as globlib
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import click
import yaml
from rich.console import Console
from rich.text import Text

from ash_hawk import __version__
from ash_hawk.config import get_config
from ash_hawk.context import setup_eval_logging

console = Console()


def _expand_improve_targets(target: str) -> list[str]:
    path = Path(target)
    if path.is_dir():
        scenario_files = sorted(
            str(p)
            for p in path.rglob("*")
            if p.is_file() and p.name.endswith((".scenario.yaml", ".scenario.yml"))
        )
        if scenario_files:
            return scenario_files
        yaml_files = sorted(
            str(p) for p in path.rglob("*") if p.is_file() and p.suffix in {".yaml", ".yml"}
        )
        return yaml_files

    if any(ch in target for ch in "*?[]"):
        return sorted(
            str(Path(p)) for p in globlib.glob(target, recursive=True) if Path(p).is_file()
        )

    if path.is_file() and path.suffix in {".yaml", ".yml"}:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return [target]
        if isinstance(data, dict) and isinstance(data.get("scenarios"), list):
            expanded: list[str] = []
            for item in data["scenarios"]:
                if not isinstance(item, dict):
                    continue
                scenario = item.get("scenario")
                if not isinstance(scenario, str) or not scenario.strip():
                    continue
                expanded.append(str((path.parent / scenario).resolve()))
            if expanded:
                return expanded

    return [target]


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
from ash_hawk.cli.thin_runtime import thin_runtime

cli.add_command(run)
cli.add_command(thin)
cli.add_command(thin_runtime)


@click.command(help="Run iterative improvement cycle on an eval suite or directory")
@click.argument("suite_path")
@click.option("--agent", default="build", help="Agent name to evaluate")
@click.option("--target", default=1.0, type=float, help="Target score (0.0-1.0)")
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
    "--min-yield",
    default=0.0,
    type=float,
    help="Minimum kept/tested mutation yield before stop condition triggers (0 disables)",
)
@click.option(
    "--overall-timeout",
    default=None,
    type=float,
    help="Overall wall-clock timeout for the improve run in seconds",
)
@click.option(
    "--output-dir", default=None, type=click.Path(), help="Directory for output artifacts"
)
@click.option(
    "--set-pref",
    multiple=True,
    help="Persist personal memory preference as KEY=VALUE before running improve",
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
    min_yield: float,
    overall_timeout: float | None,
    output_dir: str | None,
    set_pref: tuple[str, ...],
) -> None:
    from ash_hawk.improve.loop import improve as _improve
    from ash_hawk.improve.memory_store import MemoryStore, PersonalPreference
    from ash_hawk.improve.stop_condition import StopConditionConfig

    try:
        resolution = resolve_agent_path(agent, Path("."))
    except AgentResolutionError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1)

    stop_config = StopConditionConfig(
        max_reverts=max_reverts,
        min_mutation_yield=min_yield,
    )
    if set_pref:
        memory_dir = Path(".ash-hawk/memory")
        store = MemoryStore(base_dir=memory_dir)
        existing = {pref.key: pref for pref in store.load_personal_preferences()}
        for item in set_pref:
            if "=" not in item:
                console.print(f"[red]Error:[/red] invalid preference '{item}', expected KEY=VALUE")
                raise SystemExit(1)
            key, value = item.split("=", 1)
            existing[key] = PersonalPreference(key=key, value=value)
        store.save_personal_preferences(list(existing.values()))
    suite_paths = _expand_improve_targets(suite_path)
    if not suite_paths:
        console.print(f"[red]Error:[/red] No eval/scenario files found for target: {suite_path}")
        raise SystemExit(1)

    console.rule("[bold]Improve Run[/bold]")
    console.print(f"[bold]Suite target:[/bold] {suite_path}")
    if len(suite_paths) == 1:
        console.print(f"[bold]Resolved path:[/bold] {suite_paths[0]}")
    else:
        console.print(f"[bold]Resolved paths:[/bold] {len(suite_paths)} files")
        console.print(f"[dim]{suite_paths[0]}[/dim]")
        if len(suite_paths) > 1:
            console.print(f"[dim]... +{len(suite_paths) - 1} more[/dim]")
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
                suite_path=suite_paths,
                agent_name=resolution.name,
                agent_path=resolution.path,
                target=target,
                max_iterations=max_iterations,
                score_threshold=threshold,
                eval_repeats=eval_repeats,
                integrity_repeats=integrity_repeats,
                stop_config=stop_config,
                overall_timeout_seconds=overall_timeout,
                output_dir=Path(output_dir) if output_dir else None,
                console=console,
            )
        )

    if not result.convergence_achieved and result.final_score < target:
        raise SystemExit(1)


cli.add_command(improve)


@click.command(help="Backfill 4-layer memory from historical improve runs")
@click.option(
    "--runs-dir",
    default=".ash-hawk/improve-runs",
    type=click.Path(path_type=Path),
    help="Directory containing historical improve run artifacts",
)
@click.option(
    "--memory-dir",
    default=".ash-hawk/memory",
    type=click.Path(path_type=Path),
    help="Destination directory for rebuilt memory artifacts",
)
@click.option(
    "--include-improve-cycle",
    is_flag=True,
    help="Also ingest flat improve-cycle history under the same .ash-hawk root",
)
@click.option("--force", is_flag=True, help="Re-run backfill even if marker file exists")
def backfill_memory(
    runs_dir: Path, memory_dir: Path, include_improve_cycle: bool, force: bool
) -> None:
    from ash_hawk.improve.loop import backfill_memory as _backfill_memory

    summary = _backfill_memory(
        runs_dir,
        memory_dir,
        force=force,
        include_improve_cycle=include_improve_cycle,
    )
    console.rule("[bold]Memory Backfill[/bold]")
    console.print(f"[bold]Runs dir:[/bold] {runs_dir}")
    console.print(f"[bold]Memory dir:[/bold] {memory_dir}")
    console.print(f"[bold]Imported episodes:[/bold] {summary['imported_episodes']}")
    console.print(f"[bold]Semantic rules:[/bold] {summary['semantic_rules']}")


cli.add_command(backfill_memory)
