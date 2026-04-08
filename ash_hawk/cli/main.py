import asyncio
import logging
from pathlib import Path

import click
from rich.console import Console
from rich.text import Text

from ash_hawk import __version__
from ash_hawk.config import get_config
from ash_hawk.context import setup_eval_logging

console = Console()


def _print_banner() -> None:
    banner = Text()
    banner.append("ash-hawk", style="bold cyan")
    banner.append(" v" + __version__, style="dim")
    console.print(banner)


@click.group(invoke_without_command=True)
@click.option("--version", "-v", is_flag=True, help="Show version and exit")
@click.pass_context
def cli(ctx: click.Context, version: bool) -> None:
    # Configure logging so ASH_HAWK_LOG_LEVEL takes effect for all commands
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
from ash_hawk.auto_research.cycle_runner import CycleConfig
from ash_hawk.cli.run import run
from ash_hawk.cli.thin import thin

cli.add_command(run)
cli.add_command(thin)


@click.command(help="Run iterative improvement cycle on an eval suite or directory")
@click.argument("suite_path")
@click.option("--agent", default="build", help="Agent name to evaluate")
@click.option("--target", default=1.0, type=float, help="Target pass rate (0.0-1.0)")
@click.option("--max-iterations", default=5, type=int, help="Maximum improvement iterations")
@click.option("--threshold", default=0.02, type=float, help="Min score delta to keep a change")
@click.option("--eval-repeats", default=3, type=int, help="Eval runs per iteration")
@click.option("--train-ratio", default=0.7, type=float, help="Train/holdout split ratio")
@click.option("--seed", default=42, type=int, help="Random seed for train/holdout split")
@click.option(
    "--promotion-threshold", default=3, type=int, help="Consecutive successes to promote a lesson"
)
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
    train_ratio: float,
    seed: int,
    promotion_threshold: int,
    output_dir: str | None,
) -> None:
    from ash_hawk.auto_research.cycle_runner import run_cycle

    try:
        resolution = resolve_agent_path(agent, Path("."))
    except AgentResolutionError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1)

    config = _build_cycle_config(
        target=target,
        max_iterations=max_iterations,
        threshold=threshold,
        eval_repeats=eval_repeats,
        train_ratio=train_ratio,
        seed=seed,
        promotion_threshold=promotion_threshold,
        output_dir=Path(output_dir) if output_dir else None,
    )

    result = asyncio.run(
        run_cycle(
            suite_path=suite_path,
            agent_name=resolution.name,
            agent_path=resolution.path,
            config=config,
        )
    )

    if not result.success:
        raise SystemExit(1)


def _build_cycle_config(
    target: float,
    max_iterations: int,
    threshold: float,
    eval_repeats: int,
    train_ratio: float,
    seed: int,
    promotion_threshold: int,
    output_dir: Path | None,
) -> CycleConfig:
    from ash_hawk.auto_research.knowledge_promotion import PromotionCriteria
    from ash_hawk.improvement.guardrails import GuardrailConfig

    return CycleConfig(
        max_iterations=max_iterations,
        target_pass_rate=target,
        score_threshold=threshold,
        eval_repeats=eval_repeats,
        train_ratio=train_ratio,
        seed=seed,
        guardrail_config=GuardrailConfig(
            max_consecutive_holdout_drops=3,
            max_reverts=max_iterations,
            plateau_window=5,
            plateau_threshold=0.02,
        ),
        convergence_window=5,
        convergence_variance_threshold=0.001,
        max_iterations_without_improvement=10,
        promotion_criteria=PromotionCriteria(
            min_improvement=0.05,
            min_consecutive_successes=promotion_threshold,
            max_regression=0.02,
        ),
        lessons_dir=Path(".ash-hawk/lessons"),
        output_dir=output_dir,
    )


cli.add_command(improve)
