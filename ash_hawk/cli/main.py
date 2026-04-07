import click
from rich.console import Console
from rich.text import Text

from ash_hawk import __version__

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
    if version:
        console.print(f"ash-hawk {__version__}")
        return
    if ctx.invoked_subcommand is None:
        _print_banner()
        console.print(ctx.get_help())


import asyncio

from ash_hawk.cli.run import run
from ash_hawk.cli.thin import thin

cli.add_command(run)
cli.add_command(thin)


@click.command(help="Run iterative improvement loop on an eval suite")
@click.argument("suite_path")
@click.option("--agent", default="build", help="Agent name to evaluate")
@click.option("--target", default=1.0, type=float, help="Target pass rate (0.0-1.0)")
@click.option("--max-iterations", default=5, type=int, help="Maximum improvement iterations")
@click.option("--trace-dir", default=None, type=click.Path(), help="Directory for trace output")
@click.option("--output-dir", default=None, type=click.Path(), help="Directory for patch output")
def improve(
    suite_path: str,
    agent: str,
    target: float,
    max_iterations: int,
    trace_dir: str | None,
    output_dir: str | None,
) -> None:
    from pathlib import Path

    from ash_hawk.improve import improve as _improve

    result = asyncio.run(
        _improve(
            suite_path=suite_path,
            agent_name=agent,
            target=target,
            max_iterations=max_iterations,
            trace_dir=Path(trace_dir) if trace_dir else None,
            output_dir=Path(output_dir) if output_dir else None,
        )
    )

    console.print("[bold]Improvement complete[/bold]")
    console.print(f"  Iterations: {result.iterations}")
    console.print(f"  Initial pass rate: {result.initial_pass_rate:.0%}")
    console.print(f"  Final pass rate:   {result.final_pass_rate:.0%}")
    console.print(f"  Patches proposed:  {len(result.patches_proposed)}")


cli.add_command(improve)
