from pathlib import Path

import click
import yaml
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


@cli.command()
@click.argument("path", type=click.Path(), default="suite.yaml")
@click.option(
    "--name",
    "-n",
    default="my-suite",
    help="Name for the suite",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Overwrite existing file",
)
def init(path: str, name: str, force: bool) -> None:
    dest = Path(path)
    if dest.exists() and not force:
        console.print(f"[red]Error:[/red] File {path} already exists. Use --force to overwrite.")
        raise SystemExit(1)

    sample_suite = {
        "id": name,
        "name": name.replace("-", " ").replace("_", " ").title(),
        "description": "Sample evaluation suite",
        "version": "1.0.0",
        "tasks": [
            {
                "id": "task-001",
                "description": "Simple greeting task",
                "input": "Hello! Please respond with a greeting.",
                "expected_output": "A greeting response",
                "grader_specs": [
                    {
                        "grader_type": "string_match",
                        "config": {"contains": ["hello", "hi"]},
                    }
                ],
            },
            {
                "id": "task-002",
                "description": "Math question task",
                "input": "What is 2 + 2?",
                "expected_output": "4",
                "grader_specs": [
                    {
                        "grader_type": "string_match",
                        "config": {"contains": ["4", "four"]},
                    }
                ],
            },
        ],
    }

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        yaml.dump(sample_suite, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]Created:[/green] {path}")
    console.print("[dim]Edit the file to customize your evaluation suite.[/dim]")


from ash_hawk.cli.calibrate import calibrate
from ash_hawk.cli.list import list_cmd
from ash_hawk.cli.report import report
from ash_hawk.cli.run import run
from ash_hawk.cli.scenario import scenario

cli.add_command(run)
cli.add_command(list_cmd, name="list")
cli.add_command(report)
cli.add_command(calibrate)
cli.add_command(scenario)
