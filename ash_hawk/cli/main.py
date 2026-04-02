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


from ash_hawk.auto_research.cli import auto_research
from ash_hawk.cli.run import run
from ash_hawk.cli.thin import thin
from ash_hawk.research.cli import research as research_cmd

cli.add_command(run)
cli.add_command(auto_research, name="improve")
cli.add_command(thin)
cli.add_command(research_cmd)
