import asyncio
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.table import Table

from ash_hawk.config import get_config
from ash_hawk.storage import FileStorage

console = Console()


@click.command(name="list")
@click.option(
    "--storage",
    "-s",
    type=click.Path(),
    default=None,
    help="Storage path (default from config)",
)
@click.option(
    "--suite",
    type=str,
    default=None,
    help="Filter by suite ID",
)
@click.option(
    "--runs",
    is_flag=True,
    help="List runs instead of suites",
)
def list_cmd(storage: str | None, suite: str | None, runs: bool) -> None:
    _list_items(storage, suite, runs)


def _list_items(storage_path: str | None, suite_id: str | None, show_runs: bool) -> None:
    asyncio.run(_list_items_async(storage_path, suite_id, show_runs))


async def _list_items_async(
    storage_path: str | None,
    suite_id: str | None,
    show_runs: bool,
) -> None:
    config = get_config()
    effective_storage_path = storage_path or str(config.storage_path_resolved())

    storage = FileStorage(base_path=effective_storage_path)

    if show_runs:
        await _list_runs(storage, suite_id)
    else:
        await _list_suites(storage, effective_storage_path)


async def _list_suites(storage: FileStorage, storage_path: str) -> None:
    suite_ids = await storage.list_suites()

    if not suite_ids:
        console.print(f"[dim]No suites found in {storage_path}[/dim]")
        return

    table = Table(
        title="Evaluation Suites",
        show_header=True,
        header_style="bold cyan",
        row_styles=["", "dim"],
    )
    table.add_column("ID", style="green")
    table.add_column("Name")
    table.add_column("Tasks", justify="right")
    table.add_column("Runs", justify="right")

    for sid in suite_ids:
        suite = await storage.load_suite(sid)
        if suite:
            runs = await storage.list_runs(sid)
            table.add_row(
                suite.id,
                suite.name,
                str(len(suite.tasks)),
                str(len(runs)),
            )

    console.print(table)


async def _list_runs(storage: FileStorage, suite_id: str | None) -> None:
    if suite_id:
        suite_ids = [suite_id]
    else:
        suite_ids = await storage.list_suites()

    if not suite_ids:
        console.print("[dim]No suites found[/dim]")
        return

    table = Table(
        title="Evaluation Runs",
        show_header=True,
        header_style="bold cyan",
        row_styles=["", "dim"],
    )
    table.add_column("Run ID", style="green")
    table.add_column("Suite")
    table.add_column("Agent")
    table.add_column("Model")
    table.add_column("Created")

    for sid in suite_ids:
        run_ids = await storage.list_runs(sid)
        for run_id in run_ids:
            envelope = await storage.load_run_envelope(sid, run_id)
            if envelope:
                table.add_row(
                    run_id,
                    sid,
                    envelope.agent_name,
                    envelope.model,
                    envelope.created_at[:19] if envelope.created_at else "N/A",
                )

    if table.row_count == 0:
        console.print("[dim]No runs found[/dim]")
    else:
        console.print(table)
