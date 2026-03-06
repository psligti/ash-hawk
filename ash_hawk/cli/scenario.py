import asyncio
import json
import tempfile
from pathlib import Path

import click
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ash_hawk.config import get_config
from ash_hawk.scenario.loader import discover_scenarios, load_scenario
from ash_hawk.scenario.reporting import excerpt_trace, locate_trace_jsonl
from ash_hawk.scenario.runner import run_scenarios, run_scenarios_async
from ash_hawk.storage import FileStorage
from ash_hawk.types import EvalStatus

console = Console()


def _require_file_storage() -> Path:
    config = get_config()
    if config.storage_backend != "file":
        console.print(
            "[red]Error:[/red] Scenario CLI requires file storage backend. "
            "Set ASH_HAWK_STORAGE_BACKEND=file."
        )
        raise SystemExit(1)
    return config.storage_path_resolved()


def _collect_scenario_paths(path: str) -> list[Path]:
    try:
        scenario_paths = discover_scenarios(path)
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1)

    if not scenario_paths:
        console.print(f"[red]Error:[/red] No scenario files found in {path}")
        raise SystemExit(1)
    return scenario_paths


def _maybe_override_sut(
    scenario_paths: list[Path], sut: str
) -> tuple[list[str], tempfile.TemporaryDirectory | None]:
    adjusted_paths: list[str] = []
    temp_dir: tempfile.TemporaryDirectory | None = None

    for idx, scenario_path in enumerate(scenario_paths):
        scenario = load_scenario(scenario_path)
        if scenario.sut.adapter != sut:
            if temp_dir is None:
                temp_dir = tempfile.TemporaryDirectory()
            updated_sut = scenario.sut.model_copy(update={"adapter": sut})
            updated = scenario.model_copy(update={"sut": updated_sut})
            filename = f"{idx:03d}-{scenario_path.name}"
            dest = Path(temp_dir.name) / filename
            dest.write_text(
                yaml.safe_dump(updated.model_dump(), sort_keys=False),
                encoding="utf-8",
            )
            adjusted_paths.append(str(dest))
        else:
            adjusted_paths.append(str(scenario_path))

    return adjusted_paths, temp_dir


def _render_trial_rows(table: Table, summary) -> bool:
    any_failed = False
    for trial in summary.trials:
        result = trial.result
        if result is None:
            passed = False
            status = trial.status.value
            score = 0.0
            graders = "n/a"
        else:
            passed = result.aggregate_passed and result.outcome.status == EvalStatus.COMPLETED
            status = result.outcome.status.value
            score = result.aggregate_score
            if result.grader_results:
                graders = ", ".join(
                    f"{gr.grader_type}={gr.score:.2f} ({'pass' if gr.passed else 'fail'})"
                    for gr in result.grader_results
                )
            else:
                graders = "n/a"

        pass_label = "PASS" if passed else "FAIL"
        pass_style = "green" if passed else "red"
        table.add_row(
            trial.task_id,
            f"[{pass_style}]{pass_label}[/{pass_style}]",
            status,
            f"{score:.2f}",
            graders,
        )
        if not passed:
            any_failed = True

    return any_failed


@click.group(help="Scenario workflows.")
def scenario() -> None:
    pass


@scenario.command(help="Validate scenario YAML files.")
@click.argument("path", type=click.Path(exists=True))
def validate(path: str) -> None:
    _require_file_storage()
    scenario_paths = _collect_scenario_paths(path)

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Scenario")
    table.add_column("Status")
    table.add_column("Details")

    any_failed = False
    for scenario_path in scenario_paths:
        try:
            load_scenario(scenario_path)
        except (ValidationError, ValueError, FileNotFoundError) as exc:
            table.add_row(
                str(scenario_path),
                "[red]FAIL[/red]",
                str(exc),
            )
            any_failed = True
        else:
            table.add_row(str(scenario_path), "[green]PASS[/green]", "")

    console.print(table)
    if any_failed:
        raise SystemExit(1)


@scenario.command(help="Run scenario YAML files.")
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--sut",
    default="mock_adapter",
    show_default=True,
    help="Scenario adapter name to run.",
)
@click.option("--record", is_flag=True, help="Record tool calls during scenario run.")
@click.option("--replay", is_flag=True, help="Replay tool calls from recorded traces.")
def run(path: str, sut: str, record: bool, replay: bool) -> None:
    storage_path = _require_file_storage()
    scenario_paths = _collect_scenario_paths(path)

    if record and replay:
        console.print("[red]Error:[/red] --record and --replay are mutually exclusive.")
        raise SystemExit(1)

    tooling_mode = "record" if record else "replay" if replay else "mock"

    temp_dir: tempfile.TemporaryDirectory | None = None
    try:
        adjusted_paths, temp_dir = _maybe_override_sut(scenario_paths, sut)
        if tooling_mode == "mock":
            summary = run_scenarios(adjusted_paths)
        else:
            summary = asyncio.run(run_scenarios_async(adjusted_paths, tooling_mode=tooling_mode))
    except (ValidationError, ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    console.print(f"[dim]Run ID: {summary.envelope.run_id}[/dim]")
    console.print(f"[dim]Storage: {storage_path}[/dim]")
    console.print()

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Scenario")
    table.add_column("Passed")
    table.add_column("Status")
    table.add_column("Score", justify="right")
    table.add_column("Graders")

    any_failed = _render_trial_rows(table, summary)
    console.print(table)

    if any_failed:
        raise SystemExit(1)


async def _find_run_summary(storage: FileStorage, run_id: str):
    suite_ids = await storage.list_suites()
    for suite_id in suite_ids:
        run_ids = await storage.list_runs(suite_id)
        if run_id in run_ids:
            summary = await storage.load_summary(suite_id, run_id)
            if summary is not None:
                return summary
    return None


@scenario.command(help="Show a scenario run report.")
@click.option("--run", "run_id", required=True, help="Run ID to report on.")
@click.option(
    "--format",
    "format_",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format.",
)
def report(run_id: str, format_: str) -> None:
    _require_file_storage()
    asyncio.run(_report_async(run_id, format_))


async def _report_async(run_id: str, format_: str) -> None:
    config = get_config()
    storage_path = config.storage_path_resolved()
    storage = FileStorage(base_path=storage_path)

    summary = await _find_run_summary(storage, run_id)
    if summary is None:
        console.print(f"[red]Error:[/red] Run {run_id} not found in storage")
        raise SystemExit(1)

    if format_.lower() == "json":
        report_payload = {
            "run_id": summary.envelope.run_id,
            "suite_id": summary.envelope.suite_id,
            "created_at": summary.envelope.created_at,
            "trials": [],
        }
        for trial in summary.trials:
            result = trial.result
            passed = bool(result and result.aggregate_passed)
            graders = []
            if result:
                graders = [
                    {
                        "grader_type": gr.grader_type,
                        "score": gr.score,
                        "passed": gr.passed,
                    }
                    for gr in result.grader_results
                ]
            trace_excerpt = ""
            if result and not passed:
                trace_path = locate_trace_jsonl(
                    summary.envelope.suite_id, run_id, trial.id, storage_path
                )
                if trace_path is not None:
                    trace_excerpt = excerpt_trace(trace_path)

            report_payload["trials"].append(
                {
                    "trial_id": trial.id,
                    "scenario_id": trial.task_id,
                    "status": trial.status.value,
                    "passed": passed,
                    "score": result.aggregate_score if result else 0.0,
                    "graders": graders,
                    "trace_excerpt": trace_excerpt,
                }
            )

        console.print(json.dumps(report_payload, indent=2))
        return

    header = f"Scenario Run Report: {summary.envelope.run_id}"
    console.print(Panel(header, expand=False))
    console.print(f"[dim]Suite: {summary.envelope.suite_id}[/dim]")
    console.print(f"[dim]Created: {summary.envelope.created_at[:19]}[/dim]")
    console.print()

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Scenario")
    table.add_column("Passed")
    table.add_column("Status")
    table.add_column("Score", justify="right")
    table.add_column("Graders")

    any_failed = _render_trial_rows(table, summary)
    console.print(table)

    if any_failed:
        console.print()
        for trial in summary.trials:
            result = trial.result
            passed = bool(result and result.aggregate_passed)
            if passed:
                continue
            trace_path = locate_trace_jsonl(
                summary.envelope.suite_id, run_id, trial.id, storage_path
            )
            excerpt = ""
            if trace_path is not None:
                excerpt = excerpt_trace(trace_path)
            excerpt = excerpt or "No trace excerpt available."
            console.print(Panel(excerpt, title=f"Trace excerpt: {trial.task_id}", expand=False))
