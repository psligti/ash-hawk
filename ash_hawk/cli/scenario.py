import asyncio
import hashlib
import json
import tempfile
import uuid
from pathlib import Path
from typing import Literal

import click
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ash_hawk.config import get_config
from ash_hawk.scenario.loader import discover_scenarios, load_scenario
from ash_hawk.scenario.models import ScenarioV1
from ash_hawk.scenario.reporting import excerpt_trace, locate_trace_jsonl
from ash_hawk.scenario.runner import run_scenarios, run_scenarios_async
from ash_hawk.storage import FileStorage
from ash_hawk.types import EvalRunSummary, EvalStatus

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
    scenario_paths: list[Path],
    sut: str | None,
    policy_mode: str | None = None,
    model: str | None = None,
    policy: str | None = None,
) -> tuple[list[str], tempfile.TemporaryDirectory[str] | None]:
    adjusted_paths: list[str] = []
    temp_dir: tempfile.TemporaryDirectory[str] | None = None

    for idx, scenario_path in enumerate(scenario_paths):
        scenario = load_scenario(scenario_path)
        updated = _apply_sut_overrides(
            scenario,
            sut=sut,
            policy_mode=policy_mode,
            model=model,
            policy=policy,
        )
        if updated != scenario:
            if temp_dir is None:
                temp_dir = tempfile.TemporaryDirectory[str]()
            filename = f"{idx:03d}-{scenario_path.name}"
            dest = Path(temp_dir.name) / filename
            _write_scenario(dest, updated)
            adjusted_paths.append(str(dest))
        else:
            adjusted_paths.append(str(scenario_path))

    return adjusted_paths, temp_dir


def _override_sut_in_place(
    scenario_paths: list[Path],
    sut: str | None,
    policy_mode: str | None = None,
    model: str | None = None,
    policy: str | None = None,
) -> tuple[list[str], list[Path]]:
    adjusted_paths: list[str] = []
    temp_paths: list[Path] = []

    for scenario_path in scenario_paths:
        scenario = load_scenario(scenario_path)
        updated = _apply_sut_overrides(
            scenario,
            sut=sut,
            policy_mode=policy_mode,
            model=model,
            policy=policy,
        )
        if updated != scenario:
            filename = f"{scenario_path.stem}.sut-{uuid.uuid4().hex[:8]}.scenario.yaml"
            dest = scenario_path.parent / filename
            _write_scenario(dest, updated)
            adjusted_paths.append(str(dest))
            temp_paths.append(dest)
        else:
            adjusted_paths.append(str(scenario_path))

    return adjusted_paths, temp_paths


def _apply_sut_overrides(
    scenario: ScenarioV1,
    sut: str | None,
    policy_mode: str | None = None,
    model: str | None = None,
    policy: str | None = None,
) -> ScenarioV1:
    updates: dict[str, object] = {}
    if sut is not None and scenario.sut.adapter != sut:
        updates["adapter"] = sut

    config_updates: dict[str, object] = {}
    if policy_mode:
        config_updates["policy_mode"] = policy_mode
    if model:
        config_updates["model"] = model
    if policy:
        config_updates["policy"] = policy

    if config_updates:
        updated_config = dict(scenario.sut.config)
        updated_config.update(config_updates)
        updates["config"] = updated_config

    if not updates:
        return scenario

    updated_sut = scenario.sut.model_copy(update=updates)
    return scenario.model_copy(update={"sut": updated_sut})


def _write_scenario(path: Path, scenario: ScenarioV1) -> None:
    path.write_text(
        yaml.safe_dump(scenario.model_dump(), sort_keys=False),
        encoding="utf-8",
    )


def _parse_csv_option(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def _summary_hash(summary: EvalRunSummary) -> str:
    normalized_trials: list[dict[str, object]] = []
    for trial in summary.trials:
        result = trial.result
        graders: list[dict[str, object]] = []
        if result and result.grader_results:
            graders = [
                {
                    "grader_type": gr.grader_type,
                    "score": round(gr.score, 6),
                    "passed": gr.passed,
                }
                for gr in sorted(result.grader_results, key=lambda g: g.grader_type)
            ]
        normalized_trials.append(
            {
                "task_id": trial.task_id,
                "attempt": trial.attempt_number,
                "status": trial.status.value,
                "outcome": result.outcome.status.value if result else None,
                "failure_mode": (
                    result.outcome.failure_mode.value
                    if result and result.outcome.failure_mode is not None
                    else None
                ),
                "aggregate_score": round(result.aggregate_score, 6) if result else 0.0,
                "aggregate_passed": result.aggregate_passed if result else False,
                "graders": graders,
            }
        )
    normalized_trials.sort(key=lambda item: (item["task_id"], item["attempt"]))

    metrics = summary.metrics
    normalized_payload: dict[str, object] = {
        "suite_id": summary.envelope.suite_id,
        "metrics": {
            "total_tasks": metrics.total_tasks,
            "completed_tasks": metrics.completed_tasks,
            "passed_tasks": metrics.passed_tasks,
            "failed_tasks": metrics.failed_tasks,
            "pass_rate": round(metrics.pass_rate, 6),
            "mean_score": round(metrics.mean_score, 6),
        },
        "trials": normalized_trials,
    }
    payload = json.dumps(normalized_payload, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _scenario_paths_from_summary(summary: EvalRunSummary) -> list[str]:
    scenario_paths: list[str] = []
    for trial in summary.trials:
        snapshot = trial.input_snapshot
        if not isinstance(snapshot, dict):
            continue
        scenario_path = snapshot.get("scenario_path")
        if isinstance(scenario_path, str) and scenario_path.strip():
            scenario_paths.append(scenario_path)
    return sorted(set(scenario_paths))


def _average_cost_and_latency(summary: EvalRunSummary) -> tuple[float, float]:
    results = [trial.result for trial in summary.trials if trial.result is not None]
    if not results:
        return 0.0, 0.0
    total_cost = sum(result.transcript.cost_usd for result in results)
    total_latency = sum(result.transcript.duration_seconds for result in results)
    count = len(results)
    return total_cost / count, total_latency / count


def _render_trial_rows(table: Table, summary: EvalRunSummary) -> bool:
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
    default=None,
    help="Override scenario adapter (default: use adapter specified in scenario YAML).",
)
@click.option(
    "--policy-mode",
    type=click.Choice(["fsm", "rules", "react", "plan_execute", "router"], case_sensitive=False),
    default=None,
    help="Policy mode for SDK-backed adapters.",
)
@click.option("--record", is_flag=True, help="Record tool calls during scenario run.")
@click.option("--replay", is_flag=True, help="Replay tool calls from recorded traces.")
def run(path: str, sut: str, policy_mode: str | None, record: bool, replay: bool) -> None:
    storage_path = _require_file_storage()
    scenario_paths = _collect_scenario_paths(path)

    if record and replay:
        console.print("[red]Error:[/red] --record and --replay are mutually exclusive.")
        raise SystemExit(1)

    tooling_mode: Literal["mock", "record", "replay"]
    if record:
        tooling_mode = "record"
    elif replay:
        tooling_mode = "replay"
    else:
        tooling_mode = "mock"

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        adjusted_paths, temp_dir = _maybe_override_sut(
            scenario_paths,
            sut,
            policy_mode=policy_mode,
        )
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


@scenario.command(help="Run a matrix of scenarios across policies and models.")
@click.argument("path", type=click.Path(exists=True))
@click.option("--sut", required=True, help="Scenario adapter name to run.")
@click.option("--policies", required=True, help="Comma-separated policy names.")
@click.option("--models", required=True, help="Comma-separated model identifiers.")
@click.option(
    "--policy-mode",
    type=click.Choice(["fsm", "rules", "react", "plan_execute", "router"], case_sensitive=False),
    default=None,
    help="Policy mode for SDK-backed adapters.",
)
def matrix(
    path: str,
    sut: str,
    policies: str,
    models: str,
    policy_mode: str | None,
) -> None:
    storage_path = _require_file_storage()
    scenario_paths = _collect_scenario_paths(path)
    policy_list = _parse_csv_option(policies)
    model_list = _parse_csv_option(models)

    if not policy_list:
        console.print("[red]Error:[/red] --policies must include at least one value")
        raise SystemExit(1)
    if not model_list:
        console.print("[red]Error:[/red] --models must include at least one value")
        raise SystemExit(1)

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Policy")
    table.add_column("Model")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Avg Cost", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Run ID")

    any_failed = False
    for policy in policy_list:
        for model in model_list:
            temp_dir: tempfile.TemporaryDirectory[str] | None = None
            try:
                adjusted_paths, temp_dir = _maybe_override_sut(
                    scenario_paths,
                    sut,
                    policy_mode=policy_mode,
                    model=model,
                    policy=policy,
                )
                summary = run_scenarios(adjusted_paths)
            except (ValidationError, ValueError, FileNotFoundError) as exc:
                console.print(f"[red]Error:[/red] {exc}")
                raise SystemExit(1)
            finally:
                if temp_dir is not None:
                    temp_dir.cleanup()

            avg_cost, avg_latency = _average_cost_and_latency(summary)
            pass_rate = summary.metrics.pass_rate
            table.add_row(
                policy,
                model,
                f"{pass_rate:.1%}",
                f"${avg_cost:.4f}",
                f"{avg_latency:.2f}s",
                summary.envelope.run_id,
            )
            if summary.metrics.failed_tasks > 0:
                any_failed = True

    console.print(table)
    console.print(f"[dim]Storage: {storage_path}[/dim]")
    if any_failed:
        raise SystemExit(1)


@scenario.command(help="Create a new scenario template.")
@click.option(
    "--type",
    "scenario_type",
    type=click.Choice(["coding_agent", "agentic_sdk"], case_sensitive=False),
    required=True,
    help="Scenario type to generate.",
)
@click.option("--name", required=True, help="Scenario name/slug.")
@click.option(
    "--dir",
    "base_dir",
    type=click.Path(),
    default="evals/scenarios",
    show_default=True,
    help="Base directory for new scenarios.",
)
def new(scenario_type: str, name: str, base_dir: str) -> None:
    root = Path(base_dir) / name
    scenario_path = root / f"{name}.scenario.yaml"
    if scenario_path.exists():
        console.print(f"[red]Error:[/red] Scenario already exists: {scenario_path}")
        raise SystemExit(1)

    root.mkdir(parents=True, exist_ok=True)
    tool_mocks_dir = root / "tool_mocks" / name
    tool_mocks_dir.mkdir(parents=True, exist_ok=True)

    if scenario_type == "coding_agent":
        scenario = {
            "schema_version": "v1",
            "id": name,
            "description": "Describe the coding task",
            "sut": {
                "type": "coding_agent",
                "adapter": "coding_agent_subprocess",
                "config": {
                    "command": "uv run pytest -q",
                    "verify_commands": [],
                },
            },
            "inputs": {
                "prompt": "Describe the change you want implemented.",
                "repo_fixture": "path/to/fixture",
            },
        }
    else:
        scenario = {
            "schema_version": "v1",
            "id": name,
            "description": "Describe the agentic SDK scenario",
            "sut": {
                "type": "agentic_sdk",
                "adapter": "sdk_dawn_kestrel",
                "config": {
                    "provider": "anthropic",
                    "model": "claude-3-5-sonnet-20241022",
                    "runner_kwargs": {},
                    "run_config": {},
                },
            },
            "inputs": {"prompt": "Say hello."},
        }

    scenario.update(
        {
            "tools": {
                "allowed_tools": [],
                "mocks": {},
                "fault_injection": {},
            },
            "budgets": {
                "max_steps": 5,
                "max_tool_calls": 10,
                "max_tokens": 1024,
                "max_time_seconds": 60.0,
            },
            "expectations": {
                "must_events": [],
                "must_not_events": [],
                "ordering_rules": [],
                "diff_assertions": [],
                "output_assertions": [],
            },
            "graders": [],
        }
    )

    scenario_path.write_text(yaml.safe_dump(scenario, sort_keys=False), encoding="utf-8")
    console.print(f"[green]Created:[/green] {scenario_path}")
    console.print(f"[dim]Tool mocks dir: {tool_mocks_dir}[/dim]")


@scenario.command(help="Record tool calls for scenario runs.")
@click.argument("path", type=click.Path(exists=True))
@click.option("--sut", required=True, help="Scenario adapter name to run.")
@click.option(
    "--policy-mode",
    type=click.Choice(["fsm", "rules", "react", "plan_execute", "router"], case_sensitive=False),
    default=None,
    help="Policy mode for SDK-backed adapters.",
)
def record(path: str, sut: str, policy_mode: str | None) -> None:
    storage_path = _require_file_storage()
    scenario_paths = _collect_scenario_paths(path)

    temp_paths: list[Path] = []
    try:
        adjusted_paths, temp_paths = _override_sut_in_place(
            scenario_paths,
            sut,
            policy_mode=policy_mode,
        )
        summary = asyncio.run(
            run_scenarios_async(adjusted_paths, tooling_mode="record", storage_path=storage_path)
        )
    except (ValidationError, ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1)
    finally:
        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)

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
    console.print()

    roots = sorted({Path(path).resolve().parent for path in adjusted_paths})
    for root in roots:
        console.print(f"[dim]Recorded tool mocks under: {root / 'tool_mocks'}[/dim]")

    if any_failed:
        raise SystemExit(1)


@scenario.command(help="Replay a recorded scenario run.")
@click.option("--run", "run_id", required=True, help="Run ID to replay.")
def replay(run_id: str) -> None:
    storage_path = _require_file_storage()
    storage = FileStorage(base_path=storage_path)
    summary = asyncio.run(_find_run_summary(storage, run_id))
    if summary is None:
        console.print(f"[red]Error:[/red] Run {run_id} not found in storage")
        raise SystemExit(1)

    scenario_paths = _scenario_paths_from_summary(summary)
    if not scenario_paths:
        console.print("[red]Error:[/red] No scenario paths found in run summary")
        raise SystemExit(1)

    original_hash = _summary_hash(summary)
    replay_summary = asyncio.run(
        run_scenarios_async(scenario_paths, tooling_mode="replay", storage_path=storage_path)
    )
    replay_hash = _summary_hash(replay_summary)

    console.print(f"[dim]Original summary hash: {original_hash}[/dim]")
    console.print(f"[dim]Replay summary hash:   {replay_hash}[/dim]")
    console.print(f"[dim]Replay Run ID: {replay_summary.envelope.run_id}[/dim]")
    console.print()

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Scenario")
    table.add_column("Passed")
    table.add_column("Status")
    table.add_column("Score", justify="right")
    table.add_column("Graders")

    any_failed = _render_trial_rows(table, replay_summary)
    console.print(table)

    if original_hash != replay_hash:
        console.print("[red]Error:[/red] Replay summary hash did not match original")
        raise SystemExit(1)
    if any_failed:
        raise SystemExit(1)


async def _find_run_summary(storage: FileStorage, run_id: str) -> EvalRunSummary | None:
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
        trials: list[dict[str, object]] = []
        report_payload: dict[str, object] = {
            "run_id": summary.envelope.run_id,
            "suite_id": summary.envelope.suite_id,
            "created_at": summary.envelope.created_at,
            "trials": trials,
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

            trials.append(
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
