import asyncio
import hashlib
import importlib
import importlib.util
import platform
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, cast

import click
import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from ash_hawk.config import get_config
from ash_hawk.execution import EvalRunner, FixtureResolver, TrialExecutor
from ash_hawk.execution.fast_eval import FastEvalRunner
from ash_hawk.reporting.fast_eval_report import render_fast_eval_table
from ash_hawk.storage import FileStorage
from ash_hawk.types import (
    EvalAgentConfig,
    EvalSuite,
    FastEvalSuite,
    RunEnvelope,
)

console = Console()


def _compute_hash(data: dict[str, Any]) -> str:
    import json

    content = json.dumps(data, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _create_run_envelope(suite: EvalSuite, agent_config: dict[str, Any]) -> RunEnvelope:
    from ash_hawk import __version__

    config = get_config()
    policy = config.default_tool_policy

    return RunEnvelope(
        run_id=f"run-{uuid.uuid4().hex[:8]}",
        suite_id=suite.id,
        suite_hash=_compute_hash(suite.model_dump()),
        harness_version=__version__,
        git_commit=None,
        agent_name=agent_config.get("agent_name", "unknown"),
        agent_version=agent_config.get("agent_version"),
        provider=agent_config.get("provider", "unknown"),
        model=agent_config.get("model", "unknown"),
        model_params=agent_config.get("model_params", {}),
        seed=agent_config.get("seed"),
        tool_policy_hash=_compute_hash(policy.model_dump()),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        os_info=platform.platform(),
        config_snapshot={},
        created_at=datetime.now(UTC).isoformat(),
    )


@click.command()
@click.argument("suite", type=click.Path(exists=True))
@click.option(
    "--parallelism",
    "-p",
    type=int,
    default=None,
    help="Number of parallel workers (default from config)",
)
@click.option(
    "--storage",
    "-s",
    type=click.Path(),
    default=None,
    help="Storage path for results (default from config)",
)
@click.option(
    "--agent",
    "-a",
    type=str,
    default=None,
    help="Agent name/identifier (overrides suite agent)",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default=None,
    help="Model identifier override (optional)",
)
@click.option(
    "--provider",
    type=str,
    default=None,
    help="LLM provider override (optional)",
)
@click.option(
    "--agent-class",
    type=str,
    default=None,
    help="Runner class override (module:Class or module.Class)",
)
@click.option(
    "--agent-location",
    type=click.Path(),
    default=None,
    help="Optional file path for loading agent runner class",
)
@click.option(
    "--lessons",
    is_flag=True,
    default=False,
    help="Enable lesson injection from curated improvements",
)
@click.option(
    "--strategy",
    "-S",
    type=str,
    default=None,
    help="Filter lessons by strategy (e.g., policy-quality, tool-quality)",
)
@click.option(
    "--sub-strategy",
    "-ss",
    type=str,
    default=None,
    help="Filter lessons by sub-strategy (e.g., tool-efficiency, error-recovery)",
)
def run(
    suite: str,
    parallelism: int | None,
    storage: str | None,
    agent: str | None,
    model: str | None,
    provider: str | None,
    agent_class: str | None,
    agent_location: str | None,
    lessons: bool,
    strategy: str | None,
    sub_strategy: str | None,
) -> None:
    _run_suite(
        suite,
        parallelism,
        storage,
        agent,
        model,
        provider,
        agent_class,
        agent_location,
        lessons,
        strategy,
        sub_strategy,
    )


def _run_suite(
    suite_path: str,
    parallelism: int | None,
    storage_path: str | None,
    agent: str | None,
    model: str | None,
    provider: str | None,
    agent_class: str | None,
    agent_location: str | None,
    lessons: bool = False,
    strategy: str | None = None,
    sub_strategy: str | None = None,
) -> None:
    asyncio.run(
        _run_suite_async(
            suite_path,
            parallelism,
            storage_path,
            agent,
            model,
            provider,
            agent_class,
            agent_location,
            lessons,
            strategy,
            sub_strategy,
        )
    )


async def _run_suite_async(
    suite_path: str,
    parallelism: int | None,
    storage_path: str | None,
    agent: str | None,
    model: str | None,
    provider: str | None,
    agent_class: str | None,
    agent_location: str | None,
    lessons: bool = False,
    strategy: str | None = None,
    sub_strategy: str | None = None,
) -> None:
    suite_file = Path(suite_path)
    if not suite_file.exists():
        console.print(f"[red]Error:[/red] Suite file not found: {suite_path}")
        raise SystemExit(1)

    with open(suite_file) as f:
        suite_data = yaml.safe_load(f)

    # Apply conftest.yaml inheritance if available
    from ash_hawk.configs.conftest import ConftestLoader, apply_conftest_to_suite

    loader = ConftestLoader(search_root=suite_file.parent)
    conftest = loader.load_for_suite(suite_file)
    suite_data = apply_conftest_to_suite(suite_data or {}, conftest)

    if "evals" in suite_data and "tasks" not in suite_data:
        try:
            fast_suite = FastEvalSuite.model_validate(suite_data)
        except Exception as e:
            console.print(f"[red]Error parsing fast eval suite:[/red] {e}")
            raise SystemExit(1)

        if provider is not None:
            fast_suite.provider = provider
        if model is not None:
            fast_suite.model = model
        if parallelism is not None:
            fast_suite.parallelism = parallelism

        console.print()
        console.print(f"[bold cyan]Fast Eval Suite:[/bold cyan] {fast_suite.name}")
        console.print(f"[dim]ID: {fast_suite.id} | Evals: {len(fast_suite.evals)}[/dim]")
        console.print()

        fast_runner = FastEvalRunner(suite=fast_suite, parallelism=parallelism)
        fast_result = await fast_runner.run_suite()
        render_fast_eval_table(console, fast_result)

        if fast_result.failed_evals > 0:
            raise SystemExit(1)
        return

    try:
        suite = EvalSuite.model_validate(suite_data)
    except Exception as e:
        console.print(f"[red]Error parsing suite:[/red] {e}")
        raise SystemExit(1)

    console.print()
    console.print(f"[bold cyan]Suite:[/bold cyan] {suite.name}")
    console.print(f"[dim]ID: {suite.id} | Tasks: {len(suite.tasks)}[/dim]")
    console.print()

    config = get_config()
    effective_parallelism = parallelism or config.parallelism
    effective_storage_path = storage_path or str(config.storage_path_resolved())

    storage_backend = FileStorage(base_path=effective_storage_path)
    await storage_backend.save_suite(suite)

    try:
        agent_config = await _resolve_effective_agent_config(
            suite=suite,
            suite_file=suite_file,
            cli_agent=agent,
            cli_provider=provider,
            cli_model=model,
            cli_agent_class=agent_class,
            cli_agent_location=agent_location,
        )
    except ValueError as e:
        console.print(f"[red]Agent configuration error:[/red] {e}")
        raise SystemExit(1)

    envelope = _create_run_envelope(suite, agent_config)
    await storage_backend.save_run_envelope(suite.id, envelope)

    console.print(f"[dim]Run ID: {envelope.run_id}[/dim]")
    console.print(f"[dim]Storage: {effective_storage_path}[/dim]")
    console.print()

    policy = config.default_tool_policy
    fixture_resolver = FixtureResolver(suite_file, suite)

    agent_runner = _build_agent_runner(agent_config, suite_file)

    if hasattr(agent_runner, "set_lesson_injector"):
        from ash_hawk.services import DawnKestrelInjector, LessonInjector

        dk_injector = DawnKestrelInjector(project_root=suite_file.parent.resolve())
        agent_runner.set_lesson_injector(dk_injector)

        if lessons:
            db_injector = LessonInjector(
                strategy_filter=strategy,
                sub_strategy_filter=sub_strategy,
            )
            agent_runner.set_lesson_injector(db_injector)
            console.print("[dim]Lesson injection enabled (database)[/dim]")
            if strategy:
                console.print(f"[dim]Strategy filter: {strategy}[/dim]")
            if sub_strategy:
                console.print(f"[dim]Sub-strategy filter: {sub_strategy}[/dim]")
        else:
            console.print("[dim]Dawn-Kestrel file injection enabled[/dim]")

    trial_executor = TrialExecutor(
        storage_backend, policy, agent_runner=agent_runner, fixture_resolver=fixture_resolver
    )

    from ash_hawk.config import EvalConfig

    run_config = EvalConfig(
        parallelism=effective_parallelism,
        default_timeout_seconds=config.default_timeout_seconds,
        storage_backend=config.storage_backend,
        storage_path=config.storage_path,
        log_level=config.log_level,
        default_tool_policy=config.default_tool_policy,
    )
    runner = EvalRunner(run_config, storage_backend, trial_executor)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("[cyan]Running trials...", total=len(suite.tasks))

        summary = await runner.run_suite(
            suite=suite,
            agent_config=agent_config,
            run_envelope=envelope,
        )
        progress.update(task_id, completed=len(suite.tasks))

    console.print()

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    passed = summary.metrics.passed_tasks
    total = summary.metrics.total_tasks
    pass_rate = summary.metrics.pass_rate

    table.add_row("Total Tasks", str(total))
    table.add_row("Completed", str(summary.metrics.completed_tasks))
    table.add_row(
        "Passed",
        f"[green]{passed}[/green]" if pass_rate >= 0.8 else f"[yellow]{passed}[/yellow]",
    )
    table.add_row(
        "Pass Rate",
        f"[green]{pass_rate:.1%}[/green]"
        if pass_rate >= 0.8
        else f"[yellow]{pass_rate:.1%}[/yellow]",
    )
    table.add_row("Mean Score", f"{summary.metrics.mean_score:.2f}")
    table.add_row("Duration", f"{summary.metrics.total_duration_seconds:.2f}s")
    table.add_row("Total Tokens", f"{summary.metrics.total_tokens.total:,}")
    table.add_row("Total Cost", f"${summary.metrics.total_cost_usd:.4f}")

    console.print(table)
    console.print()
    console.print(
        f"[dim]Results saved to: {effective_storage_path}/{suite.id}/runs/{envelope.run_id}/[/dim]"
    )


def _pick_non_empty(primary: str | None, secondary: str | None) -> str | None:
    if primary is not None:
        value = primary.strip()
        if value != "":
            return value
    if secondary is not None:
        value = secondary.strip()
        if value != "":
            return value
    return None


async def _resolve_effective_agent_config(
    suite: EvalSuite,
    suite_file: Path,
    cli_agent: str | None,
    cli_provider: str | None,
    cli_model: str | None,
    cli_agent_class: str | None,
    cli_agent_location: str | None,
) -> dict[str, Any]:
    del suite_file

    suite_agent = suite.agent or EvalAgentConfig()

    resolved_name = _pick_non_empty(cli_agent, suite_agent.name)
    resolved_provider = _pick_non_empty(cli_provider, suite_agent.provider)
    resolved_model = _pick_non_empty(cli_model, suite_agent.model)
    resolved_class = _pick_non_empty(cli_agent_class, suite_agent.class_name)
    resolved_location = _pick_non_empty(cli_agent_location, suite_agent.location)

    if resolved_name is None and resolved_class is None and resolved_location is None:
        raise ValueError("No agent configured. Pass --agent or set suite 'agent' in YAML.")

    agents_registry_module = importlib.import_module("dawn_kestrel.agents.registry")
    settings_module = importlib.import_module("dawn_kestrel.core.settings")
    create_agent_registry = cast(
        Callable[[], Any], getattr(agents_registry_module, "create_agent_registry")
    )
    get_settings = cast(Callable[[], Any], getattr(settings_module, "get_settings"))

    settings = get_settings()
    default_account = settings.get_default_account()

    registry_model: dict[str, Any] | None = None
    if resolved_name is not None:
        registry = create_agent_registry()
        registry_agent = await registry.get_agent(resolved_name)
        if registry_agent is None and resolved_class is None and resolved_location is None:
            raise ValueError(f"Agent not found in dawn-kestrel registry: {resolved_name}")
        if registry_agent is not None:
            raw_model = getattr(registry_agent, "model", None)
            if isinstance(raw_model, dict):
                registry_model = cast(dict[str, Any], raw_model)

    if resolved_provider is None and registry_model is not None:
        provider_candidate = registry_model.get("provider") or registry_model.get("provider_id")
        if isinstance(provider_candidate, str) and provider_candidate.strip() != "":
            resolved_provider = provider_candidate.strip()

    if resolved_provider is None:
        if default_account is not None:
            resolved_provider = str(default_account.provider_id.value)
        else:
            resolved_provider = str(settings.get_default_provider().value)

    if resolved_model is None and registry_model is not None:
        model_candidate = registry_model.get("model")
        if isinstance(model_candidate, str) and model_candidate.strip() != "":
            resolved_model = model_candidate.strip()

    if resolved_model is None:
        default_model_value = settings.get_default_model(resolved_provider)
        if isinstance(default_model_value, str) and default_model_value.strip() != "":
            resolved_model = default_model_value.strip()

    if resolved_name is not None:
        resolved_identifier = resolved_name
    elif resolved_class is not None:
        resolved_identifier = resolved_class
    else:
        resolved_identifier = cast(str, resolved_location)

    return {
        "agent_name": resolved_identifier,
        "agent_lookup_name": resolved_name,
        "provider": resolved_provider,
        "model": resolved_model,
        "agent_class": resolved_class,
        "agent_location": resolved_location,
        "agent_kwargs": dict(suite_agent.kwargs),
        "mcp_servers": [server.model_dump(exclude_none=True) for server in suite_agent.mcp_servers],
    }


def _load_runner_type(
    class_name: str | None,
    location: str | None,
    suite_file: Path,
) -> type[Any]:
    if location is not None:
        if class_name is None:
            raise ValueError("agent.class is required when agent.location is provided")
        location_path = Path(location).expanduser()
        if not location_path.is_absolute():
            location_path = (suite_file.parent / location_path).resolve()
        if not location_path.exists():
            raise ValueError(f"Agent location not found: {location_path}")

        module_name = f"ash_hawk_agent_runner_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, location_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load module from: {location_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class_key = class_name.split(":")[-1].split(".")[-1]
        loaded_type = getattr(module, class_key, None)
        if not isinstance(loaded_type, type):
            raise ValueError(f"Runner class '{class_key}' not found in {location_path}")
        return loaded_type

    if class_name is None:
        raise ValueError("agent.class is required for custom runner loading")

    if ":" in class_name:
        module_path, class_key = class_name.split(":", 1)
    elif "." in class_name:
        module_path, class_key = class_name.rsplit(".", 1)
    else:
        raise ValueError("agent.class must be module:Class or module.Class")

    module = importlib.import_module(module_path)
    loaded_type = getattr(module, class_key, None)
    if not isinstance(loaded_type, type):
        raise ValueError(f"Runner class '{class_key}' not found in module '{module_path}'")
    return loaded_type


def _build_agent_runner(agent_config: dict[str, Any], suite_file: Path) -> Any:
    class_name = cast(str | None, agent_config.get("agent_class"))
    location = cast(str | None, agent_config.get("agent_location"))
    kwargs_raw = agent_config.get("agent_kwargs", {})
    kwargs: dict[str, Any] = (
        {str(key): value for key, value in kwargs_raw.items()}
        if isinstance(kwargs_raw, dict)
        else {}
    )
    mcp_servers_raw = agent_config.get("mcp_servers")
    mcp_servers: list[dict[str, Any]] = (
        [server for server in mcp_servers_raw if isinstance(server, dict)]
        if isinstance(mcp_servers_raw, list)
        else []
    )

    provider = cast(str, agent_config.get("provider"))
    model = cast(str, agent_config.get("model"))
    agent_name = cast(str, agent_config.get("agent_name"))

    if mcp_servers and "mcp_servers" not in kwargs:
        kwargs["mcp_servers"] = mcp_servers

    if class_name is None and location is None:
        from ash_hawk.agents import DawnKestrelAgentRunner

        default_runner_kwargs = dict(kwargs)
        default_runner_kwargs.pop("provider", None)
        default_runner_kwargs.pop("model", None)
        return DawnKestrelAgentRunner(provider=provider, model=model, **default_runner_kwargs)

    kwargs.setdefault("provider", provider)
    kwargs.setdefault("model", model)
    kwargs.setdefault("agent_name", agent_name)

    runner_type = _load_runner_type(class_name=class_name, location=location, suite_file=suite_file)

    try:
        runner = runner_type(**kwargs)
    except TypeError as e:
        raise ValueError(f"Could not instantiate runner class '{runner_type.__name__}': {e}") from e

    if not hasattr(runner, "run") or not callable(getattr(runner, "run")):
        raise ValueError(
            f"Runner class '{runner_type.__name__}' must provide an async run() method"
        )

    return runner
