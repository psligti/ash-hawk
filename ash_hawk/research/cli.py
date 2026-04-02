from __future__ import (
    annotations,
)  # type-hygiene: skip-file  # mypy: misc (untyped click decorators)

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console

if TYPE_CHECKING:
    from ash_hawk.research.types import ResearchLoopResult

console = Console()
logger = logging.getLogger(__name__)


def _create_llm_client() -> object | None:
    try:
        import importlib

        from ash_hawk.config import get_config

        settings_module = importlib.import_module("dawn_kestrel.core.settings")
        llm_module = importlib.import_module("dawn_kestrel.llm.client")

        get_settings = getattr(settings_module, "get_settings", None)
        llm_client_factory = getattr(llm_module, "LLMClient", None)
        if get_settings is None or llm_client_factory is None:
            logger.warning("dawn_kestrel settings or LLM client unavailable")
            return None

        settings = get_settings()
        account = settings.get_default_account()

        if not account or not account.api_key:
            logger.warning("No default account or API key configured")
            return None

        config = get_config()
        return llm_client_factory(
            provider_id=account.provider_id,
            model=account.model,
            api_key=account.api_key.get_secret_value(),
            timeout_seconds=config.auto_research_llm_timeout_seconds,
            max_retries=config.auto_research_llm_max_retries,
            use_queue=config.llm_use_queue,
        )
    except ImportError as exc:
        logger.warning(f"dawn_kestrel not available: {exc}")
        return None


@click.group(name="research")
def research() -> None:
    pass


@research.command(name="run")
@click.option(
    "--scenario",
    "-s",
    multiple=True,
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Scenario file to evaluate (REQUIRED, can specify multiple)",
)
@click.option(
    "--iterations",
    "-i",
    default=10,
    type=int,
    help="Max research iterations (default: 10)",
)
@click.option(
    "--uncertainty-threshold",
    default=0.6,
    type=float,
    help="Uncertainty threshold for observe vs fix (default: 0.6)",
)
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    help="Project root directory",
)
@click.option(
    "--storage",
    type=click.Path(path_type=Path),
    help="Storage path for research artifacts (default: .ash-hawk/research)",
)
@click.option(
    "--human-approval/--no-human-approval",
    default=True,
    help="Require human approval for mutations (default: enabled)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output JSON file for results",
)
def run(
    scenario: tuple[Path, ...],
    iterations: int,
    uncertainty_threshold: float,
    project: Path | None,
    storage: Path | None,
    human_approval: bool,
    output: Path | None,
) -> None:
    from ash_hawk.research.research_loop import ResearchLoop
    from ash_hawk.research.types import ResearchLoopConfig

    scenarios = list(scenario)
    if not scenarios:
        console.print("[red]No scenarios specified. Use -s to provide scenario files.[/red]")
        raise SystemExit(1)

    project_root = project or Path.cwd()
    storage_path = storage or Path(".ash-hawk/research")

    config = ResearchLoopConfig(
        iterations=iterations,
        uncertainty_threshold=uncertainty_threshold,
        human_approval_required=human_approval,
        storage_path=storage_path,
    )

    async def _run() -> ResearchLoopResult:
        console.print("[cyan]Running Research Supervisor...[/cyan]")
        llm_client = _create_llm_client()
        if llm_client is None:
            console.print("[yellow]No LLM client available — diagnoses will be limited[/yellow]")
        else:
            console.print("[cyan]Creating LLM client...[/cyan]")
        loop = ResearchLoop(config=config, llm_client=llm_client, storage_path=storage_path)
        return await loop.run(scenarios=scenarios, project_root=project_root)

    result = asyncio.run(_run())

    _print_result(result)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        _save_result(result, output)
        console.print(f"\n[cyan]Results saved to: {output}[/cyan]")


def _print_result(result: ResearchLoopResult) -> None:
    console.print()
    console.rule("[bold]Research Supervisor Results[/bold]")
    console.print(f"Decisions: {result.total_decisions}")
    console.print(f"Diagnoses: {result.diagnoses_count}")
    console.print(f"Uncertainty: {result.uncertainty_before:.3f} -> {result.uncertainty_after:.3f}")
    console.print(f"Improvement delta: {result.improvement_delta:+.3f}")
    console.print(f"Strategies promoted: {len(result.strategies_promoted)}")

    if result.decisions:
        observe_count = sum(1 for d in result.decisions if d.action.value == "observe")
        fix_count = sum(1 for d in result.decisions if d.action.value == "fix")
        console.print(f"Observe vs Fix ratio: {observe_count}/{fix_count}")

    if result.error_message:
        console.print(f"[red]Error: {result.error_message}[/red]")


def _save_result(result: ResearchLoopResult, output: Path) -> None:
    import json

    data: dict[str, object] = {
        "diagnoses_count": result.diagnoses_count,
        "strategies_promoted": list(result.strategies_promoted),
        "uncertainty_before": result.uncertainty_before,
        "uncertainty_after": result.uncertainty_after,
        "improvement_delta": result.improvement_delta,
        "error_message": result.error_message,
    }

    if result.started_at is not None:
        data["started_at"] = result.started_at.isoformat()
    if result.completed_at is not None:
        data["completed_at"] = result.completed_at.isoformat()

    serialized: list[dict[str, object]] = []
    for d in result.decisions:
        entry: dict[str, object] = {
            "action": d.action.value,
            "rationale": d.rationale,
            "target": d.target,
            "expected_info_gain": d.expected_info_gain,
            "confidence": d.confidence,
        }
        entry["timestamp"] = d.timestamp.isoformat()
        serialized.append(entry)
    data["decisions"] = serialized

    output.write_text(json.dumps(data, indent=2, default=str))


__all__ = ["research", "run"]
