"""CLI for auto-research improvement cycle."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import click
from rich.console import Console

from ash_hawk.auto_research.cycle_runner import run_cycle
from ash_hawk.auto_research.types import CycleResult
from ash_hawk.config import get_config
from ash_hawk.execution.queue import LLMRequestQueue, register_llm_queue

console = Console()


def _serialize_datetime(obj: object) -> str | object:
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def _save_cycle_result(result: CycleResult, storage: Path) -> Path:
    cycles_dir = storage / "cycles"
    cycles_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    cycle_path = cycles_dir / f"cycle_{result.agent_name}_{timestamp}.json"

    data = asdict(result)
    data["started_at"] = result.started_at.isoformat()
    if result.completed_at:
        data["completed_at"] = result.completed_at.isoformat()
    data["status"] = result.status.value

    for iter_data in data.get("iterations", []):
        if "timestamp" in iter_data and isinstance(iter_data["timestamp"], datetime):
            iter_data["timestamp"] = iter_data["timestamp"].isoformat()

    cycle_path.write_text(json.dumps(data, indent=2, default=_serialize_datetime))
    return cycle_path


@click.group(name="auto-research")
def auto_research() -> None:
    """Auto-research improvement cycle commands."""
    pass


@auto_research.command(name="run")
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
    default=100,
    type=int,
    help="Maximum iterations (default: 100)",
)
@click.option(
    "--threshold",
    default=0.02,
    type=float,
    help="Minimum improvement to keep change (default: 0.02)",
)
def run(
    scenario: tuple[Path, ...],
    iterations: int,
    threshold: float,
) -> None:
    """Run auto-research improvement cycle.

    Discovers the skill/tool file to improve from the scenario configuration.
    Improvements are written directly to the discovered target file.

    Examples:
        ash-hawk auto-research run -s evals/scenario1.yaml
        ash-hawk auto-research run -s evals/*.yaml -i 50
    """
    config = get_config()

    queue = LLMRequestQueue(
        max_workers=config.llm_max_workers,
        timeout_seconds=config.llm_timeout_seconds,
    )
    register_llm_queue(queue)

    scenarios = list(scenario)
    storage = Path(".ash-hawk/auto-research")

    async def _run() -> None:
        result = await run_cycle(
            scenarios=scenarios,
            iterations=iterations,
            threshold=threshold,
            storage_path=storage,
        )

        cycle_path = _save_cycle_result(result, storage)

        console.print()
        console.rule("[bold]Final Result[/bold]")
        console.print(f"Status: {result.status.value}")
        console.print(f"Target: {result.target_path}")
        console.print(f"Iterations: {result.total_iterations}")
        console.print(f"Baseline: {result.initial_score:.3f}")
        console.print(f"Final: {result.final_score:.3f}")
        console.print(f"Improvement: {result.improvement_delta:+.3f}")
        console.print(f"[cyan]Cycle results:[/cyan] {cycle_path}")
        console.print(f"[cyan]Iteration artifacts:[/cyan] {storage / 'iterations'}")

    asyncio.run(_run())


__all__ = ["auto_research", "run"]
