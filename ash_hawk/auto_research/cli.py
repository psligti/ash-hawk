"""CLI for auto-research improvement cycle."""

# type-hygiene: skip-file  # pre-existing Any — CLI module with heterogeneous config handling

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import click
from rich.console import Console

from ash_hawk.auto_research.cycle_runner import run_cycle
from ash_hawk.auto_research.enhanced_cycle_runner import run_enhanced_cycle
from ash_hawk.auto_research.types import (
    DEFAULT_LEVER_SPACE,
    CycleResult,
    EnhancedCycleConfig,
)
from ash_hawk.config import get_config

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


def _load_previous_final_score(
    storage: Path,
    current_path: Path,
    *,
    agent_name: str,
    target_path: str,
) -> float | None:
    cycles_dir = storage / "cycles"
    if not cycles_dir.exists():
        return None

    previous_files = sorted(
        [path for path in cycles_dir.glob("cycle_*.json") if path != current_path],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not previous_files:
        return None

    for candidate in previous_files:
        previous_data = json.loads(candidate.read_text())
        if previous_data.get("agent_name") != agent_name:
            continue
        if previous_data.get("target_path") != target_path:
            continue

        final_score = previous_data.get("final_score")
        if isinstance(final_score, float | int):
            return float(final_score)

    return None


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
@click.option(
    "--rate-limit",
    is_flag=True,
    default=False,
    help="Enable per-provider rate limiting (uses dawn-kestrel LocalRateLimitTracker)",
)
@click.option(
    "--providers",
    multiple=True,
    default=["anthropic"],
    help="Providers for rate limiting (default: anthropic)",
)
@click.option(
    "--max-concurrent",
    default=10,
    type=int,
    help="Max concurrent LLM calls when rate limiting (default: 10)",
)
@click.option(
    "--scenario-timeout-seconds",
    default=None,
    type=float,
    help="Override scenario max_time_seconds for all trials in this run",
)
@click.option(
    "--thin-bridge/--no-thin-bridge",
    default=True,
    help="Run scenario evaluations through thin runner bridge (default: enabled)",
)
@click.option(
    "--candidate-target-updates",
    default=3,
    type=int,
    help="How many candidate targets to evaluate per iteration (default: 3)",
)
@click.option(
    "--evolvable/--no-evolvable",
    default=False,
    help="Enable evolvable block-coordinate optimization (default: disabled)",
)
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    help="Project root directory",
)
def run(
    scenario: tuple[Path, ...],
    iterations: int,
    threshold: float,
    rate_limit: bool,
    providers: tuple[str, ...],
    max_concurrent: int,
    scenario_timeout_seconds: float | None,
    thin_bridge: bool,
    candidate_target_updates: int,
    evolvable: bool,
    project: Path | None,
) -> None:
    """Run auto-research improvement cycle.

    Discovers the skill/tool file to improve from the scenario configuration.
    Improvements are written directly to the discovered target file.

    Examples:
        ash-hawk improve run -s evals/scenario1.yaml
        ash-hawk improve run -s evals/*.yaml -i 50
        ash-hawk improve run -s evals/*.yaml --evolvable
        ash-hawk improve run -s evals/*.yaml --rate-limit --providers anthropic openai
    """
    scenarios = list(scenario)
    storage = Path(".ash-hawk/auto-research")
    project_root = project or Path.cwd()

    if evolvable:
        asyncio.run(_run_evolvable(scenarios, iterations, threshold, storage, project_root))
    else:
        asyncio.run(
            _run_standard(
                scenarios,
                iterations,
                threshold,
                storage,
                scenario_timeout_seconds,
                thin_bridge,
                candidate_target_updates,
                max_concurrent,
            )
        )


async def _run_standard(
    scenarios: list[Path],
    iterations: int,
    threshold: float,
    storage: Path,
    scenario_timeout_seconds: float | None,
    thin_bridge: bool,
    candidate_target_updates: int,
    max_concurrent: int,
) -> None:
    result = await run_cycle(
        scenarios=scenarios,
        iterations=iterations,
        threshold=threshold,
        storage_path=storage,
        scenario_timeout_seconds=scenario_timeout_seconds,
        use_thin_bridge=thin_bridge,
        candidate_target_updates=max(1, candidate_target_updates),
        parallelism=max_concurrent,
    )

    cycle_path = _save_cycle_result(result, storage)
    previous_final_score = _load_previous_final_score(
        storage,
        cycle_path,
        agent_name=result.agent_name,
        target_path=result.target_path,
    )
    kept_count = len(result.applied_iterations)
    reverted_count = result.total_iterations - kept_count

    console.print()
    console.rule("[bold]Final Result[/bold]")
    console.print(f"Status: {result.status.value}")
    if result.error_message:
        console.print(f"[red]Error: {result.error_message}[/red]")
    console.print(f"Target: {result.target_path}")
    console.print(f"Iterations: {result.total_iterations}")
    console.print(f"Baseline: {result.initial_score:.3f}")
    console.print(f"Final: {result.final_score:.3f}")
    console.print(f"Improvement: {result.improvement_delta:+.3f}")
    if previous_final_score is not None:
        console.print(f"Delta vs previous run: {result.final_score - previous_final_score:+.3f}")
    console.print(f"Kept iterations: {kept_count}")
    console.print(f"Reverted iterations: {reverted_count}")
    console.print(f"[cyan]Cycle results:[/cyan] {cycle_path}")
    console.print(f"[cyan]Iteration artifacts:[/cyan] {storage / 'iterations'}")


async def _run_evolvable(
    scenarios: list[Path],
    iterations: int,
    threshold: float,
    storage: Path,
    project_root: Path,
) -> None:
    enhanced_config = EnhancedCycleConfig(
        enable_multi_target=True,
        max_parallel_targets=4,
        enable_lever_search=True,
        lever_space=dict(DEFAULT_LEVER_SPACE),
        enable_intent_analysis=False,
        enable_knowledge_promotion=False,
        enable_skill_cleanup=False,
        note_lark_enabled=False,
        iterations_per_target=iterations,
        improvement_threshold=threshold,
    )

    result = await run_enhanced_cycle(
        scenarios=scenarios,
        config=enhanced_config,
        storage_path=storage,
        project_root=project_root,
    )

    console.print()
    console.rule("[bold]Evolvable Optimization Result[/bold]")
    console.print(f"Status: {result.status.value}")
    if result.error_message:
        console.print(f"[red]Error: {result.error_message}[/red]")
    console.print(f"Targets: {len(result.target_results)}")
    console.print(f"Total iterations: {result.total_iterations}")
    console.print(f"Overall improvement: {result.overall_improvement:+.3f}")
    console.print(f"Converged: {result.converged}")
    if result.lever_result is not None:
        from ash_hawk.auto_research.types import EvolvableCycleResult

        evolvable_result = result.lever_result
        if isinstance(evolvable_result, EvolvableCycleResult):
            console.print(f"Evolvable experiments: {evolvable_result.total_experiments}")
            console.print(f"Evolvable improvement: {evolvable_result.improvement:+.4f}")
            console.print(f"Reverted experiments: {evolvable_result.reverted_experiments}")
    console.print(f"Duration: {result.total_duration_seconds:.1f}s")


__all__ = ["auto_research", "run"]
