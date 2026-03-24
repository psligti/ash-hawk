"""Auto-research improvement cycle - stripped down core loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from rich.console import Console

from ash_hawk.auto_research.llm import generate_improvement
from ash_hawk.auto_research.types import (
    CycleResult,
    CycleStatus,
    IterationResult,
)
from ash_hawk.scenario import load_scenario, run_scenarios_async
from ash_hawk.services.dawn_kestrel_injector import DawnKestrelInjector

logger = logging.getLogger(__name__)
console = Console()


class TargetType(StrEnum):
    AGENT = "agent"
    SKILL = "skill"
    TOOL = "tool"


@dataclass
class ImprovementTarget:
    target_type: TargetType
    name: str
    discovered_path: Path
    injector: DawnKestrelInjector

    @property
    def structured_path(self) -> Path:
        if self.target_type == TargetType.AGENT:
            return self.injector.get_agent_path(self.name)
        elif self.target_type == TargetType.SKILL:
            return self.injector.get_skill_path(self.name)
        return self.injector.get_tool_path(self.name)

    def read_content(self) -> str:
        if self.structured_path.exists():
            return self.structured_path.read_text(encoding="utf-8")
        if self.discovered_path.exists():
            return self.discovered_path.read_text(encoding="utf-8")
        return ""

    def save_content(self, content: str) -> Path:
        if self.target_type == TargetType.AGENT:
            return self.injector.save_agent_content(self.name, content)
        elif self.target_type == TargetType.SKILL:
            return self.injector.save_skill_content(self.name, content)
        return self.injector.save_tool_content(self.name, content)


SKILL_SEARCH_PATHS = [
    ".dawn-kestrel/skills/skill.txt",
    ".dawn-kestrel/skills/skill.md",
    ".opencode/skills/skill.md",
    ".claude/skills/skill.md",
]

TOOL_SEARCH_PATHS = [
    ".dawn-kestrel/tools",
    ".opencode/tools",
    ".claude/tools",
]


def _create_llm_client() -> Any:
    try:
        from dawn_kestrel.core.settings import get_settings
        from dawn_kestrel.llm.client import LLMClient

        from ash_hawk.config import get_config

        settings = get_settings()
        account = settings.get_default_account()

        if not account or not account.api_key:
            logger.warning("No default account or API key configured")
            return None

        config = get_config()
        return LLMClient(
            provider_id=account.provider_id,
            model=account.model,
            api_key=account.api_key.get_secret_value(),
            timeout_seconds=config.llm_timeout_seconds,
        )
    except ImportError as e:
        logger.warning(f"dawn_kestrel not available: {e}")
        return None


def _discover_improvement_target(
    scenarios: list[Path],
    project_root: Path | None = None,
) -> ImprovementTarget | None:
    if not scenarios:
        return None

    first_scenario = load_scenario(scenarios[0])
    scenario_root = scenarios[0].parent

    resolved_root = project_root or _find_project_root(scenario_root)
    if resolved_root is None:
        return None

    injector = DawnKestrelInjector(project_root=resolved_root)

    skill_target = _find_skill_file(resolved_root)
    if skill_target:
        skill_name = _infer_name_from_path(skill_target)
        return ImprovementTarget(
            target_type=TargetType.SKILL,
            name=skill_name,
            discovered_path=skill_target,
            injector=injector,
        )

    tool_target = _find_primary_tool(resolved_root, first_scenario.tools.allowed_tools)
    if tool_target:
        tool_name = _infer_name_from_path(tool_target)
        return ImprovementTarget(
            target_type=TargetType.TOOL,
            name=tool_name,
            discovered_path=tool_target,
            injector=injector,
        )

    return None


def _find_project_root(scenario_dir: Path) -> Path | None:
    current = scenario_dir.resolve()
    for _ in range(10):
        if (current / "pyproject.toml").exists():
            return current
        if (current / ".git").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    return None


def _find_skill_file(project_root: Path) -> Path | None:
    for search_path in SKILL_SEARCH_PATHS:
        candidate = project_root / search_path
        if candidate.exists():
            return candidate
    return None


def _find_primary_tool(project_root: Path, allowed_tools: list[str]) -> Path | None:
    if not allowed_tools:
        return None

    primary_tool = allowed_tools[0]
    for tool_dir in TOOL_SEARCH_PATHS:
        tools_path = project_root / tool_dir
        if tools_path.exists():
            for ext in (".txt", ".md"):
                candidate = tools_path / f"{primary_tool}{ext}"
                if candidate.exists():
                    return candidate
    return None


def _infer_name_from_path(path: Path) -> str:
    stem = path.stem
    if stem.lower() in ("skill", "tool", "agent"):
        return "default"
    return stem


def _infer_agent_name(target: Path | ImprovementTarget) -> str:
    if isinstance(target, ImprovementTarget):
        return target.name
    parts = target.parts
    for p in parts:
        if p in ("skills", "policies", "tools"):
            idx = parts.index(p)
            if idx + 1 < len(parts):
                return parts[idx + 1].replace(".md", "").replace(".txt", "")
    return "unknown"


async def run_cycle(
    scenarios: list[Path],
    iterations: int = 100,
    threshold: float = 0.02,
    storage_path: Path | None = None,
    llm_client: Any = None,
    project_root: Path | None = None,
) -> CycleResult:
    if llm_client is None:
        llm_client = _create_llm_client()

    storage = storage_path or Path(".ash-hawk/auto-research")

    target = _discover_improvement_target(scenarios, project_root)
    if target is None:
        return CycleResult(
            agent_name="unknown",
            target_path="",
            scenario_paths=[str(s) for s in scenarios],
            status=CycleStatus.ERROR,
            error_message="Could not discover improvement target from scenarios",
        )

    result = CycleResult(
        agent_name=_infer_agent_name(target),
        target_path=str(target.structured_path),
        scenario_paths=[str(s) for s in scenarios],
        status=CycleStatus.RUNNING,
    )

    console.rule("[bold]Auto-Research Cycle[/bold]")
    console.print(f"[dim]Target: {target.structured_path} ({target.target_type.value})[/dim]")
    console.print(f"[dim]Name: {target.name}[/dim]")
    console.print(f"[dim]Scenarios: {len(scenarios)}[/dim]")
    console.print(f"[dim]Iterations: {iterations}[/dim]")
    console.print(f"[dim]Threshold: {threshold}[/dim]")

    try:
        console.print("\n[cyan]Baseline evaluation...[/cyan]")
        score, transcripts_raw = await _run_evaluation(scenarios, storage)
        result.initial_score = score
        console.print(f"  [dim]Baseline: {score:.3f}[/dim]")

        transcripts: list[Any] | None = transcripts_raw
        current_score = score

        for i in range(iterations):
            console.print(f"\n[cyan]Iteration {i + 1}/{iterations}[/cyan]")

            iter_result = await _run_iteration(
                iteration_num=i,
                target=target,
                scenarios=scenarios,
                storage=storage,
                llm_client=llm_client,
                score_before=current_score,
                cached_transcripts=transcripts if i == 0 else None,
                threshold=threshold,
            )

            result.iterations.append(iter_result)
            current_score = iter_result.score_after

            if iter_result.applied:
                transcripts = None

            console.print(
                f"  [dim]Score: {iter_result.score_before:.3f} → {iter_result.score_after:.3f} "
                f"({iter_result.delta:+.3f})[/dim]"
            )

            if iter_result.applied:
                console.print(f"  [green]✓ Kept: {iter_result.improvement_text[:50]}...[/green]")
            else:
                console.print("  [yellow]✗ Reverted[/yellow]")

            if _check_convergence(result.iterations):
                console.print("  [green]✓ Converged[/green]")
                break

        result.status = CycleStatus.COMPLETED
        result.final_score = current_score
        result.completed_at = datetime.now(UTC)

        console.print()
        console.rule("[bold]Cycle Complete[/bold]")
        console.print(f"[dim]Iterations: {result.total_iterations}[/dim]")
        console.print(f"[dim]Baseline: {result.initial_score:.3f}[/dim]")
        console.print(f"[dim]Final: {result.final_score:.3f}[/dim]")
        console.print(f"[dim]Improvement: {result.improvement_delta:+.3f}[/dim]")
        console.print(f"[dim]Saved to: {target.structured_path}[/dim]")

    except Exception as e:
        logger.error(f"Cycle failed: {e}")
        result.status = CycleStatus.ERROR
        result.error_message = str(e)
        result.completed_at = datetime.now(UTC)
        raise

    return result


async def _run_iteration(
    iteration_num: int,
    target: ImprovementTarget,
    scenarios: list[Path],
    storage: Path,
    llm_client: Any,
    score_before: float,
    cached_transcripts: list[Any] | None,
    threshold: float,
) -> IterationResult:
    current_content = target.read_content()

    transcripts = cached_transcripts
    if transcripts is None:
        _, transcripts = await _run_evaluation(scenarios, storage, show_failure_patterns=False)

    console.print("  [dim]Generating improvement...[/dim]")
    improved = await generate_improvement(llm_client, current_content, transcripts)

    if not improved:
        console.print("  [yellow]No improvement generated[/yellow]")
        return IterationResult(
            iteration_num=iteration_num,
            score_before=score_before,
            score_after=score_before,
        )

    saved_path = target.save_content(improved)

    score_after, _ = await _run_evaluation(scenarios, storage, show_failure_patterns=False)
    delta = score_after - score_before

    applied = delta >= threshold
    improvement_text = improved.split("\n")[0][:80] if improved else ""

    artifacts_dir = storage / "iterations"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = (
        artifacts_dir / f"iter_{iteration_num:03d}_{'kept' if applied else 'reverted'}.md"
    )
    artifact_path.write_text(improved)

    console.print(f"  [cyan]Proposal:[/cyan] {improvement_text}")
    console.print(f"  [dim]Saved to: {saved_path}[/dim]")
    console.print(f"  [dim]Artifact: {artifact_path}[/dim]")

    if not applied:
        target.save_content(current_content)
        score_after = score_before

    return IterationResult(
        iteration_num=iteration_num,
        score_before=score_before,
        score_after=score_after,
        improvement_text=improvement_text,
        applied=applied,
    )


async def _run_evaluation(
    scenarios: list[Path],
    storage: Path,
    show_failure_patterns: bool = True,
) -> tuple[float, list[Any]]:
    try:
        summary = await run_scenarios_async(
            paths=[str(p) for p in scenarios],
            storage_path=storage,
            show_failure_patterns=show_failure_patterns,
        )

        transcripts = []
        for trial in summary.trials:
            if trial.result and trial.result.transcript:
                transcripts.append(trial.result.transcript)

        return summary.metrics.mean_score, transcripts
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 0.0, []


def _check_convergence(
    iterations: list[IterationResult],
    window: int = 5,
    variance_threshold: float = 0.001,
) -> bool:
    if len(iterations) < window:
        return False

    recent = [i.score_after for i in iterations[-window:]]
    mean = sum(recent) / len(recent)
    variance = sum((s - mean) ** 2 for s in recent) / len(recent)

    return variance < variance_threshold


__all__ = ["run_cycle", "ImprovementTarget", "TargetType"]
