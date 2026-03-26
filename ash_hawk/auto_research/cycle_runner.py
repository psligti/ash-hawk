"""Auto-research improvement cycle - stripped down core loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from ash_hawk.auto_research.llm import extract_skill_name, generate_improvement
from ash_hawk.auto_research.types import (
    CycleResult,
    CycleStatus,
    IterationResult,
    TargetType,
)
from ash_hawk.improvement.guardrails import GuardrailChecker, GuardrailConfig
from ash_hawk.scenario import load_scenario, run_scenarios_async
from ash_hawk.services.dawn_kestrel_injector import (
    DAWN_KESTREL_DIR,
    DawnKestrelInjector,
)

logger = logging.getLogger(__name__)
console = Console()


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
        if self.target_type == TargetType.SKILL:
            return self.injector.get_skill_path(self.name)
        if self.target_type == TargetType.POLICY:
            return self.injector.get_policy_path(self.name)
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
        if self.target_type == TargetType.SKILL:
            return self.injector.save_skill_content(self.name, content)
        if self.target_type == TargetType.POLICY:
            return self.injector.save_policy_content(self.name, content)
        return self.injector.save_tool_content(self.name, content)

    def delete_content(self) -> bool:
        if self.target_type == TargetType.SKILL:
            return self.injector.delete_skill_content(self.name)
        if self.target_type == TargetType.TOOL:
            return self.injector.delete_tool_content(self.name)
        if self.target_type == TargetType.AGENT:
            return self.injector.delete_agent_content(self.name)
        if self.target_type == TargetType.POLICY:
            return self.injector.delete_policy_content(self.name)
        return False


SKILL_SEARCH_DIRS = [
    DAWN_KESTREL_DIR / "skills",
    Path(".opencode/skills"),
    Path(".claude/skills"),
]

TOOL_SEARCH_DIRS = [
    DAWN_KESTREL_DIR / "tools",
    Path(".opencode/tools"),
    Path(".claude/tools"),
]

POLICY_SEARCH_DIRS = [
    DAWN_KESTREL_DIR / "policies",
    Path(".opencode/policies"),
    Path(".claude/policies"),
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
            timeout_seconds=config.auto_research_llm_timeout_seconds,
            max_retries=config.auto_research_llm_max_retries,
            use_queue=config.llm_use_queue,
            max_concurrent=config.llm_queue_max_concurrent,
        )
    except ImportError as e:
        logger.warning(f"dawn_kestrel not available: {e}")
        return None


def _discover_improvement_target(
    scenarios: list[Path],
    project_root: Path | None,
    strategy: Any = None,
) -> ImprovementTarget | None:
    if not scenarios:
        return None

    first_scenario = load_scenario(scenarios[0])
    scenario_root = scenarios[0].parent

    resolved_root = project_root or _find_project_root(scenario_root)
    if resolved_root is None:
        logger.warning("Could not find project root")
        return None

    injector = DawnKestrelInjector(project_root=resolved_root, strategy=strategy)

    skill_target = _find_skill_file(resolved_root)
    if skill_target:
        skill_name = _infer_name_from_path(skill_target)
        injector.current_skill_name = skill_name
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
    for search_dir in SKILL_SEARCH_DIRS:
        skills_path = project_root / search_dir
        if skills_path.exists() and skills_path.is_dir():
            for skill_dir in skills_path.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists():
                        return skill_file
    return None


def _discover_all_skills(project_root: Path) -> list[Path]:
    skill_files: list[Path] = []
    for search_dir in SKILL_SEARCH_DIRS:
        skills_path = project_root / search_dir
        if skills_path.exists() and skills_path.is_dir():
            for skill_dir in sorted(skills_path.iterdir()):
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists():
                        skill_files.append(skill_file)
    return skill_files


def _discover_all_tools(project_root: Path) -> list[Path]:
    tool_files: list[Path] = []
    for search_dir in TOOL_SEARCH_DIRS:
        tools_path = project_root / search_dir
        if tools_path.exists() and tools_path.is_dir():
            for tool_dir in sorted(tools_path.iterdir()):
                if tool_dir.is_dir():
                    tool_file = tool_dir / "TOOL.md"
                    if tool_file.exists():
                        tool_files.append(tool_file)
    return tool_files


def _discover_all_policies(project_root: Path) -> list[Path]:
    policy_files: list[Path] = []
    for search_dir in POLICY_SEARCH_DIRS:
        policies_path = project_root / search_dir
        if policies_path.exists() and policies_path.is_dir():
            for policy_dir in sorted(policies_path.iterdir()):
                if policy_dir.is_dir():
                    policy_file = policy_dir / "POLICY.md"
                    if policy_file.exists():
                        policy_files.append(policy_file)
    return policy_files


def _load_existing_skill_names(project_root: Path) -> list[str]:
    skill_names: list[str] = []
    for search_dir in SKILL_SEARCH_DIRS:
        skills_path = project_root / search_dir
        if skills_path.exists() and skills_path.is_dir():
            for skill_dir in sorted(skills_path.iterdir()):
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists():
                        skill_names.append(skill_dir.name)
    return skill_names


def _discover_all_targets(project_root: Path) -> list[tuple[TargetType, Path]]:
    targets: list[tuple[TargetType, Path]] = []
    targets.extend((TargetType.AGENT, p) for p in _discover_all_agents(project_root))
    targets.extend((TargetType.POLICY, p) for p in _discover_all_policies(project_root))
    targets.extend((TargetType.SKILL, p) for p in _discover_all_skills(project_root))
    targets.extend((TargetType.TOOL, p) for p in _discover_all_tools(project_root))
    return targets


def _discover_all_agents(project_root: Path) -> list[Path]:
    agent_files: list[Path] = []
    agents_dirs = [
        DAWN_KESTREL_DIR / "agents",
        Path(".opencode/agents"),
        Path(".claude/agents"),
    ]
    for search_dir in agents_dirs:
        agents_path = project_root / search_dir
        if agents_path.exists() and agents_path.is_dir():
            for agent_dir in sorted(agents_path.iterdir()):
                if agent_dir.is_dir():
                    agent_file = agent_dir / "AGENT.md"
                    if agent_file.exists():
                        agent_files.append(agent_file)
    return agent_files


def _find_primary_tool(project_root: Path, allowed_tools: list[str]) -> Path | None:
    if not allowed_tools:
        return None

    primary_tool = allowed_tools[0]
    for tool_dir in TOOL_SEARCH_DIRS:
        tools_path = project_root / tool_dir
        if tools_path.exists() and tools_path.is_dir():
            for tool_subdir in tools_path.iterdir():
                if tool_subdir.is_dir() and tool_subdir.name == primary_tool:
                    tool_file = tool_subdir / "TOOL.md"
                    if tool_file.exists():
                        return tool_file
    return None


def _infer_name_from_path(path: Path) -> str:
    stem = path.stem
    if stem.upper() in ("SKILL", "TOOL", "AGENT", "POLICY"):
        return path.parent.name
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


def _create_target_from_path(
    target_path: Path, project_root: Path, strategy: Any = None
) -> ImprovementTarget | None:
    resolved = target_path.resolve()
    parts = resolved.parts

    target_type = TargetType.SKILL
    for p in parts:
        if p == "skills":
            target_type = TargetType.SKILL
            break
        elif p == "tools":
            target_type = TargetType.TOOL
            break
        elif p == "agents":
            target_type = TargetType.AGENT
            break
        elif p == "policies":
            target_type = TargetType.POLICY
            break

    injector = DawnKestrelInjector(project_root=project_root, strategy=strategy)
    name = _infer_name_from_path(resolved)
    if target_type == TargetType.SKILL:
        injector.current_skill_name = name

    return ImprovementTarget(
        target_type=target_type,
        name=name,
        discovered_path=resolved,
        injector=injector,
    )


async def run_cycle(
    scenarios: list[Path],
    iterations: int = 100,
    threshold: float = 0.02,
    storage_path: Path | None = None,
    llm_client: Any = None,
    project_root: Path | None = None,
    strategy_name: str | None = None,
    use_policy_adaptation: bool = False,
    explicit_targets: list[Path] | None = None,
    heldout_scenarios: list[Path] | None = None,
    guardrail_config: GuardrailConfig | None = None,
) -> CycleResult:
    if llm_client is None:
        llm_client = _create_llm_client()

    storage = storage_path or Path(".ash-hawk/auto-research")

    strategy = None
    if strategy_name:
        try:
            from dawn_kestrel.agents.context import (
                CompositeContextStrategy,
                ContextStrategyRegistry,
                DynamicContextStrategy,
                FileBasedContextStrategy,
                get_registry,
            )

            registry = get_registry()
            if registry.has_strategy(strategy_name):
                strategy_result = registry.get(strategy_name)
                if strategy_result.is_ok():
                    strategy = strategy_result.unwrap()
            else:
                if strategy_name == "file-based":
                    strategy = FileBasedContextStrategy(project_root)
                elif strategy_name == "dynamic":
                    strategy = DynamicContextStrategy(project_root)
                elif strategy_name == "composite":
                    strategy = CompositeContextStrategy(
                        strategies=[
                            FileBasedContextStrategy(project_root),
                            DynamicContextStrategy(project_root),
                        ],
                        project_root=project_root,
                    )
                else:
                    logger.warning(f"Unknown strategy: {strategy_name}, using file-based")
                    strategy = FileBasedContextStrategy(project_root)
        except ImportError:
            logger.warning("Context strategy not available, using file-based behavior")

    target: ImprovementTarget | None = None
    if explicit_targets:
        target = _create_target_from_path(explicit_targets[0], project_root or Path.cwd(), strategy)
    if target is None:
        target = _discover_improvement_target(scenarios, project_root, strategy)
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
        target_type=target.target_type,
        scenario_paths=[str(s) for s in scenarios],
        status=CycleStatus.RUNNING,
    )
    console.rule("[bold]Auto-Research Cycle[/bold]")
    console.print(f"[dim]Agent: {_infer_agent_name(target)}[/dim]")
    console.print(f"[dim]Skills: {len(_discover_all_skills(project_root or Path.cwd()))}[/dim]")
    console.print(f"[dim]Tools: {len(_discover_all_tools(project_root or Path.cwd()))}[/dim]")
    console.print(f"[dim]Policies: {len(_discover_all_policies(project_root or Path.cwd()))}[/dim]")
    console.print(f"[dim]Target: {target.structured_path} ({target.target_type.value})[/dim]")
    console.print(f"[dim]Name: {target.name}[/dim]")
    console.print(f"[dim]Scenarios: {len(scenarios)}[/dim]")
    if heldout_scenarios:
        console.print(f"[dim]Held-out: {len(heldout_scenarios)}[/dim]")
    console.print(f"[dim]Iterations: {iterations}[/dim]")
    console.print(f"[dim]Threshold: {threshold}[/dim]")

    existing_skills = _load_existing_skill_names(project_root or Path.cwd())
    if existing_skills:
        console.print(f"[dim]Existing skills for context: {len(existing_skills)}[/dim]")

    guardrail_checker = GuardrailChecker(guardrail_config)

    try:
        console.print("\n[cyan]Baseline evaluation...[/cyan]")
        score, transcripts_raw = await _run_evaluation(scenarios, storage, injector=target.injector)
        result.initial_score = score
        console.print(f"  [dim]Baseline: {score:.3f}[/dim]")

        heldout_score: float | None = None
        if heldout_scenarios:
            heldout_score, _ = await _run_evaluation(
                heldout_scenarios, storage / "heldout", injector=target.injector
            )
            console.print(f"  [dim]Held-out baseline: {heldout_score:.3f}[/dim]")
            guardrail_checker.record_iteration(heldout_score, applied=True)

        transcripts: list[Any] | None = transcripts_raw
        current_score = score
        consecutive_failures = 0
        for i in range(iterations):
            console.print(f"\n[cyan][{target.name}] Iteration {i + 1}/{iterations}[/cyan]")
            failed_proposals = [
                iter.improvement_text
                for iter in result.iterations
                if not iter.applied and iter.improvement_text
            ]
            iter_result = await _run_iteration(
                iteration_num=i,
                target=target,
                scenarios=scenarios,
                storage=storage,
                llm_client=llm_client,
                score_before=current_score,
                cached_transcripts=transcripts if i == 0 else None,
                threshold=threshold,
                failed_proposals=failed_proposals,
                consecutive_failures=consecutive_failures,
                existing_skills=existing_skills,
            )
            result.iterations.append(iter_result)
            current_score = iter_result.score_after
            if iter_result.applied:
                transcripts = None
                consecutive_failures = 0
            else:
                consecutive_failures += 1
            console.print(
                f"  [dim]Score: {iter_result.score_before:.3f} → {iter_result.score_after:.3f} "
                f"({iter_result.delta:+.3f})[/dim]"
            )
            if iter_result.applied:
                console.print(f"  [green]✓ Kept: {iter_result.improvement_text[:50]}...[/green]")
            else:
                console.print("  [yellow]✗ Reverted[/yellow]")

            if heldout_scenarios and iter_result.applied:
                heldout_score, _ = await _run_evaluation(
                    heldout_scenarios, storage / "heldout", injector=target.injector
                )
                console.print(f"  [dim]Held-out: {heldout_score:.3f}[/dim]")
                guardrail_checker.record_iteration(heldout_score, applied=True)
                if guardrail_checker.should_stop():
                    console.print(
                        f"  [yellow]⚠ Guardrail triggered: {guardrail_checker.stop_reason}[/yellow]"
                    )
                    break

            if (
                guardrail_checker.state.total_reverts
                >= (guardrail_config or GuardrailConfig()).max_reverts
            ):
                console.print("  [yellow]⚠ Max reverts reached[/yellow]")
                break

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
    failed_proposals: list[str] | None = None,
    consecutive_failures: int = 0,
    existing_skills: list[str] | None = None,
) -> IterationResult:
    original_target = target
    original_content = target.read_content()
    if not original_content:
        console.print("  [yellow]No current content to improve[/yellow]")
        return IterationResult(
            iteration_num=iteration_num,
            score_before=score_before,
            score_after=score_before,
        )
    transcripts = cached_transcripts
    if transcripts is None:
        _, transcripts = await _run_evaluation(
            scenarios, storage, show_failure_patterns=False, injector=target.injector
        )
    console.print("  [dim]Generating improvement...[/dim]")
    improved = await generate_improvement(
        llm_client,
        original_content,
        transcripts,
        failed_proposals,
        consecutive_failures,
        existing_skills=existing_skills if target.target_type == TargetType.SKILL else None,
        target_type=target.target_type.value,
    )
    if not improved:
        console.print("  [yellow]No improvement generated[/yellow]")
        return IterationResult(
            iteration_num=iteration_num,
            score_before=score_before,
            score_after=score_before,
        )
    target_name = extract_skill_name(improved)
    new_target = target
    if target_name and target_name != target.name:
        new_target = ImprovementTarget(
            target_type=target.target_type,
            name=target_name,
            discovered_path=target.discovered_path,
            injector=target.injector,
        )
        if target.target_type == TargetType.SKILL:
            new_target.injector.current_skill_name = target_name
    saved_path = new_target.save_content(improved)
    score_after, _ = await _run_evaluation(
        scenarios, storage, show_failure_patterns=False, injector=new_target.injector
    )
    delta = score_after - score_before
    applied = delta >= threshold
    improvement_text = target_name or improved.split("\n")[0][:80] if improved else ""
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
        if new_target.name != original_target.name:
            new_target.delete_content()
            if original_target.target_type == TargetType.SKILL:
                new_target.injector.current_skill_name = original_target.name
            console.print(
                f"  [dim]Deleted failed {new_target.target_type.value}: {new_target.name}[/dim]"
            )
        else:
            original_target.save_content(original_content)
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
    injector: Any | None = None,
) -> tuple[float, list[Any]]:
    try:
        summary = await run_scenarios_async(
            paths=[str(p) for p in scenarios],
            storage_path=storage,
            show_failure_patterns=show_failure_patterns,
            injector=injector,
            grader_config_overrides={"quiet": True},
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
    return False  # Convergence detection disabled
