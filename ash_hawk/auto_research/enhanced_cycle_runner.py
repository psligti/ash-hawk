"""Enhanced auto-research cycle runner with multi-target, intent analysis, and knowledge promotion."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

from rich.console import Console

from ash_hawk.auto_research.cycle_runner import (
    ImprovementTarget,
    _create_llm_client,
    _find_project_root,
)
from ash_hawk.auto_research.intent_analyzer import IntentAnalyzer
from ash_hawk.auto_research.knowledge_promotion import KnowledgePromoter, PromotionCriteria
from ash_hawk.auto_research.lever_matrix import LeverMatrixSearch
from ash_hawk.auto_research.multi_target_runner import MultiTargetCycleRunner, TargetCandidate
from ash_hawk.auto_research.skill_cleanup import CleanupConfig, CleanupResult, SkillCleaner
from ash_hawk.auto_research.target_discovery import TargetDiscovery
from ash_hawk.auto_research.types import (
    ConvergenceReason,
    CycleResult,
    CycleStatus,
    EnhancedCycleConfig,
    EnhancedCycleResult,
    IntentPatterns,
    PromotedLesson,
    PromotionStatus,
    TargetType,
)
from ash_hawk.scenario import run_scenarios_async
from ash_hawk.scenario.loader import load_scenario

logger = logging.getLogger(__name__)
console = Console()


def _resolve_agent_startup_details(scenarios: list[Path]) -> tuple[str, str]:
    agent_name = "unknown"
    agent_version = "unknown"

    if not scenarios:
        return agent_name, agent_version

    try:
        scenario = load_scenario(scenarios[0])
        sut_config = scenario.sut.config
        run_config_raw = sut_config.get("run_config")
        run_config = run_config_raw if isinstance(run_config_raw, dict) else {}

        candidate_agent = (
            run_config.get("agent_name") or sut_config.get("agent") or sut_config.get("agent_name")
        )
        if isinstance(candidate_agent, str) and candidate_agent.strip():
            agent_name = candidate_agent.strip()

        candidate_version = (
            run_config.get("agent_version")
            or sut_config.get("agent_version")
            or sut_config.get("version")
        )
        if isinstance(candidate_version, str) and candidate_version.strip():
            agent_version = candidate_version.strip()
        else:
            adapter = scenario.sut.adapter
            package_name = "dawn-kestrel"
            if adapter == "bolt_merlin":
                package_name = "bolt-merlin"

            try:
                agent_version = version(package_name)
            except PackageNotFoundError:
                agent_version = "unknown"
    except Exception:
        return agent_name, agent_version

    return agent_name, agent_version


async def run_enhanced_cycle(
    scenarios: list[Path],
    config: EnhancedCycleConfig | None = None,
    storage_path: Path | None = None,
    llm_client: Any = None,
    project_root: Path | None = None,
) -> EnhancedCycleResult:
    config = config or EnhancedCycleConfig()
    storage = storage_path or Path(".ash-hawk/enhanced-auto-research")
    storage.mkdir(parents=True, exist_ok=True)

    if llm_client is None:
        llm_client = _create_llm_client()

    if project_root is None:
        project_root = _find_project_root(scenarios[0].parent if scenarios else Path.cwd())
        if project_root is None:
            project_root = Path.cwd()

    result = EnhancedCycleResult(
        agent_name="enhanced-runner",
        config=config,
        started_at=datetime.now(UTC),
    )

    console.rule("[bold]Enhanced Auto-Research Cycle[/bold]")
    startup_agent, startup_version = _resolve_agent_startup_details(scenarios)
    console.print(f"[bold]Agent:[/bold] {startup_agent}")
    console.print(f"[bold]Agent Version:[/bold] {startup_version}")
    console.print(f"[dim]Project: {project_root}[/dim]")
    console.print(f"[dim]Scenarios: {len(scenarios)}[/dim]")
    console.print(f"[dim]Multi-target: {config.enable_multi_target}[/dim]")
    console.print(f"[dim]Intent analysis: {config.enable_intent_analysis}[/dim]")
    console.print(f"[dim]Knowledge promotion: {config.enable_knowledge_promotion}[/dim]")
    console.print(f"[dim]Skill cleanup: {config.enable_skill_cleanup}[/dim]")

    try:
        target_discovery = TargetDiscovery(project_root=project_root)
        targets = target_discovery.discover_all_targets()

        if not targets:
            console.print("[yellow]No improvement targets found[/yellow]")
            result.status = CycleStatus.ERROR
            result.error_message = "No improvement targets discovered"
            return result

        console.print(f"[dim]Discovered targets: {len(targets)}[/dim]")
        for t in targets:
            console.print(f"  - {t.name} ({t.target_type.value})")

        if config.enable_intent_analysis and llm_client:
            console.print("\n[cyan]Running baseline evaluation for intent analysis...[/cyan]")
            baseline_summary = await run_scenarios_async(
                paths=[str(s) for s in scenarios],
                storage_path=storage / "baseline",
            )
            baseline_transcripts = []
            for trial in baseline_summary.trials:
                if trial.result and trial.result.transcript:
                    baseline_transcripts.append(trial.result.transcript)

            if baseline_transcripts:
                intent_analyzer = IntentAnalyzer(llm_client=llm_client)
                result.intent_patterns = await intent_analyzer.analyze_transcripts(
                    baseline_transcripts
                )
                if result.intent_patterns and result.intent_patterns.inferred_intent:
                    console.print(
                        f"[dim]Inferred intent: {result.intent_patterns.inferred_intent[:100]}...[/dim]"
                    )

        targets_to_run = targets
        max_concurrent = config.max_parallel_targets
        if config.enable_multi_target:
            console.print("\n[cyan]Running multi-target improvement...[/cyan]")
        else:
            selected_target = targets[0]
            targets_to_run = [selected_target]
            max_concurrent = 1
            console.print("\n[cyan]Running single-target improvement...[/cyan]")
            console.print(f"[dim]Selected target: {selected_target.name}[/dim]")

        multi_runner = MultiTargetCycleRunner(
            project_root=project_root,
            llm_client=llm_client,
            max_concurrent=max_concurrent,
            convergence_window=config.convergence_window,
            convergence_variance_threshold=config.convergence_variance_threshold,
        )

        multi_result = await multi_runner.run_all_targets(
            scenarios=scenarios,
            targets=cast(list[TargetCandidate], targets_to_run),
            iterations_per_target=config.iterations_per_target,
            threshold=config.improvement_threshold,
            storage_path=storage,
        )

        result.target_results = multi_result.target_results
        result.overall_improvement = multi_result.overall_improvement
        result.converged = multi_result.converged
        result.convergence_reason = multi_result.convergence_reason

        for target_result in result.target_results.values():
            result.total_iterations += target_result.total_iterations

        if config.enable_lever_search and llm_client:
            console.print("\n[cyan]Running lever matrix search...[/cyan]")
            lever_search = LeverMatrixSearch(lever_space=config.lever_space)
            optimize = getattr(lever_search, "optimize", None)
            if callable(optimize):
                lever_result = await optimize(
                    scenarios=scenarios,
                    storage_path=storage / "lever-search",
                    max_iterations=min(20, config.iterations_per_target // 2),
                )
            else:
                lever_result = None
            result.lever_result = lever_result
            if lever_result and lever_result.best_configuration:
                console.print(
                    f"[dim]Best lever config: {lever_result.best_configuration.to_config_dict()}[/dim]"
                )

        if config.enable_knowledge_promotion:
            console.print("\n[cyan]Promoting validated lessons...[/cyan]")
            promoter = KnowledgePromoter(
                criteria=PromotionCriteria(
                    min_improvement=config.min_improvement_for_promotion,
                    min_consecutive_successes=config.min_consecutive_successes,
                ),
                note_lark_enabled=config.note_lark_enabled,
                project_name=config.project_name,
            )

            for target_name, cycle_result in result.target_results.items():
                for iteration in cycle_result.applied_iterations:
                    should_promote, reason = await promoter.should_promote(
                        iteration=iteration,
                        all_iterations=cycle_result.iterations,
                        cycle_result=cycle_result,
                    )
                    if should_promote and cycle_result.target_type is not None:
                        lesson = PromotedLesson(
                            lesson_id=f"{target_name}_{iteration.iteration_num}",
                            source_experiment=target_name,
                            target_name=target_name,
                            target_type=cycle_result.target_type,
                            improvement_text=iteration.improvement_text,
                            score_delta=iteration.delta,
                            promotion_status=PromotionStatus.PENDING,
                        )
                        promoted = await promoter.promote_lesson(
                            lesson=lesson,
                            intent_patterns=result.intent_patterns,
                        )
                        if promoted:
                            result.promoted_lessons.append(lesson)
                            console.print(
                                f"[green]Promoted: {iteration.improvement_text[:50]}...[/green]"
                            )

        if config.enable_skill_cleanup:
            console.print("\n[cyan]Cleaning up low-value skills...[/cyan]")
            cleaner = SkillCleaner(
                project_root=project_root,
                config=CleanupConfig(
                    enabled=True,
                    remove_unused=True,
                    remove_negative_impact=True,
                ),
            )
            baseline_skills = cleaner.get_baseline_skills()
            cleanup_result = cleaner.cleanup_enhanced_result(result, baseline_skills)
            result.cleanup_result = cleanup_result

            if cleanup_result.cleaned_skills:
                console.print(
                    f"[dim]Cleanup: removed {len(cleanup_result.cleaned_skills)} skills "
                    f"({', '.join(cleanup_result.cleaned_skills[:5])}"
                    f"{'...' if len(cleanup_result.cleaned_skills) > 5 else ''})[/dim]"
                )
            else:
                console.print("[dim]Cleanup: no low-value skills found[/dim]")

        result.status = CycleStatus.COMPLETED if not result.converged else CycleStatus.CONVERGED

    except Exception as e:
        logger.error(f"Enhanced cycle failed: {e}")
        result.status = CycleStatus.ERROR
        result.error_message = str(e)
        raise

    finally:
        result.completed_at = datetime.now(UTC)
        if result.started_at:
            result.total_duration_seconds = (
                result.completed_at - result.started_at
            ).total_seconds()

    console.rule("[bold]Enhanced Cycle Complete[/bold]")
    console.print(f"[dim]Status: {result.status.value}[/dim]")
    console.print(f"[dim]Targets: {len(result.target_results)}[/dim]")
    console.print(f"[dim]Total iterations: {result.total_iterations}[/dim]")
    console.print(f"[dim]Overall improvement: {result.overall_improvement:+.3f}[/dim]")
    console.print(f"[dim]Promoted lessons: {result.total_promoted}[/dim]")
    if result.cleanup_result:
        console.print(f"[dim]Cleaned skills: {len(result.cleanup_result.cleaned_skills)}[/dim]")
    console.print(f"[dim]Duration: {result.total_duration_seconds:.1f}s[/dim]")

    return result


__all__ = ["run_enhanced_cycle"]
