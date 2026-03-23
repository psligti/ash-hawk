"""Auto-research improvement cycle runner."""

from __future__ import annotations

import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from ash_hawk.auto_research.agentic_analyst import AgenticAnalyst
from ash_hawk.auto_research.agentic_improver import AgenticImprover
from ash_hawk.auto_research.discovery import (
    discover_repo_config,
    generate_experiment_id,
)
from ash_hawk.auto_research.tool_lifecycle import ToolLifecycleManager
from ash_hawk.auto_research.types import (
    AnalysisResult,
    CycleResult,
    CycleStatus,
    ImprovementResult,
    ImprovementType,
    IterationResult,
    RepoConfig,
)
from ash_hawk.scenario import run_scenarios_async
from ash_hawk.types import EvalTranscript

logger = logging.getLogger(__name__)
console = Console()


def _create_default_llm_client() -> Any:
    try:
        from dawn_kestrel.core.settings import get_settings
        from dawn_kestrel.llm.client import LLMClient

        settings = get_settings()
        account = settings.get_default_account()

        if not account:
            logger.warning("No default account configured in dawn-kestrel")
            return None

        if not account.api_key:
            logger.warning(f"No API key for account {account.account_name}")
            return None

        return LLMClient(
            provider_id=account.provider_id,
            model=account.model,
            api_key=account.api_key.get_secret_value(),
        )
    except ImportError as e:
        logger.warning(f"dawn_kestrel not available: {e}")
        return None


class CycleRunner:
    """Runs N-iteration improvement loop with agentic analysis and improvement."""

    def __init__(
        self,
        repo_config: RepoConfig | None = None,
        targets: list[Path] | None = None,
        scenarios: list[Path] | None = None,
        max_iterations: int = 100,
        improvement_threshold: float = 0.02,
        promotion_threshold: int = 3,
        convergence_variance: float = 0.001,
        convergence_window: int = 5,
        llm_client: Any = None,
        storage_path: Path | None = None,
    ):
        self.repo_config = repo_config or discover_repo_config()
        self.targets = targets or self.repo_config.improvement_targets
        self.scenarios = scenarios or self.repo_config.scenarios
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.promotion_threshold = promotion_threshold
        self.convergence_variance = convergence_variance
        self.convergence_window = convergence_window
        self.llm_client = llm_client or _create_default_llm_client()
        self.storage_path = storage_path or Path(".ash-hawk/auto-research")

        self._experiment_id: str = ""
        self._consecutive_improvements: dict[str, int] = {}
        self._backup_dir: Path | None = None
        self._tool_lifecycle: ToolLifecycleManager | None = None

    def _initialize(self) -> None:
        self._experiment_id = generate_experiment_id(
            self.repo_config.agent_name,
            self.targets,
        )
        self._backup_dir = self.storage_path / self._experiment_id / "backups"
        self._backup_dir.mkdir(parents=True, exist_ok=True)

        self._tool_lifecycle = ToolLifecycleManager(
            repo_root=self.repo_config.pyproject_path.parent
            if self.repo_config.pyproject_path
            else None
        )
        for target in self.targets:
            target_type = self._infer_target_type(target)
            self._tool_lifecycle.create_working_copy(target, target_type)
            lifecycle = self._tool_lifecycle.get_all_lifecycles()[-1]
            console.print(f"  [dim]Working copy: {lifecycle.copy_path}[/dim]")

    async def run(self) -> CycleResult:
        if not self.scenarios:
            return CycleResult(
                experiment_id="unknown",
                agent_name=self.repo_config.agent_name or "unknown",
                status=CycleStatus.ERROR,
                error_message="No scenarios discovered",
            )

        if not self.targets:
            return CycleResult(
                experiment_id="unknown",
                agent_name=self.repo_config.agent_name or "unknown",
                status=CycleStatus.ERROR,
                error_message="No improvement targets discovered",
            )

        self._initialize()

        result = CycleResult(
            experiment_id=self._experiment_id,
            agent_name=self.repo_config.agent_name or "unknown",
            status=CycleStatus.RUNNING,
            started_at=datetime.now(UTC),
        )

        console.rule("[bold]Auto-Research Cycle[/bold]")
        console.print(f"[dim]Experiment: {self._experiment_id}[/dim]")
        console.print(f"[dim]Agent: {self.repo_config.agent_name}[/dim]")
        console.print(f"[dim]Scenarios: {len(self.scenarios)}[/dim]")
        console.print(f"[dim]Targets: {len(self.targets)}[/dim]")

        try:
            console.print("\n[cyan]Step 1: Running baseline evaluation...[/cyan]")
            baseline_score, baseline_transcripts = await self._run_evaluation()
            result.initial_score = baseline_score
            console.print(f"  [dim]Baseline score: {baseline_score:.3f}[/dim]")

            current_score = baseline_score

            for iteration in range(self.max_iterations):
                console.print(f"\n[cyan]Iteration {iteration + 1}/{self.max_iterations}[/cyan]")

                iter_result = await self._run_iteration(
                    iteration,
                    current_score,
                    baseline_transcripts if iteration == 0 else None,
                )

                if iter_result:
                    result.iterations.append(iter_result)
                    current_score = iter_result.score_after

                    if iter_result.improvements and iter_result.applied:
                        result.total_lessons_created += 1

                if self._check_convergence(result.iterations):
                    console.print("  [green]✓ Converged![/green]")
                    result.converged = True
                    break

            result.status = CycleStatus.COMPLETED
            result.completed_at = datetime.now(UTC)
            result.final_score = current_score
            result.best_score = max(
                (i.score_after for i in result.iterations), default=current_score
            )
            result.improvement_delta = result.final_score - result.initial_score

            console.print()
            console.rule("[bold]Cycle Complete[/bold]")
            console.print(f"[dim]Status: {result.status.value}[/dim]")
            console.print(f"[dim]Iterations: {result.total_iterations}[/dim]")
            console.print(f"[dim]Baseline: {result.initial_score:.3f}[/dim]")
            console.print(f"[dim]Final: {result.final_score:.3f}[/dim]")
            console.print(f"[dim]Improvement: {result.improvement_delta:+.3f}[/dim]")
            console.print(f"[dim]Lessons created: {result.total_lessons_created}[/dim]")

        except Exception as e:
            logger.error(f"Cycle failed with error: {e}")
            result.status = CycleStatus.ERROR
            result.error_message = str(e)
            result.completed_at = datetime.now(UTC)
            if self._tool_lifecycle:
                self._tool_lifecycle.cleanup_all_failed()
                console.print("  [yellow]Cleaned up working copies due to error[/yellow]")
            raise

        return result

    async def _run_evaluation(self) -> tuple[float, list[EvalTranscript]]:
        """Run all scenarios and return (mean_score, transcripts)."""
        try:
            summary = await run_scenarios_async(
                paths=[str(p) for p in self.scenarios],
                storage_path=self.storage_path / self._experiment_id,
            )

            transcripts: list[EvalTranscript] = []
            for trial in summary.trials:
                if trial.result and trial.result.transcript:
                    transcripts.append(trial.result.transcript)

            return summary.metrics.mean_score, transcripts
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return 0.0, []

    async def _run_iteration(
        self,
        iteration: int,
        score_before: float,
        cached_transcripts: list[EvalTranscript] | None = None,
    ) -> IterationResult | None:
        if not self.targets:
            return None

        target = self.targets[0]
        target_type = self._infer_target_type(target)
        current_content = target.read_text() if target.exists() else ""

        console.print("  [dim]Analyzing failures...[/dim]")

        analyst = AgenticAnalyst(self.llm_client)
        analyses: list[AnalysisResult] = []

        transcripts = cached_transcripts or []
        if not transcripts:
            _, transcripts = await self._run_evaluation()

        for i, transcript in enumerate(transcripts[:5]):
            scenario_name = (
                transcript.messages[0].get("content", "")[:50]
                if transcript.messages
                else f"scenario-{i}"
            )
            analysis = await analyst.analyze_failure(
                transcript=transcript,
                scenario_name=scenario_name,
                scenario_goal="Complete the evaluation task successfully",
                current_content=current_content,
            )
            if analysis:
                analyses.append(analysis)
                console.print(f"    [dim]Analyzed: {scenario_name[:30]}...[/dim]")

        if not analyses:
            console.print("  [yellow]No analyses generated, skipping iteration[/yellow]")
            return IterationResult(
                iteration_num=iteration,
                score_before=score_before,
                score_after=score_before,
            )

        console.print("  [dim]Generating improvement...[/dim]")
        improver = AgenticImprover(self.llm_client)
        improvement = await improver.generate_improvement(
            analysis=analyses[0],
            target_type=target_type,
            target_path=target,
        )

        if not improvement:
            console.print("  [yellow]No improvement generated, skipping iteration[/yellow]")
            return IterationResult(
                iteration_num=iteration,
                score_before=score_before,
                score_after=score_before,
                analyses=analyses,
            )

        console.print(f"  [green]Change: {improvement.change_name}[/green]")

        console.print("  [dim]Injecting improvement...[/dim]")
        self._inject_improvement(target, improvement)

        console.print("  [dim]Measuring improvement...[/dim]")
        score_after, _ = await self._run_evaluation()
        improvement_delta = score_after - score_before

        console.print(
            f"  [dim]Score: {score_before:.3f} → {score_after:.3f} ({improvement_delta:+.3f})[/dim]"
        )

        applied = False
        if improvement_delta >= self.improvement_threshold:
            applied = True
            console.print(
                f"  [green]✓ Improvement kept (delta >= {self.improvement_threshold})[/green]"
            )
            self._consecutive_improvements[str(target)] = (
                self._consecutive_improvements.get(str(target), 0) + 1
            )
            if self._tool_lifecycle:
                self._tool_lifecycle.cleanup_success(target)
        else:
            console.print(
                f"  [yellow]✗ Improvement reverted (delta < {self.improvement_threshold})[/yellow]"
            )
            self._revert_improvement(target, improvement)
            self._consecutive_improvements[str(target)] = 0
            score_after = score_before
            if self._tool_lifecycle:
                self._tool_lifecycle.cleanup_failed(target)

        return IterationResult(
            iteration_num=iteration,
            score_before=score_before,
            score_after=score_after,
            score_delta=improvement_delta if applied else 0.0,
            analyses=analyses,
            improvements=[improvement],
            applied=applied,
        )

    def _infer_target_type(self, target: Path) -> ImprovementType:
        """Infer improvement type from path."""
        path_str = str(target).lower()
        if "skill" in path_str:
            return ImprovementType.SKILL
        elif "policy" in path_str:
            return ImprovementType.POLICY
        elif "tool" in path_str:
            return ImprovementType.TOOL
        return ImprovementType.SKILL

    def _inject_improvement(self, target: Path, improvement: ImprovementResult) -> None:
        write_path = target
        if self._tool_lifecycle:
            working_copy = self._tool_lifecycle.get_working_copy(target)
            if working_copy:
                write_path = working_copy

        if self._backup_dir:
            backup_path = (
                self._backup_dir / f"{write_path.name}.{datetime.now(UTC).strftime('%H%M%S')}.bak"
            )
            if write_path.exists():
                shutil.copy2(write_path, backup_path)

        write_path.parent.mkdir(parents=True, exist_ok=True)
        write_path.write_text(improvement.updated_content)
        console.print(f"    [dim]Wrote to: {write_path}[/dim]")

    def _revert_improvement(self, target: Path, improvement: ImprovementResult) -> None:
        write_path = target
        if self._tool_lifecycle:
            working_copy = self._tool_lifecycle.get_working_copy(target)
            if working_copy:
                write_path = working_copy

        write_path.write_text(improvement.original_content)
        console.print(f"    [dim]Reverted: {write_path}[/dim]")

    def _check_convergence(self, iterations: list[IterationResult]) -> bool:
        if len(iterations) < self.convergence_window:
            return False

        recent_scores = [i.score_after for i in iterations[-self.convergence_window :]]
        if len(recent_scores) < 2:
            return False

        mean_score = sum(recent_scores) / len(recent_scores)
        variance = sum((s - mean_score) ** 2 for s in recent_scores) / len(recent_scores)

        return variance < self.convergence_variance


__all__ = ["CycleRunner"]
