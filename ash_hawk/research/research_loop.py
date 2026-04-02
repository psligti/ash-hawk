from __future__ import annotations  # type-hygiene: skip-file

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence, cast

from rich.console import Console

from ash_hawk.auto_research.types import IterationResult
from ash_hawk.research.diagnosis import DiagnosisEngine, DiagnosisReport
from ash_hawk.research.strategy_promoter import (
    PromotedStrategy,
    StrategyPattern,
    StrategyPromoter,
)
from ash_hawk.research.target_registry import TargetEntry, TargetRegistry
from ash_hawk.research.types import (
    ResearchAction,
    ResearchDecision,
    ResearchLoopConfig,
    ResearchLoopResult,
    TargetSurface,
)
from ash_hawk.research.uncertainty import UncertaintyModel
from ash_hawk.scenario import run_scenarios_async

logger = logging.getLogger(__name__)
_console = Console()

THROBBER_CHARS = ".+/"


def _format_elapsed(elapsed: float) -> str:
    if elapsed >= 3600:
        hours = int(elapsed // 3600)
        mins = int((elapsed % 3600) // 60)
        secs = int(elapsed % 60)
        return f"{hours}h{mins}m{secs}s"
    if elapsed >= 60:
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        return f"{mins}m{secs}s"
    return f"{elapsed:.1f}s"


@asynccontextmanager
async def progress_indicator(message: str = "") -> AsyncIterator[None]:
    from rich.live import Live
    from rich.text import Text

    start_time = time.time()
    throbber_idx = 0

    def _render() -> Text:
        nonlocal throbber_idx
        elapsed = time.time() - start_time
        time_str = _format_elapsed(elapsed)
        char = THROBBER_CHARS[throbber_idx]
        return Text(f"  {message} {char} [{time_str}]", style="dim cyan")

    with Live(_render(), console=_console, transient=True, refresh_per_second=4) as live:

        async def _tick() -> None:
            nonlocal throbber_idx
            while True:
                await asyncio.sleep(0.25)
                throbber_idx = (throbber_idx + 1) % len(THROBBER_CHARS)
                live.update(_render())

        tick_task = asyncio.create_task(_tick())
        try:
            yield
        finally:
            tick_task.cancel()
            try:
                await tick_task
            except asyncio.CancelledError:
                pass
            elapsed = time.time() - start_time
            _console.print(f"[dim]  {message} done [{_format_elapsed(elapsed)}][/dim]")


@dataclass
class EvaluationSnapshot:
    mean_score: float
    eval_results: dict[str, float]
    trace_events: list[dict[str, str | int | float | bool | list[str]]]
    scores: dict[str, float]
    category_scores: dict[str, float]
    previous_score: float
    score_delta: float


class ResearchLoop:
    """Orchestrates the research cycle: diagnose → decide → act → learn."""

    def __init__(
        self,
        config: ResearchLoopConfig | None = None,
        llm_client: object | None = None,
        storage_path: Path | None = None,
    ) -> None:
        self._config = config or ResearchLoopConfig()
        self._llm_client = llm_client
        self._storage_path = storage_path or self._config.storage_path
        self._diagnosis_engine = DiagnosisEngine(llm_client=llm_client)
        self._uncertainty = UncertaintyModel(
            storage_path=storage_path / "uncertainty" if storage_path else None,
            max_hypotheses=self._config.max_hypotheses,
            max_evidence=self._config.max_evidence_per_hypothesis,
        )
        self._target_registry = TargetRegistry(
            storage_path=storage_path / "targets" if storage_path else None,
            min_active=self._config.min_active_targets,
        )
        self._strategy_promoter = StrategyPromoter(
            storage_path=storage_path / "strategies" if storage_path else None,
        )
        self._diagnosis_count = 0
        self._score_history: list[float] = []
        self._iteration_results: list[IterationResult] = []
        self._latest_eval: EvaluationSnapshot | None = None

    async def run(
        self,
        scenarios: list[Path],
        project_root: Path | None = None,
    ) -> ResearchLoopResult:
        """Run the full research loop.

        1. Initialize components, load persistent state
        2. Run baseline evaluation
        3. For each iteration:
           a. Diagnose: why did current state happen?
           b. Update uncertainty model
           c. Decide: observe, fix, experiment, or promote?
           d. Execute action (with human approval gate if enabled)
           e. Every d_step_interval: discover new targets
           f. Every prune_interval: prune low-correlation targets
        4. Save all state
        5. Return ResearchLoopResult
        """
        result = ResearchLoopResult(started_at=datetime.now(UTC))
        self._storage_path.mkdir(parents=True, exist_ok=True)

        if self._storage_path:
            self._uncertainty = UncertaintyModel.load(
                self._storage_path / "uncertainty",
                max_hypotheses=self._config.max_hypotheses,
                max_evidence=self._config.max_evidence_per_hypothesis,
            )
            self._target_registry = TargetRegistry.load(
                self._storage_path / "targets",
                min_active=self._config.min_active_targets,
            )
            self._strategy_promoter = StrategyPromoter.load(
                self._storage_path / "strategies",
            )

        result.uncertainty_before = self._uncertainty.uncertainty_level

        for i in range(self._config.iterations):
            _console.print(f"[cyan]Research iteration {i + 1}/{self._config.iterations}[/cyan]")
            if self._diagnosis_count >= self._config.max_diagnoses_per_run:
                logger.info("LLM budget exhausted, stopping research loop")
                break

            diagnosis = await self._diagnose(scenarios, i)
            if diagnosis is None:
                continue

            evaluation = self._latest_eval
            if evaluation:
                _console.print(
                    f"[dim]  Score: {evaluation.mean_score:.3f} | "
                    f"delta: {evaluation.score_delta:+.3f}[/dim]"
                )

            cause_categories = (
                ", ".join([category.value for category in diagnosis.cause_categories]) or "unknown"
            )
            hypothesis_summary = ""
            if diagnosis.hypotheses:
                top = max(diagnosis.hypotheses, key=lambda h: h.confidence)
                short = top.description[:80] + ("..." if len(top.description) > 80 else "")
                hypothesis_summary = f" | hypothesis: {short}"
            _console.print(
                f"[dim]  Causes: {cause_categories} | "
                f"LLM action: {diagnosis.recommended_action} | "
                f"uncertainty: {diagnosis.uncertainty_level:.3f}{hypothesis_summary}[/dim]"
            )

            self._uncertainty.update_from_diagnosis(diagnosis)

            decision = self._decide(diagnosis, i)
            result.decisions.append(decision)

            _console.print(
                f"[dim]  Decision: {decision.action.value} "
                f"(confidence: {decision.confidence:.3f})[/dim]"
            )

            await self._execute_decision(decision, scenarios, project_root, i, diagnosis, result)

            if i > 0 and i % self._config.d_step_interval == 0 and project_root:
                new_targets = self._target_registry.discover_targets(project_root)
                _console.print(f"[dim]  Discovered {len(new_targets)} new targets[/dim]")
                logger.info("Discovered %d new targets", len(new_targets))

            if i > 0 and i % self._config.prune_interval == 0:
                pruned = self._target_registry.prune_low_correlation()
                _console.print(f"[dim]  Pruned {len(pruned)} low-correlation targets[/dim]")
                if pruned:
                    logger.info("Pruned %d low-correlation targets", len(pruned))

        await self._save_state()

        result.diagnoses_count = self._diagnosis_count
        result.uncertainty_after = self._uncertainty.uncertainty_level
        result.improvement_delta = result.uncertainty_before - result.uncertainty_after
        result.completed_at = datetime.now(UTC)
        return result

    async def _diagnose(self, scenarios: list[Path], iteration: int) -> DiagnosisReport | None:
        """Run diagnosis. Returns None if budget exhausted or no data."""
        if self._diagnosis_count >= self._config.max_diagnoses_per_run:
            return None

        evaluation = await self._evaluate_scenarios(scenarios, iteration)
        if evaluation is None:
            return None

        self._latest_eval = evaluation
        self._diagnosis_count += 1

        async with progress_indicator(f"Iter {iteration + 1}: Diagnosing"):
            report = await self._diagnosis_engine.diagnose(
                eval_results=evaluation.eval_results,
                trace_events=evaluation.trace_events,
                scores=evaluation.scores,
                experiment_log_path=None,
            )
        return report

    def _decide(self, diagnosis: DiagnosisReport, iteration: int) -> ResearchDecision:
        """Decide next action based on diagnosis and uncertainty state."""
        effective_uncertainty = max(
            diagnosis.uncertainty_level,
            self._uncertainty.uncertainty_level,
        )

        if effective_uncertainty > self._config.uncertainty_threshold:
            action = ResearchAction.OBSERVE
        elif diagnosis.recommended_action == "promote":
            action = ResearchAction.PROMOTE
        elif self._has_competing_hypotheses():
            action = ResearchAction.EXPERIMENT
        elif diagnosis.recommended_action == "experiment" and effective_uncertainty > 0.3:
            action = ResearchAction.EXPERIMENT
        else:
            action = ResearchAction.FIX

        target: str | None = None
        if action is not ResearchAction.OBSERVE:
            active_targets = self._target_registry.get_active_targets()
            if active_targets:
                target = active_targets[0].name

        return ResearchDecision(
            action=action,
            rationale=diagnosis.recommended_action,
            target=target,
            expected_info_gain=effective_uncertainty,
            confidence=1.0 - effective_uncertainty,
        )

    def _has_competing_hypotheses(self) -> bool:
        competing = self._uncertainty.get_competing_hypotheses()
        return len(competing) >= 2 and all(h.confidence < 0.4 for h in competing)

    async def _execute_decision(
        self,
        decision: ResearchDecision,
        scenarios: list[Path],
        project_root: Path | None,
        iteration: int,
        diagnosis: DiagnosisReport,
        result: ResearchLoopResult,
    ) -> None:
        """Execute the decided action."""
        evaluation = self._latest_eval
        score_delta = evaluation.score_delta if evaluation else 0.0
        score_before = evaluation.previous_score if evaluation else 0.0
        score_after = evaluation.mean_score if evaluation else 0.0

        improvement_text = self._summarize_diagnosis(diagnosis)
        self._iteration_results.append(
            IterationResult(
                iteration_num=iteration,
                score_before=score_before,
                score_after=score_after,
                improvement_text=improvement_text,
                applied=decision.action is not ResearchAction.OBSERVE,
                category_scores=evaluation.category_scores if evaluation else None,
            )
        )

        if decision.target:
            self._target_registry.register(
                TargetEntry(
                    name=decision.target,
                    surface=TargetSurface.PROMPT,
                    description="Research loop target",
                )
            )
            if decision.action is not ResearchAction.OBSERVE:
                self._target_registry.update_correlation(decision.target, score_delta)

        if score_delta < self._config.safety_threshold:
            logger.warning(
                "Iteration %d score regression %.3f below safety threshold",
                iteration,
                score_delta,
            )

        if not diagnosis.hypotheses and self._uncertainty.uncertainty_level >= 1.0:
            logger.warning("Uncertainty stayed at 1.0 — no hypotheses left, observe only")

        promoted: list[PromotedStrategy] = []
        if decision.action == ResearchAction.PROMOTE:
            async with progress_indicator("Promoting strategies"):
                promoted = await self._execute_promote()
            _console.print(f"[green]  Promoted {len(promoted)} strategies[/green]")
        elif decision.action == ResearchAction.OBSERVE:
            _console.print(
                f"[yellow]  Observing (uncertainty > {self._config.uncertainty_threshold})[/yellow]"
            )
            logger.info("Iteration %d: observe (high uncertainty)", iteration)
        elif decision.action == ResearchAction.EXPERIMENT:
            _console.print("[cyan]  Experimenting (competing hypotheses)[/cyan]")
            logger.info("Iteration %d: experiment (competing hypotheses)", iteration)
        elif decision.action == ResearchAction.FIX:
            if self._config.human_approval_required:
                _console.print("[yellow]  Fix requires human approval[/yellow]")
                logger.info(
                    "Iteration %d: fix requires human approval (skipping in auto mode)",
                    iteration,
                )
            else:
                target_msg = f" → {decision.target}" if decision.target else ""
                _console.print(
                    f"[green]  Fix recommended{target_msg} (advisory — no auto-mutation)[/green]"
                )
                logger.info("Iteration %d: fix recommended (advisory)", iteration)

        if promoted:
            result.strategies_promoted.extend([strategy.name for strategy in promoted])
            logger.info("Promoted %d new strategies", len(promoted))
            await self._save_state()

    async def _execute_promote(self) -> list[PromotedStrategy]:
        """Check for patterns to promote."""
        patterns = self._strategy_promoter.detect_patterns(self._iteration_results)
        candidates: dict[str, StrategyPattern] = {
            pattern.pattern_id: pattern for pattern in patterns
        }
        for pattern in self._strategy_promoter.get_candidate_patterns():
            candidates.setdefault(pattern.pattern_id, pattern)

        promoted: list[PromotedStrategy] = []
        for pattern in candidates.values():
            if self._strategy_promoter.should_promote(pattern):
                promoted.append(await self._strategy_promoter.promote(pattern))
        return promoted

    async def _save_state(self) -> None:
        """Save all persistent state."""
        await self._uncertainty.save()
        await self._target_registry.save()
        await self._strategy_promoter.save()

    async def _evaluate_scenarios(
        self,
        scenarios: list[Path],
        iteration: int,
    ) -> EvaluationSnapshot | None:
        if not scenarios:
            return self._build_snapshot(0.0, {}, [], {}, iteration)

        try:
            async with progress_indicator(f"Iter {iteration + 1}: Evaluating"):
                summary = await run_scenarios_async(
                    paths=[str(p) for p in scenarios],
                    storage_path=self._storage_path,
                    show_failure_patterns=False,
                )
        except Exception as exc:
            logger.warning(
                "Iteration %d evaluation failed: %s — skipping diagnosis", iteration, exc
            )
            return None

        eval_results: dict[str, float] = {}
        trace_events: list[dict[str, str | int | float | bool | list[str]]] = []
        category_scores: dict[str, float] = {}

        for trial in summary.trials:
            result = trial.result
            if result is None:
                continue
            eval_results[trial.task_id] = float(result.aggregate_score)
            trace_events.extend(self._normalize_trace_events(result.transcript.trace_events))
            category_scores = self._merge_category_scores(category_scores, result.grader_results)

        mean_score = float(summary.metrics.mean_score)
        return self._build_snapshot(
            mean_score, eval_results, trace_events, category_scores, iteration
        )

    def _build_snapshot(
        self,
        mean_score: float,
        eval_results: dict[str, float],
        trace_events: list[dict[str, str | int | float | bool | list[str]]],
        category_scores: dict[str, float],
        iteration: int,
    ) -> EvaluationSnapshot:
        if not eval_results:
            eval_results["mean_score"] = mean_score
        eval_results.setdefault("score", mean_score)
        eval_results.setdefault("mean_score", mean_score)

        previous_score = self._score_history[-1] if self._score_history else mean_score
        score_delta = mean_score - previous_score if self._score_history else 0.0
        self._score_history.append(mean_score)

        scores = {
            "iteration": float(iteration),
            "mean_score": mean_score,
            "score": mean_score,
            "previous_score": previous_score,
            "score_delta": score_delta,
        }

        return EvaluationSnapshot(
            mean_score=mean_score,
            eval_results=eval_results,
            trace_events=trace_events,
            scores=scores,
            category_scores=category_scores,
            previous_score=previous_score,
            score_delta=score_delta,
        )

    @staticmethod
    def _normalize_trace_events(
        trace_events: Sequence[dict[str, Any]],
    ) -> list[dict[str, str | int | float | bool | list[str]]]:
        normalized: list[dict[str, str | int | float | bool | list[str]]] = []
        for event in trace_events:
            cleaned: dict[str, str | int | float | bool | list[str]] = {}
            for key, value in event.items():
                if isinstance(value, str | int | float | bool):
                    cleaned[str(key)] = value
                elif isinstance(value, list):
                    values = [item for item in cast(list[object], value) if isinstance(item, str)]
                    if values:
                        cleaned[str(key)] = values
            if cleaned:
                normalized.append(cleaned)
        return normalized

    @staticmethod
    def _merge_category_scores(
        category_scores: dict[str, float],
        grader_results: Sequence[object],
    ) -> dict[str, float]:
        for grader_result in grader_results:
            details = getattr(grader_result, "details", None)
            if not isinstance(details, dict):
                continue
            details_dict = cast(dict[str, Any], details)
            summary = details_dict.get("category_summary")
            if not isinstance(summary, dict):
                continue
            summary_dict = cast(dict[str, Any], summary)
            for cat_id, cat_score in summary_dict.items():
                if not isinstance(cat_score, int | float):
                    continue
                if cat_id in category_scores:
                    category_scores[cat_id] = (category_scores[cat_id] + float(cat_score)) / 2
                else:
                    category_scores[cat_id] = float(cat_score)
        return category_scores

    @staticmethod
    def _summarize_diagnosis(diagnosis: DiagnosisReport) -> str:
        if diagnosis.hypotheses:
            top = max(diagnosis.hypotheses, key=lambda h: h.confidence)
            if top.description:
                return top.description
        if diagnosis.primary_hypothesis:
            return diagnosis.primary_hypothesis
        return diagnosis.recommended_action or "observe"
