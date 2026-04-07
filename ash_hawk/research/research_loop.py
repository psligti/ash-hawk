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
from ash_hawk.types import EvalTranscript

logger = logging.getLogger(__name__)
_console = Console()

THROBBER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


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


class ScenarioProgressTracker:
    def __init__(self, total: int, scenario_names: list[str] | None = None) -> None:
        self.total = total
        self.scenario_names = scenario_names or []
        self.completed_ok: int = 0
        self.completed_error: int = 0
        self.running: int = 0
        self._current_task_id: str = ""

    def on_started(self, task_id: str = "") -> None:
        self.running += 1
        self._current_task_id = task_id

    def on_completed(self, task_id: str, score: float | None = None) -> None:
        self.running = max(0, self.running - 1)
        if score is not None and score >= 0:
            self.completed_ok += 1
        else:
            self.completed_error += 1

    def render_glyphs(self, spinner_idx: int) -> str:
        parts: list[str] = []
        parts.extend(["."] * self.completed_ok)
        parts.extend(["-"] * self.completed_error)
        parts.extend([SPINNER_CHARS[spinner_idx % len(SPINNER_CHARS)]] * self.running)
        remaining = self.total - self.completed_ok - self.completed_error - self.running
        parts.extend(["·"] * remaining)
        return "".join(parts)


@asynccontextmanager
async def progress_indicator(
    message: str = "",
    tracker: ScenarioProgressTracker | None = None,
) -> AsyncIterator[ScenarioProgressTracker | None]:
    from rich.live import Live
    from rich.text import Text

    start_time = time.time()
    spinner_idx = 0

    def _render() -> Text:
        nonlocal spinner_idx
        elapsed = time.time() - start_time
        time_str = _format_elapsed(elapsed)

        if tracker is not None:
            glyphs = tracker.render_glyphs(spinner_idx)
            return Text(f"  {message} {glyphs} [{time_str}]", style="dim cyan")
        char = THROBBER_CHARS[spinner_idx % len(THROBBER_CHARS)]
        return Text(f"  {message} {char} [{time_str}]", style="dim cyan")

    with Live(_render(), console=_console, transient=True, refresh_per_second=4) as live:

        async def _tick() -> None:
            nonlocal spinner_idx
            while True:
                await asyncio.sleep(0.25)
                spinner_idx += 1
                live.update(_render())

        tick_task = asyncio.create_task(_tick())
        try:
            yield tracker
        finally:
            tick_task.cancel()
            try:
                await tick_task
            except asyncio.CancelledError:
                pass
            elapsed = time.time() - start_time
            if tracker is not None:
                glyphs = tracker.render_glyphs(spinner_idx)
                _console.print(f"[dim]  {message} done [{_format_elapsed(elapsed)}] {glyphs}[/dim]")
            else:
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
    grader_details: dict[str, Any] | None
    transcripts: list[EvalTranscript] | None = None


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
        self._project_root: Path | None = None
        self._pending_revert: tuple[str, str] | None = None
        self._consecutive_observes: int = 0

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
        self._project_root = project_root

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

        has_promotable = any(
            self._strategy_promoter.should_promote(p)
            for p in self._strategy_promoter.get_candidate_patterns()
        )
        async with progress_indicator(f"Iter {iteration + 1}: Diagnosing"):
            report = await self._diagnosis_engine.diagnose(
                eval_results=evaluation.eval_results,
                trace_events=evaluation.trace_events,
                scores=evaluation.scores,
                experiment_log_path=None,
                grader_details=evaluation.grader_details,
                has_promotable_patterns=has_promotable,
            )
        return report

    _MAX_CONSECUTIVE_OBSERVES = 2

    def _decide(self, diagnosis: DiagnosisReport, iteration: int) -> ResearchDecision:
        """Decide next action based on diagnosis and uncertainty state."""
        effective_uncertainty = max(
            diagnosis.uncertainty_level,
            self._uncertainty.uncertainty_level,
        )

        # Force FIX after too many consecutive observes to break death spirals.
        # Without this, fallback diagnoses with no hypotheses trap the loop forever.
        if (
            self._consecutive_observes >= self._MAX_CONSECUTIVE_OBSERVES
            and self._project_root is not None
        ):
            action = ResearchAction.FIX
        elif effective_uncertainty > self._config.uncertainty_threshold:
            action = ResearchAction.OBSERVE
        elif diagnosis.recommended_action == "promote":
            action = ResearchAction.PROMOTE
        elif self._has_competing_hypotheses():
            action = ResearchAction.EXPERIMENT
        elif diagnosis.recommended_action == "experiment" and effective_uncertainty > 0.3:
            action = ResearchAction.EXPERIMENT
        else:
            action = ResearchAction.FIX

        if action is ResearchAction.OBSERVE:
            self._consecutive_observes += 1
        else:
            self._consecutive_observes = 0

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
            await self._execute_fix(decision, diagnosis, iteration)

        if promoted:
            result.strategies_promoted.extend([strategy.name for strategy in promoted])
            logger.info("Promoted %d new strategies", len(promoted))
            await self._save_state()

    async def _execute_fix(
        self,
        decision: ResearchDecision,
        diagnosis: DiagnosisReport,
        iteration: int,
    ) -> None:
        """Apply a targeted mutation to the improvement target.

        Resolves the target file from the target registry, reads its current
        content, generates an improvement via LLM, and saves the result.
        The next iteration's evaluation naturally checks whether the mutation
        improved or regressed scores.  If regression exceeds safety_threshold,
        the content is reverted.
        """
        if self._project_root is None:
            _console.print("[yellow]  Fix skipped: no project root[/yellow]")
            logger.info("Iteration %d: fix skipped — no project root", iteration)
            return

        if self._pending_revert is not None:
            await self._revert_pending()

        from ash_hawk.auto_research.llm import generate_improvement
        from ash_hawk.auto_research.target_discovery import TargetDiscovery

        discovery = TargetDiscovery(self._project_root)
        discovered = discovery.discover_all_targets()
        if not discovered:
            _console.print("[yellow]  Fix skipped: no targets on disk[/yellow]")
            logger.info("Iteration %d: fix skipped — no targets on disk", iteration)
            return

        target_name = decision.target
        if target_name:
            match = next((t for t in discovered if t.name == target_name), None)
        else:
            match = None

        if match is None:
            # No specific target or target not found — pick best from registry or first available
            active_targets = self._target_registry.get_active_targets()
            if active_targets:
                best = active_targets[0]
                match = next((t for t in discovered if t.name == best.name), None)

            if match is None:
                # Fallback: pick the first discovered target
                match = discovered[0]

        original_content = match.read_content()
        if not original_content:
            _console.print("[yellow]  Fix skipped: empty target content[/yellow]")
            logger.info("Iteration %d: fix skipped — empty content for %s", iteration, match.name)
            return

        evaluation = self._latest_eval
        transcripts: list[EvalTranscript] = (
            evaluation.transcripts if evaluation and evaluation.transcripts else []
        )
        category_scores = evaluation.category_scores if evaluation else None

        async with progress_indicator(f"Fix → {match.name}"):
            improved = await generate_improvement(
                self._llm_client,
                original_content,
                transcripts,
                failed_proposals=None,
                consecutive_failures=0,
                target_type=match.target_type.value,
                category_scores=category_scores,
            )

        if not improved:
            _console.print(f"[yellow]  Fix: no improvement generated for {match.name}[/yellow]")
            logger.info("Iteration %d: fix — no improvement generated", iteration)
            return

        saved_path = match.save_content(improved)
        self._pending_revert = (match.name, original_content)

        short_summary = improved.split("\n")[0][:80]
        _console.print(f"[green]  Fix applied → {match.name}[/green]")
        _console.print(f"[dim]  {short_summary}[/dim]")
        logger.info(
            "Iteration %d: fix applied to %s (saved: %s)", iteration, match.name, saved_path
        )

    async def _revert_pending(self) -> None:
        """Revert the pending mutation if it didn't improve scores."""
        if self._pending_revert is None:
            return

        name, original_content = self._pending_revert
        self._pending_revert = None

        evaluation = self._latest_eval
        if evaluation is None:
            return

        if evaluation.score_delta < self._config.safety_threshold:
            from ash_hawk.auto_research.target_discovery import TargetDiscovery

            if self._project_root is not None:
                discovery = TargetDiscovery(self._project_root)
                for target in discovery.discover_all_targets():
                    if target.name == name:
                        target.save_content(original_content)
                        _console.print(f"[yellow]  Reverted {name} (score regressed)[/yellow]")
                        logger.info("Reverted %s due to score regression", name)
                        break

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
            return self._build_snapshot(0.0, {}, [], {}, None, iteration, [])

        scenario_names = [p.stem for p in scenarios]
        tracker = ScenarioProgressTracker(
            total=len(scenarios),
            scenario_names=scenario_names,
        )

        async def _on_trial_progress(
            completed: int, running: int, _total: int, _task_id: str
        ) -> None:
            tracker.running = running
            tracker.completed_ok = completed - tracker.completed_error

        try:
            logger.info(
                "Research iter %d: _evaluate_scenarios started with %d scenarios",
                iteration,
                len(scenarios),
            )
            async with progress_indicator(f"Iter {iteration + 1}: Evaluating", tracker=tracker):
                summary = await run_scenarios_async(
                    paths=[str(p) for p in scenarios],
                    storage_path=self._storage_path,
                    show_failure_patterns=False,
                    on_trial_progress=_on_trial_progress,
                )
                tracker.completed_ok = 0
                tracker.completed_error = 0
                tracker.running = 0
                for trial in summary.trials:
                    if trial.result is not None:
                        if trial.status == "error":
                            tracker.completed_error += 1
                        else:
                            tracker.completed_ok += 1
        except Exception as exc:
            logger.warning(
                "Iteration %d evaluation failed: %s — skipping diagnosis", iteration, exc
            )
            return None

        eval_results: dict[str, float] = {}
        trace_events: list[dict[str, str | int | float | bool | list[str]]] = []
        category_scores: dict[str, float] = {}
        grader_details: dict[str, Any] | None = None
        transcripts: list[EvalTranscript] = []

        for trial in summary.trials:
            result = trial.result
            if result is None:
                continue
            eval_results[trial.task_id] = float(result.aggregate_score)
            trace_events.extend(self._normalize_trace_events(result.transcript.trace_events))
            category_scores = self._merge_category_scores(category_scores, result.grader_results)
            if grader_details is None:
                grader_details = self._extract_emotional_details(result.grader_results)
            transcripts.append(result.transcript)

        mean_score = float(summary.metrics.mean_score)

        all_zero = eval_results and all(v == 0.0 for v in eval_results.values())
        if all_zero:
            error_traces = [
                t.result.transcript.error_trace
                for t in summary.trials
                if t.result is not None and t.result.transcript.error_trace
            ]
            if error_traces:
                first_error = error_traces[0][:200]
                _console.print(
                    f"[red]  All scores 0.0 — adapter likely failed: {first_error}[/red]"
                )
            else:
                _console.print("[red]  All scores 0.0 — no agent output produced[/red]")
            logger.warning(
                "Iteration %d: all evaluation scores are 0.0 — check adapter configuration",
                iteration,
            )
        return self._build_snapshot(
            mean_score,
            eval_results,
            trace_events,
            category_scores,
            grader_details,
            iteration,
            transcripts,
        )

    def _build_snapshot(
        self,
        mean_score: float,
        eval_results: dict[str, float],
        trace_events: list[dict[str, str | int | float | bool | list[str]]],
        category_scores: dict[str, float],
        grader_details: dict[str, Any] | None,
        iteration: int,
        transcripts: list[EvalTranscript] | None = None,
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
            grader_details=grader_details,
            transcripts=transcripts or [],
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
    def _extract_emotional_details(
        grader_results: Sequence[object],
    ) -> dict[str, Any] | None:
        for grader_result in grader_results:
            grader_type = getattr(grader_result, "grader_type", None)
            if grader_type != "emotional":
                continue
            details = getattr(grader_result, "details", None)
            if isinstance(details, dict):
                return cast(dict[str, Any], details)
        return None

    @staticmethod
    def _summarize_diagnosis(diagnosis: DiagnosisReport) -> str:
        if diagnosis.hypotheses:
            top = max(diagnosis.hypotheses, key=lambda h: h.confidence)
            if top.description:
                return top.description
        if diagnosis.primary_hypothesis:
            return diagnosis.primary_hypothesis
        return diagnosis.recommended_action or "observe"
