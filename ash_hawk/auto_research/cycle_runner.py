"""Orchestrated improvement cycle runner.

Implements the 5-step improvement workflow:
1. Run scenarios, capture telemetry/traces/transcripts
2. Run graders, capture scores and feedback
3. Generate multiple hypotheses, rank them, consult past lessons
4. Apply changes one-by-one, re-run, compare, keep or revert
5. Track lessons learned persistently

This module is the top-level orchestrator that wires together:
- improve/loop.py (core improvement loop with one-by-one testing)
- improve/lesson_store.py (persistent lesson tracking)
- improve/hypothesis_ranker.py (multi-hypothesis ranking)
- improvement/fixture_splitter.py (train/holdout split)
- improvement/guardrails.py (safety rails)
- auto_research/convergence.py (convergence detection)
- auto_research/knowledge_promotion.py (knowledge promotion)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from rich.console import Console

from ash_hawk.auto_research.convergence import (
    ConvergenceDetector,
    ConvergenceResult,
    ScoreRecord,
)
from ash_hawk.auto_research.knowledge_promotion import (
    KnowledgePromoter,
    PromotionCriteria,
)
from ash_hawk.improve.lesson_store import LessonStore
from ash_hawk.improvement.fixture_splitter import FixtureSplitter
from ash_hawk.improvement.guardrails import (
    GuardrailChecker,
    GuardrailConfig,
    IterationRecord,
)

logger = logging.getLogger(__name__)


class CycleStatus(StrEnum):
    """Status of an improvement cycle."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CONVERGED = "converged"
    GUARDRAIL_STOPPED = "guardrail_stopped"
    ERROR = "error"


@dataclass
class CycleConfig:
    """Configuration for the improvement cycle."""

    # Iteration settings
    max_iterations: int = 100
    target_pass_rate: float = 1.0
    score_threshold: float = 0.02
    eval_repeats: int = 3
    iteration_timeout_seconds: float = 300.0

    # Train/holdout split
    train_ratio: float = 0.7
    seed: int = 42

    # Guardrails
    guardrail_config: GuardrailConfig | None = None

    # Convergence
    convergence_window: int = 5
    convergence_variance_threshold: float = 0.001
    max_iterations_without_improvement: int = 10

    # Knowledge promotion
    promotion_criteria: PromotionCriteria | None = None
    note_lark_enabled: bool = False

    # Storage
    lessons_dir: Path | None = None
    output_dir: Path | None = None
    storage_path: Path | None = None


@dataclass
class CycleResult:
    """Result of an improvement cycle."""

    cycle_id: str
    agent_name: str
    status: CycleStatus
    initial_score: float = 0.0
    final_score: float = 0.0
    total_iterations: int = 0
    applied_count: int = 0
    reverted_count: int = 0
    promoted_lessons: int = 0
    convergence_result: ConvergenceResult | None = None
    guardrail_reason: str | None = None
    duration_seconds: float = 0.0
    train_scenarios: list[str] = field(default_factory=list)
    holdout_scenarios: list[str] = field(default_factory=list)

    @property
    def improvement_delta(self) -> float:
        """Improvement from initial to final score."""
        return self.final_score - self.initial_score

    @property
    def success(self) -> bool:
        """Whether the cycle completed successfully or converged."""
        return self.status in (CycleStatus.COMPLETED, CycleStatus.CONVERGED)


async def run_cycle(
    suite_path: str | Path,
    agent_name: str = "build",
    agent_path: Path | None = None,
    config: CycleConfig | None = None,
) -> CycleResult:
    """Run a complete improvement cycle.

    This is the main entry point that orchestrates the 5-step workflow:

    1. **Setup**: Initialize all components (LessonStore, ConvergenceDetector,
       GuardrailChecker, KnowledgePromoter). Split scenarios into train/holdout.
    2. **Iterate**: For each iteration up to max_iterations:
       a. Run improvement loop (which internally does: diagnose, rank hypotheses,
          test one-by-one, keep/revert, capture lessons)
       b. Check convergence (plateau, no improvement, regression)
       c. Check guardrails (max reverts, holdout drops)
       d. Promote validated lessons
    3. **Report**: Return CycleResult with full statistics.

    Logs extensively at every step for user visibility:
    - logger.info("=== Improvement Cycle Started === cycle_id=%s agent=%s", ...)
    - logger.info("Step 1: Running evaluation (iteration %d/%d)", ...)
    - logger.info("Step 2: Grading complete - score=%.4f", ...)
    - logger.info("Step 3: Generating hypotheses - %d ranked, %d filtered", ...)
    - logger.info("Step 4: Testing hypothesis rank=%d - delta=%.4f - %s", ...)
    - logger.info("Step 5: Lesson captured - id=%s outcome=%s", ...)
    - logger.warning("Guardrail triggered: %s", reason)
    - logger.info("Convergence detected: %s", reason)

    Args:
        suite_path: Path to the eval suite YAML or directory of scenarios.
        agent_name: Name of the agent to improve.
        agent_path: Path to the agent directory.
        config: Cycle configuration (uses defaults if None).

    Returns:
        CycleResult with full cycle statistics.
    """
    cfg = config or CycleConfig()
    cycle_id = f"cycle-{uuid.uuid4().hex[:8]}"
    start_time = time.time()

    console = Console()

    logger.debug("=" * 60)
    logger.debug("=== Improvement Cycle Started ===")
    logger.debug(
        "  cycle_id=%s agent=%s max_iterations=%d",
        cycle_id,
        agent_name,
        cfg.max_iterations,
    )
    logger.debug(
        "  train_ratio=%.2f seed=%d score_threshold=%.4f",
        cfg.train_ratio,
        cfg.seed,
        cfg.score_threshold,
    )
    logger.debug("=" * 60)

    console.rule("[bold cyan]Improvement Cycle[/bold cyan]")
    console.print(f"  [bold]Cycle:[/bold]           {cycle_id}")
    console.print(f"  [bold]Agent:[/bold]            {agent_name}")
    console.print(f"  [bold]Max iterations:[/bold]   {cfg.max_iterations}")
    console.print(f"  [bold]Score threshold:[/bold]  {cfg.score_threshold}")
    console.print(
        f"  [bold]Train/holdout:[/bold]     {cfg.train_ratio:.0%}/{1 - cfg.train_ratio:.0%}"
    )

    # --- Initialize components ---
    lesson_store = LessonStore(lessons_dir=cfg.lessons_dir)
    convergence = ConvergenceDetector(
        window_size=cfg.convergence_window,
        variance_threshold=cfg.convergence_variance_threshold,
        max_iterations_without_improvement=cfg.max_iterations_without_improvement,
    )
    guardrails = GuardrailChecker(config=cfg.guardrail_config)
    promoter = KnowledgePromoter(
        criteria=cfg.promotion_criteria,
        storage_dir=(cfg.lessons_dir.parent / "promoted" if cfg.lessons_dir else None),
        note_lark_enabled=cfg.note_lark_enabled,
    )

    # --- Split scenarios if directory ---
    train_paths: list[str]
    holdout_paths: list[str]
    suite_p = Path(suite_path)
    if suite_p.is_dir():
        # Discover scenarios and split
        from ash_hawk.scenario.loader import discover_scenarios

        all_scenarios = sorted(discover_scenarios(suite_p))
        if len(all_scenarios) >= 10:
            splitter = FixtureSplitter(seed=cfg.seed, train_ratio=cfg.train_ratio)
            split = splitter.split(all_scenarios)
            train_paths = [str(p) for p in split.train]
            holdout_paths = [str(p) for p in split.heldout]
            logger.debug(
                "Split %d scenarios: train=%d holdout=%d",
                split.total,
                len(split.train),
                len(split.heldout),
            )
        else:
            train_paths = [str(p) for p in all_scenarios]
            holdout_paths = []
            logger.debug(
                "Too few scenarios (%d) for splitting, using all for training",
                len(all_scenarios),
            )
    else:
        train_paths = [str(suite_p)]
        holdout_paths = []
        logger.debug("Single scenario mode: %s", suite_p)

    if train_paths:
        console.print(
            f"  [bold]Scenarios:[/bold]        "
            f"{len(train_paths)} train, {len(holdout_paths)} holdout"
        )

    # --- Run improvement loop ---
    from ash_hawk.improve.loop import improve as _improve

    result = CycleResult(
        cycle_id=cycle_id,
        agent_name=agent_name,
        status=CycleStatus.RUNNING,
        train_scenarios=train_paths,
        holdout_scenarios=holdout_paths,
    )

    try:
        # Run the core improvement loop with lesson tracking
        for suite in train_paths:
            logger.info("--- Running improvement on: %s ---", suite)

            improvement = await _improve(
                suite_path=suite,
                agent_name=agent_name,
                agent_path=agent_path,
                target=cfg.target_pass_rate,
                max_iterations=cfg.max_iterations,
                trace_dir=cfg.output_dir,
                output_dir=cfg.output_dir,
                iteration_timeout_seconds=cfg.iteration_timeout_seconds,
                eval_repeats=cfg.eval_repeats,
                score_threshold=cfg.score_threshold,
                lessons_dir=cfg.lessons_dir,
                console=console,
            )

            # Update cycle result from improvement result
            if result.initial_score == 0.0:
                result.initial_score = improvement.initial_pass_rate
            result.final_score = improvement.final_pass_rate
            result.total_iterations += improvement.iterations

            # Count applied vs reverted from mutation history
            for record in improvement.mutation_history:
                if record.get("kept", False):
                    result.applied_count += 1
                    convergence.record(
                        ScoreRecord(
                            iteration=record.get("iteration", 0),
                            score=record.get("mean_pass_rate_after", 0.0),
                            applied=True,
                            delta=record.get("improvement", 0.0),
                        )
                    )
                else:
                    result.reverted_count += 1
                    convergence.record(
                        ScoreRecord(
                            iteration=record.get("iteration", 0),
                            score=record.get("mean_pass_rate_before", 0.0),
                            applied=False,
                            delta=record.get("improvement", 0.0),
                        )
                    )

                # Check guardrails after each mutation
                guardrail_result = guardrails.record_iteration(
                    IterationRecord(
                        iteration=record.get("iteration", 0),
                        score=(
                            record.get("mean_pass_rate_after", 0.0)
                            if record.get("kept")
                            else record.get("mean_pass_rate_before", 0.0)
                        ),
                        applied=record.get("kept", False),
                    )
                )

                if guardrail_result.should_stop:
                    logger.warning("Guardrail triggered: %s", guardrail_result.reason)
                    console.print(
                        f"  [bold yellow]⚠ Guardrail: {guardrail_result.reason}[/bold yellow]"
                    )
                    result.status = CycleStatus.GUARDRAIL_STOPPED
                    result.guardrail_reason = guardrail_result.reason
                    break

            # Check convergence
            conv_result = convergence.check()
            if conv_result.converged:
                logger.info(
                    "Convergence detected: reason=%s confidence=%.2f",
                    conv_result.reason,
                    conv_result.confidence,
                )
                console.print(
                    f"  [bold cyan]⏹ Converged: {conv_result.reason} "
                    f"(confidence={conv_result.confidence:.2f})[/bold cyan]"
                )
                result.status = CycleStatus.CONVERGED
                result.convergence_result = conv_result
                break

            # Check target reached
            if improvement.convergence_achieved:
                logger.info(
                    "Target pass rate reached: %.4f >= %.4f",
                    improvement.final_pass_rate,
                    cfg.target_pass_rate,
                )
                result.status = CycleStatus.COMPLETED
                break

            if result.status == CycleStatus.GUARDRAIL_STOPPED:
                break

        # --- Promote validated lessons ---
        if result.status in (CycleStatus.COMPLETED, CycleStatus.CONVERGED):
            all_lessons = lesson_store.load_all()
            promoted = 0
            for lesson in all_lessons:
                should, reason = promoter.should_promote(lesson)
                if should:
                    promoted_lesson = promoter.promote(lesson, tags=[agent_name, cycle_id])
                    if promoted_lesson is not None:
                        promoted += 1
                        logger.info(
                            "Promoted lesson %s: %s",
                            promoted_lesson.lesson_id,
                            reason,
                        )
            result.promoted_lessons = promoted
            logger.info("Promoted %d/%d lessons", promoted, len(all_lessons))

            if promoted > 0:
                console.print(
                    f"  [bold green]★ Promoted {promoted} lesson(s) to knowledge base[/bold green]"
                )

        if result.status == CycleStatus.RUNNING:
            result.status = CycleStatus.COMPLETED

    except Exception:
        logger.exception("Improvement cycle failed: cycle_id=%s", cycle_id)
        result.status = CycleStatus.ERROR

    finally:
        result.duration_seconds = time.time() - start_time

    # --- Final report ---
    logger.debug("=" * 60)
    logger.debug("=== Improvement Cycle Complete ===")
    logger.debug("  cycle_id=%s status=%s", cycle_id, result.status.value)
    logger.debug(
        "  initial_score=%.4f final_score=%.4f delta=%.4f",
        result.initial_score,
        result.final_score,
        result.improvement_delta,
    )
    logger.debug(
        "  iterations=%d applied=%d reverted=%d",
        result.total_iterations,
        result.applied_count,
        result.reverted_count,
    )
    logger.debug(
        "  promoted_lessons=%d duration=%.1fs",
        result.promoted_lessons,
        result.duration_seconds,
    )
    logger.debug("=" * 60)

    console.rule("[bold]Cycle Summary[/bold]")

    status_color = "green" if result.success else "red"
    console.print(
        f"  [bold]Status:[/bold]            [{status_color}]{result.status.value}[/{status_color}]"
    )
    console.print(
        f"  [bold]Score:[/bold]             "
        f"{result.initial_score:.2%} → {result.final_score:.2%}  "
        f"[bold]Δ={result.improvement_delta:+.2%}[/bold]"
    )
    console.print(f"  [bold]Iterations:[/bold]        {result.total_iterations}")
    console.print(
        f"  [bold]Applied:[/bold]           [green]{result.applied_count}[/green]  "
        f"[bold]Reverted:[/bold] [red]{result.reverted_count}[/red]"
    )
    console.print(f"  [bold]Lessons stored:[/bold]    {lesson_store.lesson_count()}")
    console.print(f"  [bold]Lessons promoted:[/bold]  {result.promoted_lessons}")
    console.print(f"  [bold]Duration:[/bold]          {result.duration_seconds:.1f}s")

    if result.convergence_result:
        console.print(f"  [bold]Convergence:[/bold]       {result.convergence_result.reason}")

    if result.guardrail_reason:
        console.print(
            f"  [bold]Guardrail:[/bold]         [yellow]{result.guardrail_reason}[/{'yellow'}]"
        )

    # --- Triage paths ---
    _print_triage_paths(console, train_paths, cfg)

    return result


def _print_triage_paths(
    console: Console,
    scenario_paths: list[str],
    cfg: CycleConfig,
) -> None:
    from ash_hawk.tracing import TelemetryLog

    telemetry_path = TelemetryLog().path
    console.rule("[bold]Triage[/bold]")
    console.print(f"  [bold]Telemetry:[/bold]    {telemetry_path}")

    storage_root = cfg.storage_path or Path(".ash-hawk")
    for scenario_path in scenario_paths:
        stem = Path(scenario_path).stem
        suite_dir = storage_root / f"scenario-{stem}"
        runs_dir = suite_dir / "runs"
        if runs_dir.exists():
            run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
            if run_dirs:
                latest = run_dirs[-1]
                summary = latest / "summary.json"
                console.print(f"  [bold]Latest run:[/bold]   {summary}")
                return

    console.print("  [dim]No run artifacts found[/dim]")
