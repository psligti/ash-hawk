# type-hygiene: skip-file
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console

from ash_hawk.improve.diagnose import diagnose_failures
from ash_hawk.improve.patch import ProposedPatch, propose_patch, write_patch
from ash_hawk.tracing import get_telemetry

if TYPE_CHECKING:
    from ash_hawk.agents.agent_mutator import AgentMutator
    from ash_hawk.types import EvalRunSummary

logger = logging.getLogger(__name__)


@dataclass
class ImprovementResult:
    iterations: int
    initial_pass_rate: float
    final_pass_rate: float
    patches_proposed: list[ProposedPatch]
    patches_applied: list[str]
    trace_path: Path | None
    mutation_history: list[dict[str, Any]] = field(default_factory=list)
    convergence_achieved: bool = False


async def improve(
    suite_path: str,
    agent_name: str = "build",
    agent_path: Path | None = None,
    target: float = 1.0,
    max_iterations: int = 5,
    trace_dir: Path | None = None,
    output_dir: Path | None = None,
    iteration_timeout_seconds: float = 300.0,
    eval_repeats: int = 3,
    score_threshold: float = 0.02,
    lessons_dir: Path | None = None,
    console: Console | None = None,
) -> ImprovementResult:
    from ash_hawk.improve.hypothesis_ranker import HypothesisRanker
    from ash_hawk.improve.lesson_store import Lesson, LessonStore

    lesson_store = LessonStore(lessons_dir=lessons_dir)
    ranker = HypothesisRanker(lesson_store=lesson_store)

    patches: list[ProposedPatch] = []
    initial_pass_rate = 0.0
    final_pass_rate = 0.0
    mutation_history: list[dict[str, Any]] = []
    convergence_achieved = False

    mutator: AgentMutator | None = None
    snapshot_hash: str | None = None
    agent_content: dict[str, str] | None = None

    if agent_path is not None:
        from ash_hawk.agents.agent_mutator import AgentMutator

        uuid_hex = uuid.uuid4().hex
        mutator = AgentMutator(agent_path, run_id=f"improve-{uuid_hex[:8]}")
        snapshot_hash = mutator.snapshot()
        agent_content = mutator.scan()

    try:
        for i in range(max_iterations):
            # --- Run evals N times and compute mean pass rate ---
            pass_rates: list[float] = []
            eval_errors: list[str] = []
            for repeat_idx in range(eval_repeats):
                try:
                    summary = await _run_eval(
                        suite_path, agent_name, iteration_timeout_seconds, agent_path
                    )
                    pass_rates.append(summary.metrics.pass_rate)
                    get_telemetry().emit(
                        "improve.eval_repeat",
                        iteration=i,
                        repeat=repeat_idx,
                        suite=suite_path,
                        trial_count=len(summary.trials),
                        pass_rate=round(summary.metrics.pass_rate, 4),
                        mean_score=round(summary.metrics.mean_score, 4),
                        completed=summary.metrics.completed_tasks,
                    )
                except Exception as exc:
                    logger.warning("Eval run failed in iteration %d", i, exc_info=True)
                    eval_errors.append(str(exc))

            if not pass_rates:
                logger.warning("Iteration %d: all eval runs failed, skipping", i)
                get_telemetry().emit(
                    "improve.iteration_all_evals_failed",
                    iteration=i,
                    suite=suite_path,
                    errors=eval_errors,
                )
                continue

            mean_pass_rate = sum(pass_rates) / len(pass_rates)
            if i == 0:
                initial_pass_rate = mean_pass_rate
            final_pass_rate = mean_pass_rate

            logger.debug(
                "Iteration %d: mean_pass_rate=%.4f (runs=%d) target=%.2f",
                i,
                mean_pass_rate,
                len(pass_rates),
                target,
            )

            if console is not None:
                color = "green" if mean_pass_rate >= target else "yellow"
                console.print(
                    f"  [bold]Iteration {i + 1}/{max_iterations}[/bold]  "
                    f"score=[{color}]{mean_pass_rate:.2%}[/{color}]  "
                    f"target={target:.0%}"
                )

            if mean_pass_rate >= target:
                logger.info("Target reached at iteration %d", i)
                if console is not None:
                    console.print(
                        f"  [bold green]✓ Target reached: "
                        f"{mean_pass_rate:.2%} >= {target:.0%}[/bold green]"
                    )
                convergence_achieved = True
                break

            # --- Gather failures from the last eval run ---
            last_summary = await _run_eval(
                suite_path, agent_name, iteration_timeout_seconds, agent_path
            )
            get_telemetry().emit(
                "improve.failure_check",
                iteration=i,
                suite=suite_path,
                trial_count=len(last_summary.trials),
                completed=last_summary.metrics.completed_tasks,
                pass_rate=round(last_summary.metrics.pass_rate, 4),
            )
            if not last_summary.trials:
                logger.warning(
                    "No trials completed in re-evaluation for iteration %d, continuing", i
                )
                get_telemetry().emit(
                    "improve.no_trials_from_reeval",
                    iteration=i,
                    suite=suite_path,
                )
                continue

            failures = [
                t
                for t in last_summary.trials
                if t.result is not None and not t.result.aggregate_passed
            ]
            get_telemetry().emit(
                "improve.failures_detected",
                iteration=i,
                failure_count=len(failures),
                trial_ids=[t.id for t in failures],
            )
            if not failures:
                logger.info("No failures found in iteration %d, stopping", i)
                if console is not None:
                    console.print("  [bold green]✓ All scenarios passing[/bold green]")
                break

            # --- Diagnose with agent content context + lesson history ---
            try:
                diagnoses = await diagnose_failures(
                    failures, trace_dir, agent_content, lesson_store=lesson_store
                )
            except Exception:
                logger.warning("Diagnosis failed in iteration %d", i, exc_info=True)
                continue

            logger.info("Diagnosed %d failures in iteration %d", len(diagnoses), i)

            # --- Rank hypotheses, test one at a time ---
            ranking = ranker.rank(diagnoses)

            logger.debug(
                "Iteration %d: %d hypotheses ranked (%d filtered as already tried)",
                i,
                ranking.total_candidates,
                ranking.filtered_as_tried,
            )

            if console is not None and ranking.hypotheses:
                console.print(
                    f"  [cyan]Hypotheses:[/cyan] "
                    f"{ranking.total_candidates} generated, "
                    f"{ranking.filtered_as_tried} filtered (already tried)"
                )

            mean_old = mean_pass_rate
            for hyp in ranking.hypotheses:
                logger.debug(
                    "Testing hypothesis rank=%d trial=%s impact=%.2f novel=%.2f: %s",
                    hyp.rank,
                    hyp.diagnosis.trial_id,
                    hyp.estimated_impact,
                    hyp.novelty_score,
                    hyp.diagnosis.failure_summary[:80],
                )

                if console is not None:
                    files_str = ", ".join(hyp.diagnosis.target_files[:3])
                    console.print(
                        f"    [dim]Testing rank={hyp.rank}[/dim]  "
                        f"impact={hyp.estimated_impact:.2f}  "
                        f"files=[cyan]{files_str}[/cyan]"
                    )

                try:
                    patch = await propose_patch(hyp.diagnosis, agent_content)
                    patches.append(patch)

                    if mutator is not None and patch.agent_relative_path and patch.content:
                        mutator.write_file(patch.agent_relative_path, patch.content)
                        logger.info("Applied mutation: %s", patch.agent_relative_path)
                        if console is not None:
                            console.print(
                                f"    [dim]Applied mutation to {patch.agent_relative_path}[/dim]"
                            )
                    else:
                        patch_path = write_patch(patch, output_dir)
                        logger.info("Patch proposed: %s for %s", patch_path, patch.file_path)
                        continue  # Can't test without mutator

                    # Re-evaluate after single hypothesis
                    new_pass_rates: list[float] = []
                    for _ in range(eval_repeats):
                        try:
                            new_summary = await _run_eval(
                                suite_path,
                                agent_name,
                                iteration_timeout_seconds,
                                agent_path,
                            )
                            new_pass_rates.append(new_summary.metrics.pass_rate)
                        except Exception:
                            logger.warning("Post-mutation eval failed iter %d", i, exc_info=True)

                    if not new_pass_rates:
                        logger.warning(
                            "All post-mutation evals failed for hypothesis rank=%d",
                            hyp.rank,
                        )
                        mutator.revert_all()
                        if snapshot_hash is not None:
                            snapshot_hash = mutator.snapshot()
                            agent_content = mutator.scan()
                        continue

                    mean_new = sum(new_pass_rates) / len(new_pass_rates)
                    delta = mean_new - mean_old

                    kept = delta > score_threshold
                    logger.debug(
                        "Hypothesis result: rank=%d delta=%.4f (%.4f -> %.4f) %s",
                        hyp.rank,
                        delta,
                        mean_old,
                        mean_new,
                        "KEPT" if kept else "REVERTED",
                    )

                    if console is not None:
                        if kept:
                            console.print(
                                f"    [green]✓ KEPT[/green]  "
                                f"delta=[green]{delta:+.4f}[/green]  "
                                f"({mean_old:.4f} → {mean_new:.4f})"
                            )
                        else:
                            console.print(
                                f"    [red]✗ REVERTED[/red]  "
                                f"delta=[red]{delta:+.4f}[/red]  "
                                f"({mean_old:.4f} → {mean_new:.4f})"
                            )

                    lesson = Lesson(
                        lesson_id=uuid.uuid4().hex[:12],
                        trial_id=hyp.diagnosis.trial_id,
                        hypothesis_summary=hyp.diagnosis.failure_summary,
                        root_cause=hyp.diagnosis.root_cause,
                        target_files=hyp.diagnosis.target_files,
                        outcome="kept" if kept else "reverted",
                        score_before=mean_old,
                        score_after=mean_new,
                        score_delta=delta,
                        iteration=i,
                        agent_path=str(agent_path) if agent_path else None,
                    )
                    lesson_store.save(lesson)

                    if console is not None:
                        outcome = "kept" if kept else "reverted"
                        console.print(
                            f"    [dim]Lesson saved: {lesson.lesson_id} "
                            f"({outcome} Δ={delta:+.4f})[/dim]"
                        )

                    iteration_record: dict[str, Any] = {
                        "iteration": i,
                        "hypothesis_rank": hyp.rank,
                        "trial_id": hyp.diagnosis.trial_id,
                        "mean_pass_rate_before": mean_old,
                        "mean_pass_rate_after": mean_new,
                        "improvement": delta,
                        "kept": kept,
                        "lesson_id": lesson.lesson_id,
                    }
                    mutation_history.append(iteration_record)

                    if kept:
                        final_pass_rate = mean_new
                        snapshot_hash = mutator.snapshot()
                        agent_content = mutator.scan()
                        break
                    else:
                        mutator.revert_all()
                        if snapshot_hash is not None:
                            snapshot_hash = mutator.snapshot()
                            agent_content = mutator.scan()
                except Exception:
                    logger.warning("Hypothesis %s failed", hyp.diagnosis.trial_id, exc_info=True)

    finally:
        if mutator is not None:
            mutator.cleanup()

    return ImprovementResult(
        iterations=max_iterations,
        initial_pass_rate=initial_pass_rate,
        final_pass_rate=final_pass_rate,
        patches_proposed=patches,
        patches_applied=[],
        trace_path=trace_dir,
        mutation_history=mutation_history,
        convergence_achieved=convergence_achieved,
    )


async def _run_eval(
    suite_path: str,
    agent_name: str,
    timeout: float,
    agent_path: Path | None = None,
) -> EvalRunSummary:
    from ash_hawk.scenario.runner import run_scenarios_async

    return await run_scenarios_async([suite_path], agent_path=agent_path)
