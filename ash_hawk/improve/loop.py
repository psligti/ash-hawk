# type-hygiene: skip-file
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pydantic as pd
from rich.console import Console

from ash_hawk.agents.source_workspace import (
    detect_agent_config_path,
    prepare_isolated_agent_workspace,
)
from ash_hawk.improve.diagnose import diagnose_failures
from ash_hawk.improve.hypothesis_ranker import HypothesisRanker
from ash_hawk.improve.iteration_log import (
    IterationLog,
    diagnosis_to_summary,
    write_iteration_log,
)
from ash_hawk.improve.lesson_store import Lesson, LessonStore
from ash_hawk.improve.patch import (
    ProposedPatch,
    propose_patch,
    propose_patch_via_agent,
    write_patch,
)
from ash_hawk.improve.stop_condition import ScoreRecord, StopCondition, StopConditionConfig
from ash_hawk.tracing import get_telemetry

if TYPE_CHECKING:
    from ash_hawk.agents.agent_mutator import AgentMutator
    from ash_hawk.types import EvalRunSummary

logger = logging.getLogger(__name__)


def _format_path_list(paths: list[str], limit: int = 3) -> str:
    if not paths:
        return "none"
    shown = paths[:limit]
    suffix = "" if len(paths) <= limit else f" (+{len(paths) - limit} more)"
    return ", ".join(shown) + suffix


class ImprovementResult(pd.BaseModel):
    model_config = pd.ConfigDict(extra="forbid")

    iterations: int = pd.Field(description="Total iterations run")
    initial_pass_rate: float = pd.Field(
        ge=0.0,
        le=1.0,
        description="Initial improvement score (currently backed by scenario mean_score)",
    )
    final_pass_rate: float = pd.Field(
        ge=0.0,
        le=1.0,
        description="Final improvement score (currently backed by scenario mean_score)",
    )
    patches_proposed: list[dict[str, Any]] = pd.Field(
        default_factory=list, description="Serialized ProposedPatch list"
    )
    patches_applied: list[str] = pd.Field(default_factory=list)
    trace_path: Path | None = None
    mutation_history: list[dict[str, Any]] = pd.Field(default_factory=list)
    iteration_logs: list[dict[str, Any]] = pd.Field(
        default_factory=list, description="Per-iteration structured logs"
    )
    convergence_achieved: bool = False
    stop_reasons: list[str] = pd.Field(default_factory=list, description="Final stop reasons")


def _serialize_patch(patch: ProposedPatch) -> dict[str, Any]:
    return {
        "file_path": patch.file_path,
        "description": patch.description,
        "agent_relative_path": patch.agent_relative_path,
        "rationale": patch.rationale,
        "diagnosis_trial_id": patch.diagnosis.trial_id,
    }


async def improve(
    suite_path: str,
    agent_name: str = "build",
    agent_path: Path | None = None,
    target: float = 1.0,
    max_iterations: int = 5,
    trace_dir: Path | None = None,
    output_dir: Path | None = None,
    iteration_timeout_seconds: float = 300.0,
    eval_repeats: int = 1,
    integrity_repeats: int | None = None,
    score_threshold: float = 0.02,
    lessons_dir: Path | None = None,
    stop_config: StopConditionConfig | None = None,
    console: Console | None = None,
) -> ImprovementResult:
    lesson_store = LessonStore(lessons_dir=lessons_dir)
    ranker = HypothesisRanker(lesson_store=lesson_store)
    stop_condition = StopCondition(config=stop_config)

    patches: list[dict[str, Any]] = []
    initial_score = 0.0
    final_score = 0.0
    mutation_history: list[dict[str, Any]] = []
    iteration_logs: list[dict[str, Any]] = []
    convergence_achieved = False
    final_stop_reasons: list[str] = []
    applied_files: set[str] = set()
    actual_iterations = 0
    agent_content: dict[str, str] | None = None
    original_mutator: AgentMutator | None = None

    if integrity_repeats is None:
        integrity_repeats = max(eval_repeats, 3)

    if agent_path is not None:
        from ash_hawk.agents.agent_mutator import AgentMutator

        uuid_hex = uuid.uuid4().hex
        original_mutator = AgentMutator(agent_path, run_id=f"improve-{uuid_hex[:8]}")
        agent_content = original_mutator.scan()

    if console is not None:
        console.print(
            f"[cyan]Run plan:[/cyan] baseline eval x{eval_repeats}, integrity eval x{integrity_repeats}, target {target:.0%}"
        )
        if agent_content is not None:
            console.print(
                f"[cyan]Loaded agent source:[/cyan] {len(agent_content)} text file(s) ready for mutation analysis"
            )
        console.print()

    try:
        for i in range(max_iterations):
            actual_iterations += 1
            if console is not None:
                console.print(
                    f"  [bold]Outer pass {i + 1}/{max_iterations}[/bold]  "
                    f"[dim]Step 1: baseline evaluation[/dim]"
                )
                console.print(
                    "  [dim]Each outer pass runs the full suite, then diagnoses only the failures from that fresh baseline run.[/dim]"
                )

            mean_score, last_summary, eval_errors = await _run_eval_n_times(
                suite_path,
                agent_name,
                iteration_timeout_seconds,
                agent_path,
                eval_repeats,
                console=console,
                phase_label="baseline evaluation",
            )

            if mean_score is None:
                logger.warning("Iteration %d: all eval runs failed, skipping", i)
                if console is not None:
                    console.print(
                        f"  [bold red]✗ Iteration {i + 1}:[/bold red] "
                        f"All {len(eval_errors)} eval run(s) failed"
                    )
                get_telemetry().emit(
                    "improve.iteration_all_evals_failed",
                    iteration=i,
                    suite=suite_path,
                    errors=eval_errors,
                )
                iter_log = IterationLog(
                    iteration=i,
                    baseline_score=0.0,
                    baseline_repeats=eval_repeats,
                    error=f"All {len(eval_errors)} eval runs failed",
                )
                iteration_logs.append(iter_log.model_dump())
                write_iteration_log(iter_log, output_dir)
                continue

            if i == 0:
                initial_score = mean_score
            final_score = mean_score

            logger.debug(
                "Iteration %d: mean_score=%.4f (runs=%d) target=%.2f",
                i,
                mean_score,
                eval_repeats,
                target,
            )

            if console is not None:
                color = "green" if mean_score >= target else "yellow"
                trial_info = ""
                if last_summary is not None:
                    passed = last_summary.metrics.passed_tasks
                    total = last_summary.metrics.completed_tasks
                    trial_info = f"  trials=[{passed}/{total} passed]"
                console.print(
                    f"  [bold]Baseline result:[/bold] outer pass {i + 1}/{max_iterations}  "
                    f"score=[{color}]{mean_score:.2%}[/{color}]"
                    f"{trial_info}  "
                    f"target={target:.0%}"
                )

            if mean_score >= target:
                logger.info("Target reached at iteration %d", i)
                if console is not None:
                    console.print(
                        f"  [bold green]✓ Target reached: "
                        f"{mean_score:.2%} >= {target:.0%}[/bold green]"
                    )
                convergence_achieved = True

                iter_log = IterationLog(
                    iteration=i,
                    baseline_score=mean_score,
                    baseline_repeats=eval_repeats,
                    stop_reasons=["target_reached"],
                )
                iteration_logs.append(iter_log.model_dump())
                write_iteration_log(iter_log, output_dir)
                break

            if last_summary is None:
                logger.warning("No eval summary available for iteration %d, continuing", i)
                continue

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
                t for t in last_summary.trials if t.result is None or not t.result.aggregate_passed
            ]
            failure_by_id = {t.id: t for t in failures}
            get_telemetry().emit(
                "improve.failures_detected",
                iteration=i,
                failure_count=len(failures),
                trial_ids=[t.id for t in failures],
            )
            if console is not None:
                failed_ids = [t.id for t in failures]
                console.print(
                    f"  [yellow]Failures found:[/yellow] {len(failures)}  ids={_format_path_list(failed_ids)}"
                )
                console.print(
                    "  [dim]These failures come from the latest baseline suite run. New trial ids here mean the suite was rerun, not that the same hypothesis is on a second sub-trial.[/dim]"
                )
            if not failures:
                logger.info("No failures found in iteration %d, stopping", i)
                if console is not None:
                    console.print("  [bold green]✓ All scenarios passing[/bold green]")
                iter_log = IterationLog(
                    iteration=i,
                    baseline_score=mean_score,
                    baseline_repeats=eval_repeats,
                    stop_reasons=["all_passing"],
                )
                iteration_logs.append(iter_log.model_dump())
                write_iteration_log(iter_log, output_dir)
                break

            if console is not None:
                console.print(f"  [dim]Step 2: diagnosing {len(failures)} failure(s)...[/dim]")
            try:
                diagnoses = await diagnose_failures(
                    failures,
                    trace_dir,
                    agent_content,
                    agent_path=agent_path,
                    lesson_store=lesson_store,
                    console=console,
                )
            except Exception:
                logger.warning("Diagnosis failed in iteration %d", i, exc_info=True)
                if console is not None:
                    console.print(f"  [bold red]✗ Diagnosis failed in iteration {i + 1}[/bold red]")
                continue

            logger.info("Diagnosed %d failures in iteration %d", len(diagnoses), i)
            actionable_diagnoses = [d for d in diagnoses if d.actionable]
            non_actionable_diagnoses = [d for d in diagnoses if not d.actionable]
            if console is not None:
                console.print(
                    f"  [cyan]Diagnoses:[/cyan] "
                    f"{len(diagnoses)} generated from {len(failures)} failure(s)"
                )
                if len(diagnoses) == len(failures) == 1:
                    console.print(
                        "  [dim]One failing trial produced one diagnosis, so this pass has one hypothesis candidate to test.[/dim]"
                    )
                elif len(diagnoses) > len(failures):
                    console.print(
                        "  [dim]At least one failing trial produced multiple diagnosis ideas, so this pass will explore a broader hypothesis set.[/dim]"
                    )
                elif len(diagnoses) < len(failures):
                    console.print(
                        "  [dim]Some failures did not turn into diagnoses, so they will not produce hypotheses in this pass.[/dim]"
                    )
                if non_actionable_diagnoses:
                    reasons = sorted(
                        {
                            diagnosis.degraded_reason or diagnosis.diagnosis_mode
                            for diagnosis in non_actionable_diagnoses
                        }
                    )
                    console.print(
                        f"  [yellow]Non-actionable diagnoses:[/yellow] {len(non_actionable_diagnoses)}  reasons={_format_path_list(reasons)}"
                    )

            if not actionable_diagnoses:
                if console is not None:
                    console.print(
                        "  [bold yellow]⚠ No actionable diagnoses this pass, so mutation testing is skipped.[/bold yellow]"
                    )
                    console.print(
                        "  [dim]The improver will stop here because rerunning the same suite without target files or a usable diagnosis would only create more placeholder output.[/dim]"
                    )
                final_stop_reasons = ["no_actionable_diagnoses"]
                iter_log = IterationLog(
                    iteration=i,
                    baseline_score=mean_score,
                    baseline_repeats=eval_repeats,
                    failures=[t.id for t in failures],
                    diagnoses=[diagnosis_to_summary(d) for d in diagnoses],
                    hypothesis_ranked=0,
                    hypothesis_outcome="no_actionable_diagnoses",
                    stop_reasons=final_stop_reasons,
                )
                iteration_logs.append(iter_log.model_dump())
                write_iteration_log(iter_log, output_dir)
                break

            ranking = ranker.rank(actionable_diagnoses)

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
                top_targets = [
                    _format_path_list(h.diagnosis.target_files, limit=2)
                    for h in ranking.hypotheses[:3]
                ]
                console.print(
                    f"  [dim]Top targets:[/dim] {_format_path_list(top_targets, limit=3)}"
                )

            if not ranking.hypotheses:
                if console is not None:
                    console.print(
                        "  [bold yellow]⚠ No hypotheses remain after filtering already-tried ideas.[/bold yellow]"
                    )
                final_stop_reasons = ["no_ranked_hypotheses"]
                iter_log = IterationLog(
                    iteration=i,
                    baseline_score=mean_score,
                    baseline_repeats=eval_repeats,
                    failures=[t.id for t in failures],
                    diagnoses=[diagnosis_to_summary(d) for d in diagnoses],
                    hypothesis_ranked=0,
                    hypothesis_outcome="no_ranked_hypotheses",
                    stop_reasons=final_stop_reasons,
                )
                iteration_logs.append(iter_log.model_dump())
                write_iteration_log(iter_log, output_dir)
                break

            mean_old = mean_score

            iteration_kept: bool | None = None
            iteration_hypothesis_attempted: str | None = None
            iteration_hypothesis_outcome: str | None = None
            iteration_hypothesis_tested: str | None = None
            iteration_hypothesis_score: float | None = None
            iteration_delta: float | None = None
            iteration_lesson_id: str | None = None

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
                        f"    [dim]Step 3: testing rank={hyp.rank}[/dim]  "
                        f"impact={hyp.estimated_impact:.2f}  "
                        f"files=[cyan]{files_str}[/cyan]"
                    )

                workspace = None
                hypothesis_mutator = None
                try:
                    iteration_hypothesis_attempted = hyp.diagnosis.trial_id
                    if agent_path is not None:
                        if console is not None:
                            console.print(
                                f"    [dim]Creating disposable worktree for rank={hyp.rank}...[/dim]"
                            )
                        workspace = prepare_isolated_agent_workspace(
                            agent_path,
                            run_id="improve",
                            workspace_id=f"iter-{i}-rank-{hyp.rank}",
                        )
                        hypothesis_agent_path = workspace.workspace_agent_path
                        if console is not None:
                            console.print(
                                f"    [dim]Worktree ready:[/dim] {workspace.workspace_root}"
                            )
                        from ash_hawk.agents.agent_mutator import AgentMutator

                        hypothesis_mutator = AgentMutator(
                            hypothesis_agent_path,
                            run_id=f"improve-iter-{i}-rank-{hyp.rank}",
                        )
                        hypothesis_snapshot = hypothesis_mutator.snapshot()
                        grader_details = ""
                        transcript_excerpt = ""
                        failure_trial = failure_by_id.get(hyp.diagnosis.trial_id)
                        if failure_trial is not None and failure_trial.result is not None:
                            grader_details = json.dumps(
                                [r.model_dump() for r in failure_trial.result.grader_results],
                                default=str,
                            )
                            tc = failure_trial.result.transcript
                            msgs = getattr(tc, "messages", [])
                            tool_calls = getattr(tc, "tool_calls", [])
                            lines: list[str] = []
                            for m in msgs:
                                if isinstance(m, dict):
                                    role = str(m.get("role", "?"))
                                    content = str(m.get("content", ""))
                                else:
                                    role = str(getattr(m, "role", "?"))
                                    content = str(getattr(m, "content", ""))
                                lines.append(f"[{role}]\n{content}\n")
                            for tc_item in tool_calls:
                                if isinstance(tc_item, dict):
                                    name = str(tc_item.get("name") or tc_item.get("tool") or "?")
                                    arguments = (
                                        tc_item.get("arguments") or tc_item.get("input") or {}
                                    )
                                else:
                                    name = str(getattr(tc_item, "name", "?"))
                                    arguments = getattr(tc_item, "arguments", {})
                                args = json.dumps(arguments, default=str)
                                lines.append(f"[tool_call] {name}({args})")
                            transcript_excerpt = "\n".join(lines)

                        hypothesis_config_path = detect_agent_config_path(hypothesis_agent_path)
                        hypothesis_repo_root = workspace.workspace_root

                        try:
                            patch = await propose_patch_via_agent(
                                hyp.diagnosis,
                                hypothesis_agent_path,
                                agent_content,
                                grader_details=grader_details,
                                transcript_excerpt=transcript_excerpt,
                                console=console,
                                config_path=hypothesis_config_path,
                                repo_root=hypothesis_repo_root,
                            )
                        except ImportError:
                            logger.warning("bolt_merlin unavailable, falling back to LLM patching")
                            patch = await propose_patch(
                                hyp.diagnosis, agent_content, console=console
                            )

                        patches.append(_serialize_patch(patch))

                        if patch.agent_relative_path and patch.content:
                            if patch.agent_relative_path == "(agent-edited)":
                                logger.info(
                                    "Agent directly edited files for %s",
                                    hyp.diagnosis.trial_id,
                                )
                                if console is not None:
                                    console.print(
                                        f"    [dim]Agent mutation applied: "
                                        f"{patch.description[:80]}[/dim]"
                                    )
                            else:
                                hypothesis_mutator.write_file(
                                    patch.agent_relative_path,
                                    patch.content,
                                )
                                logger.info("Applied mutation: %s", patch.agent_relative_path)
                                if console is not None:
                                    console.print(
                                        f"    [dim]Applied mutation to "
                                        f"{patch.agent_relative_path}[/dim]"
                                    )
                        else:
                            iteration_hypothesis_outcome = "mutation_generation_failed"
                            patch_path = write_patch(patch, output_dir)
                            logger.info(
                                "Patch proposed: %s for %s",
                                patch_path,
                                patch.file_path,
                            )
                            if console is not None:
                                console.print(f"    [dim]Patch written: {patch_path.name}[/dim]")
                                console.print(
                                    "    [yellow]No mutation was applied, so no validation run will happen for this hypothesis.[/yellow]"
                                )
                            workspace.cleanup()
                            hypothesis_mutator.cleanup()
                            continue
                    else:
                        patch = await propose_patch(hyp.diagnosis, agent_content, console=console)
                        patches.append(_serialize_patch(patch))
                        patch_path = write_patch(patch, output_dir)
                        logger.info("Patch proposed: %s for %s", patch_path, patch.file_path)
                        if console is not None:
                            console.print(f"    [dim]Patch written: {patch_path.name}[/dim]")
                        continue

                    changed_paths = sorted(
                        hypothesis_mutator.diff_since_snapshot(hypothesis_snapshot).keys()
                    )
                    if not changed_paths:
                        iteration_hypothesis_outcome = "no_file_changes"
                        logger.info(
                            "Mutation produced no file changes for hypothesis rank=%d", hyp.rank
                        )
                        if console is not None:
                            console.print(
                                f"    [yellow]↷ No-op mutation for rank={hyp.rank}; skipping[/yellow]"
                            )
                        workspace.cleanup()
                        hypothesis_mutator.cleanup()
                        continue

                    if console is not None:
                        console.print(
                            f"    [dim]Mutation changed {len(changed_paths)} file(s):[/dim] {_format_path_list(changed_paths)}"
                        )

                    if console is not None:
                        console.print(
                            "    [dim]Step 4: fast validation on mutated worktree...[/dim]"
                        )

                    new_mean, _, _ = await _run_eval_n_times(
                        suite_path,
                        agent_name,
                        iteration_timeout_seconds,
                        hypothesis_agent_path,
                        eval_repeats,
                        console=console,
                        phase_label="fast validation",
                    )

                    if new_mean is None:
                        iteration_hypothesis_outcome = "post_mutation_eval_failed"
                        logger.warning(
                            "All post-mutation evals failed for hypothesis rank=%d",
                            hyp.rank,
                        )
                        if console is not None:
                            console.print(
                                f"    [bold red]✗ Post-mutation eval failed "
                                f"for rank={hyp.rank}[/bold red]"
                            )
                        workspace.cleanup()
                        hypothesis_mutator.cleanup()
                        continue

                    evaluated_mean = new_mean
                    fast_delta = new_mean - mean_old

                    if fast_delta > score_threshold and integrity_repeats > eval_repeats:
                        if console is not None:
                            console.print(
                                f"    [dim]Step 5: integrity validation ({integrity_repeats} repeats)...[/dim]"
                            )
                        integrity_mean, _, integrity_errors = await _run_eval_n_times(
                            suite_path,
                            agent_name,
                            iteration_timeout_seconds,
                            hypothesis_agent_path,
                            integrity_repeats,
                            console=console,
                            phase_label="integrity validation",
                        )
                        if integrity_mean is None:
                            logger.warning(
                                "Integrity pass failed for hypothesis rank=%d: %s",
                                hyp.rank,
                                integrity_errors,
                            )
                            if console is not None:
                                console.print(
                                    f"    [bold red]✗ Integrity pass failed for rank={hyp.rank}[/bold red]"
                                )
                            workspace.cleanup()
                            hypothesis_mutator.cleanup()
                            continue
                        evaluated_mean = integrity_mean

                    delta = evaluated_mean - mean_old
                    kept = delta > score_threshold
                    iteration_hypothesis_outcome = "kept" if kept else "reverted"

                    logger.debug(
                        "Hypothesis result: rank=%d delta=%.4f (%.4f -> %.4f) %s",
                        hyp.rank,
                        delta,
                        mean_old,
                        evaluated_mean,
                        "KEPT" if kept else "REVERTED",
                    )

                    if console is not None:
                        if kept:
                            console.print(
                                f"    [green]✓ KEPT[/green]  "
                                f"delta=[green]{delta:+.4f}[/green]  "
                                f"({mean_old:.4f} → {evaluated_mean:.4f})"
                            )
                        else:
                            console.print(
                                f"    [red]✗ REVERTED[/red]  "
                                f"delta=[red]{delta:+.4f}[/red]  "
                                f"({mean_old:.4f} → {evaluated_mean:.4f})"
                            )

                    lesson = Lesson(
                        lesson_id=uuid.uuid4().hex[:12],
                        trial_id=hyp.diagnosis.trial_id,
                        hypothesis_summary=hyp.diagnosis.failure_summary,
                        root_cause=hyp.diagnosis.root_cause,
                        target_files=hyp.diagnosis.target_files,
                        outcome="kept" if kept else "reverted",
                        score_before=mean_old,
                        score_after=evaluated_mean,
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

                    mutation_history.append(
                        {
                            "iteration": i,
                            "hypothesis_rank": hyp.rank,
                            "trial_id": hyp.diagnosis.trial_id,
                            "mean_score_before": mean_old,
                            "mean_score_after": evaluated_mean,
                            "fast_mean_score_after": new_mean,
                            "applied_files": changed_paths,
                            "improvement": delta,
                            "kept": kept,
                            "lesson_id": lesson.lesson_id,
                        }
                    )

                    stop_result = stop_condition.record(
                        ScoreRecord(
                            iteration=i,
                            score=evaluated_mean if kept else mean_old,
                            applied=kept,
                            delta=delta,
                        )
                    )

                    iteration_hypothesis_tested = hyp.diagnosis.trial_id
                    iteration_hypothesis_score = evaluated_mean
                    iteration_delta = delta
                    iteration_lesson_id = lesson.lesson_id
                    iteration_kept = kept

                    if kept:
                        final_score = evaluated_mean
                        synced_paths = workspace.sync_back()
                        applied_files.update(synced_paths)
                        if original_mutator is not None:
                            agent_content = original_mutator.scan()
                        mean_old = evaluated_mean
                        if console is not None:
                            console.print(
                                f"    [green]Step 6: synced kept mutation back to agent[/green]  {_format_path_list(synced_paths)}"
                            )
                            console.print(
                                f"    [dim]Kept patch raised score floor to {evaluated_mean:.2%}; continuing to next ranked hypothesis from the updated baseline.[/dim]"
                            )
                        workspace.cleanup()
                        hypothesis_mutator.cleanup()
                        continue
                    else:
                        if console is not None:
                            console.print(
                                "    [yellow]Discarding hypothesis worktree and trying next candidate[/yellow]"
                            )
                        workspace.cleanup()
                        hypothesis_mutator.cleanup()

                    if stop_result.should_stop:
                        logger.warning("Stop condition: %s", "; ".join(stop_result.reasons))
                        if console is not None:
                            console.print(
                                f"  [bold yellow]⚠ Stop: {stop_result.reasons[0]}[/bold yellow]"
                            )
                        final_stop_reasons = stop_result.reasons
                        break

                except Exception:
                    iteration_hypothesis_outcome = "hypothesis_exception"
                    logger.warning("Hypothesis %s failed", hyp.diagnosis.trial_id, exc_info=True)
                    if workspace is not None:
                        workspace.cleanup()
                    if hypothesis_mutator is not None:
                        hypothesis_mutator.cleanup()
                    if console is not None:
                        console.print(
                            f"    [bold red]✗ Hypothesis rank={hyp.rank} "
                            f"failed with exception[/bold red]"
                        )

            iter_log = IterationLog(
                iteration=i,
                baseline_score=mean_score,
                baseline_repeats=eval_repeats,
                failures=[t.id for t in failures],
                diagnoses=[diagnosis_to_summary(d) for d in diagnoses],
                hypothesis_ranked=ranking.ranked_count,
                hypothesis_attempted=iteration_hypothesis_attempted,
                hypothesis_outcome=iteration_hypothesis_outcome,
                hypothesis_tested=iteration_hypothesis_tested,
                hypothesis_score=iteration_hypothesis_score,
                delta=iteration_delta,
                kept=iteration_kept,
                lesson_id=iteration_lesson_id,
                stop_reasons=final_stop_reasons,
            )
            iteration_logs.append(iter_log.model_dump())
            write_iteration_log(iter_log, output_dir)

            if console is not None:
                console.print(
                    f"  [dim]Outer pass {i + 1} complete: "
                    f"score {final_score:.2%}  "
                    f"tested={len(mutation_history)}  kept={sum(1 for m in mutation_history if m.get('kept'))}[/dim]"
                )
                console.print()

            if final_stop_reasons:
                break

    finally:
        if original_mutator is not None:
            original_mutator.cleanup()

    if console is not None:
        console.rule("[bold]Improve Summary[/bold]")
        console.print(
            f"  [bold]Score:[/bold]     {initial_score:.2%} → "
            f"{final_score:.2%}  "
            f"[bold]Δ={final_score - initial_score:+.2%}[/bold]"
        )
        console.print(
            f"  [bold]Patches:[/bold]   {len(patches)} proposed  {len(mutation_history)} tested"
        )
        kept_count = sum(1 for m in mutation_history if m.get("kept"))
        console.print(
            f"  [bold]Results:[/bold]   [green]{kept_count} kept[/green]  "
            f"[red]{len(mutation_history) - kept_count} reverted[/red]"
        )
        if convergence_achieved:
            console.print("  [bold]Status:[/bold]    [green]✓ Converged[/green]")
        if final_stop_reasons:
            console.print(f"  [bold]Stopped:[/bold]   [yellow]{final_stop_reasons[0]}[/yellow]")

    return ImprovementResult(
        iterations=actual_iterations,
        initial_pass_rate=initial_score,
        final_pass_rate=final_score,
        patches_proposed=patches,
        patches_applied=sorted(applied_files),
        trace_path=trace_dir,
        mutation_history=mutation_history,
        iteration_logs=iteration_logs,
        convergence_achieved=convergence_achieved,
        stop_reasons=final_stop_reasons,
    )


async def _run_eval_n_times(
    suite_path: str,
    agent_name: str,
    timeout: float,
    agent_path: Path | None,
    n: int,
    console: Console | None = None,
    phase_label: str = "evaluation",
) -> tuple[float | None, EvalRunSummary | None, list[str]]:
    scores: list[float] = []
    errors: list[str] = []
    last_summary: EvalRunSummary | None = None

    for repeat_idx in range(n):
        if console is not None and n > 1:
            console.print(
                f"    [dim]{phase_label.capitalize()} repeat {repeat_idx + 1}/{n}...[/dim]"
            )
        try:
            summary = await _run_eval(suite_path, agent_name, timeout, agent_path)
            scores.append(summary.metrics.mean_score)
            last_summary = summary
            if console is not None and n > 1:
                console.print(
                    f"    [dim]{phase_label.capitalize()} repeat {repeat_idx + 1}/{n} score {summary.metrics.mean_score:.2%}[/dim]"
                )
            get_telemetry().emit(
                "improve.eval_repeat",
                repeat=repeat_idx,
                suite=suite_path,
                trial_count=len(summary.trials),
                pass_rate=round(summary.metrics.pass_rate, 4),
                mean_score=round(summary.metrics.mean_score, 4),
                completed=summary.metrics.completed_tasks,
            )
        except Exception as exc:
            logger.warning("Eval run failed (repeat %d)", repeat_idx, exc_info=True)
            errors.append(str(exc))
            if console is not None:
                console.print(
                    f"    [yellow]{phase_label.capitalize()} repeat {repeat_idx + 1}/{n} failed:[/yellow] {exc}"
                )

    if not scores:
        return None, None, errors

    return sum(scores) / len(scores), last_summary, errors


async def _run_eval(
    suite_path: str,
    agent_name: str,
    timeout: float,
    agent_path: Path | None = None,
) -> EvalRunSummary:
    from ash_hawk.scenario.runner import run_scenarios_async

    return await run_scenarios_async(
        [suite_path],
        agent_path=agent_path,
        adapter_override=_resolve_adapter_override(agent_name),
    )


def _resolve_adapter_override(agent_name: str) -> str | None:
    from ash_hawk.scenario.registry import get_default_adapter_registry

    registry = get_default_adapter_registry()
    candidates = [agent_name]
    normalized = agent_name.replace("-", "_")
    if normalized not in candidates:
        candidates.append(normalized)

    for candidate in candidates:
        if registry.get(candidate) is not None:
            return candidate

    return None
