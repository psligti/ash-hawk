"""Iteration coordinator for improvement cycles."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

from ash_hawk.contracts import CuratedLesson, RunArtifact, ToolCallRecord
from ash_hawk.curation.experiment_store import ExperimentStore
from ash_hawk.cycle.convergence import ConvergenceChecker
from ash_hawk.cycle.types import (
    ConvergenceStatus,
    CycleCheckpoint,
    CycleConfig,
    CycleResult,
    CycleStatus,
    IterationResult,
)
from ash_hawk.experiments.registry import ExperimentRegistry
from ash_hawk.improve_cycle.models import CuratedLesson as ImproveCuratedLesson
from ash_hawk.improve_cycle.models import MetricValue as ImproveMetricValue
from ash_hawk.improve_cycle.models import ProposalType
from ash_hawk.improve_cycle.models import ReviewFinding as ImproveReviewFinding
from ash_hawk.improve_cycle.models import RunArtifactBundle as ImproveRunArtifactBundle
from ash_hawk.improve_cycle.orchestrator import ImproveCycleOrchestrator, ImproveCycleResult
from ash_hawk.improve_cycle.storage import ImproveCycleStorage
from ash_hawk.storage import FileStorage
from ash_hawk.strategies import Strategy

logger = logging.getLogger(__name__)


def _to_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _to_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _to_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in cast(list[object], value):
        if isinstance(item, str):
            items.append(item)
    return items


class IterationCoordinator:
    """Coordinates single iteration execution within a cycle."""

    def __init__(
        self,
        config: CycleConfig,
        storage: FileStorage,
        experiment_store: ExperimentStore,
    ) -> None:
        self._config = config
        self._storage = storage
        self._experiment_store = experiment_store
        improve_root = Path(".ash-hawk") / "improve-cycle" / config.experiment_id
        self._improve_storage = ImproveCycleStorage(improve_root)
        self._improve_orchestrator = ImproveCycleOrchestrator(
            storage=self._improve_storage,
            enable_competitor=bool(config.metadata.get("enable_competitor", True)),
            enable_triage=bool(config.metadata.get("enable_triage", True)),
            enable_verifier=bool(config.metadata.get("enable_verifier", True)),
            enable_adversary=bool(config.metadata.get("enable_adversary", True)),
            min_verification_runs=_to_int(config.metadata.get("min_verification_runs"), 3),
            max_latency_delta_pct=_to_float(config.metadata.get("max_latency_delta_pct"), 15.0),
            max_token_delta_pct=_to_float(config.metadata.get("max_token_delta_pct"), 10.0),
            cross_pack_eval_pack_ids=_to_str_list(config.metadata.get("cross_pack_eval_pack")),
            promotion_scope=str(config.metadata.get("promotion_scope", "agent-specific")),
        )
        self._artifact_adapter = _CycleArtifactAdapter(storage)

    async def run_iteration(
        self,
        iteration_num: int,
        lessons_to_apply: list[CuratedLesson],
    ) -> IterationResult:
        started_at = datetime.now(UTC)
        logger.info(f"Starting iteration {iteration_num}")

        try:
            run_id = f"cycle-{self._config.cycle_id}-iter-{iteration_num}"
            scenario_eval = await self._evaluate_scenarios()
            artifact = await self._generate_run_artifact(run_id, lessons_to_apply, scenario_eval)

            run_bundle = self._to_improve_run_bundle(artifact, scenario_eval, lessons_to_apply)
            improve_result = await asyncio.to_thread(
                self._improve_orchestrator.run_cycle, run_bundle
            )
            lessons = [
                self._to_legacy_curated_lesson(lesson) for lesson in improve_result.curated_lessons
            ]
            if any(report.regression_count > 0 for report in improve_result.verification_reports):
                lessons = [
                    lesson.model_copy(update={"validation_status": "rolled_back"})
                    for lesson in lessons
                ]
            role_summaries = self._summarize_improve_cycle(improve_result)

            for lesson in lessons:
                self._experiment_store.store(lesson, self._config.experiment_id)

            score = self._derive_iteration_score(scenario_eval, improve_result.verification_reports)

            result = IterationResult(
                iteration_num=iteration_num,
                run_artifact_id=artifact.run_id,
                score=score,
                lessons_generated=len(lessons),
                lessons_applied=len(lessons_to_apply),
                status=CycleStatus.COMPLETED,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                metadata={
                    "scenario_eval": scenario_eval,
                    "lesson_titles": [lesson.title for lesson in lessons],
                    "lesson_ids": [lesson.lesson_id for lesson in lessons],
                    "tested_change_titles": [lesson.title for lesson in lessons_to_apply],
                    "tested_change_ids": [lesson.lesson_id for lesson in lessons_to_apply],
                    "tested_change_rationales": [
                        lesson.description or str(lesson.lesson_payload.get("hypothesis", ""))
                        for lesson in lessons_to_apply
                    ],
                    "role_summaries": role_summaries,
                }
                if scenario_eval is not None
                else {
                    "lesson_titles": [lesson.title for lesson in lessons],
                    "lesson_ids": [lesson.lesson_id for lesson in lessons],
                    "tested_change_titles": [lesson.title for lesson in lessons_to_apply],
                    "tested_change_ids": [lesson.lesson_id for lesson in lessons_to_apply],
                    "tested_change_rationales": [
                        lesson.description or str(lesson.lesson_payload.get("hypothesis", ""))
                        for lesson in lessons_to_apply
                    ],
                    "role_summaries": role_summaries,
                },
            )

            logger.info(
                f"Iteration {iteration_num} complete: score={score:.3f}, lessons={len(lessons)}"
            )
            return result

        except Exception as e:
            logger.error(f"Iteration {iteration_num} failed: {e}")
            return IterationResult(
                iteration_num=iteration_num,
                run_artifact_id="",
                score=0.0,
                status=CycleStatus.FAILED,
                error_message=str(e),
                started_at=started_at,
                completed_at=datetime.now(UTC),
            )

    async def _generate_run_artifact(
        self,
        run_id: str,
        lessons: list[CuratedLesson],
        scenario_eval: dict[str, object] | None,
    ) -> RunArtifact:
        tool_calls = [
            ToolCallRecord(
                tool_name="policy_load",
                outcome="success",
                duration_ms=20,
                input_args={"lesson_count": len(lessons)},
            ),
            ToolCallRecord(
                tool_name="policy_apply",
                outcome="success",
                duration_ms=30,
                error_message=None,
                input_args={"lesson_count": len(lessons)},
                output={
                    "applied": bool(lessons),
                    "reason": "No lessons selected for this iteration" if not lessons else None,
                },
            ),
        ]

        metrics: dict[str, float] = {"lessons_applied": float(len(lessons))}
        outcome = "success"

        if scenario_eval is not None:
            mean_score = _to_float(scenario_eval.get("mean_score"), 0.0)
            failed_tasks = _to_int(scenario_eval.get("failed_tasks"), 0)
            total_tasks = _to_int(scenario_eval.get("total_tasks"), 0)
            normalized_errors = _to_str_list(scenario_eval.get("error_messages"))
            metrics["scenario_mean_score"] = mean_score
            metrics["scenario_failed_tasks"] = float(failed_tasks)
            metrics["scenario_total_tasks"] = float(total_tasks)

            tool_calls.append(
                ToolCallRecord(
                    tool_name="scenario_eval",
                    outcome="success" if failed_tasks == 0 else "failure",
                    duration_ms=100,
                    error_message=("; ".join(normalized_errors[:3]) if normalized_errors else None),
                    input_args={
                        "scenario_paths": self._config.scenario_paths,
                        "total_tasks": total_tasks,
                    },
                    output={
                        "mean_score": mean_score,
                        "failed_tasks": failed_tasks,
                        "total_tasks": total_tasks,
                    },
                )
            )
            outcome = "success" if failed_tasks == 0 and mean_score >= 0.5 else "failure"

        return RunArtifact(
            run_id=run_id,
            agent_id=self._config.target_agent,
            task_type="improvement_cycle",
            outcome=outcome,
            tool_calls=tool_calls,
            metrics=metrics,
            created_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )

    def _derive_iteration_score(
        self,
        scenario_eval: dict[str, object] | None,
        verification_reports: list[Any],
    ) -> float:
        if scenario_eval is not None:
            return _to_float(scenario_eval.get("mean_score"), 0.0)
        if not verification_reports:
            return 0.0
        deltas = [
            float(report.score_delta)
            for report in verification_reports
            if getattr(report, "score_delta", None) is not None
        ]
        if deltas:
            avg_delta = sum(deltas) / len(deltas)
            return max(0.0, min(1.0, 0.5 + avg_delta))
        passed_count = sum(1 for report in verification_reports if getattr(report, "passed", False))
        return passed_count / len(verification_reports)

    def _summarize_improve_cycle(self, improve_result: ImproveCycleResult) -> dict[str, str]:
        finding_count = len(
            self._improve_storage.findings.list_all(model_type=ImproveReviewFinding)
        )
        coach_count = sum(
            1 for proposal in improve_result.proposals if proposal.source_role == "coach"
        )
        architect_count = sum(
            1 for proposal in improve_result.proposals if proposal.source_role == "architect"
        )
        competitor_enabled = bool(self._config.metadata.get("enable_competitor", True))
        summaries = {
            "competitor": "Replay comparison complete"
            if competitor_enabled
            else "Competitor disabled",
            "translator": f"Normalized findings={finding_count}",
            "analyst": improve_result.history.trend_notes[0]
            if improve_result.history.trend_notes
            else "Analysis complete",
            "coach": f"Behavior proposals={coach_count}",
            "architect": f"Infra proposals={architect_count}",
            "curator": f"Curated lessons={len(improve_result.curated_lessons)}",
            "verifier": f"Verification reports={len(improve_result.verification_reports)}",
            "promotion_manager": f"Decisions={len(improve_result.promotion_decisions)}",
        }
        return summaries

    def _to_improve_run_bundle(
        self,
        artifact: RunArtifact,
        scenario_eval: dict[str, object] | None,
        lessons_to_apply: list[CuratedLesson],
    ) -> ImproveRunArtifactBundle:
        score_value = _to_float(artifact.metrics.get("scenario_mean_score", 0.0), 0.0)
        if scenario_eval is not None:
            score_value = _to_float(scenario_eval.get("mean_score"), score_value)
        tool_traces = [
            {
                "tool_name": call.tool_name,
                "outcome": call.outcome,
                "duration_ms": call.duration_ms,
                "error_message": call.error_message,
            }
            for call in artifact.tool_calls
        ]
        return ImproveRunArtifactBundle(
            run_id=artifact.run_id,
            experiment_id=self._config.experiment_id,
            agent_id=self._config.target_agent,
            eval_pack_id=self._config.eval_pack or "default",
            scenario_ids=list(self._config.scenario_paths) or ["default"],
            timestamp=datetime.now(UTC).isoformat(),
            transcripts=[message.get("content", "") for message in artifact.messages],
            tool_traces=tool_traces,
            outputs=[{"outcome": artifact.outcome, "error_message": artifact.error_message}],
            metrics=[ImproveMetricValue(name="score", value=score_value)],
            active_lessons=[lesson.lesson_id for lesson in lessons_to_apply],
        )

    def _to_legacy_curated_lesson(self, lesson: ImproveCuratedLesson) -> CuratedLesson:
        lesson_type: Literal["policy", "skill", "tool", "harness", "eval"] = "policy"
        if lesson.proposal_type in {
            ProposalType.POLICY_PATCH,
            ProposalType.POLICY_REORDER,
            ProposalType.PLAYBOOK_UPDATE,
            ProposalType.PROMPT_GUARDRAIL,
            ProposalType.BEHAVIORAL_CHECKLIST,
        }:
            lesson_type = "policy"
        elif lesson.proposal_type in {ProposalType.SKILL_CREATE, ProposalType.SKILL_REVISE}:
            lesson_type = "skill"
        elif lesson.proposal_type in {
            ProposalType.TOOL_CREATE,
            ProposalType.TOOL_REVISE,
            ProposalType.TOOL_WRAPPER_UPDATE,
            ProposalType.OBSERVABILITY_IMPROVEMENT,
            ProposalType.CONFIG_ADJUSTMENT,
        }:
            lesson_type = "tool"
        elif lesson.proposal_type == ProposalType.HARNESS_PATCH:
            lesson_type = "harness"
        elif lesson.proposal_type in {ProposalType.EVAL_PATCH, ProposalType.EVAL_EXPANSION}:
            lesson_type = "eval"

        if lesson_type == "harness":
            strategy = Strategy.HARNESS_QUALITY
        elif lesson_type == "eval":
            strategy = Strategy.EVAL_QUALITY
        elif lesson_type in {"policy", "skill"}:
            strategy = Strategy.AGENT_BEHAVIOR
        else:
            strategy = Strategy.TOOL_QUALITY
        return CuratedLesson(
            lesson_id=lesson.lesson_id,
            source_proposal_id=lesson.proposal_id,
            applies_to_agents=[self._config.target_agent],
            lesson_type=lesson_type,
            title=lesson.title,
            description=lesson.summary,
            lesson_payload={
                "target_surface": lesson.target_surface,
                "curation_notes": lesson.curation_notes,
                "lineage": lesson.lineage,
            },
            validation_status="approved",
            version=1,
            created_at=datetime.now(UTC),
            experiment_id=self._config.experiment_id,
            strategy=strategy,
        )

    async def _evaluate_scenarios(self) -> dict[str, object] | None:
        if not self._config.scenario_paths:
            return None

        try:
            from ash_hawk.scenario.runner import run_scenarios_async

            logger.info(
                "Running %d scenario(s) with parallelism=%s",
                len(self._config.scenario_paths),
                self._config.scenario_parallelism,
            )

            scenario_task = asyncio.create_task(
                run_scenarios_async(
                    paths=self._config.scenario_paths,
                    tooling_mode="record",
                    parallelism=self._config.scenario_parallelism,
                )
            )
            heartbeat_task = asyncio.create_task(self._log_scenario_heartbeat(scenario_task))
            try:
                summary = await scenario_task
            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
            logger.info(
                "Scenario run complete: passed=%d failed=%d total=%d mean_score=%.3f",
                int(summary.metrics.passed_tasks),
                int(summary.metrics.failed_tasks),
                int(summary.metrics.total_tasks),
                float(summary.metrics.mean_score),
            )
            has_errors = any(str(trial.status) in {"error", "timeout"} for trial in summary.trials)
            collected_errors: list[str] = []
            for trial in summary.trials:
                if str(trial.status) not in {"error", "timeout"}:
                    continue
                if trial.result is None:
                    continue
                error_message = trial.result.outcome.error_message
                if isinstance(error_message, str):
                    collected_errors.append(error_message)

            if has_errors:
                if collected_errors:
                    logger.error(f"Scenario errors: {'; '.join(collected_errors[:3])}")
                if self._config.fail_on_scenario_error:
                    raise RuntimeError(
                        f"Scenario execution failed with {summary.metrics.failed_tasks} error(s)"
                    )
            return {
                "mean_score": float(summary.metrics.mean_score),
                "failed_tasks": int(summary.metrics.failed_tasks),
                "passed_tasks": int(summary.metrics.passed_tasks),
                "total_tasks": int(summary.metrics.total_tasks),
                "error_messages": collected_errors,
            }
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning(f"Scenario scoring failed: {exc}")
            if self._config.fail_on_scenario_error:
                raise RuntimeError(f"Scenario scoring failed: {exc}") from exc
            return None

    async def _log_scenario_heartbeat(self, scenario_task: asyncio.Task[object]) -> None:
        elapsed = 0
        while not scenario_task.done():
            await asyncio.sleep(30)
            elapsed += 30
            if scenario_task.done():
                break
            logger.info(
                "Scenario run still in progress (%ds elapsed, parallelism=%s)",
                elapsed,
                self._config.scenario_parallelism,
            )


class CycleRunner:
    """Runs complete improvement cycles with checkpointing."""

    def __init__(self, config: CycleConfig, storage_path: str = ".ash-hawk") -> None:
        self._config = config
        self._storage = FileStorage(storage_path)
        self._experiment_store = ExperimentStore()
        self._registry = ExperimentRegistry()
        self._convergence = ConvergenceChecker(config)
        self._coordinator = IterationCoordinator(
            config,
            self._storage,
            self._experiment_store,
        )
        self._checkpoints_path = Path(".ash-hawk") / "cycles" / config.cycle_id
        self._checkpoints_path.mkdir(parents=True, exist_ok=True)

    async def run_cycle(self) -> CycleResult:
        result = CycleResult(
            cycle_id=self._config.cycle_id,
            config=self._config,
            total_iterations=0,
            status=CycleStatus.RUNNING,
            started_at=datetime.now(UTC),
        )

        self._registry.get_or_create(
            self._config.experiment_id,
            {
                "target_agent": self._config.target_agent,
                "cycle_id": self._config.cycle_id,
            },
        )

        checkpoint = self._load_checkpoint()
        start_iteration = checkpoint.current_iteration if checkpoint else 0

        if checkpoint:
            self._restore_convergence_state(checkpoint)
            result.iterations = checkpoint.iterations
            logger.info(f"Resuming from iteration {start_iteration}")

        for iteration_num in range(start_iteration + 1, self._config.max_iterations + 1):
            lessons = self._get_lessons_for_iteration(iteration_num)

            iteration_result = await self._coordinator.run_iteration(
                iteration_num,
                lessons,
            )

            if iteration_num > 1 and result.iterations:
                prev_score = result.iterations[-1].score
                iteration_result.score_delta = iteration_result.score - prev_score

            result.iterations.append(iteration_result)
            self._convergence.add_score(iteration_result.score)

            convergence = self._convergence.check_convergence()
            if convergence == ConvergenceStatus.CONVERGED and self._config.stop_on_convergence:
                result.convergence_status = convergence
                result.status = CycleStatus.CONVERGED
                result.total_iterations = iteration_num
                logger.info(f"Cycle converged at iteration {iteration_num}")
                break

            result.convergence_status = convergence

            if self._convergence.should_promote_lessons():
                promoted = await self._promote_lessons(lessons)
                result.lessons_promoted.extend(promoted)
                self._convergence.reset_improvement_counter()

            if iteration_num % self._config.checkpoint_interval == 0:
                self._save_checkpoint(result, iteration_num)

            result.total_iterations = iteration_num

        result.best_score = self._convergence.get_best_score()
        result.final_score = self._convergence.get_latest_score() or 0.0
        result.initial_score = result.iterations[0].score if result.iterations else 0.0
        if result.total_iterations == 0:
            result.total_iterations = len(result.iterations)
        result.total_lessons_generated = sum(i.lessons_generated for i in result.iterations)

        if result.status == CycleStatus.RUNNING:
            result.status = CycleStatus.COMPLETED

        result.completed_at = datetime.now(UTC)
        self._registry.update_status(self._config.experiment_id, "completed")

        return result

    def _get_lessons_for_iteration(self, iteration_num: int) -> list[CuratedLesson]:
        available_lessons = self._experiment_store.get_for_agent(
            self._config.target_agent,
            self._config.experiment_id,
        )
        return self.select_lessons_for_iteration(
            available_lessons,
            iteration_num,
            self._config.max_lessons_per_iteration,
        )

    @staticmethod
    def select_lessons_for_iteration(
        lessons: list[CuratedLesson],
        iteration_num: int,
        max_lessons: int,
    ) -> list[CuratedLesson]:
        if max_lessons <= 0 or not lessons:
            return []

        if len(lessons) <= max_lessons:
            return sorted(
                lessons,
                key=lambda lesson: (
                    lesson.created_at.isoformat(),
                    lesson.lesson_id,
                ),
            )

        by_bucket: dict[str, list[CuratedLesson]] = {}
        for lesson in lessons:
            strategy_value = (
                lesson.strategy.value if lesson.strategy is not None else lesson.lesson_type
            )
            by_bucket.setdefault(strategy_value, []).append(lesson)

        for bucket_lessons in by_bucket.values():
            bucket_lessons.sort(
                key=lambda lesson: (
                    lesson.created_at.isoformat(),
                    lesson.lesson_id,
                )
            )

        selected: list[CuratedLesson] = []
        selected_ids: set[str] = set()

        newest_lessons = sorted(
            lessons,
            key=lambda lesson: (
                lesson.created_at.isoformat(),
                lesson.lesson_id,
            ),
            reverse=True,
        )

        newest = newest_lessons[0]
        selected.append(newest)
        selected_ids.add(newest.lesson_id)

        bucket_keys = sorted(by_bucket)
        if bucket_keys:
            start_bucket = (iteration_num - 1) % len(bucket_keys)
            bucket_positions: dict[str, int] = {}
            for key in bucket_keys:
                bucket = by_bucket[key]
                bucket_positions[key] = (iteration_num - 1) % len(bucket)

            loop_budget = len(lessons) * 3
            while len(selected) < max_lessons and loop_budget > 0:
                for offset in range(len(bucket_keys)):
                    key = bucket_keys[(start_bucket + offset) % len(bucket_keys)]
                    bucket = by_bucket[key]
                    if not bucket:
                        continue
                    position = bucket_positions[key] % len(bucket)
                    candidate = bucket[position]
                    bucket_positions[key] = (position + 1) % len(bucket)
                    if candidate.lesson_id in selected_ids:
                        continue
                    selected.append(candidate)
                    selected_ids.add(candidate.lesson_id)
                    if len(selected) >= max_lessons:
                        break
                loop_budget -= 1

        if len(selected) < max_lessons:
            for candidate in newest_lessons:
                if candidate.lesson_id in selected_ids:
                    continue
                selected.append(candidate)
                selected_ids.add(candidate.lesson_id)
                if len(selected) >= max_lessons:
                    break

        return selected

    async def _promote_lessons(self, lessons: list[CuratedLesson]) -> list[str]:
        if not lessons:
            return []

        promoted_ids: list[str] = []
        for lesson in lessons:
            if lesson.validation_status == "approved":
                try:
                    ids = self._experiment_store.promote_to_global(
                        self._config.experiment_id,
                        [lesson.lesson_id],
                    )
                    promoted_ids.extend(ids)
                    logger.info(f"Promoted lesson {lesson.lesson_id} to global store")
                except Exception as e:
                    logger.warning(f"Failed to promote lesson {lesson.lesson_id}: {e}")

        return promoted_ids

    def _save_checkpoint(self, result: CycleResult, iteration: int) -> None:
        checkpoint = CycleCheckpoint(
            cycle_id=self._config.cycle_id,
            config=self._config,
            iterations=result.iterations,
            current_iteration=iteration,
            status=result.status,
            saved_at=datetime.now(UTC),
            consecutive_improvements=self._convergence.get_consecutive_improvements(),
        )

        checkpoint_file = self._checkpoints_path / "checkpoint.json"
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint.model_dump(mode="json"), f, indent=2, default=str)

        logger.debug(f"Saved checkpoint at iteration {iteration}")

    def _load_checkpoint(self) -> CycleCheckpoint | None:
        checkpoint_file = self._checkpoints_path / "checkpoint.json"
        if not checkpoint_file.exists():
            return None

        try:
            with open(checkpoint_file) as f:
                data = json.load(f)
            return CycleCheckpoint(**data)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def _restore_convergence_state(self, checkpoint: CycleCheckpoint) -> None:
        for iteration in checkpoint.iterations:
            self._convergence.add_score(iteration.score)


class _CycleArtifactAdapter:
    """Simple adapter for loading artifacts during cycle execution."""

    def __init__(self, storage: FileStorage) -> None:
        self._storage = storage

    async def load_run_artifact(self, artifact_id: str) -> RunArtifact | None:
        return await self._storage.load_run_artifact(artifact_id)


def create_cycle_id() -> str:
    return f"cycle-{uuid4().hex[:8]}"


__all__ = [
    "CycleRunner",
    "IterationCoordinator",
    "create_cycle_id",
]
