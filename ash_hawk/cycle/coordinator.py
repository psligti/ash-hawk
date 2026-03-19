"""Iteration coordinator for improvement cycles."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from ash_hawk.contracts import CuratedLesson, ReviewRequest, RunArtifact, ToolCallRecord
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
from ash_hawk.pipeline.orchestrator import PipelineOrchestrator
from ash_hawk.services.review_service import ReviewService
from ash_hawk.storage import FileStorage

if TYPE_CHECKING:
    pass

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
        self._review_service = ReviewService()
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

            review_request = ReviewRequest(
                run_artifact_id=artifact.run_id,
                target_agent=self._config.target_agent,
                eval_suite=[self._config.eval_pack] if self._config.eval_pack else [],
                review_mode="standard",
                persistence_mode="curate",
                experiment_id=self._config.experiment_id,
                baseline_run_id=self._config.baseline_run_id,
            )

            orchestrator = PipelineOrchestrator()
            lessons = orchestrator.run(review_request, artifact)

            for lesson in lessons:
                self._experiment_store.store(lesson, self._config.experiment_id)

            score = self._extract_score(orchestrator)
            if scenario_eval is not None:
                score = _to_float(scenario_eval.get("mean_score"), 0.0)

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
                    "tested_change_titles": [lesson.title for lesson in lessons_to_apply],
                }
                if scenario_eval is not None
                else {
                    "lesson_titles": [lesson.title for lesson in lessons],
                    "tested_change_titles": [lesson.title for lesson in lessons_to_apply],
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
                outcome="success" if lessons else "failure",
                duration_ms=30,
                error_message=None if lessons else "No approved policy lessons available",
                input_args={"lesson_count": len(lessons)},
            ),
        ]

        metrics: dict[str, float] = {"lessons_applied": float(len(lessons))}
        outcome = "success" if lessons else "failure"

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

    def _extract_score(self, orchestrator: PipelineOrchestrator) -> float:
        from ash_hawk.pipeline.types import PipelineRole

        analyst_step = orchestrator.get_step_result(PipelineRole.ANALYST)
        if analyst_step and analyst_step.outputs:
            metrics = analyst_step.outputs.get("metrics", {})
            return float(metrics.get("score", 0.0))
        return 0.0

    async def _evaluate_scenarios(self) -> dict[str, object] | None:
        if not self._config.scenario_paths:
            return None

        try:
            from ash_hawk.scenario.runner import run_scenarios_async

            summary = await run_scenarios_async(
                paths=self._config.scenario_paths,
                tooling_mode="record",
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
            lessons = self._get_lessons_for_iteration()

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

    def _get_lessons_for_iteration(self) -> list[CuratedLesson]:
        return self._experiment_store.get_for_agent(
            self._config.target_agent,
            self._config.experiment_id,
        )

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
