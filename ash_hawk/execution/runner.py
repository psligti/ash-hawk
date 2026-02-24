"""Parallel suite runner for Ash-Hawk evaluation harness.

This module provides the EvalRunner class for executing evaluation suites
with controlled parallelism, timeout management, and graceful cancellation.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ash_hawk.config import EvalConfig
from ash_hawk.events import AHEvents, get_bus
from ash_hawk.execution.trial import TrialExecutor
from ash_hawk.types import (
    EvalOutcome,
    EvalRunSummary,
    EvalStatus,
    EvalSuite,
    EvalTask,
    EvalTrial,
    FailureMode,
    RunEnvelope,
    SuiteMetrics,
    TokenUsage,
    TrialEnvelope,
    TrialResult,
)

if TYPE_CHECKING:
    from ash_hawk.storage import StorageBackend


class ResourceTracker:
    """Tracks resource usage during suite execution."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.total_tokens = TokenUsage()
        self.total_cost_usd = 0.0
        self.total_duration_seconds = 0.0

    async def add_trial_usage(
        self,
        tokens: TokenUsage,
        cost_usd: float,
        duration_seconds: float,
    ) -> None:
        """Add trial resource usage to totals."""
        async with self._lock:
            self.total_tokens = TokenUsage(
                input=self.total_tokens.input + tokens.input,
                output=self.total_tokens.output + tokens.output,
                reasoning=self.total_tokens.reasoning + tokens.reasoning,
                cache_read=self.total_tokens.cache_read + tokens.cache_read,
                cache_write=self.total_tokens.cache_write + tokens.cache_write,
            )
            self.total_cost_usd += cost_usd
            self.total_duration_seconds += duration_seconds


class EvalRunner:
    """Parallel suite runner with timeout, budget, and cancellation support.

    This class manages the execution of evaluation suites with controlled
    parallelism, resource tracking, and graceful cancellation handling.

    Key features:
    - Semaphore-based concurrency control
    - Per-trial timeout enforcement
    - Resource usage tracking (tokens, cost, duration)
    - Graceful cancellation with partial artifact storage
    - Suite-level metrics aggregation

    Example:
        >>> from ash_hawk.config import EvalConfig
        >>> from ash_hawk.storage import FileStorage
        >>> from ash_hawk.execution import TrialExecutor
        >>>
        >>> config = EvalConfig(parallelism=4)
        >>> storage = FileStorage(base_path="./results")
        >>> policy = ToolSurfacePolicy()
        >>> trial_executor = TrialExecutor(storage, policy)
        >>>
        >>> runner = EvalRunner(config, storage, trial_executor)
        >>> summary = await runner.run_suite(suite, agent_config, run_envelope)
    """

    def __init__(
        self,
        config: EvalConfig,
        storage: StorageBackend,
        trial_executor: TrialExecutor,
    ) -> None:
        """Initialize the EvalRunner.

        Args:
            config: Evaluation configuration with parallelism and timeout settings.
            storage: Storage backend for persisting results.
            trial_executor: Executor for running individual trials.
        """
        self._config = config
        self._storage = storage
        self._trial_executor = trial_executor
        self._semaphore = asyncio.Semaphore(config.parallelism)
        self._cancelled = False
        self._resource_tracker = ResourceTracker()
        self._trials: list[EvalTrial] = []
        self._trial_durations: list[float] = []

    @property
    def is_cancelled(self) -> bool:
        """Check if the runner has been cancelled."""
        return self._cancelled

    def cancel(self) -> None:
        """Request cancellation of the suite run."""
        self._cancelled = True

    async def run_suite(
        self,
        suite: EvalSuite,
        agent_config: dict[str, Any],
        run_envelope: RunEnvelope,
    ) -> EvalRunSummary:
        """Execute all tasks in an evaluation suite.

        Runs tasks in parallel with semaphore-controlled concurrency,
        tracks resource usage, handles cancellation, and aggregates
        results into a suite-level summary.

        Args:
            suite: The evaluation suite to run.
            agent_config: Agent configuration for all trials.
            run_envelope: Reproducibility metadata for this run.

        Returns:
            EvalRunSummary containing all trial results and aggregated metrics.
        """
        self._cancelled = False
        self._resource_tracker = ResourceTracker()
        self._trials = []
        self._trial_durations = []

        await get_bus().publish(
            AHEvents.SUITE_STARTED,
            {
                "suite_id": suite.id,
                "run_id": run_envelope.run_id,
                "task_count": len(suite.tasks),
            },
        )

        start_time = time.time()

        try:
            tasks = [
                self._run_with_semaphore(
                    task=task,
                    agent_config=agent_config,
                    run_envelope=run_envelope,
                )
                for task in suite.tasks
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                task = suite.tasks[i]
                if isinstance(result, Exception):
                    trial = self._create_failed_trial(
                        task=task,
                        run_envelope=run_envelope,
                        error=str(result),
                    )
                    self._trials.append(trial)
                elif isinstance(result, EvalTrial):
                    self._trials.append(result)

        except asyncio.CancelledError:
            self._cancelled = True

        finally:
            end_time = time.time()
            suite_duration = end_time - start_time

        metrics = self._build_metrics(
            suite=suite,
            run_envelope=run_envelope,
            suite_duration=suite_duration,
        )

        summary = EvalRunSummary(
            envelope=run_envelope,
            metrics=metrics,
            trials=self._trials,
        )

        try:
            await self._storage.save_summary(
                suite_id=suite.id,
                run_id=run_envelope.run_id,
                summary=summary,
            )
        except Exception:
            pass

        await get_bus().publish(
            AHEvents.SUITE_COMPLETED,
            {
                "suite_id": suite.id,
                "run_id": run_envelope.run_id,
                "total_tasks": metrics.total_tasks,
                "completed_tasks": metrics.completed_tasks,
                "passed_tasks": metrics.passed_tasks,
                "pass_rate": metrics.pass_rate,
            },
        )

        return summary

    async def _run_with_semaphore(
        self,
        task: EvalTask,
        agent_config: dict[str, Any],
        run_envelope: RunEnvelope,
    ) -> EvalTrial:
        """Run a single task with semaphore-controlled concurrency.

        Args:
            task: The task to execute.
            agent_config: Agent configuration.
            run_envelope: Run envelope for trial metadata.

        Returns:
            EvalTrial containing the result.

        Raises:
            asyncio.CancelledError: Re-raised if cancellation was requested.
        """
        if self._cancelled:
            trial = self._create_cancelled_trial(task, run_envelope)
            self._trials.append(trial)
            return trial

        async with self._semaphore:
            if self._cancelled:
                trial = self._create_cancelled_trial(task, run_envelope)
                self._trials.append(trial)
                return trial

            trial_start = time.time()

            try:
                result = await self._trial_executor.execute(
                    task=task,
                    agent_config=agent_config,
                    run_envelope=run_envelope,
                )

                trial_end = time.time()
                trial_duration = trial_end - trial_start
                self._trial_durations.append(trial_duration)

                await self._resource_tracker.add_trial_usage(
                    tokens=result.transcript.token_usage,
                    cost_usd=result.transcript.cost_usd,
                    duration_seconds=result.transcript.duration_seconds,
                )

                trial = EvalTrial(
                    id=result.trial_id,
                    task_id=task.id,
                    status=self._outcome_to_status(result.outcome),
                    attempt_number=1,
                    input_snapshot=task.input,
                    result=result,
                    envelope=await self._get_trial_envelope(run_envelope, result.trial_id),
                )

                return trial

            except asyncio.CancelledError:
                trial = self._create_cancelled_trial(task, run_envelope)
                self._trials.append(trial)
                raise

    async def _get_trial_envelope(
        self,
        run_envelope: RunEnvelope,
        trial_id: str,
    ) -> TrialEnvelope | None:
        try:
            stored = await self._storage.load_trial(
                suite_id=run_envelope.suite_id,
                run_id=run_envelope.run_id,
                trial_id=trial_id,
            )
            if stored:
                return stored.envelope
        except Exception:
            pass
        return None

    def _create_failed_trial(
        self,
        task: EvalTask,
        run_envelope: RunEnvelope,
        error: str,
    ) -> EvalTrial:
        del run_envelope
        trial_id = f"trial-{uuid.uuid4().hex[:8]}"
        return EvalTrial(
            id=trial_id,
            task_id=task.id,
            status=EvalStatus.ERROR,
            attempt_number=1,
            input_snapshot=task.input,
            result=TrialResult(
                trial_id=trial_id,
                outcome=EvalOutcome.failure(
                    FailureMode.AGENT_ERROR,
                    error,
                ),
            ),
        )

    def _create_cancelled_trial(
        self,
        task: EvalTask,
        run_envelope: RunEnvelope,
    ) -> EvalTrial:
        del run_envelope
        trial_id = f"trial-{uuid.uuid4().hex[:8]}"
        return EvalTrial(
            id=trial_id,
            task_id=task.id,
            status=EvalStatus.CANCELLED,
            attempt_number=1,
            input_snapshot=task.input,
            result=TrialResult(
                trial_id=trial_id,
                outcome=EvalOutcome.failure(
                    FailureMode.CRASH,
                    "Suite execution was cancelled",
                ),
            ),
        )

    def _outcome_to_status(self, outcome: EvalOutcome) -> EvalStatus:
        return outcome.status

    def _build_metrics(
        self,
        suite: EvalSuite,
        run_envelope: RunEnvelope,
        suite_duration: float,
    ) -> SuiteMetrics:
        """Build aggregate metrics from all trial results."""
        total_tasks = len(suite.tasks)
        completed_tasks = 0
        passed_tasks = 0
        failed_tasks = 0
        total_score = 0.0

        for trial in self._trials:
            if trial.status == EvalStatus.COMPLETED:
                completed_tasks += 1
                if trial.result and trial.result.aggregate_passed:
                    passed_tasks += 1
                    total_score += trial.result.aggregate_score
                elif trial.result:
                    total_score += trial.result.aggregate_score
            elif trial.status == EvalStatus.ERROR:
                failed_tasks += 1
                completed_tasks += 1
            elif trial.status == EvalStatus.CANCELLED:
                failed_tasks += 1

        pass_rate = passed_tasks / completed_tasks if completed_tasks > 0 else 0.0
        mean_score = total_score / completed_tasks if completed_tasks > 0 else 0.0

        latency_p50 = None
        latency_p95 = None
        latency_p99 = None

        if self._trial_durations:
            sorted_durations = sorted(self._trial_durations)
            n = len(sorted_durations)
            latency_p50 = sorted_durations[int(n * 0.5)]
            latency_p95 = sorted_durations[int(n * 0.95)] if n >= 20 else sorted_durations[-1]
            latency_p99 = sorted_durations[int(n * 0.99)] if n >= 100 else sorted_durations[-1]

        return SuiteMetrics(
            suite_id=suite.id,
            run_id=run_envelope.run_id,
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            passed_tasks=passed_tasks,
            failed_tasks=failed_tasks,
            pass_rate=pass_rate,
            mean_score=mean_score,
            total_tokens=self._resource_tracker.total_tokens,
            total_cost_usd=self._resource_tracker.total_cost_usd,
            total_duration_seconds=suite_duration,
            latency_p50_seconds=latency_p50,
            latency_p95_seconds=latency_p95,
            latency_p99_seconds=latency_p99,
            created_at=datetime.now(UTC).isoformat(),
        )


__all__ = ["EvalRunner", "ResourceTracker"]
