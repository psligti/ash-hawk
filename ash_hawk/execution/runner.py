"""Parallel suite runner for Ash-Hawk evaluation harness.

This module provides the EvalRunner class for executing evaluation suites
with controlled parallelism, timeout management, and graceful cancellation.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ash_hawk.config import EvalConfig
from ash_hawk.events import AHEvents, get_bus
from ash_hawk.execution.queue import (
    LLMRequestQueue,
    TrialExecutionQueue,
    TrialJob,
    get_llm_queue_sync,
    register_llm_queue,
)
from ash_hawk.execution.trial import TrialExecutor
from ash_hawk.types import (
    EvalOutcome,
    EvalRunSummary,
    EvalStatus,
    EvalSuite,
    EvalTask,
    EvalTranscript,
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
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None
        self.total_tokens = TokenUsage()
        self.total_cost_usd = 0.0
        self.total_duration_seconds = 0.0
        self.total_queue_wait_seconds = 0.0
        self.peak_queue_depth = 0

    def _get_lock(self) -> asyncio.Lock:
        """Get lock, recreating if event loop changed."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._lock is None or self._lock_loop != current_loop:
            self._lock = asyncio.Lock()
            self._lock_loop = current_loop
        assert self._lock is not None
        return self._lock

    async def add_trial_usage(
        self,
        tokens: TokenUsage,
        cost_usd: float,
        duration_seconds: float,
        queue_wait_seconds: float = 0.0,
    ) -> None:
        """Add trial resource usage to totals."""
        async with self._get_lock():
            self.total_tokens = TokenUsage(
                input=self.total_tokens.input + tokens.input,
                output=self.total_tokens.output + tokens.output,
                reasoning=self.total_tokens.reasoning + tokens.reasoning,
                cache_read=self.total_tokens.cache_read + tokens.cache_read,
                cache_write=self.total_tokens.cache_write + tokens.cache_write,
            )
            self.total_cost_usd += cost_usd
            self.total_duration_seconds += duration_seconds
            self.total_queue_wait_seconds += queue_wait_seconds

    async def update_queue_depth(self, depth: int) -> None:
        """Update peak queue depth if new depth is higher."""
        async with self._get_lock():
            if depth > self.peak_queue_depth:
                self.peak_queue_depth = depth


class EvalRunner:
    """Parallel suite runner with timeout, budget, and cancellation support."""

    def __init__(
        self,
        config: EvalConfig,
        storage: StorageBackend,
        trial_executor: TrialExecutor,
        post_run_hook: Any | None = None,
        on_trial_progress: Callable[[int, int, int, str], Awaitable[None]] | None = None,
    ) -> None:
        self._config = config
        self._storage = storage
        self._trial_executor = trial_executor
        self._on_trial_progress = on_trial_progress
        existing_queue = get_llm_queue_sync()
        if existing_queue is not None:
            self._llm_queue = existing_queue
        else:
            self._llm_queue = LLMRequestQueue(
                max_workers=config.llm_max_workers,
                timeout_seconds=config.llm_timeout_seconds,
            )
            register_llm_queue(self._llm_queue)
        self._trial_queue = TrialExecutionQueue(
            max_workers=config.trial_max_workers,
            timeout_seconds=config.default_timeout_seconds,
        )
        self._cancelled = False
        self._resource_tracker = ResourceTracker()
        self._trials: list[EvalTrial] = []
        self._trial_durations: list[float] = []

        if post_run_hook is not None:
            self._trial_executor.set_post_run_hook(post_run_hook)

    @property
    def llm_queue(self) -> LLMRequestQueue:
        """Get the shared LLM request queue."""
        return self._llm_queue

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

        Runs tasks in parallel with queue-based throttling,
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
            jobs = [
                TrialJob(
                    job_id=f"job_{task.id}_{uuid.uuid4().hex[:8]}",
                    task_id=task.id,
                    task_input=task.input,
                )
                for task in suite.tasks
            ]

            async def execute_trial(job: TrialJob) -> TrialResult:
                task = next((t for t in suite.tasks if t.id == job.task_id), None)
                if task is None:
                    raise ValueError(f"Task not found: {job.task_id}")

                result = await self._trial_executor.execute(
                    task=task,
                    agent_config=agent_config,
                    run_envelope=run_envelope,
                )
                return result

            results = await self._trial_queue.run_trials(
                jobs, execute_trial, on_progress=self._on_trial_progress
            )

            for i, result in enumerate(results):
                task = suite.tasks[i]

                if isinstance(result, Exception):
                    trial = self._create_failed_trial(
                        task=task,
                        run_envelope=run_envelope,
                        error=str(result),
                    )
                    self._trials.append(trial)
                else:
                    await self._resource_tracker.add_trial_usage(
                        tokens=result.transcript.token_usage,
                        cost_usd=result.transcript.cost_usd,
                        duration_seconds=result.transcript.duration_seconds,
                        queue_wait_seconds=result.wait_time_seconds,
                    )
                    self._trial_durations.append(result.wait_time_seconds)

                    trial_id = str(agent_config.get("trial_id") or f"trial-{uuid.uuid4().hex[:8]}")
                    trial = EvalTrial(
                        id=trial_id,
                        task_id=task.id,
                        status=self._outcome_to_status(result.outcome),
                        attempt_number=1,
                        input_snapshot=task.input,
                        result=TrialResult(
                            trial_id=trial_id,
                            transcript=result.transcript,
                            outcome=result.outcome,
                            grader_results=result.grader_results or [],
                            aggregate_score=result.aggregate_score,
                            aggregate_passed=result.aggregate_passed,
                        ),
                        envelope=await self._get_trial_envelope(run_envelope, trial_id),
                    )
                    self._trials.append(trial)

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
