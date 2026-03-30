"""Parallel suite runner for Ash-Hawk evaluation harness."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ash_hawk.config import EvalConfig
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
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None
        self.total_tokens = TokenUsage()
        self.total_cost_usd = 0.0
        self.total_duration_seconds = 0.0

    def _get_lock(self) -> asyncio.Lock:
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
    ) -> None:
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
        self._cancelled = False
        self._resource_tracker = ResourceTracker()
        self._trials: list[Any] = []
        self._trial_durations: list[float] = []

        if post_run_hook is not None:
            self._trial_executor.set_post_run_hook(post_run_hook)

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True

    async def run_suite(
        self,
        suite: EvalSuite,
        agent_config: dict[str, Any],
        run_envelope: RunEnvelope,
    ) -> EvalRunSummary:
        self._cancelled = False
        self._resource_tracker = ResourceTracker()
        self._trials = []
        self._trial_durations = []

        start_time = time.time()

        try:
            semaphore = asyncio.Semaphore(self._config.trial_max_workers)

            async def execute_task(task: EvalTask) -> tuple[EvalTask, TrialResult | Exception]:
                async with semaphore:
                    if self._cancelled:
                        return task, Exception("Suite cancelled")
                    try:
                        result = await self._trial_executor.execute(
                            task=task,
                            agent_config=agent_config,
                            run_envelope=run_envelope,
                        )
                        return task, result
                    except Exception as e:
                        return task, e

            tasks = [execute_task(task) for task in suite.tasks]
            results = await asyncio.gather(*tasks)

            for task, result in results:
                if isinstance(result, Exception):
                    trial = _create_failed_trial(task, str(result))
                    self._trials.append(trial)
                else:
                    await self._resource_tracker.add_trial_usage(
                        tokens=result.transcript.token_usage,
                        cost_usd=result.transcript.cost_usd,
                        duration_seconds=result.transcript.duration_seconds,
                    )
                    self._trial_durations.append(result.transcript.duration_seconds)

                    trial_id = str(agent_config.get("trial_id") or f"trial-{uuid.uuid4().hex[:8]}")
                    trial = EvalTrial(
                        id=trial_id,
                        task_id=task.id,
                        status=result.outcome.status,
                        attempt_number=1,
                        input_snapshot=task.input,
                        result=result,
                    )
                    self._trials.append(trial)

        except asyncio.CancelledError:
            self._cancelled = True

        finally:
            end_time = time.time()
            suite_duration = end_time - start_time

        metrics = _build_metrics(
            suite=suite,
            run_envelope=run_envelope,
            suite_duration=suite_duration,
            trials=self._trials,
            resource_tracker=self._resource_tracker,
            trial_durations=self._trial_durations,
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

        return summary


def _create_failed_trial(task: EvalTask, error: str) -> Any:
    from ash_hawk.types import EvalTrial, TrialResult

    trial_id = f"trial-{uuid.uuid4().hex[:8]}"
    return EvalTrial(
        id=trial_id,
        task_id=task.id,
        status=EvalStatus.ERROR,
        attempt_number=1,
        input_snapshot=task.input,
        result=TrialResult(
            trial_id=trial_id,
            outcome=EvalOutcome.failure(FailureMode.AGENT_ERROR, error),
        ),
    )


def _build_metrics(
    suite: EvalSuite,
    run_envelope: RunEnvelope,
    suite_duration: float,
    trials: list[Any],
    resource_tracker: ResourceTracker,
    trial_durations: list[float],
) -> SuiteMetrics:
    from ash_hawk.types import EvalStatus, SuiteMetrics

    total_tasks = len(suite.tasks)
    completed_tasks = 0
    passed_tasks = 0
    failed_tasks = 0
    total_score = 0.0

    for trial in trials:
        if trial.status == EvalStatus.COMPLETED:
            completed_tasks += 1
            if trial.result and trial.result.aggregate_passed:
                passed_tasks += 1
            if trial.result:
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

    if trial_durations:
        sorted_durations = sorted(trial_durations)
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
        total_tokens=resource_tracker.total_tokens,
        total_cost_usd=resource_tracker.total_cost_usd,
        total_duration_seconds=suite_duration,
        latency_p50_seconds=latency_p50,
        latency_p95_seconds=latency_p95,
        latency_p99_seconds=latency_p99,
        created_at=datetime.now(UTC).isoformat(),
    )


__all__ = ["EvalRunner", "ResourceTracker"]
