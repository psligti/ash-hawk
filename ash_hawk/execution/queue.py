"""Queue-based throttling for LLM requests and trial execution.

This module provides semaphore-based throttling for both LLM API calls
and trial execution to control concurrency.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

from ash_hawk.types import (
    EvalOutcome,
    EvalStatus,
    EvalTranscript,
    GraderResult,
    TokenUsage,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMRequest:
    """Represents a queued LLM request."""

    request_id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    options: Any


@dataclass
class LLMResponse:
    """Result of an LLM request."""

    request_id: str
    response: Any
    wait_time_seconds: float
    token_usage: TokenUsage
    cost_usd: float


@dataclass(frozen=True)
class TrialJob:
    """Represents a queued trial execution job."""

    job_id: str
    task_id: str
    task_input: Any


@dataclass
class TrialJobResult:
    """Result of a trial execution job."""

    job_id: str
    task_id: str
    transcript: EvalTranscript
    outcome: EvalOutcome
    wait_time_seconds: float
    grader_results: list[GraderResult] | None = None
    aggregate_score: float = 0.0
    aggregate_passed: bool = False


class LLMRequestQueue:
    """Queue for throttling LLM API requests using semaphore-based concurrency.

    Example:
        queue = LLMRequestQueue(max_workers=4, timeout_seconds=300)
        response = await queue.execute(request, llm_client.complete)
    """

    def __init__(
        self,
        max_workers: int = 4,
        timeout_seconds: float = 300.0,
        poll_interval: float = 0.05,
    ) -> None:
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self.poll_interval = poll_interval
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None
        self._stats = {
            "total_requests": 0,
            "total_wait_time": 0.0,
        }

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get semaphore, recreating if event loop changed."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._semaphore is None or self._semaphore_loop != current_loop:
            self._semaphore = asyncio.Semaphore(self.max_workers)
            self._semaphore_loop = current_loop
        return self._semaphore

    async def execute(
        self,
        request: LLMRequest,
        execute: Callable[[LLMRequest], Awaitable[Any]],
    ) -> LLMResponse:
        """Execute a single LLM request with throttling.

        Args:
            request: The LLM request to execute.
            execute: Async function to execute the request.

        Returns:
            LLMResponse with the result and timing info.

        Raises:
            Exception: Re-raised from execute function.
        """
        start_time = time.time()

        async with self._get_semaphore():
            response = await execute(request)

        wait_time = time.time() - start_time
        self._stats["total_requests"] += 1
        self._stats["total_wait_time"] += wait_time

        token_usage = self._extract_token_usage(response)
        cost_usd = self._extract_cost(response)

        return LLMResponse(
            request_id=request.request_id,
            response=response,
            wait_time_seconds=wait_time,
            token_usage=token_usage,
            cost_usd=cost_usd,
        )

    async def get_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        return self._stats.copy()

    def _extract_token_usage(self, response: Any) -> TokenUsage:
        usage = TokenUsage()
        if hasattr(response, "usage") and response.usage:
            usage_data = response.usage
            usage.input = getattr(usage_data, "input", 0) or 0
            usage.output = getattr(usage_data, "output", 0) or 0
            usage.reasoning = getattr(usage_data, "reasoning", 0) or 0
            usage.cache_read = getattr(usage_data, "cache_read", 0) or 0
            usage.cache_write = getattr(usage_data, "cache_write", 0) or 0
        return usage

    def _extract_cost(self, response: Any) -> float:
        if hasattr(response, "cost"):
            cost = response.cost
            if hasattr(cost, "__float__"):
                return float(cost)
            return cost or 0.0
        return 0.0


class TrialExecutionQueue:
    """Queue for throttling trial executions using semaphore-based concurrency.

    Example:
        queue = TrialExecutionQueue(max_workers=4, timeout_seconds=600)
        results = await queue.run_trials(
            jobs=[job1, job2, job3],
            execute=trial_executor.execute,
        )
    """

    def __init__(
        self,
        max_workers: int = 4,
        timeout_seconds: float = 600.0,
        poll_interval: float = 0.05,
    ) -> None:
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self.poll_interval = poll_interval

    async def run_trials(
        self,
        jobs: list[TrialJob],
        execute: Callable[[TrialJob], Awaitable[Any]],
        on_progress: Callable[[int, int, int, str], Awaitable[None]] | None = None,
    ) -> list[TrialJobResult | Exception]:
        """Execute trial jobs with throttling.

        Args:
            jobs: List of trial jobs to execute.
            execute: Async function to execute each trial. Can return either
                (transcript, outcome) tuple or a TrialResult-like object with
                additional fields.
            on_progress: Optional callback(completed, total, running_delta, status)
                called when trials start (running_delta=+1, status="running") and
                complete (running_delta=-1, status="passed"/"failed"/"incomplete").

        Returns:
            List of results or exceptions in job order.
        """
        if not jobs:
            return []

        results: list[TrialJobResult | Exception] = [Exception("not executed") for _ in jobs]
        semaphore = asyncio.Semaphore(self.max_workers)
        completed = 0
        completed_lock = asyncio.Lock()

        async def run_one(idx: int, job: TrialJob) -> None:
            nonlocal completed
            async with semaphore:
                if on_progress:
                    await on_progress(0, len(jobs), 1, "running")

                started = time.time()
                status = "incomplete"
                try:
                    raw_result = await asyncio.wait_for(
                        execute(job),
                        timeout=self.timeout_seconds,
                    )
                    elapsed = time.time() - started
                    results[idx] = self._to_trial_job_result(job, raw_result, elapsed)
                    job_result = results[idx]
                    if isinstance(job_result, TrialJobResult):
                        if job_result.outcome.status == EvalStatus.COMPLETED:
                            status = "passed"
                        else:
                            status = "failed"
                except Exception as exc:
                    results[idx] = exc
                    status = "incomplete"

                if on_progress:
                    async with completed_lock:
                        completed += 1
                    await on_progress(completed, len(jobs), -1, status)

        await asyncio.gather(
            *(run_one(idx, job) for idx, job in enumerate(jobs)),
            return_exceptions=False,
        )
        return results

    @staticmethod
    def _to_trial_job_result(job: TrialJob, result: Any, elapsed: float) -> TrialJobResult:
        if isinstance(result, tuple):
            result_tuple = cast(tuple[object, ...], result)
            if len(result_tuple) != 2:
                raise TypeError("Trial execute() returned invalid tuple result")
            transcript_raw = result_tuple[0]
            outcome_raw = result_tuple[1]
            if not isinstance(transcript_raw, EvalTranscript) or not isinstance(
                outcome_raw, EvalOutcome
            ):
                raise TypeError("Trial execute() tuple must be (EvalTranscript, EvalOutcome)")
            return TrialJobResult(
                job_id=job.job_id,
                task_id=job.task_id,
                transcript=transcript_raw,
                outcome=outcome_raw,
                wait_time_seconds=elapsed,
            )

        transcript_value = getattr(result, "transcript", None)
        outcome_value = getattr(result, "outcome", None)
        if not isinstance(transcript_value, EvalTranscript) or not isinstance(
            outcome_value, EvalOutcome
        ):
            raise TypeError(
                "Trial execute() returned unsupported result type; expected tuple or object with "
                "EvalTranscript/EvalOutcome"
            )

        return TrialJobResult(
            job_id=job.job_id,
            task_id=job.task_id,
            transcript=transcript_value,
            outcome=outcome_value,
            wait_time_seconds=elapsed,
            grader_results=getattr(result, "grader_results", None),
            aggregate_score=getattr(result, "aggregate_score", 0.0),
            aggregate_passed=getattr(result, "aggregate_passed", False),
        )


_llm_queue: LLMRequestQueue | None = None


def register_llm_queue(queue: LLMRequestQueue) -> None:
    """Register a pre-configured LLM request queue globally."""
    global _llm_queue
    _llm_queue = queue


def get_llm_queue(
    max_workers: int = 4,
    timeout_seconds: float = 300.0,
) -> LLMRequestQueue:
    """Get or create the global LLM request queue singleton."""
    global _llm_queue
    if _llm_queue is None:
        _llm_queue = LLMRequestQueue(
            max_workers=max_workers,
            timeout_seconds=timeout_seconds,
        )
    return _llm_queue


def get_llm_queue_sync() -> LLMRequestQueue | None:
    """Get the global LLM request queue if registered (sync version)."""
    return _llm_queue


def reset_llm_queue() -> None:
    """Reset the global LLM request queue (for testing)."""
    global _llm_queue
    _llm_queue = None


__all__ = [
    "LLMRequest",
    "LLMRequestQueue",
    "LLMResponse",
    "TrialExecutionQueue",
    "TrialJob",
    "TrialJobResult",
    "get_llm_queue",
    "get_llm_queue_sync",
    "register_llm_queue",
    "reset_llm_queue",
]
