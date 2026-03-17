"""Queue-based throttling for LLM requests and trial execution.

This module provides thin wrappers around Dawn Kestrel's InMemoryAgentExecutionQueue
to throttle both LLM API calls and trial execution.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

from ash_hawk.types import (
    EvalOutcome,
    EvalTranscript,
    TokenUsage,
)


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


class LLMRequestQueue:
    """Queue for throttling LLM API requests using Dawn Kestrel's execution queue.

    Uses InMemoryAgentExecutionQueue directly for consistent throttling behavior.

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
        self._stats = {
            "total_requests": 0,
            "total_wait_time": 0.0,
        }

    def _get_queue_type(self) -> Any:
        execution_queue_module = importlib.import_module("dawn_kestrel.agents.execution_queue")
        return getattr(execution_queue_module, "InMemoryAgentExecutionQueue")

    def _get_job_type(self) -> Any:
        execution_queue_module = importlib.import_module("dawn_kestrel.agents.execution_queue")
        return getattr(execution_queue_module, "AgentExecutionJob")

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
        queue_type = self._get_queue_type()
        job_type = self._get_job_type()

        queue = queue_type(
            max_workers=self.max_workers,
            poll_interval=self.poll_interval,
            timeout_seconds=self.timeout_seconds,
        )

        queue_job = job_type(index=0, task_id=request.request_id)
        start_time = time.time()
        response: Any = None
        error: Exception | None = None

        async def execute_job(_: Any) -> str:
            nonlocal response, error
            try:
                response = await execute(request)
                return request.request_id
            except Exception as e:
                error = e
                raise

        batch_result = await queue.run_jobs([queue_job], execute_job)

        wait_time = time.time() - start_time
        self._stats["total_requests"] += 1
        self._stats["total_wait_time"] += wait_time

        if 0 in batch_result.errors_by_index:
            raise Exception(batch_result.errors_by_index[0])

        if error:
            raise error

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
    """Queue for throttling trial executions.

    Uses Dawn Kestrel's InMemoryAgentExecutionQueue directly to control
    how many trials run concurrently.

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

    def _get_queue_type(self) -> Any:
        execution_queue_module = importlib.import_module("dawn_kestrel.agents.execution_queue")
        return getattr(execution_queue_module, "InMemoryAgentExecutionQueue")

    def _get_job_type(self) -> Any:
        execution_queue_module = importlib.import_module("dawn_kestrel.agents.execution_queue")
        return getattr(execution_queue_module, "AgentExecutionJob")

    async def run_trials(
        self,
        jobs: list[TrialJob],
        execute: Callable[[TrialJob], Awaitable[tuple[EvalTranscript, EvalOutcome]]],
    ) -> list[TrialJobResult | Exception]:
        """Execute trial jobs with throttling.

        Args:
            jobs: List of trial jobs to execute.
            execute: Async function to execute each trial.

        Returns:
            List of results or exceptions in job order.
        """
        if not jobs:
            return []

        queue_type = self._get_queue_type()
        job_type = self._get_job_type()

        queue = queue_type(
            max_workers=self.max_workers,
            poll_interval=self.poll_interval,
            timeout_seconds=self.timeout_seconds,
        )

        queue_jobs = [job_type(index=i, task_id=job.task_id) for i, job in enumerate(jobs)]

        results: list[TrialJobResult | Exception] = [Exception("not executed")] * len(jobs)
        start_times: dict[str, float] = {}

        async def execute_job(queue_job: Any) -> str:
            idx = cast(int, queue_job.index)
            job: TrialJob = jobs[idx]
            job_id: str = job.job_id
            start_times[job_id] = time.time()

            try:
                transcript, outcome = await execute(job)
                elapsed = time.time() - start_times[job_id]

                results[idx] = TrialJobResult(
                    job_id=job_id,
                    task_id=job.task_id,
                    transcript=transcript,
                    outcome=outcome,
                    wait_time_seconds=elapsed,
                )
                return job_id
            except Exception as e:
                results[idx] = e
                raise

        batch_result = await queue.run_jobs(queue_jobs, execute_job)

        for idx, error_msg in batch_result.errors_by_index.items():
            if not isinstance(results[idx], TrialJobResult):
                results[idx] = Exception(error_msg)

        return results


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
