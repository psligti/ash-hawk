"""Queue-based throttling for LLM requests and trial execution.

This module provides thin wrappers around Dawn Kestrel's InMemoryAgentExecutionQueue
to throttle both LLM API calls and trial execution.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

from ash_hawk.types import (
    EvalOutcome,
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
        return getattr(
            execution_queue_module,
            "InMemoryAgentExecutionQueue",
            getattr(execution_queue_module, "AgentExecutionQueue"),
        )

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
        start_time = time.time()
        response: Any

        try:
            queue_type = self._get_queue_type()
            job_type = self._get_job_type()

            queue = queue_type(
                max_workers=self.max_workers,
                poll_interval=self.poll_interval,
                timeout_seconds=self.timeout_seconds,
            )

            queue_job = job_type(index=0, task_id=request.request_id)
            error: Exception | None = None
            queued_response: Any = None

            async def execute_job(_: Any) -> str:
                nonlocal queued_response, error
                try:
                    queued_response = await execute(request)
                    return request.request_id
                except Exception as e:
                    error = e
                    raise

            batch_result = await queue.run_jobs([queue_job], execute_job)

            if 0 in batch_result.errors_by_index:
                raise RuntimeError(batch_result.errors_by_index[0])

            if error is not None:
                raise error

            response = queued_response
        except Exception as queue_exc:
            logger.warning(
                "LLMRequestQueue backend failed; falling back to direct execution: %s",
                queue_exc,
            )
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
        return getattr(
            execution_queue_module,
            "InMemoryAgentExecutionQueue",
            getattr(execution_queue_module, "AgentExecutionQueue"),
        )

    def _get_job_type(self) -> Any:
        execution_queue_module = importlib.import_module("dawn_kestrel.agents.execution_queue")
        return getattr(execution_queue_module, "AgentExecutionJob")

    async def run_trials(
        self,
        jobs: list[TrialJob],
        execute: Callable[[TrialJob], Awaitable[Any]],
    ) -> list[TrialJobResult | Exception]:
        """Execute trial jobs with throttling.

        Args:
            jobs: List of trial jobs to execute.
            execute: Async function to execute each trial. Can return either
                (transcript, outcome) tuple or a TrialResult-like object with
                additional fields.

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

        results: list[TrialJobResult | Exception] = [Exception("not executed") for _ in jobs]
        start_times: dict[str, float] = {}

        async def execute_job(queue_job: Any) -> str:
            idx = cast(int, queue_job.index)
            job: TrialJob = jobs[idx]
            job_id: str = job.job_id
            start_times[job_id] = time.time()

            try:
                result = await execute(job)
                elapsed = time.time() - start_times[job_id]
                results[idx] = self._to_trial_job_result(job, result, elapsed)
                return job_id
            except Exception as e:
                results[idx] = e
                raise

        try:
            batch_result = await queue.run_jobs(queue_jobs, execute_job)
        except Exception as queue_exc:
            logger.warning(
                "TrialExecutionQueue backend failed; falling back to local execution: %s",
                queue_exc,
            )
            return await self._run_trials_fallback(jobs, execute)

        for idx, error_msg in batch_result.errors_by_index.items():
            if not isinstance(results[idx], TrialJobResult):
                results[idx] = Exception(error_msg)

        unresolved_indices = [
            idx for idx, result in enumerate(results) if self._is_not_executed_result(result)
        ]
        if unresolved_indices:
            unresolved_job_ids = [jobs[idx].job_id for idx in unresolved_indices]
            logger.warning(
                "TrialExecutionQueue returned unresolved jobs (%d/%d): %s; "
                "retrying unresolved jobs locally",
                len(unresolved_indices),
                len(jobs),
                ", ".join(unresolved_job_ids[:10]),
            )
            fallback_results = await self._run_trials_subset_fallback(
                jobs, unresolved_indices, execute
            )
            for idx, fallback_result in fallback_results.items():
                results[idx] = fallback_result

        final_unresolved = [
            idx for idx, result in enumerate(results) if self._is_not_executed_result(result)
        ]
        if final_unresolved:
            unresolved_job_ids = [jobs[idx].job_id for idx in final_unresolved]
            logger.error(
                "TrialExecutionQueue still unresolved after local retry (%d/%d): %s",
                len(final_unresolved),
                len(jobs),
                ", ".join(unresolved_job_ids[:10]),
            )

        return results

    @staticmethod
    def _is_not_executed_result(result: TrialJobResult | Exception) -> bool:
        if not isinstance(result, Exception):
            return False
        return str(result).strip().lower() == "not executed"

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

    async def _run_trials_fallback(
        self,
        jobs: list[TrialJob],
        execute: Callable[[TrialJob], Awaitable[Any]],
    ) -> list[TrialJobResult | Exception]:
        results: list[TrialJobResult | Exception] = [Exception("not executed") for _ in jobs]
        semaphore = asyncio.Semaphore(self.max_workers)

        async def run_one(idx: int, job: TrialJob) -> None:
            async with semaphore:
                started = time.time()
                try:
                    raw_result = await asyncio.wait_for(
                        execute(job),
                        timeout=self.timeout_seconds,
                    )
                    elapsed = time.time() - started
                    results[idx] = self._to_trial_job_result(job, raw_result, elapsed)
                except Exception as exc:
                    results[idx] = exc

        await asyncio.gather(
            *(run_one(idx, job) for idx, job in enumerate(jobs)),
            return_exceptions=False,
        )
        return results

    async def _run_trials_subset_fallback(
        self,
        jobs: list[TrialJob],
        indices: list[int],
        execute: Callable[[TrialJob], Awaitable[Any]],
    ) -> dict[int, TrialJobResult | Exception]:
        fallback_results: dict[int, TrialJobResult | Exception] = {}
        semaphore = asyncio.Semaphore(self.max_workers)

        async def run_one(idx: int) -> None:
            async with semaphore:
                job = jobs[idx]
                started = time.time()
                try:
                    raw_result = await asyncio.wait_for(
                        execute(job),
                        timeout=self.timeout_seconds,
                    )
                    elapsed = time.time() - started
                    fallback_results[idx] = self._to_trial_job_result(job, raw_result, elapsed)
                except Exception as exc:
                    fallback_results[idx] = exc

        await asyncio.gather(*(run_one(idx) for idx in indices), return_exceptions=False)
        return fallback_results


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
