"""Tests for queue-based throttling."""

from __future__ import annotations

import asyncio
import threading

import pytest

from ash_hawk.execution.queue import (
    LLMRequest,
    LLMRequestQueue,
    LLMResponse,
    TrialExecutionQueue,
    TrialJob,
    TrialJobResult,
    get_llm_queue,
    get_llm_queue_sync,
    register_llm_queue,
    reset_llm_queue,
)
from ash_hawk.types import EvalOutcome, EvalStatus, EvalTranscript


class TestLLMRequestQueue:
    """Tests for LLMRequestQueue."""

    def test_init_default_values(self) -> None:
        """Queue initializes with default values."""
        queue = LLMRequestQueue()
        assert queue.max_workers == 4
        assert queue.timeout_seconds == 300.0

    def test_init_custom_values(self) -> None:
        """Queue initializes with custom values."""
        queue = LLMRequestQueue(max_workers=8, timeout_seconds=600.0)
        assert queue.max_workers == 8
        assert queue.timeout_seconds == 600.0

    @pytest.mark.asyncio
    async def test_execute_single_request(self) -> None:
        """Execute a single LLM request with throttling."""
        queue = LLMRequestQueue(max_workers=1, timeout_seconds=5.0)

        request = LLMRequest(
            request_id="test-1",
            messages=[{"role": "user", "content": "Hello"}],
            tools=None,
            options=None,
        )

        async def mock_execute(req: LLMRequest) -> dict[str, str]:
            return {"response": f"processed {req.request_id}"}

        response = await queue.execute(request, mock_execute)

        assert isinstance(response, LLMResponse)
        assert response.request_id == "test-1"
        assert response.response == {"response": "processed test-1"}

    @pytest.mark.asyncio
    async def test_execute_concurrent_requests(self) -> None:
        """Execute multiple concurrent requests with throttling."""
        queue = LLMRequestQueue(max_workers=2, timeout_seconds=5.0)

        call_order: list[str] = []

        async def mock_execute(req: LLMRequest) -> dict[str, str]:
            call_order.append(req.request_id)
            await asyncio.sleep(0.1)
            return {"id": req.request_id}

        async def execute_one(i: int) -> LLMResponse:
            request = LLMRequest(
                request_id=f"req-{i}",
                messages=[],
                tools=None,
                options=None,
            )
            return await queue.execute(request, mock_execute)

        results = await asyncio.gather(*[execute_one(i) for i in range(4)])

        assert len(results) == 4
        assert all(isinstance(r, LLMResponse) for r in results)

    @pytest.mark.asyncio
    async def test_get_stats(self) -> None:
        """Get queue statistics."""
        queue = LLMRequestQueue(max_workers=1, timeout_seconds=5.0)

        stats = await queue.get_stats()
        assert stats["total_requests"] == 0
        assert stats["total_wait_time"] == 0.0


class TestTrialExecutionQueue:
    """Tests for TrialExecutionQueue."""

    def test_init_default_values(self) -> None:
        """Queue initializes with default values."""
        queue = TrialExecutionQueue()
        assert queue.max_workers == 4
        assert queue.timeout_seconds == 600.0

    @pytest.mark.asyncio
    async def test_run_trials_empty_list(self) -> None:
        """Run empty job list returns empty results."""
        queue = TrialExecutionQueue()

        async def mock_execute(job: TrialJob) -> tuple[EvalTranscript, EvalOutcome]:
            return EvalTranscript(), EvalOutcome.success()

        results = await queue.run_trials([], mock_execute)
        assert results == []

    @pytest.mark.asyncio
    async def test_run_trials_single_job(self) -> None:
        """Run single trial job."""
        queue = TrialExecutionQueue(max_workers=1, timeout_seconds=5.0)

        job = TrialJob(job_id="job-1", task_id="task-1", task_input={"prompt": "test"})

        async def mock_execute(j: TrialJob) -> tuple[EvalTranscript, EvalOutcome]:
            return EvalTranscript(duration_seconds=0.5), EvalOutcome.success()

        results = await queue.run_trials([job], mock_execute)

        assert len(results) == 1
        assert isinstance(results[0], TrialJobResult)
        assert results[0].job_id == "job-1"
        assert results[0].outcome.status == EvalStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_trials_with_exception(self) -> None:
        """Run trials handles exceptions."""
        queue = TrialExecutionQueue(max_workers=1, timeout_seconds=5.0)

        job = TrialJob(job_id="job-1", task_id="task-1", task_input={})

        async def mock_execute(j: TrialJob) -> tuple[EvalTranscript, EvalOutcome]:
            raise RuntimeError("Test error")

        results = await queue.run_trials([job], mock_execute)

        assert len(results) == 1
        assert isinstance(results[0], Exception)


class TestGlobalQueue:
    """Tests for global queue management."""

    def test_register_and_get_queue(self) -> None:
        """Register and retrieve global queue."""
        reset_llm_queue()

        custom_queue = LLMRequestQueue(max_workers=16, timeout_seconds=120.0)
        register_llm_queue(custom_queue)

        retrieved = get_llm_queue_sync()
        assert retrieved is not None
        assert retrieved.max_workers == 16

        reset_llm_queue()

    def test_get_llm_queue_creates_default(self) -> None:
        """Get queue creates default if not registered."""
        reset_llm_queue()

        queue = get_llm_queue(max_workers=8, timeout_seconds=60.0)
        assert queue.max_workers == 8

        reset_llm_queue()

    def test_reset_llm_queue(self) -> None:
        """Reset clears global queue."""
        custom_queue = LLMRequestQueue(max_workers=16, timeout_seconds=120.0)
        register_llm_queue(custom_queue)

        assert get_llm_queue_sync() is not None

        reset_llm_queue()

        assert get_llm_queue_sync() is None


class TestEventLoopChangeResilience:
    """Tests for async primitive resilience across event loop changes."""

    @pytest.mark.asyncio
    async def test_llm_request_queue_survives_loop_change(self) -> None:
        """Queue works after event loop changes (singleton reuse across threads)."""
        queue = LLMRequestQueue(max_workers=1, timeout_seconds=5.0)

        request = LLMRequest(
            request_id="test-loop-1",
            messages=[],
            tools=None,
            options=None,
        )

        async def mock_execute(req: LLMRequest) -> dict[str, str]:
            return {"id": req.request_id}

        result1 = await queue.execute(request, mock_execute)
        assert result1.request_id == "test-loop-1"

        loop1_id = id(asyncio.get_running_loop())

        second_loop_result: dict[str, LLMResponse | Exception] = {}

        def run_in_new_thread() -> None:
            async def run_in_new_loop() -> None:
                loop2_id = id(asyncio.get_running_loop())
                assert loop2_id != loop1_id

                request2 = LLMRequest(
                    request_id="test-loop-2",
                    messages=[],
                    tools=None,
                    options=None,
                )

                try:
                    result2 = await queue.execute(request2, mock_execute)
                    second_loop_result["result"] = result2
                except Exception as e:
                    second_loop_result["error"] = e

            asyncio.run(run_in_new_loop())

        thread = threading.Thread(target=run_in_new_thread)
        thread.start()
        thread.join(timeout=10.0)

        assert "error" not in second_loop_result, f"Got error: {second_loop_result.get('error')}"
        assert "result" in second_loop_result
        assert second_loop_result["result"].request_id == "test-loop-2"

    @pytest.mark.asyncio
    async def test_llm_request_queue_survives_multiple_loop_changes(self) -> None:
        """Queue works after multiple event loop changes."""
        queue = LLMRequestQueue(max_workers=2, timeout_seconds=5.0)

        async def mock_execute(req: LLMRequest) -> dict[str, str]:
            await asyncio.sleep(0.01)
            return {"id": req.request_id}

        for iteration in range(3):
            result_container: dict[str, LLMResponse | Exception] = {}

            def run_in_thread(i: int = iteration) -> None:
                async def run() -> None:
                    request = LLMRequest(
                        request_id=f"test-iter-{i}",
                        messages=[],
                        tools=None,
                        options=None,
                    )
                    try:
                        result = await queue.execute(request, mock_execute)
                        result_container["result"] = result
                    except Exception as e:
                        result_container["error"] = e

                asyncio.run(run())

            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join(timeout=10.0)

            assert "error" not in result_container, (
                f"Iteration {iteration} failed: {result_container.get('error')}"
            )
            assert result_container["result"].request_id == f"test-iter-{iteration}"
