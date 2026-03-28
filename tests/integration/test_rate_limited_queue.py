"""Tests for rate-limited queue integration."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio

import pytest

from ash_hawk.execution.queue import (
    LLMRequest,
    LLMResponse,
    get_llm_queue_sync,
    reset_llm_queue,
)


class TestRateLimitedLLMQueue:
    """Tests for RateLimitedLLMQueue adapter."""

    def test_init_default_values(self) -> None:
        """Queue initializes with default values."""
        from ash_hawk.integration.rate_limited_queue import RateLimitedLLMQueue

        queue = RateLimitedLLMQueue()
        assert queue._providers == ["anthropic"]
        assert queue._max_concurrent == 10
        assert queue._timeout == 300.0

    def test_init_custom_values(self) -> None:
        """Queue initializes with custom values."""
        from ash_hawk.integration.rate_limited_queue import RateLimitedLLMQueue

        queue = RateLimitedLLMQueue(
            providers=["anthropic", "openai"],
            max_concurrent=5,
            timeout_seconds=120.0,
        )
        assert queue._providers == ["anthropic", "openai"]
        assert queue._max_concurrent == 5
        assert queue._timeout == 120.0

    @pytest.mark.asyncio
    async def test_execute_single_request(self) -> None:
        """Execute a single request with rate limiting."""
        from ash_hawk.integration.rate_limited_queue import RateLimitedLLMQueue

        queue = RateLimitedLLMQueue(providers=["anthropic"], max_concurrent=1)

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
        """Execute multiple concurrent requests with rate limiting."""
        from ash_hawk.integration.rate_limited_queue import RateLimitedLLMQueue

        queue = RateLimitedLLMQueue(providers=["anthropic"], max_concurrent=2)

        async def mock_execute(req: LLMRequest) -> dict[str, str]:
            await asyncio.sleep(0.05)
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
        from ash_hawk.integration.rate_limited_queue import RateLimitedLLMQueue

        queue = RateLimitedLLMQueue()

        stats = await queue.get_stats()
        assert stats["total_requests"] == 0
        assert stats["total_wait_time"] == 0.0

    @pytest.mark.asyncio
    async def test_register_provider_mapping(self) -> None:
        """Register provider mapping for requests."""
        from ash_hawk.integration.rate_limited_queue import RateLimitedLLMQueue

        queue = RateLimitedLLMQueue(providers=["anthropic", "openai"])

        queue.register_provider("req-1", "openai")
        queue.register_provider("req-2", "anthropic")

        assert queue._provider_mapping["req-1"] == "openai"
        assert queue._provider_mapping["req-2"] == "anthropic"


class TestSetupRateLimiting:
    """Tests for setup_rate_limiting helper."""

    def setup_method(self) -> None:
        reset_llm_queue()

    def teardown_method(self) -> None:
        reset_llm_queue()

    def test_setup_registers_queue(self) -> None:
        """setup_rate_limiting registers queue globally."""
        from ash_hawk.integration.rate_limited_queue import setup_rate_limiting

        queue = setup_rate_limiting(providers=["anthropic"], max_concurrent=5)

        assert get_llm_queue_sync() is queue
        assert queue._providers == ["anthropic"]
        assert queue._max_concurrent == 5

    def test_setup_default_providers(self) -> None:
        """setup_rate_limiting uses default providers."""
        from ash_hawk.integration.rate_limited_queue import setup_rate_limiting

        queue = setup_rate_limiting()

        assert queue._providers == ["anthropic"]
