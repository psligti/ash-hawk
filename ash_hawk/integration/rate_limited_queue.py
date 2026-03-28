"""Rate-limited queue integration for ash-hawk.

Uses dawn-kestrel's LocalRateLimitTracker to add per-provider rate limiting
to ash-hawk's LLMRequestQueue interface.

Usage:
    from ash_hawk.integration.rate_limited_queue import (
        RateLimitedLLMQueue,
        setup_rate_limiting,
    )
    from ash_hawk.execution import register_llm_queue, reset_llm_queue

    # Option 1: Manual setup
    reset_llm_queue()
    queue = RateLimitedLLMQueue(providers=["anthropic", "openai"])
    register_llm_queue(queue)
    # Now DawnKestrelAgentRunner uses rate-limited queue

    # Option 2: One-liner
    setup_rate_limiting(providers=["anthropic", "openai"])
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)


class RateLimitedLLMQueue:
    """LLMRequestQueue-compatible adapter with per-provider rate limiting.

    Bridges ash-hawk's LLMRequestQueue.execute() interface with dawn-kestrel's
    LocalRateLimitTracker for rate limiting. Uses semaphore for concurrency
    and token bucket for rate limiting.

    Example:
        from ash_hawk.execution import register_llm_queue, reset_llm_queue

        reset_llm_queue()
        queue = RateLimitedLLMQueue(
            providers=["anthropic", "openai"],
            max_concurrent=10,
        )
        register_llm_queue(queue)

        runner = DawnKestrelAgentRunner(provider="anthropic", model="claude-sonnet-4")
        transcript, outcome = await runner.run(task, policy, config)
    """

    def __init__(
        self,
        providers: list[str] | None = None,
        max_concurrent: int = 10,
        timeout_seconds: float = 300.0,
    ) -> None:
        self._providers = [self._normalize_provider(p) for p in (providers or ["anthropic"])]
        self._max_concurrent = max_concurrent
        self._timeout = timeout_seconds
        self._trackers: dict[str, Any] = {}
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None
        self._provider_mapping: dict[str, str] = {}
        self._stats = {
            "total_requests": 0,
            "total_wait_time": 0.0,
            "rate_limited_waits": 0,
            "concurrent_waits": 0,
        }

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        """Normalize provider ID to match PROVIDER_LIMITS keys."""
        p = provider.lower().replace("-", "_")
        if p.startswith("providerid_"):
            p = p[11:]
        p = p.replace("z_ai", "zai")
        return p

    def _get_semaphore(self) -> asyncio.Semaphore:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._semaphore is None or self._semaphore_loop != current_loop:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
            self._semaphore_loop = current_loop
        assert self._semaphore is not None
        return self._semaphore

    def _get_tracker(self, provider: str) -> Any:
        if provider not in self._trackers:
            from dawn_kestrel.llm.provider_limits import (  # pyright: ignore[reportMissingTypeStubs]
                LocalRateLimitTracker,
            )

            self._trackers[provider] = LocalRateLimitTracker()
            logger.info(f"RateLimitedLLMQueue: created tracker for {provider}")
        return self._trackers[provider]

    async def execute(
        self,
        request: Any,
        execute: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        """Execute with rate limiting and concurrency control.

        Args:
            request: LLMRequest with request_id, messages, tools, options
            execute: Async function to execute the LLM call

        Returns:
            LLMResponse with response and timing info
        """
        from ash_hawk.execution.queue import LLMResponse  # pyright: ignore[reportMissingTypeStubs]

        start_time = time.time()
        provider = self._normalize_provider(
            self._provider_mapping.get(request.request_id, self._providers[0])
        )
        tracker = self._get_tracker(provider)

        rate_wait = 0.0
        check_result = await tracker.check_allowed(provider, cost=1)
        if check_result.is_ok():
            allowed, wait_seconds = check_result.unwrap()
            if not allowed and wait_seconds > 0:
                rate_wait = wait_seconds
                self._stats["rate_limited_waits"] += 1
                logger.debug(f"Rate limit wait for {provider}: {wait_seconds:.2f}s")
                await asyncio.sleep(wait_seconds)

        async with self._get_semaphore():
            if rate_wait == 0.0:
                self._stats["concurrent_waits"] += 1
            response = await execute(request)

        wait_time = time.time() - start_time
        self._stats["total_requests"] += 1
        self._stats["total_wait_time"] += wait_time

        return LLMResponse(
            request_id=request.request_id,
            response=response,
            wait_time_seconds=wait_time,
            token_usage=self._extract_token_usage(response),
            cost_usd=self._extract_cost(response),
        )

    def register_provider(self, request_id: str, provider: str) -> None:
        self._provider_mapping[request_id] = provider

    async def get_stats(self) -> dict[str, Any]:
        return self._stats.copy()

    def _extract_token_usage(self, response: Any) -> Any:
        from ash_hawk.execution.queue import TokenUsage  # pyright: ignore[reportMissingTypeStubs]

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


def setup_rate_limiting(
    providers: list[str] | None = None,
    max_concurrent: int = 10,
) -> RateLimitedLLMQueue:
    """One-liner to set up rate limiting for ash-hawk.

    Args:
        providers: List of provider IDs (default: ["anthropic"])
        max_concurrent: Max concurrent LLM calls

    Returns:
        The created queue (also registered globally)

    Example:
        from ash_hawk.integration.rate_limited_queue import setup_rate_limiting

        setup_rate_limiting(providers=["anthropic", "openai"], max_concurrent=10)
        # Now run evals normally - rate limiting is active
    """
    from ash_hawk.execution import register_llm_queue, reset_llm_queue  # pyright: ignore[reportMissingTypeStubs]

    reset_llm_queue()
    queue = RateLimitedLLMQueue(providers=providers, max_concurrent=max_concurrent)
    register_llm_queue(queue)  # pyright: ignore[reportArgumentType]
    logger.info(f"Rate limiting enabled for providers: {providers or ['anthropic']}")
    return queue


__all__ = [
    "RateLimitedLLMQueue",
    "setup_rate_limiting",
]
