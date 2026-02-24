"""Dawn-Kestrel agent runner implementation.

This module provides integration between ash-hawk and the dawn-kestrel
framework for agent execution with policy enforcement.
"""

from __future__ import annotations

import inspect
import time
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from ash_hawk.policy import PolicyEnforcer
from ash_hawk.types import (
    EvalOutcome,
    EvalTask,
    EvalTranscript,
    FailureMode,
    TokenUsage,
)

if TYPE_CHECKING:
    from dawn_kestrel.llm.client import LLMClient
    from dawn_kestrel.tools.framework import ToolRegistry
    from dawn_kestrel.tools.permission_filter import ToolPermissionFilter


class DawnKestrelAgentRunner:
    """Agent runner that uses the dawn-kestrel framework.

    This runner integrates dawn-kestrel's LLMClient with ash-hawk's
    policy enforcement system, filtering tool access based on the
    ToolSurfacePolicy configuration.

    Attributes:
        _provider: The LLM provider identifier.
        _model: The model identifier.
        _kwargs: Additional configuration options.
        _client: The dawn-kestrel LLMClient instance.
    """

    def __init__(self, provider: str, model: str, **kwargs: Any) -> None:
        """Initialize the dawn-kestrel agent runner.

        Args:
            provider: The LLM provider (e.g., 'anthropic', 'openai', 'zai').
            model: The model identifier (e.g., 'claude-3-5-sonnet-20241022').
            **kwargs: Additional configuration options passed to the client.
        """
        self._provider = provider
        self._model = model
        self._kwargs = kwargs
        self._client: LLMClient | None = None

    def _get_client(self) -> LLMClient:
        if self._client is None:
            from dawn_kestrel.core.settings import get_settings
            from dawn_kestrel.llm.client import LLMClient

            settings = get_settings()
            api_key_secret = settings.get_api_key_for_provider(self._provider)
            api_key = api_key_secret.get_secret_value() if api_key_secret else None

            self._client = LLMClient(
                provider_id=self._provider,
                model=self._model,
                api_key=api_key,
            )
        return self._client

    def _create_filtered_registry(
        self,
        policy_enforcer: PolicyEnforcer,
        base_registry: ToolRegistry | None = None,
    ) -> ToolRegistry:
        from dawn_kestrel.tools.framework import ToolRegistry
        from dawn_kestrel.tools.permission_filter import ToolPermissionFilter

        if base_registry is None:
            base_registry = ToolRegistry()

        policy = policy_enforcer.policy
        allowed_tools = list(policy.allowed_tools) if policy.allowed_tools else []
        denied_tools = list(policy.denied_tools) if policy.denied_tools else []

        init_params = set(inspect.signature(ToolPermissionFilter.__init__).parameters.keys())
        filter_kwargs: dict[str, Any] = {"tool_registry": base_registry}

        permissions: list[dict[str, Any]] = []
        for tool in allowed_tools:
            permissions.append({"permission": tool, "pattern": "*", "action": "allow"})
        for tool in denied_tools:
            permissions.append({"permission": tool, "pattern": "*", "action": "deny"})

        if "allowed_tools" in init_params:
            filter_kwargs["allowed_tools"] = allowed_tools
        elif "allowlist" in init_params:
            filter_kwargs["allowlist"] = allowed_tools
        elif "permissions" in init_params:
            filter_kwargs["permissions"] = permissions

        if "denied_tools" in init_params:
            filter_kwargs["denied_tools"] = denied_tools
        elif "denylist" in init_params:
            filter_kwargs["denylist"] = denied_tools
        elif "permissions" in init_params and "permissions" not in filter_kwargs:
            filter_kwargs["permissions"] = permissions

        permission_filter = ToolPermissionFilter(**filter_kwargs)

        filtered_registry = permission_filter.get_filtered_registry()
        return filtered_registry if filtered_registry else ToolRegistry()

    def _extract_token_usage(self, response: Any) -> TokenUsage:
        usage = TokenUsage()

        if hasattr(response, "usage") and response.usage:
            usage_data = response.usage
            usage.input = getattr(usage_data, "input", 0) or 0
            usage.output = getattr(usage_data, "output", 0) or 0
            usage.reasoning = getattr(usage_data, "reasoning", 0) or 0
            usage.cache_read = getattr(usage_data, "cache_read", 0) or 0
            usage.cache_write = getattr(usage_data, "cache_write", 0) or 0

        elif isinstance(response, dict):
            usage_data = response.get("usage", {})
            if isinstance(usage_data, dict):
                usage.input = usage_data.get("input", 0)
                usage.output = usage_data.get("output", 0)
                usage.reasoning = usage_data.get("reasoning", 0)
                usage.cache_read = usage_data.get("cache_read", 0)
                usage.cache_write = usage_data.get("cache_write", 0)

        return usage

    def _extract_cost(self, response: Any) -> float:
        if hasattr(response, "cost"):
            cost = response.cost
            if isinstance(cost, Decimal):
                return float(cost)
            return cost or 0.0
        if isinstance(response, dict):
            return response.get("cost", 0.0)
        return 0.0

    def _extract_messages(self, response: Any) -> list[dict[str, Any]]:
        messages = []

        if hasattr(response, "messages") and response.messages:
            for msg in response.messages:
                if hasattr(msg, "model_dump"):
                    messages.append(msg.model_dump())
                elif isinstance(msg, dict):
                    messages.append(msg)
                else:
                    messages.append({"content": str(msg)})
        elif isinstance(response, dict):
            for msg in response.get("messages", []):
                if isinstance(msg, dict):
                    messages.append(msg)
                else:
                    messages.append({"content": str(msg)})

        return messages

    def _extract_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        tool_calls = []

        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                if hasattr(tc, "model_dump"):
                    tool_calls.append(tc.model_dump())
                elif isinstance(tc, dict):
                    tool_calls.append(tc)
                else:
                    tool_calls.append({"name": str(tc)})
        elif isinstance(response, dict):
            for tc in response.get("tool_calls", []):
                if isinstance(tc, dict):
                    tool_calls.append(tc)
                else:
                    tool_calls.append({"name": str(tc)})

        return tool_calls

    def _extract_agent_response(self, response: Any) -> str | dict[str, Any] | None:
        if hasattr(response, "text") and response.text:
            return response.text
        if hasattr(response, "content"):
            return response.content
        if isinstance(response, dict):
            return response.get("text") or response.get("content")
        return None

    async def run(
        self,
        task: EvalTask,
        policy_enforcer: PolicyEnforcer,
        config: dict[str, Any],
    ) -> tuple[EvalTranscript, EvalOutcome]:
        start_time = time.time()

        try:
            client = self._get_client()
            filtered_registry = self._create_filtered_registry(policy_enforcer)

            task_input = task.input
            if isinstance(task_input, dict):
                prompt = task_input.get("prompt", task_input.get("message", str(task_input)))
            else:
                prompt = str(task_input)

            tools = (
                list((await filtered_registry.get_all()).values())
                if filtered_registry.tools
                else []
            )

            from dawn_kestrel.llm.client import LLMRequestOptions

            options = LLMRequestOptions(
                temperature=config.get("temperature", self._kwargs.get("temperature")),
                max_tokens=config.get("max_tokens", self._kwargs.get("max_tokens")),
            )

            response = await client.complete(
                messages=[{"role": "user", "content": prompt}],
                tools=tools if tools else None,
                options=options,
            )

            duration = time.time() - start_time

            transcript = EvalTranscript(
                messages=[{"role": "user", "content": prompt}] + self._extract_messages(response),
                tool_calls=self._extract_tool_calls(response),
                token_usage=self._extract_token_usage(response),
                cost_usd=self._extract_cost(response),
                duration_seconds=duration,
                agent_response=self._extract_agent_response(response),
            )

            outcome = EvalOutcome.success()

            return transcript, outcome

        except ImportError as e:
            duration = time.time() - start_time
            transcript = EvalTranscript(
                error_trace=f"dawn-kestrel not installed: {e}",
                duration_seconds=duration,
            )
            outcome = EvalOutcome.failure(
                FailureMode.AGENT_ERROR,
                f"dawn-kestrel not installed: {e}",
            )
            return transcript, outcome

        except Exception as e:
            duration = time.time() - start_time
            error_trace = str(e)
            if hasattr(e, "__traceback__") and e.__traceback__:
                import traceback

                error_trace = "".join(traceback.format_exception(type(e), e, e.__traceback__))

            transcript = EvalTranscript(
                error_trace=error_trace,
                duration_seconds=duration,
            )
            outcome = EvalOutcome.failure(
                FailureMode.AGENT_ERROR,
                str(e),
            )
            return transcript, outcome
