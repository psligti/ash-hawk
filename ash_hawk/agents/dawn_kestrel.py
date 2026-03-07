"""Dawn-Kestrel agent runner implementation.

This module provides integration between ash-hawk and the dawn-kestrel
framework for agent execution with policy enforcement.
"""

from __future__ import annotations

import inspect
import json
import asyncio
import re
import time
from decimal import Decimal
from pathlib import Path
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
        allowed_tools_override: list[str] | None = None,
        denied_tools_override: list[str] | None = None,
    ) -> ToolRegistry:
        try:
            from dawn_kestrel.tools.registry import ToolRegistry as DefaultToolRegistry
        except Exception:
            from dawn_kestrel.tools.framework import ToolRegistry as DefaultToolRegistry
        from dawn_kestrel.tools.permission_filter import ToolPermissionFilter

        if base_registry is None:
            base_registry = DefaultToolRegistry()
        if not hasattr(base_registry, "tool_metadata"):
            setattr(base_registry, "tool_metadata", {})

        policy = policy_enforcer.policy
        allowed_tools = list(policy.allowed_tools) if policy.allowed_tools else []
        denied_tools = list(policy.denied_tools) if policy.denied_tools else []
        if allowed_tools_override is not None:
            allowed_tools = list(allowed_tools_override)
        if denied_tools_override is not None:
            denied_tools = list(denied_tools_override)

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
        if filtered_registry is None:
            filtered_registry = DefaultToolRegistry()
        if not hasattr(filtered_registry, "tool_metadata"):
            setattr(filtered_registry, "tool_metadata", {})
        return filtered_registry

    def _build_tool_definitions(self, tools: dict[str, Any]) -> list[dict[str, Any]]:
        definitions: list[dict[str, Any]] = []
        for tool in tools.values():
            tool_id = getattr(tool, "id", None)
            if not isinstance(tool_id, str) or not tool_id.strip():
                continue
            tool_description = getattr(tool, "description", "")
            parameters = tool.parameters() if hasattr(tool, "parameters") else {}
            if not isinstance(parameters, dict):
                parameters = {}
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_id,
                        "description": tool_description,
                        "parameters": parameters,
                    },
                }
            )
        return definitions

    def _extract_text_tool_calls(self, text: str) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        alias_map = {
            "read_file": "read",
            "write_file": "write",
            "edit_file": "edit",
            "search_code": "grep",
            "run_bash": "bash",
        }

        pattern = re.compile(r"([a-zA-Z_][\w/]*)\(([^)]*)\)")
        arg_pattern = re.compile(r"([a-zA-Z_][\w]*)\s*=\s*([\"'])(.*?)\2")

        xml_call_pattern = re.compile(r"<tool_call>\s*([a-zA-Z_][\w]*)>?", re.IGNORECASE)
        xml_path_pattern = re.compile(r"<path>(.*?)</path>", re.IGNORECASE | re.DOTALL)
        xml_match = xml_call_pattern.search(text)
        if xml_match:
            raw_name = xml_match.group(1).strip()
            tool_name = alias_map.get(raw_name, raw_name)
            tool_input: dict[str, Any] = {}
            path_match = xml_path_pattern.search(text)
            if path_match:
                tool_input["filePath"] = path_match.group(1).strip()
            if tool_name:
                calls.append({"tool": tool_name, "input": tool_input})

        for match in pattern.finditer(text):
            raw_name = match.group(1).strip()
            raw_name = raw_name.split("/")[-1]
            tool_name = alias_map.get(raw_name, raw_name)
            args_text = match.group(2).replace('\\"', '"').replace("\\'", "'")
            tool_input: dict[str, Any] = {}
            for arg_match in arg_pattern.finditer(args_text):
                key = arg_match.group(1)
                value = arg_match.group(3)
                tool_input[key] = value

            if tool_name == "read" and "path" in tool_input and "filePath" not in tool_input:
                tool_input["filePath"] = tool_input.pop("path")
            if tool_name == "write" and "path" in tool_input and "filePath" not in tool_input:
                tool_input["filePath"] = tool_input.pop("path")
            if tool_name == "bash" and "cmd" in tool_input and "command" not in tool_input:
                tool_input["command"] = tool_input.pop("cmd")

            if tool_name:
                calls.append({"tool": tool_name, "input": tool_input})

        return calls

    async def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        filtered_registry: ToolRegistry,
        config: dict[str, Any],
        trial_id: str,
    ) -> list[dict[str, Any]]:
        from dawn_kestrel.tools.framework import ToolContext

        executed: list[dict[str, Any]] = []
        base_dir_value = config.get("workdir") or "."
        base_dir = Path(base_dir_value) if isinstance(base_dir_value, str) else Path(".")

        for idx, call in enumerate(tool_calls):
            tool_name = call.get("tool") or call.get("name")
            if not isinstance(tool_name, str) or not tool_name.strip():
                continue
            tool_name = tool_name.strip().lower()

            raw_input = call.get("input")
            if raw_input is None:
                raw_input = call.get("arguments")
            tool_input = raw_input if isinstance(raw_input, dict) else {}

            tool = filtered_registry.get(tool_name)
            if tool is None:
                executed.append(
                    {
                        "tool": tool_name,
                        "input": tool_input,
                        "error": f"Tool not available: {tool_name}",
                    }
                )
                continue

            call_id = f"{trial_id}_tool_{idx}"
            ctx = ToolContext(
                session_id=trial_id,
                message_id=trial_id,
                agent="ash_hawk_eval",
                abort=asyncio.Event(),
                messages=[],
                call_id=call_id,
                base_dir=base_dir,
            )

            try:
                result = await tool.execute(tool_input, ctx)
                executed.append(
                    {
                        "tool": tool_name,
                        "input": tool_input,
                        "output": result.output,
                        "title": result.title,
                        "metadata": result.metadata,
                    }
                )
            except Exception as exc:
                executed.append(
                    {
                        "tool": tool_name,
                        "input": tool_input,
                        "error": str(exc),
                    }
                )

        return executed

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
            return response.get("cost", 0.0)  # type: ignore[no-any-return]
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
            return response.text  # type: ignore[no-any-return]
        if hasattr(response, "content"):
            return response.content  # type: ignore[no-any-return]
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
            task_input = task.input
            override_allowed_tools: list[str] | None = None
            override_denied_tools: list[str] | None = None
            if isinstance(task_input, dict):
                policy_snapshot = task_input.get("policy_snapshot")
                if isinstance(policy_snapshot, dict):
                    raw_allowed = policy_snapshot.get("allowed_tools")
                    if isinstance(raw_allowed, list):
                        override_allowed_tools = [
                            tool for tool in raw_allowed if isinstance(tool, str)
                        ]
                    raw_denied = policy_snapshot.get("denied_tools")
                    if isinstance(raw_denied, list):
                        override_denied_tools = [
                            tool for tool in raw_denied if isinstance(tool, str)
                        ]

            filtered_registry = self._create_filtered_registry(
                policy_enforcer,
                allowed_tools_override=override_allowed_tools,
                denied_tools_override=override_denied_tools,
            )

            if isinstance(task_input, dict):
                prompt = task_input.get("prompt", task_input.get("message", str(task_input)))
            else:
                prompt = str(task_input)

            available_tools = await filtered_registry.get_all()
            tool_definitions = self._build_tool_definitions(available_tools)

            from dawn_kestrel.llm.client import LLMRequestOptions

            options = LLMRequestOptions(
                temperature=config.get("temperature", self._kwargs.get("temperature")),
                max_tokens=config.get("max_tokens", self._kwargs.get("max_tokens")),
            )

            conversation: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
            all_tool_calls: list[dict[str, Any]] = []
            trial_id = str(config.get("trial_id") or f"trial-{int(time.time() * 1000)}")
            max_iterations = int(config.get("max_tool_iterations", 3) or 3)
            response = None

            for _ in range(max_iterations):
                response = await client.complete(
                    messages=conversation,
                    tools=tool_definitions if tool_definitions else None,
                    options=options,
                )

                model_text = self._extract_agent_response(response)
                if isinstance(model_text, str) and model_text.strip():
                    conversation.append({"role": "assistant", "content": model_text})

                tool_calls = self._extract_tool_calls(response)
                if not tool_calls and isinstance(model_text, str) and model_text.strip():
                    tool_calls = self._extract_text_tool_calls(model_text)
                if not tool_calls:
                    break

                executed = await self._execute_tool_calls(
                    tool_calls=tool_calls,
                    filtered_registry=filtered_registry,
                    config=config,
                    trial_id=trial_id,
                )

                for item in executed:
                    normalized = {
                        "tool": item.get("tool"),
                        "input": item.get("input", {}),
                    }
                    if "output" in item:
                        normalized["output"] = item.get("output")
                    if "error" in item:
                        normalized["error"] = item.get("error")
                    all_tool_calls.append(normalized)

                conversation.append(
                    {
                        "role": "user",
                        "content": "Tool execution results:\n"
                        + json.dumps(executed, ensure_ascii=True, default=str),
                    }
                )

            if response is None:
                raise RuntimeError("LLM completion failed to produce a response")

            duration = time.time() - start_time

            transcript = EvalTranscript(
                messages=conversation,
                tool_calls=all_tool_calls,
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
