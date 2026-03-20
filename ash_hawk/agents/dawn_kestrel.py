"""Dawn-Kestrel agent runner implementation.

This module provides integration between ash-hawk and the dawn-kestrel
framework for agent execution with policy enforcement.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from ash_hawk.execution.queue import LLMRequest, get_llm_queue_sync
from ash_hawk.policy import PolicyEnforcer
from ash_hawk.types import (
    EvalOutcome,
    EvalTask,
    EvalTranscript,
    FailureMode,
    TokenUsage,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _McpServerConfig:
    name: str
    command: str
    args: list[str]
    env: dict[str, str]
    cwd: str | None


def _sanitize_mcp_tool_id(raw_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_id).strip("_")
    if cleaned == "":
        cleaned = "mcp_tool"
    if not re.match(r"[a-zA-Z_]", cleaned):
        cleaned = f"mcp_{cleaned}"
    return cleaned.lower()


def _mcp_result_to_text(result: dict[str, Any]) -> str:
    content = result.get("content")
    if isinstance(content, list):
        text_chunks: list[str] = []
        for item_dict in cast(list[dict[str, Any]], content):
            if not isinstance(item_dict, dict):
                continue
            item_type = item_dict.get("type")
            text_value = item_dict.get("text")
            if item_type == "text" and isinstance(text_value, str):
                text_chunks.append(text_value)
            elif item_type == "json" and "json" in item_dict:
                text_chunks.append(json.dumps(item_dict["json"], ensure_ascii=True, default=str))
            elif isinstance(text_value, str):
                text_chunks.append(text_value)
        if text_chunks:
            return "\n".join(text_chunks)

    if "structuredContent" in result:
        return json.dumps(result["structuredContent"], ensure_ascii=True, default=str)

    return json.dumps(result, ensure_ascii=True, default=str)


class _McpStdioClient:
    def __init__(self, config: _McpServerConfig) -> None:
        self._config = config
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._request_lock = asyncio.Lock()

    async def start(self) -> None:
        env = os.environ.copy()
        env.update(self._config.env)
        self._process = await asyncio.create_subprocess_exec(
            self._config.command,
            *self._config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._config.cwd,
            env=env,
        )

        await self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "ash-hawk",
                    "version": "0.1.0",
                },
            },
        )
        await self.notify("notifications/initialized", {})

    async def close(self) -> None:
        if self._process is None:
            return
        process = self._process
        self._process = None

        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except TimeoutError:
                process.kill()
                await process.wait()

    async def list_tools(self) -> list[dict[str, Any]]:
        response = await self.request("tools/list", {})
        tools = response.get("tools") if isinstance(response, dict) else None
        if isinstance(tools, list):
            return [tool for tool in tools if isinstance(tool, dict)]
        return []

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        response = await self.request(
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments,
            },
        )
        if isinstance(response, dict):
            return response
        return {"content": [{"type": "text", "text": str(response)}]}

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        await self._write_message(payload)

    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        async with self._request_lock:
            self._request_id += 1
            request_id = self._request_id
            payload: dict[str, Any] = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
            }
            if params is not None:
                payload["params"] = params
            await self._write_message(payload)

            while True:
                message = await self._read_message()
                if not isinstance(message, dict):
                    continue
                if message.get("id") != request_id:
                    continue
                if "error" in message:
                    raise RuntimeError(
                        f"MCP request failed for server '{self._config.name}': {message['error']}"
                    )
                return message.get("result", {})

    async def _write_message(self, payload: dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError(f"MCP server '{self._config.name}' is not running")
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self._process.stdin.write(header + body)
        await self._process.stdin.drain()

    async def _read_message(self) -> dict[str, Any]:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError(f"MCP server '{self._config.name}' is not running")

        content_length = 0
        while True:
            line = await self._process.stdout.readline()
            if line == b"":
                raise RuntimeError(f"MCP server '{self._config.name}' closed stdout unexpectedly")
            stripped = line.strip()
            if stripped == b"":
                break
            lower_line = stripped.lower()
            if lower_line.startswith(b"content-length:"):
                _, value = stripped.split(b":", 1)
                content_length = int(value.strip())

        if content_length <= 0:
            raise RuntimeError(f"MCP server '{self._config.name}' sent invalid content length")

        body = await self._process.stdout.readexactly(content_length)
        parsed = json.loads(body.decode("utf-8"))
        if isinstance(parsed, dict):
            return cast(dict[str, Any], parsed)
        raise RuntimeError(f"MCP server '{self._config.name}' sent invalid JSON-RPC message")


class _McpToolProxy:
    def __init__(
        self,
        *,
        tool_id: str,
        server_name: str,
        mcp_tool_name: str,
        description: str,
        input_schema: dict[str, Any],
        client: _McpStdioClient,
    ) -> None:
        self.id = tool_id
        self.description = description
        self._server_name = server_name
        self._mcp_tool_name = mcp_tool_name
        self._input_schema = input_schema
        self._client = client

    def parameters(self) -> dict[str, Any]:
        if isinstance(self._input_schema, dict) and self._input_schema:
            return self._input_schema
        return {"type": "object", "properties": {}, "additionalProperties": True}

    async def execute(self, args: dict[str, Any], ctx: Any) -> Any:
        del ctx
        tools_framework_module = importlib.import_module("dawn_kestrel.tools.framework")
        tool_result_type = getattr(tools_framework_module, "ToolResult")

        result = await self._client.call_tool(self._mcp_tool_name, args)
        return tool_result_type(
            title=f"{self.id} completed",
            output=_mcp_result_to_text(result),
            metadata={
                "mcp_server": self._server_name,
                "mcp_tool": self._mcp_tool_name,
            },
        )


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
        self._provider = provider
        self._model = model
        self._mcp_servers = self._parse_mcp_servers(kwargs.pop("mcp_servers", None))
        self._kwargs = kwargs
        self._client: Any | None = None
        self._lesson_injector: Any | None = None
        self._llm_queue: Any | None = None
        self._post_run_hook: Any | None = None

    def set_lesson_injector(self, injector: Any) -> None:
        self._lesson_injector = injector

    def get_lesson_injector(self) -> Any | None:
        return self._lesson_injector

    def set_post_run_hook(self, hook: Any) -> None:
        self._post_run_hook = hook

    def get_post_run_hook(self) -> Any | None:
        return self._post_run_hook

    def _parse_mcp_servers(self, raw_servers: Any) -> list[_McpServerConfig]:
        if raw_servers is None:
            return []
        if not isinstance(raw_servers, list):
            raise ValueError("mcp_servers must be a list")

        parsed: list[_McpServerConfig] = []
        for idx, item in enumerate(raw_servers):
            if not isinstance(item, dict):
                raise ValueError(f"mcp_servers[{idx}] must be a mapping")

            name = item.get("name")
            command = item.get("command")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"mcp_servers[{idx}].name must be a non-empty string")
            if not isinstance(command, str) or not command.strip():
                raise ValueError(f"mcp_servers[{idx}].command must be a non-empty string")

            raw_args = item.get("args", [])
            if not isinstance(raw_args, list) or not all(isinstance(arg, str) for arg in raw_args):
                raise ValueError(f"mcp_servers[{idx}].args must be a list of strings")

            raw_env = item.get("env", {})
            if not isinstance(raw_env, dict):
                raise ValueError(f"mcp_servers[{idx}].env must be a mapping of strings")
            env: dict[str, str] = {}
            for key, value in raw_env.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise ValueError(f"mcp_servers[{idx}].env must contain only string keys/values")
                env[key] = value

            raw_cwd = item.get("cwd")
            if raw_cwd is not None and not isinstance(raw_cwd, str):
                raise ValueError(f"mcp_servers[{idx}].cwd must be a string when provided")

            parsed.append(
                _McpServerConfig(
                    name=name.strip(),
                    command=command.strip(),
                    args=[arg for arg in raw_args if arg.strip() != ""],
                    env=env,
                    cwd=raw_cwd,
                )
            )

        return parsed

    def _create_base_registry(self) -> Any:
        try:
            tools_registry_module = importlib.import_module("dawn_kestrel.tools.registry")
            default_tool_registry = getattr(tools_registry_module, "ToolRegistry")
        except Exception:
            tools_framework_module = importlib.import_module("dawn_kestrel.tools.framework")
            default_tool_registry = getattr(tools_framework_module, "ToolRegistry")

        registry = default_tool_registry()
        if not hasattr(registry, "tool_metadata"):
            setattr(registry, "tool_metadata", {})
        return registry

    async def _register_mcp_tools(self, base_registry: Any) -> list[_McpStdioClient]:
        clients: list[_McpStdioClient] = []
        if not self._mcp_servers:
            return clients

        try:
            for server in self._mcp_servers:
                client = _McpStdioClient(server)
                await client.start()
                tool_specs = await client.list_tools()

                for tool_spec in tool_specs:
                    if not isinstance(tool_spec, dict):
                        continue
                    tool_name = tool_spec.get("name")
                    if not isinstance(tool_name, str) or not tool_name.strip():
                        continue

                    tool_id = _sanitize_mcp_tool_id(f"{server.name}_{tool_name}")
                    description_raw = tool_spec.get("description")
                    description = description_raw if isinstance(description_raw, str) else ""
                    input_schema_raw = tool_spec.get("inputSchema")
                    input_schema = input_schema_raw if isinstance(input_schema_raw, dict) else {}

                    proxy = _McpToolProxy(
                        tool_id=tool_id,
                        server_name=server.name,
                        mcp_tool_name=tool_name,
                        description=description,
                        input_schema=input_schema,
                        client=client,
                    )
                    await base_registry.register(
                        proxy,
                        tool_id,
                        metadata={
                            "mcp_server": server.name,
                            "mcp_tool": tool_name,
                        },
                    )

                clients.append(client)

            return clients
        except Exception:
            await self._close_mcp_clients(clients)
            raise

    async def _close_mcp_clients(self, clients: list[_McpStdioClient]) -> None:
        for client in clients:
            try:
                await client.close()
            except Exception:
                continue

    def _get_client(self) -> Any:
        if self._client is None:
            settings_module = importlib.import_module("dawn_kestrel.core.settings")
            llm_client_module = importlib.import_module("dawn_kestrel.llm.client")
            get_settings = getattr(settings_module, "get_settings")
            llm_client_type = getattr(llm_client_module, "LLMClient")

            settings = get_settings()
            api_key_secret = settings.get_api_key_for_provider(self._provider)
            api_key = api_key_secret.get_secret_value() if api_key_secret else None

            self._client = llm_client_type(
                provider_id=self._provider,
                model=self._model,
                api_key=api_key,
            )
        return self._client

    def _create_filtered_registry(
        self,
        policy_enforcer: PolicyEnforcer,
        base_registry: Any = None,
        allowed_tools_override: list[str] | None = None,
        denied_tools_override: list[str] | None = None,
    ) -> Any:
        try:
            tools_registry_module = importlib.import_module("dawn_kestrel.tools.registry")
            default_tool_registry = getattr(tools_registry_module, "ToolRegistry")
        except Exception:
            tools_framework_module = importlib.import_module("dawn_kestrel.tools.framework")
            default_tool_registry = getattr(tools_framework_module, "ToolRegistry")
        permission_filter_module = importlib.import_module("dawn_kestrel.tools.permission_filter")
        tool_permission_filter = getattr(permission_filter_module, "ToolPermissionFilter")

        if base_registry is None:
            base_registry = default_tool_registry()
        if not hasattr(base_registry, "tool_metadata"):
            setattr(base_registry, "tool_metadata", {})

        policy = policy_enforcer.policy
        allowed_tools = list(policy.allowed_tools) if policy.allowed_tools else []
        denied_tools = list(policy.denied_tools) if policy.denied_tools else []
        if allowed_tools_override is not None:
            allowed_tools = list(allowed_tools_override)
        if denied_tools_override is not None:
            denied_tools = list(denied_tools_override)

        init_params = set(inspect.signature(tool_permission_filter.__init__).parameters.keys())
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

        permission_filter = tool_permission_filter(**filter_kwargs)

        filtered_registry = permission_filter.get_filtered_registry()
        if filtered_registry is None:
            filtered_registry = default_tool_registry()
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

        pattern = re.compile(r"([a-zA-Z_][\w/-]*)\(([^)]*)\)")
        arg_pattern = re.compile(r"([a-zA-Z_][\w]*)\s*=\s*([\"'])(.*?)\2")

        xml_call_pattern = re.compile(r"<tool_call>\s*([a-zA-Z_][\w-]*)>?", re.IGNORECASE)
        xml_path_pattern = re.compile(r"<path>(.*?)</path>", re.IGNORECASE | re.DOTALL)
        xml_match = xml_call_pattern.search(text)
        if xml_match:
            raw_name = xml_match.group(1).strip()
            tool_name = alias_map.get(raw_name, raw_name)
            xml_tool_input: dict[str, Any] = {}
            path_match = xml_path_pattern.search(text)
            if path_match:
                xml_tool_input["filePath"] = path_match.group(1).strip()
            if tool_name:
                calls.append({"tool": tool_name, "input": xml_tool_input})

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
        filtered_registry: Any,
        config: dict[str, Any],
        trial_id: str,
    ) -> list[dict[str, Any]]:
        tools_framework_module = importlib.import_module("dawn_kestrel.tools.framework")
        tool_context_type = getattr(tools_framework_module, "ToolContext")

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
            ctx = tool_context_type(
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
        mcp_clients: list[_McpStdioClient] = []

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

            base_registry = self._create_base_registry()
            mcp_clients = await self._register_mcp_tools(base_registry)
            filtered_registry = self._create_filtered_registry(
                policy_enforcer,
                base_registry=base_registry,
                allowed_tools_override=override_allowed_tools,
                denied_tools_override=override_denied_tools,
            )

            if isinstance(task_input, dict):
                prompt = task_input.get("prompt", task_input.get("message", str(task_input)))
            else:
                prompt = str(task_input)

            agent_id = str(config.get("agent_name", "default"))
            if self._lesson_injector is not None:
                prompt = self._lesson_injector.inject_into_prompt(agent_id, prompt)

            available_tools = await filtered_registry.get_all()
            tool_definitions = self._build_tool_definitions(available_tools)

            llm_client_module = importlib.import_module("dawn_kestrel.llm.client")
            llm_request_options = getattr(llm_client_module, "LLMRequestOptions")

            options = llm_request_options(
                temperature=config.get("temperature", self._kwargs.get("temperature")),
                max_tokens=config.get("max_tokens", self._kwargs.get("max_tokens")),
            )

            conversation: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
            all_tool_calls: list[dict[str, Any]] = []
            trial_id = str(config.get("trial_id") or f"trial-{int(time.time() * 1000)}")
            max_iterations = int(config.get("max_tool_iterations", 3) or 3)
            response = None

            for iteration in range(max_iterations):
                llm_queue = get_llm_queue_sync()
                if llm_queue is not None:
                    request_id = f"{trial_id}_llm_{iteration}"
                    request = LLMRequest(
                        request_id=request_id,
                        messages=conversation,
                        tools=tool_definitions if tool_definitions else None,
                        options=options,
                    )

                    async def execute_llm(req: LLMRequest) -> Any:
                        return await client.complete(
                            messages=req.messages,
                            tools=req.tools,
                            options=req.options,
                        )

                    llm_response = await llm_queue.execute(request, execute_llm)
                    response = llm_response.response
                else:
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

            if self._post_run_hook is not None:
                try:
                    suite_id = str(config.get("suite_id", ""))
                    run_id = str(config.get("run_id", ""))
                    self._post_run_hook.on_transcript_complete(
                        transcript, run_id=run_id, suite_id=suite_id
                    )
                except Exception as hook_error:
                    logger.warning(
                        f"Post-run hook failed (non-blocking): {hook_error}",
                        exc_info=True,
                    )

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
        finally:
            await self._close_mcp_clients(mcp_clients)
