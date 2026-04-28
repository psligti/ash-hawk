# type-hygiene: skip-file  # dawn-kestrel integration uses dynamic Any for SDK compatibility
"""Dawn-Kestrel agent runner implementation.

This module provides integration between ash-hawk and the dawn-kestrel
framework for agent execution with policy enforcement.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ash_hawk.dawn_kestrel_skills import prepare_skill_runtime, resolve_skill_project_root
from ash_hawk.policy import PolicyEnforcer
from ash_hawk.scenario.trace import (
    DEFAULT_TRACE_TS,
    ModelMessageEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from ash_hawk.types import (
    EvalOutcome,
    EvalTask,
    EvalTranscript,
    FailureMode,
    TokenUsage,
    ToolPermission,
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


def _ensure_eval_command_allowlist() -> None:
    try:
        importlib.import_module("dawn_kestrel.base.config")
    except ImportError:
        return
    # No ALLOWED_SHELL_COMMANDS in new SDK — function is effectively a no-op


def _tool_allowed_by_policy(policy_enforcer: PolicyEnforcer, tool_name: str) -> bool:
    return policy_enforcer.policy.is_tool_allowed(tool_name) is ToolPermission.ALLOW


def _mcp_policy_aliases(tool_name: str, metadata: dict[str, Any]) -> list[str]:
    aliases = [tool_name]
    server = metadata.get("mcp_server")
    mcp_tool = metadata.get("mcp_tool")
    if (
        isinstance(server, str)
        and server.strip()
        and isinstance(mcp_tool, str)
        and mcp_tool.strip()
    ):
        aliases.extend([f"{server}_{mcp_tool}", f"{server}-{mcp_tool}", mcp_tool])
    deduped: list[str] = []
    for alias in aliases:
        if alias not in deduped:
            deduped.append(alias)
    return deduped


def _is_allowed_tool_name(
    policy_enforcer: PolicyEnforcer,
    tool_name: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> bool:
    permissions = [
        policy_enforcer.policy.is_tool_allowed(alias)
        for alias in _mcp_policy_aliases(tool_name, metadata or {})
    ]
    if ToolPermission.DENY in permissions or ToolPermission.ASK in permissions:
        return False
    return ToolPermission.ALLOW in permissions


def _trace_ts(raw_timestamp: object) -> str:
    if isinstance(raw_timestamp, int | float):
        return f"{float(raw_timestamp):.6f}"
    return DEFAULT_TRACE_TS


def _session_trace_events(session: Any) -> list[dict[str, Any]]:
    trace_events: list[dict[str, Any]] = []
    for message in getattr(session, "messages", []):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if isinstance(role, str) and isinstance(content, str):
            trace_events.append(
                ModelMessageEvent.create(
                    DEFAULT_TRACE_TS, {"role": role, "content": content}
                ).model_dump()
            )
    for event in getattr(session, "events", []):
        event_type = getattr(event, "event_type", "")
        event_ts = _trace_ts(getattr(event, "timestamp", None))
        if event_type == "llm_call":
            trace_events.append(
                ModelMessageEvent.create(
                    event_ts,
                    {
                        "role": "assistant",
                        "content": getattr(event, "text", ""),
                        "provider": getattr(event, "provider", ""),
                        "model": getattr(event, "model", ""),
                        "finish_reason": getattr(event, "finish_reason", ""),
                        "had_tool_calls": getattr(event, "had_tool_calls", False),
                    },
                ).model_dump()
            )
            continue
        if event_type != "tool_call":
            continue
        tool_name = getattr(event, "tool_name", "")
        tool_input = getattr(event, "tool_input", {})
        trace_events.append(
            ToolCallEvent.create(
                event_ts,
                {"name": tool_name, "arguments": tool_input},
            ).model_dump()
        )
        trace_events.append(
            ToolResultEvent.create(
                event_ts,
                {
                    "tool_name": tool_name,
                    "result": getattr(event, "tool_result", ""),
                    "error": getattr(event, "error", None),
                    "duration_ms": getattr(event, "duration_ms", 0.0),
                    "output_preview": getattr(event, "output_preview", ""),
                },
            ).model_dump()
        )
    return trace_events


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
        if self._input_schema:
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
        project_root = kwargs.pop("project_root", None)
        self._kwargs = kwargs
        self._client: Any | None = None
        self._llm_queue: Any | None = None
        self._post_run_hook: Any | None = None
        self._skill_project_root = (
            resolve_skill_project_root(project_root) if project_root else None
        )

    def set_post_run_hook(self, hook: Any) -> None:
        self._post_run_hook = hook

    def get_post_run_hook(self) -> Any | None:
        return self._post_run_hook

    def _resolve_requested_skills(self, config: dict[str, Any]) -> list[str]:
        requested: list[str] = []
        for key in ("skill_name", "skill"):
            value = config.get(key)
            if isinstance(value, str) and value.strip() and value.strip() not in requested:
                requested.append(value.strip())
        raw_skill_names = config.get("skill_names")
        if isinstance(raw_skill_names, list):
            for item in raw_skill_names:
                if isinstance(item, str) and item.strip() and item.strip() not in requested:
                    requested.append(item.strip())
        skill_path = config.get("skill_path")
        if isinstance(skill_path, str) and skill_path.strip():
            path = Path(skill_path.strip())
            if path.name == "SKILL.md" and path.parent.name and path.parent.name not in requested:
                requested.append(path.parent.name)
        return requested

    def _resolve_skill_project_root(self, config: dict[str, Any]) -> Path:
        project_root = config.get("project_root")
        if isinstance(project_root, str) and project_root.strip():
            return resolve_skill_project_root(project_root)
        if self._skill_project_root is not None:
            return self._skill_project_root
        return resolve_skill_project_root()

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
        except ImportError:
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
                    metadata = {
                        "mcp_server": server.name,
                        "mcp_tool": tool_name,
                    }

                    register_method = getattr(base_registry, "register", None)
                    if callable(register_method):
                        result = register_method(proxy, tool_id, metadata=metadata)
                        if hasattr(result, "__await__"):
                            await cast(Any, result)
                    else:
                        tools_attr = getattr(base_registry, "tools", None)
                        if isinstance(tools_attr, dict):
                            tools_attr[tool_id] = proxy
                        tool_metadata_attr = getattr(base_registry, "tool_metadata", None)
                        if isinstance(tool_metadata_attr, dict):
                            tool_metadata_attr[tool_id] = metadata

                clients.append(client)

            return clients
        except Exception:
            await self._close_mcp_clients(clients)
            raise

    async def _close_mcp_clients(self, clients: list[_McpStdioClient]) -> None:
        for client in clients:
            try:
                await client.close()
            except Exception:  # nosec B112
                continue

    def _get_client(self) -> Any:
        if self._client is None:
            llm_client_module = importlib.import_module("dawn_kestrel.provider.llm_client")
            llm_client_type = getattr(llm_client_module, "LLMClient")
            api_key = None  # LLMClient auto-resolves from agent_config.yaml

            env_timeout = os.getenv("ASH_HAWK_LLM_TIMEOUT_SECONDS")
            timeout_value = self._kwargs.get("timeout_seconds", env_timeout)
            timeout_seconds = float(timeout_value) if timeout_value is not None else None

            client_kwargs: dict[str, Any] = {
                "provider_id": self._provider,
                "model": self._model,
                "api_key": api_key,
            }
            if timeout_seconds is not None:
                client_kwargs["timeout_seconds"] = timeout_seconds

            self._client = llm_client_type(**client_kwargs)
        return self._client

    def _create_filtered_registry(
        self,
        policy_enforcer: PolicyEnforcer,
        base_registry: Any = None,
        allowed_tools_override: list[str] | None = None,
        denied_tools_override: list[str] | None = None,
        use_policy_filters: bool = False,
    ) -> Any:
        try:
            tools_registry_module = importlib.import_module("dawn_kestrel.tools.registry")
            default_tool_registry = getattr(tools_registry_module, "ToolRegistry")
        except ImportError:
            tools_framework_module = importlib.import_module("dawn_kestrel.tools.framework")
            default_tool_registry = getattr(tools_framework_module, "ToolRegistry")

        if base_registry is None:
            base_registry = default_tool_registry()
        if not hasattr(base_registry, "tool_metadata"):
            setattr(base_registry, "tool_metadata", {})
        return base_registry

    async def run(
        self,
        task: EvalTask,
        policy_enforcer: PolicyEnforcer,
        config: dict[str, Any],
    ) -> tuple[EvalTranscript, EvalOutcome]:
        start_time = time.time()
        mcp_clients: list[_McpStdioClient] = []
        conversation: list[dict[str, Any]] = []
        all_tool_calls: list[dict[str, Any]] = []

        try:
            agent_module = importlib.import_module("dawn_kestrel.agent.loop")
            agent_types_module = importlib.import_module("dawn_kestrel.agent.types")

            run_agent = getattr(agent_module, "run_agent")
            LoopConfig = getattr(agent_types_module, "LoopConfig")

            _ensure_eval_command_allowlist()

            client = self._get_client()
            task_input = task.input

            base_registry = self._create_base_registry()
            mcp_clients = await self._register_mcp_tools(base_registry)
            filtered_registry = self._create_filtered_registry(
                policy_enforcer,
                base_registry=base_registry,
                use_policy_filters=False,
            )

            if isinstance(task_input, dict):
                prompt_value = task_input.get("prompt", task_input.get("message", str(task_input)))
                prompt = prompt_value if isinstance(prompt_value, str) else str(prompt_value)
            else:
                prompt = str(task_input)

            requested_skills = self._resolve_requested_skills(config)
            agent_path = config.get("agent_path")
            if agent_path is not None:
                # Read agent instructions directly from the agent_path directory
                agent_md = Path(agent_path) / "AGENT.md"
                if agent_md.exists():
                    agent_content = agent_md.read_text(encoding="utf-8")
                    prompt = (
                        f"<agent-instructions>\n{agent_content}\n</agent-instructions>\n\n{prompt}"
                    )
            skill_project_root = self._resolve_skill_project_root(config)

            available_tools_raw: Any = {}
            if hasattr(filtered_registry, "tools") and isinstance(
                getattr(filtered_registry, "tools"), dict
            ):
                available_tools_raw = dict(getattr(filtered_registry, "tools"))
            elif hasattr(filtered_registry, "get_all") and callable(
                getattr(filtered_registry, "get_all")
            ):
                try:
                    maybe_awaitable = filtered_registry.get_all()
                    if hasattr(maybe_awaitable, "__await__"):
                        available_tools_raw = await maybe_awaitable
                    else:
                        available_tools_raw = maybe_awaitable
                except Exception:
                    available_tools_raw = {}

            tool_metadata = getattr(filtered_registry, "tool_metadata", {})

            if isinstance(available_tools_raw, dict):
                available_tools = {
                    k: v
                    for k, v in available_tools_raw.items()
                    if _is_allowed_tool_name(
                        policy_enforcer,
                        k,
                        metadata=tool_metadata.get(k) if isinstance(tool_metadata, dict) else None,
                    )
                }
            else:
                available_tools = {
                    k: v
                    for k, v in available_tools_raw
                    if _is_allowed_tool_name(
                        policy_enforcer,
                        k,
                        metadata=tool_metadata.get(k) if isinstance(tool_metadata, dict) else None,
                    )
                }

            max_loop_iterations = int(config.get("max_iterations", 1) or 1)
            if max_loop_iterations < 1:
                max_loop_iterations = 1

            tools_registry_module = importlib.import_module("dawn_kestrel.tools.registry")
            tool_registry = tools_registry_module.ToolRegistry()
            for tool_impl in available_tools.values():
                tool_registry.register(tool_impl)

            loop_config = LoopConfig(max_iterations=max_loop_iterations)

            skill_runtime, preloaded_skill_messages = await prepare_skill_runtime(
                project_root=skill_project_root,
                preactivate=requested_skills,
                strict_preactivate=bool(requested_skills),
            )

            result = await run_agent(
                client=client,
                messages=[
                    *preloaded_skill_messages,
                    {"role": "user", "content": prompt},
                ],
                tools=tool_registry,
                config=loop_config,
                skills=skill_runtime,
            )

            if result.session is not None:
                conversation = list(result.session.messages)
                for event in result.session.filter_by_type("tool_call"):
                    normalized_tc: dict[str, Any] = {
                        "tool": event.tool_name,
                        "name": event.tool_name,
                        "input": event.tool_input,
                        "arguments": event.tool_input,
                    }
                    if hasattr(event, "output_preview") and event.output_preview:
                        normalized_tc["output"] = event.output_preview
                    if hasattr(event, "error") and event.error is not None:
                        normalized_tc["error"] = event.error
                    all_tool_calls.append(normalized_tc)

            duration = time.time() - start_time
            total_usage = result.total_usage or {}

            total_cost_usd = 0.0
            trace_events: list[dict[str, Any]] = []
            if result.session is not None:
                for evt in result.session.filter_by_type("llm_call"):
                    total_cost_usd += getattr(evt, "cost_usd", 0.0)
                trace_events = _session_trace_events(result.session)

            transcript = EvalTranscript(
                messages=conversation,
                tool_calls=all_tool_calls,
                trace_events=trace_events,
                token_usage=TokenUsage(
                    input=int(total_usage.get("input", 0)),
                    output=int(total_usage.get("output", 0)),
                    reasoning=int(total_usage.get("reasoning", 0)),
                    cache_read=int(total_usage.get("cache_read", 0)),
                    cache_write=int(total_usage.get("cache_write", 0)),
                ),
                cost_usd=total_cost_usd,
                duration_seconds=duration,
                agent_response=result.response.text if result.response else None,
                error_trace=result.error,
            )

            if result.error is not None:
                outcome = EvalOutcome.failure(FailureMode.AGENT_ERROR, result.error)
            else:
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
                messages=conversation,
                tool_calls=all_tool_calls,
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
                messages=conversation,
                tool_calls=all_tool_calls,
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
