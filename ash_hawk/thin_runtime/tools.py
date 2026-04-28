from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from ash_hawk.thin_runtime.error_signatures import (
    missing_contexts_error,
    no_handler_error,
    tool_denied_error,
)
from ash_hawk.thin_runtime.models import ToolCall, ToolResult, ToolSpec
from ash_hawk.types import ToolPermission, ToolSurfacePolicy

ToolHandler = Callable[[ToolCall], ToolResult]


def default_tool_handler(call: ToolCall) -> ToolResult:
    return ToolResult(
        tool_name=call.tool_name,
        success=False,
        error=no_handler_error(call.tool_name),
    )


@dataclass
class RegisteredTool:
    spec: ToolSpec
    handler: ToolHandler


class ToolRegistry:
    def __init__(self, tools: list[ToolSpec]) -> None:
        self._tools = {
            tool.name: RegisteredTool(spec=tool, handler=self._load_handler(tool)) for tool in tools
        }

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name].spec
        except KeyError as exc:
            raise ValueError(f"Unknown thin runtime tool: {name}") from exc

    def register_handler(self, name: str, handler: ToolHandler) -> None:
        self._tools[name].handler = handler

    def list_tools(self) -> list[ToolSpec]:
        return [item.spec for item in self._tools.values()]

    def resolve_allowed(self, tool_names: list[str], policy: ToolSurfacePolicy) -> list[ToolSpec]:
        allowed: list[ToolSpec] = []
        for tool_name in tool_names:
            permission = policy.is_tool_allowed(tool_name)
            if permission is ToolPermission.DENY:
                continue
            allowed.append(self.get(tool_name))
        return allowed

    def invoke(
        self,
        call: ToolCall,
        policy: ToolSurfacePolicy,
        *,
        available_contexts: set[str] | None = None,
    ) -> ToolResult:
        permission = policy.is_tool_allowed(call.tool_name)
        if permission is ToolPermission.DENY:
            return ToolResult(
                tool_name=call.tool_name,
                success=False,
                error=tool_denied_error(call.tool_name),
            )
        spec = self._tools[call.tool_name].spec
        missing_contexts = sorted(
            context_name
            for context_name in spec.required_contexts
            if available_contexts is not None and context_name not in available_contexts
        )
        if missing_contexts:
            return ToolResult(
                tool_name=call.tool_name,
                success=False,
                error=missing_contexts_error(missing_contexts),
            )
        return self._tools[call.tool_name].handler(call)

    def _load_handler(self, tool: ToolSpec) -> ToolHandler:
        if not tool.entrypoint:
            return default_tool_handler
        module_path = tool.entrypoint
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            raise ValueError(
                f"Unable to import tool module for {tool.name}: {module_path}"
            ) from exc
        handler = getattr(module, tool.callable, None)
        if handler is None or not callable(handler):
            raise ValueError(
                f"Tool callable '{tool.callable}' not found for {tool.name} in {module_path}"
            )
        return cast(ToolHandler, handler)
