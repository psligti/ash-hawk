from __future__ import annotations

NO_HANDLER_TEMPLATE = "No handler registered for tool: {tool_name}"
TOOL_DENIED_TEMPLATE = "Tool denied by policy: {tool_name}"
MISSING_CONTEXTS_TEMPLATE = "Missing required contexts: {contexts}"


def no_handler_error(tool_name: str) -> str:
    return NO_HANDLER_TEMPLATE.format(tool_name=tool_name)


def tool_denied_error(tool_name: str) -> str:
    return TOOL_DENIED_TEMPLATE.format(tool_name=tool_name)


def missing_contexts_error(contexts: list[str]) -> str:
    return MISSING_CONTEXTS_TEMPLATE.format(contexts=", ".join(contexts))
