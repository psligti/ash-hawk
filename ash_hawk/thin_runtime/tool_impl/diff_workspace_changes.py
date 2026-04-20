from __future__ import annotations

from ash_hawk.thin_runtime.models import ToolCall, ToolResult
from ash_hawk.thin_runtime.tool_command import (
    ToolCommand,
    basic_input_schema,
    context_input_schema,
    delegation_input_schema,
    standard_output_schema,
)
from ash_hawk.thin_runtime.tool_types import ToolExecutionPayload, WorkspaceToolContext


def _execute(call: ToolCall) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    payload = ToolExecutionPayload(
        workspace_updates=WorkspaceToolContext(changed_files=call.context.workspace.changed_files)
    )
    return True, payload, "Diffed workspace changes", []


COMMAND = ToolCommand(
    name="diff_workspace_changes",
    summary="Diff workspace changes.",
    when_to_use=["When workspace state must be inspected or updated"],
    when_not_to_use=["When no workspace context exists"],
    input_schema=context_input_schema(),
    output_schema=standard_output_schema(),
    side_effects=["filesystem"],
    risk_level="low",
    timeout_seconds=30,
    completion_criteria=[
        "Output matches declared schema",
        "Execution stays within timeout and permission bounds",
        "Errors are explicit and actionable when failure occurs",
    ],
    escalation_rules=[
        "Escalate when confidence in output validity is low",
        "Escalate when the request exceeds tool permissions",
        "Escalate when repeated retries produce the same failure",
    ],
    executor=_execute,
)


def run(call: ToolCall) -> ToolResult:
    return COMMAND.run(call)
