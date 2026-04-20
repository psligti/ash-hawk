from __future__ import annotations

import asyncio
from pathlib import Path

from ash_hawk.thin_runtime.models import ToolCall, ToolResult
from ash_hawk.thin_runtime.tool_command import (
    ToolCommand,
    context_input_schema,
    standard_output_schema,
)
from ash_hawk.thin_runtime.tool_impl._workspace_targets import preferred_workspace_target
from ash_hawk.thin_runtime.tool_types import (
    AuditToolContext,
    ToolExecutionPayload,
    WorkspaceToolContext,
)


def _execute(call: ToolCall) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    try:
        from bolt_merlin.agent.tools.write import WriteTool
        from dawn_kestrel.tools.framework import ToolContext as DKToolContext
    except ImportError:
        return (
            False,
            ToolExecutionPayload(),
            "bolt_merlin write tool is not available",
            [
                "bolt_merlin_unavailable",
            ],
        )

    workdir = Path(call.context.workspace.workdir or str(Path.cwd()))
    candidates = call.context.workspace.allowed_target_files or call.context.workspace.changed_files
    if not candidates:
        return (
            False,
            ToolExecutionPayload(),
            "No target file available for write",
            ["no_target_file"],
        )
    target = preferred_workspace_target(candidates)
    if target is None:
        return (
            False,
            ToolExecutionPayload(),
            "No target file available for write",
            ["no_target_file"],
        )

    tool = WriteTool()
    ctx = DKToolContext(session_id=call.goal_id, working_dir=workdir)
    result = asyncio.run(tool.execute({"file_path": target, "content": ""}, ctx))
    success = result.error is None
    payload = ToolExecutionPayload(
        workspace_updates=WorkspaceToolContext(mutated_files=[target]),
        audit_updates=AuditToolContext(tool_usage=["write"]),
    )
    errors = [result.error] if result.error else []
    return success, payload, result.output, errors


COMMAND = ToolCommand(
    name="write",
    summary="Write a file using the Dawn Kestrel tool surface.",
    when_to_use=["When a complete file write is the intended atomic action"],
    when_not_to_use=["When no target file is available in workspace context"],
    input_schema=context_input_schema(),
    output_schema=standard_output_schema(),
    side_effects=["filesystem"],
    risk_level="medium",
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
