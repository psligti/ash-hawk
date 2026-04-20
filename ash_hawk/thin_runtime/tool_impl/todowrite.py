from __future__ import annotations

import asyncio
from pathlib import Path

from ash_hawk.thin_runtime.models import ToolCall, ToolResult
from ash_hawk.thin_runtime.tool_command import (
    ToolCommand,
    context_input_schema,
    standard_output_schema,
)
from ash_hawk.thin_runtime.tool_types import AuditToolContext, ToolExecutionPayload


def _execute(call: ToolCall) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    try:
        from bolt_merlin.agent.tools.todowrite import TodoWriteTool
        from dawn_kestrel.tools.framework import ToolContext as DKToolContext
    except ImportError:
        return (
            False,
            ToolExecutionPayload(),
            "bolt_merlin todowrite tool is not available",
            [
                "bolt_merlin_unavailable",
            ],
        )

    tool = TodoWriteTool()
    workdir = Path(call.context.workspace.workdir or str(Path.cwd()))
    ctx = DKToolContext(session_id=call.goal_id, working_dir=workdir)
    result = asyncio.run(
        tool.execute(
            {"todos": [{"content": call.goal_id, "status": "in_progress", "priority": "high"}]},
            ctx,
        )
    )
    success = result.error is None
    payload = ToolExecutionPayload(audit_updates=AuditToolContext(tool_usage=["todowrite"]))
    errors = [result.error] if result.error else []
    return success, payload, result.output, errors


COMMAND = ToolCommand(
    name="todowrite",
    summary="Write todo state using the Dawn Kestrel tool surface.",
    when_to_use=["When todo state should be updated as an atomic action"],
    when_not_to_use=["When no todo state change is needed"],
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
