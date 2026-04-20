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
from ash_hawk.thin_runtime.tool_impl.load_workspace_state import load_scenario_brief
from ash_hawk.thin_runtime.tool_types import AuditToolContext, ToolExecutionPayload


def _execute(call: ToolCall) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    try:
        from bolt_merlin.agent.tools.read import ReadTool
        from dawn_kestrel.tools.framework import ToolContext as DKToolContext
    except ImportError:
        return (
            False,
            ToolExecutionPayload(),
            "bolt_merlin read tool is not available",
            [
                "bolt_merlin_unavailable",
            ],
        )

    workdir = Path(call.context.workspace.workdir or str(Path.cwd()))
    candidates = call.context.workspace.allowed_target_files or call.context.workspace.changed_files
    if not candidates:
        scenario_summary = call.context.workspace.scenario_summary
        if not scenario_summary:
            _, _, scenario_summary = load_scenario_brief(
                call.context.workspace.scenario_path, workdir
            )
        if isinstance(scenario_summary, str) and scenario_summary.strip():
            payload = ToolExecutionPayload(audit_updates=AuditToolContext(tool_usage=["read"]))
            return True, payload, scenario_summary, []
        return (
            False,
            ToolExecutionPayload(),
            "No target file available for read",
            ["no_target_file"],
        )
    else:
        preferred_target = preferred_workspace_target(candidates)
        if preferred_target is None:
            return (
                False,
                ToolExecutionPayload(),
                "No target file available for read",
                ["no_target_file"],
            )
        target = preferred_target

    tool = ReadTool()
    ctx = DKToolContext(session_id=call.goal_id, working_dir=workdir)
    result = asyncio.run(tool.execute({"file_path": target}, ctx))
    success = result.error is None
    payload = ToolExecutionPayload(audit_updates=AuditToolContext(tool_usage=["read"]))
    errors = [result.error] if result.error else []
    return success, payload, result.output, errors


COMMAND = ToolCommand(
    name="read",
    summary="Read a file using the Dawn Kestrel tool surface.",
    when_to_use=["When exact file contents are needed"],
    when_not_to_use=["When no target file is available in workspace context"],
    input_schema=context_input_schema(),
    output_schema=standard_output_schema(),
    side_effects=["none"],
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
