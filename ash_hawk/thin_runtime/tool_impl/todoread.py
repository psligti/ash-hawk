from __future__ import annotations

from pathlib import Path

from ash_hawk.thin_runtime.models import ToolCall, ToolResult
from ash_hawk.thin_runtime.tool_command import (
    ToolCommand,
    context_input_schema,
    standard_output_schema,
)
from ash_hawk.thin_runtime.tool_impl._native_tooling import (
    check_workspace_path_error,
    format_error,
    format_todoread_output,
    todo_file_path,
)
from ash_hawk.thin_runtime.tool_types import (
    AuditToolContext,
    SchemaFieldType,
    ToolExecutionPayload,
    ToolFieldSpec,
    ToolSchemaSpec,
)


def _execute(call: ToolCall) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    workdir = Path(call.context.workspace.workdir or str(Path.cwd()))
    file_arg = call.tool_args.get("file_path")
    file_path = todo_file_path(
        workdir, call.goal_id, file_arg if isinstance(file_arg, str) else None
    )
    resolved_file_path, workspace_err = check_workspace_path_error(file_path, workdir, "todoread")
    if workspace_err or resolved_file_path is None:
        detail = (workspace_err or "❌ Access denied: path validation failed.").removeprefix("❌ ")
        msg = format_error("todoread", detail)
        return False, ToolExecutionPayload(), msg, [msg]
    file_path = resolved_file_path
    try:
        if not file_path.exists():
            output = format_todoread_output("", file_path)
        else:
            output = format_todoread_output(file_path.read_text(encoding="utf-8"), file_path)
    except Exception as exc:  # noqa: BLE001
        msg = format_error("todoread", f"Failed to read todos: {exc}")
        return False, ToolExecutionPayload(), msg, [msg]
    payload = ToolExecutionPayload(audit_updates=AuditToolContext(tool_usage=["todoread"]))
    return True, payload, output, []


COMMAND = ToolCommand(
    name="todoread",
    summary="Read the current todo list.",
    when_to_use=["When current todo state is needed"],
    when_not_to_use=["When no todo state is relevant"],
    input_schema=context_input_schema(),
    model_input_schema=ToolSchemaSpec(
        properties=[
            ToolFieldSpec(
                name="file_path",
                type=SchemaFieldType.STRING,
                description="Optional todo file path override",
            )
        ],
        required=[],
    ),
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
