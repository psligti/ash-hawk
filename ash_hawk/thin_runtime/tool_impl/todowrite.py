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
    format_todowrite_output,
    todo_file_path,
    write_todo_markdown,
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
    raw_todos = call.tool_args.get("todos", [])
    file_arg = call.tool_args.get("file_path")
    if not isinstance(raw_todos, list):
        return (
            False,
            ToolExecutionPayload(),
            "Field 'todos' expected array[object]",
            ["invalid_todos"],
        )
    todos = [todo for todo in raw_todos if isinstance(todo, dict)]
    file_path = todo_file_path(
        workdir, call.goal_id, file_arg if isinstance(file_arg, str) else None
    )
    resolved_file_path, workspace_err = check_workspace_path_error(file_path, workdir, "todowrite")
    if workspace_err or resolved_file_path is None:
        detail = (workspace_err or "❌ Access denied: path validation failed.").removeprefix("❌ ")
        msg = format_error("todowrite", detail)
        return False, ToolExecutionPayload(), msg, [msg]
    file_path = resolved_file_path
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(write_todo_markdown(todos), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        msg = format_error("todowrite", f"Failed to write todos: {exc}")
        return False, ToolExecutionPayload(), msg, [msg]
    payload = ToolExecutionPayload(audit_updates=AuditToolContext(tool_usage=["todowrite"]))
    return True, payload, format_todowrite_output(todos, file_path), []


COMMAND = ToolCommand(
    name="todowrite",
    summary="Write or update the todo list.",
    when_to_use=["When todo state should be updated as an atomic action"],
    when_not_to_use=["When no todo state change is needed"],
    input_schema=context_input_schema(),
    model_input_schema=ToolSchemaSpec(
        properties=[
            ToolFieldSpec(
                name="todos",
                type=SchemaFieldType.ARRAY,
                item_type=SchemaFieldType.OBJECT,
                description="Todo items to write",
            ),
            ToolFieldSpec(
                name="file_path",
                type=SchemaFieldType.STRING,
                description="Optional todo file path override",
            ),
        ],
        required=[],
    ),
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
