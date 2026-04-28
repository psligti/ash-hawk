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
    format_bash_error,
    format_bash_output,
    is_forbidden_command,
    resolve_path,
    run_shell_command,
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
    command = call.tool_args.get("command")
    timeout = call.tool_args.get("timeout")
    requested_workdir = call.tool_args.get("workdir")
    if not isinstance(command, str) or not command.strip():
        return (
            False,
            ToolExecutionPayload(),
            "Missing required tool argument: command",
            ["missing_command"],
        )
    cwd = (
        resolve_path(requested_workdir, workdir)
        if isinstance(requested_workdir, str) and requested_workdir.strip()
        else workdir
    )
    payload = ToolExecutionPayload(audit_updates=AuditToolContext(tool_usage=["bash"]))
    resolved_cwd, workspace_err = check_workspace_path_error(cwd, workdir, "bash")
    if workspace_err or resolved_cwd is None:
        message = workspace_err or "❌ Access denied: path validation failed."
        return False, payload, message, [message]
    cwd = resolved_cwd
    forbidden = is_forbidden_command(command)
    if forbidden:
        msg = f"❌ Access denied: command references forbidden pattern '{forbidden}'. Do not access evaluation infrastructure."
        return False, payload, msg, [msg]
    timeout_seconds = timeout if isinstance(timeout, int) and timeout > 0 else 120
    exit_code, output, exec_error = run_shell_command(command, cwd=cwd, timeout=timeout_seconds)
    if exec_error == f"timeout after {timeout_seconds}s":
        msg = format_bash_error(f"Timeout after {timeout_seconds}s: {command}", "", -1)
        return False, payload, msg, [msg]
    if exec_error is not None:
        msg = format_bash_error(f"Bash failed: {exec_error}", "", -1)
        return False, payload, msg, [msg]
    if exit_code != 0:
        msg = format_bash_error(command, output, exit_code)
        return False, payload, msg, [f"Command exited with code {exit_code}"]
    return True, payload, format_bash_output(command, output), []


COMMAND = ToolCommand(
    name="bash",
    summary="Run shell commands and capture combined output.",
    when_to_use=["When a simple shell command is the correct atomic action"],
    when_not_to_use=["When a dedicated file or test tool is more appropriate"],
    input_schema=context_input_schema(),
    model_input_schema=ToolSchemaSpec(
        properties=[
            ToolFieldSpec(
                name="command",
                type=SchemaFieldType.STRING,
                description="Shell command to execute",
                required=True,
            ),
            ToolFieldSpec(
                name="timeout",
                type=SchemaFieldType.INTEGER,
                description="Optional timeout in seconds",
            ),
            ToolFieldSpec(
                name="workdir",
                type=SchemaFieldType.STRING,
                description="Optional working directory override",
            ),
        ],
        required=["command"],
    ),
    output_schema=standard_output_schema(),
    side_effects=["subprocess"],
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
