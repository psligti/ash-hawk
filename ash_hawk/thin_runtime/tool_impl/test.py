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
    detect_test_command,
    format_error,
    format_test_output,
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
    path_arg = call.tool_args.get("path")
    cwd = (
        resolve_path(path_arg, workdir)
        if isinstance(path_arg, str) and path_arg.strip()
        else workdir
    )
    resolved_cwd, workspace_err = check_workspace_path_error(cwd, workdir, "test")
    payload = ToolExecutionPayload(audit_updates=AuditToolContext(tool_usage=["test"]))
    if workspace_err or resolved_cwd is None:
        detail = (workspace_err or "❌ Access denied: path validation failed.").removeprefix("❌ ")
        msg = format_error("test", detail)
        return False, payload, msg, [msg]
    cwd = resolved_cwd
    command = call.tool_args.get("command")
    resolved_command = (
        command if isinstance(command, str) and command.strip() else detect_test_command(cwd)
    )
    exit_code, output, exec_error = run_shell_command(resolved_command, cwd=cwd, timeout=300)
    if exec_error is not None:
        msg = format_error("test", f"Test failed: {exec_error}")
        return False, payload, msg, [msg]
    if exit_code != 0:
        return (
            False,
            payload,
            format_test_output(output, passed=False),
            [f"Tests failed with exit code {exit_code}"],
        )
    return True, payload, format_test_output(output, passed=True), []


COMMAND = ToolCommand(
    name="test",
    summary="Run project tests.",
    when_to_use=["When verification should be delegated to the environment test runner"],
    when_not_to_use=["When no testable state change exists"],
    input_schema=context_input_schema(),
    model_input_schema=ToolSchemaSpec(
        properties=[
            ToolFieldSpec(
                name="command",
                type=SchemaFieldType.STRING,
                description="Optional test command override",
            ),
            ToolFieldSpec(
                name="path",
                type=SchemaFieldType.STRING,
                description="Optional directory to run tests in",
            ),
        ],
        required=[],
    ),
    output_schema=standard_output_schema(),
    side_effects=["subprocess"],
    risk_level="medium",
    timeout_seconds=300,
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
