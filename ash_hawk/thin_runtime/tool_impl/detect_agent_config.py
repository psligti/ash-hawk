from __future__ import annotations

from pathlib import Path

from ash_hawk.thin_runtime.models import ToolCall, ToolResult
from ash_hawk.thin_runtime.tool_command import (
    ToolCommand,
    basic_input_schema,
    context_input_schema,
    delegation_input_schema,
    standard_output_schema,
)
from ash_hawk.thin_runtime.tool_types import (
    AuditToolContext,
    ToolExecutionPayload,
    WorkspaceToolContext,
)


def _execute(call: ToolCall) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    from ash_hawk.agents.source_workspace import (
        detect_agent_config_path,
        detect_package_name,
        detect_source_root,
    )

    workdir = Path(call.context.workspace.workdir or str(Path.cwd()))
    config_path = detect_agent_config_path(workdir)
    source_root = detect_source_root(workdir)
    package_name = detect_package_name(workdir) or _discover_package_name(source_root)
    payload = ToolExecutionPayload(
        workspace_updates=WorkspaceToolContext(
            agent_config=str(config_path),
            source_root=str(source_root),
            package_name=package_name,
        ),
        audit_updates=AuditToolContext(
            run_summary={
                "agent_config": str(config_path),
                "source_root": str(source_root),
                "package_name": package_name or "",
            },
            tool_usage=["detect_agent_config"],
        ),
    )
    package_fragment = f" for package {package_name}" if isinstance(package_name, str) else ""
    return True, payload, f"Detected agent config at {config_path}{package_fragment}", []


def _discover_package_name(source_root: Path) -> str | None:
    for child in sorted(source_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if not (child / "__init__.py").is_file():
            continue
        if (child / "agent").is_dir() or (child / "cli").is_dir():
            return child.name
    return None


COMMAND = ToolCommand(
    name="detect_agent_config",
    summary="Detect agent config.",
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
