from __future__ import annotations

import asyncio
import difflib
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
    RuntimeToolContext,
    ToolExecutionPayload,
    WorkspaceToolContext,
)


def _execute(call: ToolCall) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    workdir = Path(call.context.workspace.workdir or str(Path.cwd()))
    required_files = list(call.context.workspace.scenario_required_files)
    existing_required_files = [
        path for path in required_files if (workdir / path).exists() and (workdir / path).is_file()
    ]
    candidates = (
        existing_required_files
        or call.context.workspace.allowed_target_files
        or call.context.workspace.changed_files
    )
    target_file = preferred_workspace_target(candidates)
    if not target_file:
        return (
            False,
            ToolExecutionPayload(),
            "No target files available for mutation",
            ["no_target_files"],
        )

    target_path = workdir / target_file
    if not target_path.exists() or not target_path.is_file():
        return (
            False,
            ToolExecutionPayload(),
            f"Target file not found: {target_file}",
            ["target_file_not_found"],
        )

    before_text = target_path.read_text(encoding="utf-8", errors="replace")
    required_fragment = ", ".join(required_files) if required_files else target_file
    capability_mode_note = ""
    if required_files and not existing_required_files:
        capability_mode_note = (
            "The scenario-required files are scenario-local outputs and may not exist in this repository. "
            "Do not claim they are already complete based on repo inspection. Instead, improve the coding agent's reusable prompt/tool behavior so future scenario runs produce those files correctly.\n\n"
        )

    explanation = "\n".join(call.context.failure.explanations)
    concepts = "\n".join(call.context.failure.concepts)
    prompt = (
        f"Improve the required workspace deliverables `{required_fragment}` for goal `{call.goal_id}`.\n\n"
        f"Primary target file for this mutation: `{target_file}`.\n\n"
        f"Scenario summary:\n{call.context.workspace.scenario_summary or 'No scenario summary available'}\n\n"
        f"{capability_mode_note}"
        f"Failure explanations:\n{explanation}\n\n"
        f"Concepts:\n{concepts}\n\n"
        "Use the available tools to inspect, edit, and verify the change. "
        "Keep the change minimal and targeted. Update the scenario-required workspace files rather than only agent guidance files."
    )
    try:
        from bolt_merlin.agent.execute import execute
    except ImportError:
        return (
            False,
            ToolExecutionPayload(),
            "bolt_merlin agent is not available",
            ["bolt_merlin_unavailable"],
        )

    config_path = workdir / ".dawn-kestrel" / "agent_config.yaml"
    result = asyncio.run(
        execute(
            prompt=prompt,
            agent_name="coding_agent",
            working_dir=workdir.resolve(),
            config_path=config_path if config_path.exists() else None,
            trace=True,
        )
    )
    if hasattr(result, "error_type"):
        return (
            False,
            ToolExecutionPayload(),
            getattr(result, "message", "bolt_merlin execution failed"),
            [str(getattr(result, "error_type", "execution_error"))],
        )

    response_text = getattr(result, "response", "")
    after_text = target_path.read_text(encoding="utf-8", errors="replace")
    diff_lines = list(
        difflib.unified_diff(
            before_text.splitlines(),
            after_text.splitlines(),
            fromfile=f"before/{target_file}",
            tofile=f"after/{target_file}",
            n=2,
            lineterm="",
        )
    )
    diff_preview = "\n".join(diff_lines[:40])
    changed = before_text != after_text
    if (not isinstance(response_text, str) or not response_text.strip()) and not changed:
        return (
            False,
            ToolExecutionPayload(),
            "bolt_merlin returned no mutation response",
            ["empty_mutation_response"],
        )
    response_summary = (
        response_text if isinstance(response_text, str) and response_text.strip() else ""
    )
    payload = ToolExecutionPayload(
        runtime_updates=RuntimeToolContext(preferred_tool="run_eval_repeated"),
        workspace_updates=WorkspaceToolContext(mutated_files=[target_file]),
        audit_updates=AuditToolContext(
            diff_report={
                "files": 1,
                "diff_line_count": len(diff_lines),
            },
            run_summary={
                "mutation_session_id": str(getattr(result, "session_id", "")),
                "target_file": target_file,
                "changed": str(changed),
                "diff_preview": diff_preview,
                "response": response_summary,
            },
            tool_usage=["mutate_agent_files"],
        ),
    )
    change_fragment = "changed" if changed else "reported success but produced no file diff"
    return True, payload, f"Mutated {target_file} via bolt_merlin agent ({change_fragment})", []


COMMAND = ToolCommand(
    name="mutate_agent_files",
    summary="Mutate agent files.",
    when_to_use=["When workspace state must be inspected or updated"],
    when_not_to_use=["When no workspace context exists"],
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
