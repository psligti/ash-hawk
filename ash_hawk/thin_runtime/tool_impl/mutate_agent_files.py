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
from ash_hawk.thin_runtime.tool_impl._native_tooling import check_workspace_path_error
from ash_hawk.thin_runtime.tool_impl._workspace_targets import preferred_workspace_target
from ash_hawk.thin_runtime.tool_types import (
    AuditToolContext,
    RuntimeToolContext,
    SchemaFieldType,
    ToolExecutionPayload,
    ToolFieldSpec,
    ToolSchemaSpec,
    WorkspaceToolContext,
)


def _execute(call: ToolCall) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    workdir = Path(call.context.workspace.workdir or str(Path.cwd()))
    if not call.context.workspace.isolated_workspace_path:
        return (
            False,
            ToolExecutionPayload(),
            "Mutation requires an isolated workspace first",
            ["missing_isolated_workspace"],
        )
    required_files = list(call.context.workspace.scenario_required_files)
    target_candidates = _mutation_candidates(call)
    target_file = preferred_workspace_target(target_candidates)
    if not target_file:
        return (
            False,
            ToolExecutionPayload(),
            "No target files available for mutation",
            ["no_target_files"],
        )

    target_path = workdir / target_file
    resolved_target_path, workspace_err = check_workspace_path_error(
        target_path, workdir, "mutate_agent_files"
    )
    if workspace_err or resolved_target_path is None:
        message = workspace_err or "❌ Access denied: path validation failed."
        return False, ToolExecutionPayload(), message, [message]
    target_path = resolved_target_path
    if not target_path.exists() or not target_path.is_file():
        return (
            False,
            ToolExecutionPayload(),
            f"Target file not found: {target_file}",
            ["target_file_not_found"],
        )

    before_text = target_path.read_text(encoding="utf-8", errors="replace")
    required_fragment = ", ".join(required_files) if required_files else target_file
    prompt = _mutation_prompt(call, target_file=target_file, required_fragment=required_fragment)

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
        runtime_updates=RuntimeToolContext(
            preferred_tool=(
                "" if call.context.runtime.active_agent == "executor" else "run_eval_repeated"
            ),
            stop_reason=(
                "candidate mutation completed in isolated workspace"
                if call.context.runtime.active_agent == "executor"
                else None
            ),
        ),
        workspace_updates=WorkspaceToolContext(mutated_files=[target_file]),
        audit_updates=AuditToolContext(
            diff_report={
                "files": 1,
                "diff_line_count": len(diff_lines),
            },
            sync_events=["candidate_mutated"],
            run_summary={
                "mutation_session_id": str(getattr(result, "session_id", "")),
                "target_file": target_file,
                "changed": str(changed),
                "diff_preview": diff_preview,
                "response": response_summary,
                "failure_family": call.context.failure.failure_family or "",
                "baseline_score": str(call.context.evaluation.baseline_summary.score),
            },
            tool_usage=["mutate_agent_files"],
        ),
        stop=(call.context.runtime.active_agent == "executor"),
    )
    change_fragment = "changed" if changed else "reported success but produced no file diff"
    return True, payload, f"Mutated {target_file} via bolt_merlin agent ({change_fragment})", []


def _mutation_candidates(call: ToolCall) -> list[str]:
    requested_target = call.tool_args.get("target_file")
    if isinstance(requested_target, str) and requested_target.strip():
        return [requested_target.strip()]
    return []


def _existing_file_candidates(workdir: Path, paths: list[str]) -> list[str]:
    return [path for path in paths if (workdir / path).exists() and (workdir / path).is_file()]


def _mutation_prompt(call: ToolCall, *, target_file: str, required_fragment: str) -> str:
    baseline = call.context.evaluation.baseline_summary
    repeat_eval = call.context.evaluation.repeat_eval_summary
    last_eval = call.context.evaluation.last_eval_summary
    latest_score = repeat_eval.score if repeat_eval.score is not None else last_eval.score
    failure_family = call.context.failure.failure_family or "Unknown"
    explanation = (
        "\n".join(call.context.failure.explanations) or "No failure explanations available"
    )
    concepts = "\n".join(call.context.failure.concepts) or "No concepts available"
    ranked = call.context.failure.ranked_hypotheses[:3]
    top_hypothesis = ranked[0] if ranked else None
    ideal_outcome = top_hypothesis.ideal_outcome if top_hypothesis else ""
    capability_mode_note = ""
    if call.context.workspace.scenario_required_files and not _existing_file_candidates(
        Path(call.context.workspace.workdir or str(Path.cwd())),
        list(call.context.workspace.scenario_required_files),
    ):
        capability_mode_note = (
            "The scenario-required files may be local outputs rather than durable repo files. "
            "Improve reusable coding-agent behavior when the diagnosis points to a durable prompt/runtime/config surface.\n\n"
        )

    hypotheses_block = (
        "\n".join(
            f"- {hypothesis.name} (score {hypothesis.score:.2f})"
            + (f"\n  Rationale: {hypothesis.rationale}" if hypothesis.rationale else "")
            + (
                f"\n  Target files: {', '.join(hypothesis.target_files)}"
                if hypothesis.target_files
                else ""
            )
            + (f"\n  Ideal outcome: {hypothesis.ideal_outcome}" if hypothesis.ideal_outcome else "")
            for hypothesis in ranked
        )
        or "- No ranked hypotheses available"
    )

    return (
        f"Improve the coding agent behavior for goal `{call.goal_id}` with one small, meaningful change.\n\n"
        f"## Primary mutation target\n{target_file}\n\n"
        f"## Current baseline\n"
        f"Baseline score: {baseline.score}\n"
        f"Latest re-eval score: {latest_score}\n"
        f"Failure family: {failure_family}\n"
        f"Regressions: {', '.join(call.context.evaluation.regressions) or 'None recorded'}\n\n"
        f"## What went wrong\n{explanation}\n\n"
        f"## What is needed\n"
        f"Required workspace deliverables: {required_fragment}\n"
        f"Ideal condition: {ideal_outcome or call.context.workspace.scenario_summary or 'Encode the missing capability durably.'}\n"
        f"Concepts: {concepts}\n\n"
        f"## Ranked hypotheses\n{hypotheses_block}\n\n"
        f"## Decision trace\n{_decision_trace(call)}\n\n"
        f"## Trace analysis\n{_trace_excerpt(call)}\n\n"
        f"## Transcript evidence\n{_transcript_excerpt(call)}\n\n"
        f"## Previous mutation / diff context\n{_diff_summary(call)}\n\n"
        f"{capability_mode_note}"
        "Use the diagnosis package above to make one targeted change. Prefer durable prompt/runtime/config/code files over workspace artifacts unless the target file itself is the graded deliverable. Read the named target directly before broadening scope."
    )


def _decision_trace(call: ToolCall) -> str:
    return "\n".join(call.context.audit.decision_trace[-5:]) or "No decision trace available"


def _trace_excerpt(call: ToolCall) -> str:
    excerpts = []
    for event in call.context.audit.events[-6:]:
        parts = [event.event_type or "event"]
        if event.tool:
            parts.append(f"tool={event.tool}")
        if event.rationale:
            parts.append(f"rationale={event.rationale}")
        if event.error:
            parts.append(f"error={event.error}")
        excerpts.append(" | ".join(parts))
    return "\n".join(excerpts) or "No trace events recorded"


def _transcript_excerpt(call: ToolCall) -> str:
    excerpts = []
    for record in call.context.audit.transcripts[-6:]:
        label = record.speaker or record.type or "entry"
        message = (record.message or "").strip()
        if message:
            excerpts.append(f"{label}: {message}")
    return "\n".join(excerpts) or "No transcript entries recorded"


def _diff_summary(call: ToolCall) -> str:
    diff_report = call.context.audit.diff_report
    run_summary = call.context.audit.run_summary
    if not diff_report and not run_summary:
        return "No prior mutation summary available"
    lines = []
    if diff_report:
        lines.append(f"Diff report: {diff_report}")
    if run_summary:
        lines.append(f"Run summary: {run_summary}")
    return "\n".join(lines)


COMMAND = ToolCommand(
    name="mutate_agent_files",
    summary="Mutate agent files.",
    when_to_use=["When a specific durable file has already been chosen for mutation"],
    when_not_to_use=["When you have not yet identified an exact target file"],
    input_schema=context_input_schema(),
    model_input_schema=ToolSchemaSpec(
        properties=[
            ToolFieldSpec(
                name="target_file",
                type=SchemaFieldType.STRING,
                description="Primary file path to mutate",
                required=True,
            )
        ],
        required=["target_file"],
    ),
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
