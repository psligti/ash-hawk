from pathlib import Path

from ash_hawk.thin_runtime.live_eval import missing_live_eval_result, run_live_scenario_eval
from ash_hawk.thin_runtime.models import ToolCall, ToolResult
from ash_hawk.thin_runtime.tool_command import (
    ToolCommand,
    context_input_schema,
    standard_output_schema,
)
from ash_hawk.thin_runtime.tool_impl._eval_manifest import verify_eval_manifest
from ash_hawk.thin_runtime.tool_impl._evaluation_signals import collect_evaluation_signals
from ash_hawk.thin_runtime.tool_impl._isolated_workspace import cleanup_isolated_workspace
from ash_hawk.thin_runtime.tool_types import (
    AuditToolContext,
    RuntimeToolContext,
    ToolExecutionPayload,
    WorkspaceToolContext,
)


def _execute(call: ToolCall) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    manifest_ok, manifest_error = verify_eval_manifest(
        manifest_path=call.context.evaluation.eval_manifest_path,
        manifest_hash=call.context.evaluation.eval_manifest_hash,
        scenario_path=call.context.workspace.scenario_path,
    )
    if not manifest_ok:
        return (
            False,
            ToolExecutionPayload(),
            manifest_error or "Eval manifest verification failed",
            ["eval_manifest_mismatch"],
        )
    scenario_path_raw = call.context.workspace.scenario_path
    if scenario_path_raw:
        scenario_path = Path(scenario_path_raw)
        if scenario_path.exists():
            success, payload, message, errors = run_live_scenario_eval(
                "run_eval_repeated",
                scenario_path,
                summary_field="repeat_eval_summary",
                repetitions=2,
            )
            payload = _post_process_repeat_eval(call, payload)
            return success, payload, message, errors
    return missing_live_eval_result(
        "run_eval_repeated",
        reason="Cannot run repeated evaluation without a valid scenario_path",
    )


def _post_process_repeat_eval(
    call: ToolCall, payload: ToolExecutionPayload
) -> ToolExecutionPayload:
    signals = collect_evaluation_signals(
        call.model_copy(
            update={
                "context": call.context.model_copy(
                    update={
                        "evaluation": payload.evaluation_updates,
                        "failure": payload.failure_updates,
                        "audit": payload.audit_updates,
                    }
                )
            }
        )
    )
    isolated_path_raw = call.context.workspace.isolated_workspace_path
    sync_events = list(payload.audit_updates.sync_events)
    workspace_updates = payload.workspace_updates
    runtime_updates = payload.runtime_updates

    if isolated_path_raw and signals.aggregate_passed and not signals.score_regressed:
        sync_events.append("candidate_validated")
        runtime_updates = runtime_updates.model_copy(
            update={"preferred_tool": "sync_workspace_changes"}
        )
    else:
        if isolated_path_raw:
            cleanup_isolated_workspace(Path(isolated_path_raw))
        sync_events.append("candidate_rejected")
        runtime_updates = runtime_updates.model_copy(
            update={"preferred_tool": "call_llm_structured"}
        )
        workspace_updates = workspace_updates.model_copy(
            update={
                "mutated_files": [],
                "isolated_workspace": False,
                "isolated_workspace_path": "",
                "scenario_path": call.context.workspace.source_scenario_path
                or call.context.workspace.scenario_path,
                "source_scenario_path": "",
            }
        )

    audit_updates = payload.audit_updates.model_copy(update={"sync_events": sync_events})
    return payload.model_copy(
        update={
            "runtime_updates": runtime_updates,
            "workspace_updates": workspace_updates,
            "audit_updates": audit_updates,
        }
    )


COMMAND = ToolCommand(
    name="run_eval_repeated",
    summary="Execute the re-evaluation step of an improvement loop.",
    when_to_use=[
        "When a candidate mutation already exists and stability should be checked",
        "When a scenario path may be available for repeated re-evaluation",
    ],
    when_not_to_use=[
        "When no scenario or evaluation context exists",
        "When the initial baseline has not been established",
        "When no mutation or candidate change has been made yet",
    ],
    input_schema=context_input_schema(),
    output_schema=standard_output_schema(),
    side_effects=["may invoke thin scenario runner"],
    risk_level="medium",
    timeout_seconds=120,
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
