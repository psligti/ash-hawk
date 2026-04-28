from pathlib import Path

from ash_hawk.thin_runtime.live_eval import missing_live_eval_result, run_live_scenario_eval
from ash_hawk.thin_runtime.models import ToolCall, ToolResult
from ash_hawk.thin_runtime.tool_command import (
    ToolCommand,
    context_input_schema,
    standard_output_schema,
)
from ash_hawk.thin_runtime.tool_impl._eval_manifest import write_eval_manifest
from ash_hawk.thin_runtime.tool_types import (
    AuditToolContext,
    EvaluationToolContext,
    ToolExecutionPayload,
)


def _execute(call: ToolCall) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    scenario_path_raw = call.context.workspace.scenario_path
    if scenario_path_raw:
        scenario_path = Path(scenario_path_raw)
        if scenario_path.exists():
            success, payload, message, errors = run_live_scenario_eval(
                "run_baseline_eval", scenario_path
            )
            manifest_path, manifest_hash = write_eval_manifest(
                workdir=Path(call.context.workspace.workdir or str(Path.cwd())).resolve(),
                run_id=call.context.runtime.run_id or call.goal_id,
                scenario_path=scenario_path_raw,
                scenario_required_files=list(call.context.workspace.scenario_required_files),
                repetitions=2,
            )
            if manifest_path is not None and manifest_hash is not None:
                payload.evaluation_updates = payload.evaluation_updates.model_copy(
                    update={
                        "eval_manifest_path": str(manifest_path),
                        "eval_manifest_hash": manifest_hash,
                    }
                )
                payload.audit_updates = payload.audit_updates.model_copy(
                    update={
                        "artifacts": list(payload.audit_updates.artifacts)
                        + [f"eval-manifest:{manifest_hash}:{manifest_path}"],
                        "run_summary": {
                            **payload.audit_updates.run_summary,
                            "eval_manifest_path": str(manifest_path),
                            "eval_manifest_hash": manifest_hash,
                        },
                    }
                )
            return success, payload, message, errors
    return missing_live_eval_result(
        "run_baseline_eval",
        reason="Cannot run baseline evaluation without a valid scenario_path",
    )


COMMAND = ToolCommand(
    name="run_baseline_eval",
    summary="Run the baseline evaluation path.",
    when_to_use=["When evaluation is needed", "When a scenario path may be available"],
    when_not_to_use=["When no scenario or evaluation context exists"],
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
