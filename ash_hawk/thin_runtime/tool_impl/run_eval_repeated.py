from pathlib import Path

from ash_hawk.thin_runtime.live_eval import missing_live_eval_result, run_live_scenario_eval
from ash_hawk.thin_runtime.models import ToolCall, ToolResult
from ash_hawk.thin_runtime.tool_command import (
    ToolCommand,
    context_input_schema,
    standard_output_schema,
)
from ash_hawk.thin_runtime.tool_types import ToolExecutionPayload


def _execute(call: ToolCall) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
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
            payload.runtime_updates.preferred_tool = _next_tool_after_repeat(call, payload)
            return success, payload, message, errors
    return missing_live_eval_result(
        "run_eval_repeated",
        reason="Cannot run repeated evaluation without a valid scenario_path",
    )


def _next_tool_after_repeat(call: ToolCall, payload: ToolExecutionPayload) -> str:
    repeat_score = payload.evaluation_updates.repeat_eval_summary.score
    baseline_score = call.context.evaluation.baseline_summary.score
    repeat_passed = payload.audit_updates.run_result.aggregate_passed

    if isinstance(repeat_passed, bool) and repeat_passed:
        return ""
    if isinstance(repeat_score, int | float) and isinstance(baseline_score, int | float):
        if repeat_score < baseline_score:
            return "call_llm_structured"
    return "call_llm_structured"


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
