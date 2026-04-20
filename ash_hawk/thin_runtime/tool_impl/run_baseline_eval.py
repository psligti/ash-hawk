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
            return run_live_scenario_eval("run_baseline_eval", scenario_path)
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
