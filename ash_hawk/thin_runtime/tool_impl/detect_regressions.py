from __future__ import annotations

from ash_hawk.thin_runtime.models import ToolCall, ToolResult
from ash_hawk.thin_runtime.tool_command import (
    ToolCommand,
    basic_input_schema,
    context_input_schema,
    delegation_input_schema,
    standard_output_schema,
)
from ash_hawk.thin_runtime.tool_impl._evaluation_signals import collect_evaluation_signals
from ash_hawk.thin_runtime.tool_types import (
    AuditToolContext,
    EvaluationToolContext,
    ToolExecutionPayload,
)


def _execute(call: ToolCall) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    signals = collect_evaluation_signals(call)
    regressions = list(signals.existing_regressions)
    if signals.score_regressed:
        regressions.append(
            f"{signals.candidate_label} regressed from {signals.baseline_score:.3f} to {signals.candidate_score:.3f}"
        )

    message = "No regressions detected"
    if regressions:
        message = f"Detected regressions: {'; '.join(regressions[:3])}"

    payload = ToolExecutionPayload(
        evaluation_updates=EvaluationToolContext(regressions=regressions),
        audit_updates=AuditToolContext(
            validation_tools=["detect_regressions"],
            run_summary={"regression_count": str(len(regressions))},
        ),
    )
    return True, payload, message, []


COMMAND = ToolCommand(
    name="detect_regressions",
    summary="Detect regressions outside target scope.",
    when_to_use=["When this exact capability is needed"],
    when_not_to_use=["When required inputs are missing"],
    input_schema=basic_input_schema(),
    output_schema=standard_output_schema(),
    side_effects=["none"],
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
