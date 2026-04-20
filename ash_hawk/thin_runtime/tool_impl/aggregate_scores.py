from __future__ import annotations

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
    EvaluationToolContext,
    ToolExecutionPayload,
)


def _execute(call: ToolCall) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    scores = [
        score
        for score in (
            call.context.evaluation.baseline_summary.score,
            call.context.evaluation.last_eval_summary.score,
            call.context.evaluation.repeat_eval_summary.score,
            call.context.evaluation.targeted_validation_summary.score,
            call.context.evaluation.integrity_summary.score,
        )
        if isinstance(score, int | float)
    ]
    if not scores:
        payload = ToolExecutionPayload(
            audit_updates=AuditToolContext(
                validation_tools=["aggregate_scores"],
                run_summary={"status": "missing_scores"},
            )
        )
        return (
            False,
            payload,
            "Cannot aggregate scores without completed evaluations",
            ["missing_scores"],
        )

    aggregated_score = sum(float(score) for score in scores) / len(scores)
    payload = ToolExecutionPayload(
        evaluation_updates=EvaluationToolContext(aggregated_score=aggregated_score),
        audit_updates=AuditToolContext(
            validation_tools=["aggregate_scores"],
            run_summary={
                "aggregated_score": f"{aggregated_score:.3f}",
                "input_score_count": str(len(scores)),
            },
        ),
    )
    return True, payload, "Aggregated evaluation scores", []


COMMAND = ToolCommand(
    name="aggregate_scores",
    summary="Aggregate repeated evaluation scores.",
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
