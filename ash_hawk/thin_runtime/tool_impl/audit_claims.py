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
    ClaimAuditStatus,
    EvaluationToolContext,
    ToolExecutionPayload,
)


def _execute(call: ToolCall) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    signals = collect_evaluation_signals(call)
    contradictions: list[str] = []
    run_result = call.context.audit.run_result

    success_claim = isinstance(run_result.message, str) and "success" in run_result.message.lower()
    if success_claim and signals.aggregate_passed is False:
        contradictions.append("run_result message claims success while aggregate eval failed")
    if signals.failure_family is not None:
        contradictions.append(f"failure family remains {signals.failure_family}")
    if signals.verification_verified is False:
        contradictions.append("verification step has not confirmed the outcome")
    if signals.score_regressed:
        contradictions.append(
            f"{signals.candidate_label} regressed from {signals.baseline_score:.3f} to {signals.candidate_score:.3f}"
        )

    aligned = not contradictions
    if aligned:
        message = "Claims are aligned with current evidence"
    else:
        message = f"Claims are not aligned: {'; '.join(contradictions[:3])}"

    payload = ToolExecutionPayload(
        evaluation_updates=EvaluationToolContext(claim_audit=ClaimAuditStatus(aligned=aligned)),
        audit_updates=AuditToolContext(
            validation_tools=["audit_claims"],
            run_summary={
                "aligned": str(aligned),
                "contradiction_count": str(len(contradictions)),
            },
        ),
    )
    return True, payload, message, []


COMMAND = ToolCommand(
    name="audit_claims",
    summary="Compare claims to evidence.",
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
