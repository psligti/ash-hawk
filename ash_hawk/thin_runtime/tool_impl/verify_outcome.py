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
    VerificationStatus,
)


def _execute(call: ToolCall) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
    signals = collect_evaluation_signals(call)
    issues: list[str] = []
    if signals.aggregate_passed is False:
        issues.append("latest eval did not pass")
    if signals.failure_family is not None:
        issues.append(f"failure family is {signals.failure_family}")
    if signals.score_regressed:
        issues.append(
            f"{signals.candidate_label} regressed from {signals.baseline_score:.3f} to {signals.candidate_score:.3f}"
        )
    if signals.existing_regressions:
        issues.extend(signals.existing_regressions)

    evidence_count = sum(
        1
        for value in (
            signals.baseline_score,
            signals.candidate_score,
            signals.aggregate_passed,
        )
        if value is not None
    ) + len(signals.failure_explanations)
    if evidence_count == 0:
        issues.append("no evaluation evidence is available")
    verified = not issues
    if verified:
        message = "Verified outcome against current evaluation context"
    else:
        message = f"Outcome not verified: {'; '.join(issues[:3])}"

    payload = ToolExecutionPayload(
        evaluation_updates=EvaluationToolContext(
            verification=VerificationStatus(
                verified=verified,
                evidence_count=evidence_count,
            )
        ),
        audit_updates=AuditToolContext(
            validation_tools=["verify_outcome"],
            run_summary={
                "verified": str(verified),
                "evidence_count": str(evidence_count),
                "candidate_label": signals.candidate_label,
            },
        ),
    )
    return True, payload, message, []


COMMAND = ToolCommand(
    name="verify_outcome",
    summary="Verify an outcome against configured checks.",
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
