from __future__ import annotations

from ash_hawk.improve_cycle.models import (
    CompetitorOutput,
    EvidenceRef,
    FailureCategory,
    ReviewFinding,
    RunArtifactBundle,
    Severity,
    TranslatorOutput,
)
from ash_hawk.improve_cycle.roles.base import BaseRoleAgent


class TranslatorRole(
    BaseRoleAgent[tuple[RunArtifactBundle, CompetitorOutput | None], TranslatorOutput]
):
    def __init__(self) -> None:
        super().__init__(
            "translator", "Normalize run artifacts into canonical findings", "deterministic", 0.0
        )

    def run(self, payload: tuple[RunArtifactBundle, CompetitorOutput | None]) -> TranslatorOutput:
        run_bundle, competitor = payload
        findings: list[ReviewFinding] = []

        def _trace_category(trace: dict[str, object]) -> tuple[FailureCategory, Severity, str]:
            if not trace:
                return (FailureCategory.TOOL_OBSERVABILITY_POOR, Severity.HIGH, "empty trace")
            lowered = " ".join(str(value).lower() for value in trace.values())
            if "missing" in lowered or "not found" in lowered:
                return (FailureCategory.TOOL_MISSING, Severity.HIGH, "missing tool signal")
            if "timeout" in lowered or "error" in lowered or "exception" in lowered:
                return (FailureCategory.TOOL_INTERFACE_POOR, Severity.HIGH, "tool runtime error")
            return (FailureCategory.TOOL_INTERFACE_POOR, Severity.MEDIUM, "interface friction")

        for idx, trace in enumerate(run_bundle.tool_traces):
            category, severity, rationale = _trace_category(trace)
            finding = ReviewFinding(
                finding_id=f"finding-{idx + 1}",
                title="Tool trace observation",
                summary=f"Observed tool trace keys: {', '.join(sorted(trace.keys())) if trace else 'none'}",
                severity=severity,
                category=category,
                evidence=[
                    EvidenceRef(
                        artifact_id=run_bundle.run_id,
                        kind="tool_trace",
                        pointer=str(idx),
                        note=rationale,
                    )
                ],
                strategy="tool-quality",
                sub_strategy="tool-efficiency",
            )
            findings.append(finding)

        for idx, output in enumerate(run_bundle.outputs):
            text = str(output).lower()
            if "policy" in text and "violation" in text:
                findings.append(
                    ReviewFinding(
                        finding_id=f"finding-output-policy-{idx + 1}",
                        title="Policy conformance issue",
                        summary="Output indicates policy violation handling weakness",
                        severity=Severity.HIGH,
                        category=FailureCategory.POLICY_GUARDRAIL,
                        evidence=[
                            EvidenceRef(
                                artifact_id=run_bundle.run_id,
                                kind="output",
                                pointer=str(idx),
                            )
                        ],
                        strategy="agent-behavior",
                        sub_strategy="policy-compliance",
                    )
                )
            elif "plan" in text and "fail" in text:
                findings.append(
                    ReviewFinding(
                        finding_id=f"finding-output-plan-{idx + 1}",
                        title="Planning failure",
                        summary="Output indicates planning breakdown",
                        severity=Severity.MEDIUM,
                        category=FailureCategory.PLANNER_FAILURE,
                        evidence=[
                            EvidenceRef(
                                artifact_id=run_bundle.run_id,
                                kind="output",
                                pointer=str(idx),
                            )
                        ],
                        strategy="agent-behavior",
                        sub_strategy="task-completion",
                    )
                )

        if competitor is not None and competitor.improved:
            findings.append(
                ReviewFinding(
                    finding_id="finding-replay-improved",
                    title="Replay showed improvement",
                    summary=competitor.summary,
                    severity=Severity.LOW,
                    category=FailureCategory.POLICY_ORDERING,
                    evidence=competitor.evidence,
                    strategy="agent-behavior",
                    sub_strategy="task-completion",
                )
            )
        elif competitor is not None and not competitor.improved:
            findings.append(
                ReviewFinding(
                    finding_id="finding-replay-no-gain",
                    title="Replay did not improve",
                    summary=competitor.summary,
                    severity=Severity.MEDIUM,
                    category=FailureCategory.NONDETERMINISM,
                    evidence=competitor.evidence,
                    strategy="reliability",
                    sub_strategy="stability",
                )
            )
        if not findings:
            findings.append(
                ReviewFinding(
                    finding_id="finding-fallback",
                    title="Insufficient artifact evidence",
                    summary="No actionable traces or outputs were available",
                    severity=Severity.LOW,
                    category=FailureCategory.MULTI_CAUSAL,
                    evidence=[EvidenceRef(artifact_id=run_bundle.run_id, kind="artifact")],
                    strategy="analysis",
                    sub_strategy="evidence-collection",
                )
            )
        return TranslatorOutput(
            normalized_findings=findings,
            schema_valid=True,
            mapping_notes=[
                f"Canonical mappings generated for {len(findings)} findings",
                f"tool_traces={len(run_bundle.tool_traces)} outputs={len(run_bundle.outputs)}",
            ],
            rejected_inputs=[],
        )
