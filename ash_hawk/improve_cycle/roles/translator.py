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
        for idx, trace in enumerate(run_bundle.tool_traces):
            finding = ReviewFinding(
                finding_id=f"finding-{idx + 1}",
                title="Tool trace observation",
                summary=f"Observed tool trace keys: {', '.join(sorted(trace.keys())) if trace else 'none'}",
                severity=Severity.MEDIUM,
                category=FailureCategory.TOOL_OBSERVABILITY_POOR
                if not trace
                else FailureCategory.TOOL_INTERFACE_POOR,
                evidence=[
                    EvidenceRef(artifact_id=run_bundle.run_id, kind="tool_trace", pointer=str(idx))
                ],
                strategy="tool-quality",
                sub_strategy="tool-efficiency",
            )
            findings.append(finding)
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
        return TranslatorOutput(
            normalized_findings=findings,
            schema_valid=True,
            mapping_notes=["Canonical mappings generated"],
            rejected_inputs=[],
        )
