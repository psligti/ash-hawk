from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from ash_hawk.contracts import ReviewFinding, ReviewMetrics, ReviewRequest, ReviewResult


class ReviewService:
    def review(self, request: ReviewRequest, run_artifact: Any) -> ReviewResult:
        tool_calls = list(getattr(run_artifact, "tool_calls", []) or [])
        total_calls = len(tool_calls)
        success_calls = sum(
            1 for tc in tool_calls if getattr(tc, "outcome", "success") == "success"
        )

        score = 1.0 if total_calls == 0 else success_calls / total_calls
        findings = self._build_findings(run_artifact, tool_calls)

        return ReviewResult(
            review_id=f"review-{uuid4().hex[:12]}",
            request_id=request.run_artifact_id,
            run_artifact_id=request.run_artifact_id,
            target_agent=request.target_agent,
            status="completed",
            findings=findings,
            metrics=ReviewMetrics(
                score=score,
                efficiency_score=score,
                quality_score=1.0
                if getattr(run_artifact, "outcome", "success") == "success"
                else 0.5,
                safety_score=1.0,
            ),
            proposal_ids=[],
            comparison=None,
            created_at=datetime.now(UTC),
            error_message=None,
        )

    def _build_findings(self, run_artifact: Any, tool_calls: list[Any]) -> list[ReviewFinding]:
        findings: list[ReviewFinding] = []

        if getattr(run_artifact, "outcome", "success") != "success":
            findings.append(
                ReviewFinding(
                    finding_id=f"finding-{uuid4().hex[:10]}",
                    category="execution",
                    severity="critical",
                    title="Run did not complete successfully",
                    description="The run outcome was marked as failure.",
                    evidence_refs=[f"run:{getattr(run_artifact, 'run_id', 'unknown')}:outcome"],
                    recommendation="Inspect failing steps and retry with targeted fixes.",
                )
            )

        for idx, tool_call in enumerate(tool_calls):
            if getattr(tool_call, "outcome", "success") == "failure":
                findings.append(
                    ReviewFinding(
                        finding_id=f"finding-{uuid4().hex[:10]}",
                        category="tooling",
                        severity="critical",
                        title=f"Tool call failed: {getattr(tool_call, 'tool_name', 'unknown')}",
                        description=getattr(
                            tool_call, "error_message", "Tool call failed without error details."
                        ),
                        evidence_refs=[f"tool_call:{idx}"],
                        recommendation="Address the immediate tool failure and add resilience for similar failures.",
                    )
                )

        return findings
