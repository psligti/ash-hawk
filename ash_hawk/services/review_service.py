from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from ash_hawk.contracts import (
    ReviewFinding,
    ReviewMetrics,
    ReviewRequest,
    ReviewResult,
    RunArtifact,
)
from ash_hawk.pipeline.types import PipelineRole

if TYPE_CHECKING:
    from ash_hawk.pipeline.orchestrator import PipelineOrchestrator


class ReviewService:
    def __init__(self) -> None:
        pass

    def review(
        self,
        request: ReviewRequest,
        artifact: RunArtifact,
    ) -> ReviewResult:
        review_id = f"review-{uuid4().hex[:8]}"

        try:
            from ash_hawk.pipeline.orchestrator import PipelineOrchestrator

            orchestrator = PipelineOrchestrator()
            orchestrator.run(request, artifact)

            analyst_step = orchestrator.get_step_result(PipelineRole.ANALYST)

            findings: list[ReviewFinding] = []
            metrics = ReviewMetrics(score=0.0)

            if analyst_step and analyst_step.outputs:
                raw_findings = analyst_step.outputs.get("findings", [])
                for rf in raw_findings:
                    if isinstance(rf, dict):
                        findings.append(
                            ReviewFinding(
                                finding_id=rf.get("finding_id", ""),
                                category=rf.get("category", ""),
                                severity=rf.get("severity", "info"),
                                title=rf.get("title", ""),
                                description=rf.get("description", ""),
                                evidence_refs=rf.get("evidence_refs", []),
                                recommendation=rf.get("recommendation"),
                            )
                        )
                raw_metrics = analyst_step.outputs.get("metrics", {})
                if raw_metrics:
                    metrics = ReviewMetrics(**raw_metrics)

            return ReviewResult(
                review_id=review_id,
                request_id=request.run_artifact_id,
                run_artifact_id=request.run_artifact_id,
                target_agent=request.target_agent,
                status="completed",
                findings=findings,
                metrics=metrics,
                proposal_ids=[p.proposal_id for p in orchestrator.get_proposals()],
                created_at=datetime.now(UTC),
            )
        except Exception as e:
            return ReviewResult(
                review_id=review_id,
                request_id=request.run_artifact_id,
                run_artifact_id=request.run_artifact_id,
                target_agent=request.target_agent,
                status="failed",
                findings=[],
                metrics=ReviewMetrics(score=0.0),
                created_at=datetime.now(UTC),
                error_message=str(e),
            )
