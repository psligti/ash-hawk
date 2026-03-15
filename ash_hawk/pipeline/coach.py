"""Coach role for policy and playbook, and behavior suggestions."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from ash_hawk.contracts import ImprovementProposal
from ash_hawk.pipeline.types import PipelineContext


class CoachRole:
    def generate_proposals(
        self, context: PipelineContext, findings: list[str]
    ) -> list[ImprovementProposal]:
        proposals = []

        for finding in findings:
            if "failure" in finding.lower() or "error" in finding.lower():
                proposals.append(
                    ImprovementProposal(
                        proposal_id=f"prop-{uuid4().hex[:8]}",
                        origin_run_id=context.run_artifact_id,
                        origin_review_id=context.review_request_id,
                        target_agent=context.target_agent,
                        proposal_type="policy",
                        title="Add error handling for failure cases",
                        rationale=f"Based on finding: {finding}",
                        expected_benefit="Improved reliability and fewer failures",
                        risk_level="low",
                        created_at=datetime.now(UTC),
                    )
                )

            if "timeout" in finding.lower():
                proposals.append(
                    ImprovementProposal(
                        proposal_id=f"prop-{uuid4().hex[:8]}",
                        origin_run_id=context.run_artifact_id,
                        origin_review_id=context.review_request_id,
                        target_agent=context.target_agent,
                        proposal_type="policy",
                        title="Add timeout handling",
                        rationale=f"Based on finding: {finding}",
                        expected_benefit="Prevent timeout failures",
                        risk_level="low",
                        diff_payload={"timeout_seconds": 30},
                        created_at=datetime.now(UTC),
                    )
                )

        return proposals
