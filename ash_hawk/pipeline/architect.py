"""Architect role for harness, tool, and extension suggestions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from ash_hawk.contracts import ImprovementProposal
from ash_hawk.pipeline.types import PipelineContext


class ArchitectRole:
    def generate_proposals(
        self, context: PipelineContext, findings: list[str]
    ) -> list[ImprovementProposal]:
        proposals = []

        for finding in findings:
            if "tool" in finding.lower():
                proposals.append(
                    ImprovementProposal(
                        proposal_id=f"prop-{uuid4().hex[:8]}",
                        origin_run_id=context.run_artifact_id,
                        origin_review_id=context.review_request_id,
                        target_agent=context.target_agent,
                        proposal_type="tool",
                        title="Improve tool usage",
                        rationale=f"Based on finding: {finding}",
                        expected_benefit="Better tool efficiency",
                        risk_level="medium",
                        created_at=datetime.now(UTC),
                    )
                )

            if "redundant" in finding.lower():
                proposals.append(
                    ImprovementProposal(
                        proposal_id=f"prop-{uuid4().hex[:8]}",
                        origin_run_id=context.run_artifact_id,
                        origin_review_id=context.review_request_id,
                        target_agent=context.target_agent,
                        proposal_type="harness",
                        title="Reduce redundant operations",
                        rationale=f"Based on finding: {finding}",
                        expected_benefit="Improved efficiency",
                        risk_level="medium",
                        created_at=datetime.now(UTC),
                    )
                )

        return proposals
