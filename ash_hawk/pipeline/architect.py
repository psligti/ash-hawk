from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

import pydantic as pd

from ash_hawk.contracts import ImprovementProposal
from ash_hawk.strategies.registry import Strategy, SubStrategy

if TYPE_CHECKING:
    from ash_hawk.pipeline.types import PipelineContext


class ArchitectRole(pd.BaseModel):
    """Architect for generating improvement proposals from findings."""

    model_config = pd.ConfigDict(extra="forbid")

    def generate_proposals(
        self,
        context: PipelineContext,
        findings: list[str],
    ) -> list[ImprovementProposal]:
        """Generate improvement proposals from findings."""
        proposals: list[ImprovementProposal] = []
        for finding in findings:
            strategy, sub_strategies = self._infer_strategy_from_finding(finding)
            proposal_type = self._strategy_to_proposal_type(strategy)
            
            proposal = ImprovementProposal(
                proposal_id=f"architect-{context.run_artifact_id}-{len(proposals)}",
                origin_run_id=context.run_artifact_id,
                target_agent=context.target_agent,
                proposal_type=proposal_type,
                title=f"Architect proposal: {finding[:50]}...",
                rationale=finding,
                expected_benefit="Improve harness or tool quality",
                risk_level="medium",
                status="pending",
                created_at=datetime.now(UTC),
                strategy=strategy,
                sub_strategies=sub_strategies,
                confidence=1.0,
            )
            proposals.append(proposal)
        
        return proposals

    def _strategy_to_proposal_type(self, strategy: Strategy) -> Literal["policy", "skill", "tool", "harness", "eval"]:
        """Map strategy to proposal type."""
        if strategy == Strategy.TOOL_QUALITY:
            return "tool"
        elif strategy == Strategy.SKILL_QUALITY:
            return "skill"
        elif strategy == Strategy.POLICY_QUALITY:
            return "policy"
        elif strategy == Strategy.HARNESS_QUALITY:
            return "harness"
        else:
            return "tool"

    def _infer_strategy_from_finding(self, finding: str) -> tuple[Strategy, list[SubStrategy]]:
        """Infer strategy and sub-strategies from finding."""
        finding_lower = finding.lower()
        
        if "tool" in finding_lower or "efficiency" in finding_lower:
            return Strategy.TOOL_QUALITY, [SubStrategy.TOOL_EFFICIENCY]
        elif "harness" in finding_lower or "grader" in finding_lower:
            return Strategy.HARNESS_QUALITY, [SubStrategy.GRADER_CALIBRATION]
        elif "redundant" in finding_lower:
            return Strategy.TOOL_QUALITY, [SubStrategy.TOOL_SELECTION]
        else:
            # Default fallback
            return Strategy.TOOL_QUALITY, [SubStrategy.TOOL_EFFICIENCY]


__all__ = ["ArchitectRole"]
