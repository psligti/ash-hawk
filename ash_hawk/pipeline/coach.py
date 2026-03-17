from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

import pydantic as pd

from ash_hawk.contracts import ImprovementProposal
from ash_hawk.strategies.registry import Strategy, SubStrategy

if TYPE_CHECKING:
    from ash_hawk.pipeline.types import PipelineContext


class CoachRole(pd.BaseModel):
    """Coach role for generating improvement proposals from failure patterns."""

    def generate_proposals(
        self,
        context: PipelineContext,
        failure_patterns: list[str],
    ) -> list[ImprovementProposal]:
        """Generate improvement proposals from failure patterns."""
        proposals: list[ImprovementProposal] = []
        for pattern in failure_patterns:
            strategy, sub_strategies = self._infer_strategy_from_pattern(pattern)
            proposal_type = self._strategy_to_proposal_type(strategy)
            
            proposal = ImprovementProposal(
                proposal_id=f"coach-{context.run_artifact_id}-{len(proposals)}",
                origin_run_id=context.run_artifact_id,
                target_agent=context.target_agent,
                proposal_type=proposal_type,
                title=f"Coach proposal: {pattern[:50]}...",
                rationale=pattern,
                expected_benefit="Improve agent performance",
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

    def _infer_strategy_from_pattern(self, pattern: str) -> tuple[Strategy, list[SubStrategy]]:
        """Infer strategy and sub-strategies from failure pattern."""
        pattern_lower = pattern.lower()
        
        if "timeout" in pattern_lower or "error" in pattern_lower:
            return Strategy.TOOL_QUALITY, [SubStrategy.ERROR_RECOVERY]
        elif "policy" in pattern_lower or "rule" in pattern_lower:
            return Strategy.POLICY_QUALITY, [SubStrategy.ENGAGEMENT_POLICY]
        elif "instruction" in pattern_lower or "clarity" in pattern_lower:
            return Strategy.SKILL_QUALITY, [SubStrategy.INSTRUCTION_CLARITY]
        else:
            # Default fallback
            return Strategy.TOOL_QUALITY, [SubStrategy.ERROR_RECOVERY]


__all__ = ["CoachRole"]
