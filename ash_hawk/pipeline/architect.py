from __future__ import annotations

from typing import Any

import pydantic as pd

from ash_hawk.contracts import ImprovementProposal
from ash_hawk.strategies.registry import Strategy, SubStrategy
from ash_hawk.types import EvalTranscript, EvalTrial


class Architect(pd.BaseModel):
    """Architect for generating improvement proposals from trial analysis."""

    model_config = pd.ConfigDict(extra="forbid")

    async def generate_proposals(
        self,
        trial: EvalTrial,
        transcript: EvalTranscript,
    ) -> list[ImprovementProposal]:
        """Generate improvement proposals from trial analysis."""
        findings = self._analyze_transcript(transcript)
        
        proposals = []
        for finding in findings:
            strategy, sub_strategies = self._infer_strategy_from_finding(finding)
            
            proposal = ImprovementProposal(
                id=f"architect-{trial.id}-{len(proposals)}",
                trial_id=trial.id,
                finding=finding,
                strategy=strategy,
                sub_strategies=sub_strategies,
                confidence=0.8,  # Default confidence
                source="architect",
            )
            proposals.append(proposal)
        
        return proposals

    def _analyze_transcript(self, transcript: EvalTranscript) -> list[str]:
        """Extract findings from transcript."""
        findings = []
        
        # Simple pattern matching for demo - in real impl would use LLM
        text = " ".join([msg.content for msg in transcript.messages])
        
        if "tool" in text.lower() or "efficiency" in text.lower():
            findings.append("Agent used inefficient tools or tool selection")
        if "harness" in text.lower() or "grader" in text.lower():
            findings.append("Harness or grader calibration issues detected")
        if "redundant" in text.lower():
            findings.append("Agent performed redundant actions")
        
        return findings

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
