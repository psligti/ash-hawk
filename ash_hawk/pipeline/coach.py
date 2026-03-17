from __future__ import annotations

from typing import Any

import pydantic as pd

from ash_hawk.contracts import ImprovementProposal
from ash_hawk.strategies.registry import Strategy, SubStrategy
from ash_hawk.types import EvalTranscript, EvalTrial


class Coach(pd.BaseModel):
    """Coach for generating improvement proposals from trial transcripts."""

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
                id=f"coach-{trial.id}-{len(proposals)}",
                trial_id=trial.id,
                finding=finding,
                strategy=strategy,
                sub_strategies=sub_strategies,
                confidence=0.8,  # Default confidence
                source="coach",
            )
            proposals.append(proposal)
        
        return proposals

    def _analyze_transcript(self, transcript: EvalTranscript) -> list[str]:
        """Extract findings from transcript."""
        findings = []
        
        # Simple pattern matching for demo - in real impl would use LLM
        text = " ".join([msg.content for msg in transcript.messages])
        
        if "timeout" in text.lower() or "error" in text.lower():
            findings.append("Agent encountered timeout or error during execution")
        if "policy" in text.lower() or "rule" in text.lower():
            findings.append("Agent violated policy or rule constraints")
        if "instruction" in text.lower() or "clarity" in text.lower():
            findings.append("Agent misunderstood instructions or lacked clarity")
        
        return findings

    def _infer_strategy_from_finding(self, finding: str) -> tuple[Strategy, list[SubStrategy]]:
        """Infer strategy and sub-strategies from finding."""
        finding_lower = finding.lower()
        
        if "timeout" in finding_lower or "error" in finding_lower:
            return Strategy.TOOL_QUALITY, [SubStrategy.ERROR_RECOVERY]
        elif "policy" in finding_lower or "rule" in finding_lower:
            return Strategy.POLICY_QUALITY, [SubStrategy.ENGAGEMENT_POLICY]
        elif "instruction" in finding_lower or "clarity" in finding_lower:
            return Strategy.SKILL_QUALITY, [SubStrategy.INSTRUCTION_CLARITY]
        else:
            # Default fallback
            return Strategy.TOOL_QUALITY, [SubStrategy.ERROR_RECOVERY]
