from __future__ import annotations

from typing import Any

import pydantic as pd

from ash_hawk.strategies.registry import STRATEGY_TO_LESSON_TYPE, Strategy, SubStrategy
from ash_hawk.types import GraderSpec


class StrategyMapper(pd.BaseModel):
    """Maps strategies to eval packs and grader specs."""

    model_config = pd.ConfigDict(extra="forbid")

    def get_eval_packs(self, strategy: Strategy) -> list[str]:
        """Get eval pack names for a strategy."""
        lesson_type = STRATEGY_TO_LESSON_TYPE.get(strategy)
        if not lesson_type:
            return []
        
        # Map lesson types to eval pack names
        # This would be expanded based on actual eval packs
        pack_mapping = {
            "tool": ["tool_selection", "tool_efficiency"],
            "skill": ["instruction_following", "reasoning"],
            "policy": ["safety", "alignment"],
            "harness": ["grader_calibration", "fixture_quality"],
            "eval": ["rubric_precision", "test_coverage"],
        }
        
        packs = []
        for lesson in lesson_type or []:
            packs.extend(pack_mapping.get(lesson, []))
        return packs

    def get_grader_specs(
        self, 
        strategy: Strategy, 
        sub_strategies: list[SubStrategy]
    ) -> list[GraderSpec]:
        """Get grader specs for strategy and sub-strategies."""
        specs = []
        
        # Base spec for strategy
        base_spec = GraderSpec(
            grader_type=f"{strategy.value}_grader",  # or "llm_judge"
            config={"strategy": strategy.value},
        )
        specs.append(base_spec)
        
        # Additional specs for sub-strategies
        for sub_strategy in sub_strategies:
            spec = GraderSpec(
                grader_type=f"{sub_strategy.value}_grader",
                config={"sub_strategy": sub_strategy.value},
            )
            specs.append(spec)
        
        return specs

    def infer_strategy_from_findings(self, findings: list[str]) -> Strategy:
        """Infer primary strategy from list of findings."""
        # Simple heuristic - count strategy mentions
        strategy_counts: dict[Strategy, int] = {}
        
        for finding in findings:
            finding_lower = finding.lower()
            
            if any(word in finding_lower for word in ["tool", "efficiency", "selection"]):
                strategy_counts[Strategy.TOOL_QUALITY] = strategy_counts.get(Strategy.TOOL_QUALITY, 0) + 1
            elif any(word in finding_lower for word in ["instruction", "clarity", "skill"]):
                strategy_counts[Strategy.SKILL_QUALITY] = strategy_counts.get(Strategy.SKILL_QUALITY, 0) + 1
            elif any(word in finding_lower for word in ["policy", "rule", "safety"]):
                strategy_counts[Strategy.POLICY_QUALITY] = strategy_counts.get(Strategy.POLICY_QUALITY, 0) + 1
            elif any(word in finding_lower for word in ["harness", "grader", "calibration"]):
                strategy_counts[Strategy.HARNESS_QUALITY] = strategy_counts.get(Strategy.HARNESS_QUALITY, 0) + 1
        
        # Return most common strategy, default to TOOL_QUALITY
        if not strategy_counts:
            return Strategy.TOOL_QUALITY
        
        return max(strategy_counts, key=lambda k: strategy_counts[k])


__all__ = ["StrategyMapper"]
