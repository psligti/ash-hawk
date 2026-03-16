"""Base classes for Evaluator Packs."""

from __future__ import annotations

from typing import Any

import pydantic as pd

from ash_hawk.strategies import Strategy


class EvalPackConfig(pd.BaseModel):
    """Configuration for a single grader within an eval pack.

    Attributes:
        grader_name: Name of the grader to use.
        weight: Weight for this grader in composite scoring (0.0-1.0).
        pass_threshold: Minimum score to consider this grader passed.
        params: Additional parameters passed to the grader.
    """

    grader_name: str = pd.Field(description="Name of the grader to use")
    weight: float = pd.Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Weight for this grader in composite scoring",
    )
    pass_threshold: float = pd.Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum score to consider this grader passed",
    )
    params: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Additional parameters passed to the grader",
    )

    model_config = pd.ConfigDict(extra="forbid")


class EvalPack(pd.BaseModel):
    """Pre-configured bundle of graders and settings for agent evaluation.

    EvalPacks define a standardized evaluation configuration that can be
    applied consistently across runs and agents. Each pack specifies:
    - Which graders to use and how to configure them
    - Pass/fail thresholds for each grader
    - Strategy focus areas for improvement tracking
    - Target agents this pack is designed for

    Attributes:
        pack_id: Unique identifier for this pack.
        name: Human-readable name.
        description: Detailed description of the pack's purpose.
        target_agents: List of agent names this pack is designed for.
        grader_configs: Mapping of grader name to its configuration.
        strategy_focus: List of improvement strategies this pack addresses.
        global_pass_threshold: Overall pass threshold for the pack.
        metadata: Additional metadata about the pack.
    """

    pack_id: str = pd.Field(description="Unique identifier for this pack")
    name: str = pd.Field(description="Human-readable name")
    description: str = pd.Field(description="Detailed description of the pack's purpose")
    target_agents: list[str] = pd.Field(
        default_factory=list,
        description="List of agent names this pack is designed for",
    )
    grader_configs: dict[str, EvalPackConfig] = pd.Field(
        default_factory=dict,
        description="Mapping of grader name to its configuration",
    )
    strategy_focus: list[Strategy] = pd.Field(
        default_factory=list,
        description="List of improvement strategies this pack addresses",
    )
    global_pass_threshold: float = pd.Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Overall pass threshold for the pack",
    )
    metadata: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="Additional metadata about the pack",
    )

    model_config = pd.ConfigDict(extra="forbid")

    def get_grader_names(self) -> list[str]:
        """Get list of grader names in this pack."""
        return list(self.grader_configs.keys())

    def get_total_weight(self) -> float:
        """Get sum of all grader weights."""
        return sum(config.weight for config in self.grader_configs.values())

    def is_for_agent(self, agent_name: str) -> bool:
        """Check if this pack applies to a given agent.

        Args:
            agent_name: Name of the agent to check.

        Returns:
            True if the pack targets this agent (or all agents).
        """
        if not self.target_agents:
            return True  # Empty target_agents means applies to all
        return agent_name in self.target_agents

    def compute_weighted_score(self, scores: dict[str, float]) -> float:
        """Compute weighted average score from individual grader scores.

        Args:
            scores: Mapping of grader name to its score.

        Returns:
            Weighted average score (0.0-1.0).
        """
        if not self.grader_configs:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for grader_name, config in self.grader_configs.items():
            if grader_name in scores:
                weighted_sum += scores[grader_name] * config.weight
                total_weight += config.weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def evaluate_pass(self, scores: dict[str, float]) -> bool:
        """Evaluate whether the overall result passes.

        Args:
            scores: Mapping of grader name to its score.

        Returns:
            True if the weighted score meets the global pass threshold.
        """
        weighted_score = self.compute_weighted_score(scores)
        return weighted_score >= self.global_pass_threshold
