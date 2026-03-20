"""Review request contract for evaluating completed agent runs."""

from __future__ import annotations

from typing import Literal

import pydantic as pd

from ash_hawk.strategies import SubStrategy


class ReviewRequest(pd.BaseModel):
    """Request to evaluate a completed agent run.

    Produced by orchestration layer and consumed by Ash Hawk's
    review service to generate improvement proposals.

    Attributes:
        run_artifact_id: ID of the run artifact to review.
        target_agent: Agent being evaluated (e.g., "iron-rook", "bolt-merlin").
        eval_suite: List of evaluator names to run.
        review_mode: Depth of review (quick, standard, deep).
        persistence_mode: What to persist (none, propose, curate).
        focus_areas: Specific areas to focus evaluation on.
        baseline_run_id: Optional baseline run ID for comparison.
        experiment_id: ID of the experiment for parallel trial isolation.
        variant: A/B test variant identifier (e.g., 'control', 'treatment-a').
        sub_strategy_focus: Specific sub-strategies to target for improvement.
    """

    run_artifact_id: str = pd.Field(
        description="ID of the run artifact to review",
    )
    target_agent: str = pd.Field(
        description="Agent being evaluated (e.g., 'iron-rook', 'bolt-merlin')",
    )
    eval_suite: list[str] = pd.Field(
        default_factory=list,
        description="List of evaluator names to run",
    )
    review_mode: Literal["quick", "standard", "deep"] = pd.Field(
        default="standard",
        description="Depth of review (quick: surface analysis, standard: full pipeline, deep: exhaustive)",
    )
    persistence_mode: Literal["none", "propose", "curate"] = pd.Field(
        default="propose",
        description="What to persist (none: transient, propose: generate proposals, curate: require approval)",
    )
    focus_areas: list[str] = pd.Field(
        default_factory=list,
        description="Specific areas to focus evaluation on (e.g., 'tool_efficiency', 'evidence_quality')",
    )
    baseline_run_id: str | None = pd.Field(
        default=None,
        description="Optional baseline run ID for before/after comparison",
    )
    experiment_id: str | None = pd.Field(
        default=None,
        description="ID of the experiment for parallel trial isolation",
    )
    variant: str | None = pd.Field(
        default=None,
        description="A/B test variant identifier (e.g., 'control', 'treatment-a')",
    )
    sub_strategy_focus: list[SubStrategy] = pd.Field(
        default_factory=list,
        description="Specific sub-strategies to target for improvement (e.g., [SubStrategy.TOOL_EFFICIENCY])",
    )

    model_config = pd.ConfigDict(extra="forbid")
