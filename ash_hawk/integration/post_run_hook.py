"""Post-run review hook base classes and default implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import pydantic as pd

if TYPE_CHECKING:
    from ash_hawk.contracts import RunArtifact


class HookConfig(pd.BaseModel):
    """Configuration for post-run review hooks.

    Attributes:
        enabled: Whether the hook is active.
        review_mode: Depth of review to perform.
        review_on_success: Whether to review successful runs.
        min_score_for_review: Minimum score to skip review (lower scores trigger review).
    """

    enabled: bool = pd.Field(default=True, description="Whether the hook is active")
    review_mode: Literal["quick", "standard", "deep"] = pd.Field(
        default="standard",
        description="Depth of review to perform",
    )
    review_on_success: bool = pd.Field(
        default=False,
        description="Whether to review successful runs",
    )
    min_score_for_review: float = pd.Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Runs scoring below this trigger review",
    )

    model_config = pd.ConfigDict(extra="forbid")


class PostRunReviewHook(ABC):
    """Abstract base class for post-run review hooks.

    Hooks are called after an agent run completes and can trigger
    the improvement pipeline based on configurable criteria.
    """

    @abstractmethod
    def on_run_complete(self, run_artifact: RunArtifact) -> None:
        """Called when a run completes.

        Args:
            run_artifact: The completed run artifact to potentially review.
        """
        ...

    @abstractmethod
    def should_review(self, run_artifact: RunArtifact) -> bool:
        """Determine if a run should be reviewed.

        Args:
            run_artifact: The run artifact to check.

        Returns:
            True if the run should trigger a review.
        """
        ...

    @abstractmethod
    def on_review_complete(self, run_artifact: RunArtifact, review_id: str) -> None:
        """Called when review completes.

        Args:
            run_artifact: The run artifact that was reviewed.
            review_id: ID of the completed review.
        """
        ...


class DefaultPostRunReviewHook(PostRunReviewHook):
    """Default implementation of post-run review hook.

    Triggers reviews based on:
    - Run outcome (failed runs always reviewed if enabled)
    - Score threshold (runs below min_score_for_review)
    - Configuration settings
    """

    def __init__(self, config: HookConfig | None = None) -> None:
        self._config = config or HookConfig()

    @property
    def config(self) -> HookConfig:
        return self._config

    def should_review(self, run_artifact: RunArtifact) -> bool:
        if not self._config.enabled:
            return False

        is_successful = run_artifact.is_successful()

        if not is_successful:
            return True

        if not self._config.review_on_success:
            return False

        score = self._extract_score(run_artifact)
        return score < self._config.min_score_for_review

    def on_run_complete(self, run_artifact: RunArtifact) -> None:
        if not self.should_review(run_artifact):
            return

        self._trigger_review(run_artifact)

    def _extract_score(self, run_artifact: RunArtifact) -> float:
        metadata = run_artifact.metadata or {}
        return float(metadata.get("score", 0.0))

    def _trigger_review(self, run_artifact: RunArtifact) -> None:
        from ash_hawk.contracts import ReviewRequest
        from ash_hawk.services.review_service import ReviewService

        service = ReviewService()
        request = ReviewRequest(
            run_artifact_id=run_artifact.run_id,
            target_agent=run_artifact.agent_name,
            eval_suite=[],
            review_mode=self._config.review_mode,
            persistence_mode="propose",
        )
        service.review(request, run_artifact)
