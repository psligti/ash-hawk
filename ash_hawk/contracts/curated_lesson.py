"""Curated lesson contract for approved behavioral improvements."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import pydantic as pd


class CuratedLesson(pd.BaseModel):
    """Approved and versioned lesson for persistence.

    Lessons are created from approved ImprovementProposals and represent
    stable, validated behavioral improvements that can be injected into
    future agent runs.

    Attributes:
        lesson_id: Unique identifier for this lesson.
        source_proposal_id: ID of the ImprovementProposal this lesson derives from.
        applies_to_agents: List of agents this lesson applies to.
        lesson_type: Type of lesson (policy, skill, tool, harness, eval).
        title: Short title describing the lesson.
        description: Detailed description of the lesson.
        lesson_payload: The actual lesson content (format depends on type).
        validation_status: Current validation status.
        version: Version number for this lesson (incremented on updates).
        parent_lesson_id: Parent lesson ID if this is an update.
        rollback_of: Lesson ID this rolls back (if applicable).
        evidence_summary: Summary of evidence supporting this lesson.
        impact_metrics: Metrics showing impact of this lesson.
        created_at: When the lesson was created.
        updated_at: When the lesson was last updated.
        applied_at: When the lesson was first applied to a run.
    """

    lesson_id: str = pd.Field(
        description="Unique identifier for this lesson",
    )
    source_proposal_id: str = pd.Field(
        description="ID of the ImprovementProposal this lesson derives from",
    )
    applies_to_agents: list[str] = pd.Field(
        default_factory=list,
        description="List of agents this lesson applies to",
    )
    lesson_type: Literal["policy", "skill", "tool", "harness", "eval"] = pd.Field(
        description="Type of lesson",
    )
    title: str = pd.Field(
        description="Short title describing the lesson",
    )
    description: str = pd.Field(
        description="Detailed description of the lesson",
    )
    lesson_payload: dict[str, Any] = pd.Field(
        default_factory=dict,
        description="The actual lesson content (format depends on type)",
    )
    validation_status: Literal["approved", "deprecated", "rolled_back"] = pd.Field(
        default="approved",
        description="Current validation status",
    )
    version: int = pd.Field(
        default=1,
        ge=1,
        description="Version number for this lesson",
    )
    parent_lesson_id: str | None = pd.Field(
        default=None,
        description="Parent lesson ID if this is an update",
    )
    rollback_of: str | None = pd.Field(
        default=None,
        description="Lesson ID this rolls back",
    )
    evidence_summary: str | None = pd.Field(
        default=None,
        description="Summary of evidence supporting this lesson",
    )
    impact_metrics: dict[str, float] = pd.Field(
        default_factory=dict,
        description="Metrics showing impact of this lesson",
    )
    created_at: datetime = pd.Field(
        description="When the lesson was created",
    )
    updated_at: datetime | None = pd.Field(
        default=None,
        description="When the lesson was last updated",
    )
    applied_at: datetime | None = pd.Field(
        default=None,
        description="When the lesson was first applied to a run",
    )

    def is_active(self) -> bool:
        """Check if this lesson is currently active and applicable."""
        return self.validation_status == "approved"

    def create_new_version(self, new_payload: dict[str, Any]) -> CuratedLesson:
        """Create a new version of this lesson with updated payload."""
        return CuratedLesson(
            lesson_id=f"{self.lesson_id}-v{self.version + 1}",
            source_proposal_id=self.source_proposal_id,
            applies_to_agents=self.applies_to_agents,
            lesson_type=self.lesson_type,
            title=self.title,
            description=self.description,
            lesson_payload=new_payload,
            validation_status="approved",
            version=self.version + 1,
            parent_lesson_id=self.lesson_id,
            created_at=datetime.now(),
        )

    model_config = pd.ConfigDict(extra="forbid")
