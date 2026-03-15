from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from ash_hawk.contracts import (
    CuratedLesson,
    ImprovementProposal,
    ReviewRequest,
    RunArtifact,
)
from ash_hawk.pipeline.types import PipelineContext, PipelineRole, PipelineStepResult

if TYPE_CHECKING:
    pass


class PipelineOrchestrator:
    """Orchestrates the improvement pipeline execution.

    Roles run in sequence:
    1. COMPETITOR: Re-attempts or replays (optional)
    2. ANALYST: Analyzes failures, generates findings
    3. COACH: Generates policy/playbook proposals
    4. ARCHITECT: Generates harness/tool proposals
    5. CURATOR: Approves/rejects proposals into lessons
    """

    def __init__(self) -> None:
        self._context: PipelineContext | None = None
        self._steps: dict[PipelineRole, PipelineStepResult] = {}
        self._proposals: list[ImprovementProposal] = []
        self._lessons: list[CuratedLesson] = []

    def run(self, request: ReviewRequest, artifact: RunArtifact) -> list[CuratedLesson]:
        """Execute the full improvement pipeline.

        Args:
            request: The review request to process.
            artifact: The run artifact to analyze.

        Returns:
            List of curated lessons from approved proposals.
        """
        review_id = f"review-{uuid4().hex[:8]}"
        self._context = PipelineContext(
            run_artifact_id=artifact.run_id,
            review_request_id=review_id,
            role=PipelineRole.COMPETITOR,
            target_agent=request.target_agent,
            inputs={"artifact": artifact.model_dump()},
        )

        self._initialize_steps()

        self._run_competitor_role(artifact)
        self._run_analyst_role(artifact)
        self._run_coach_role()
        self._run_architect_role()
        lessons = self._run_curator_role()

        return lessons

    def _initialize_steps(self) -> None:
        """Initialize step results for all roles."""
        now = datetime.now(UTC)
        for role in PipelineRole:
            self._steps[role] = PipelineStepResult(
                step_id=f"{role.value}-{uuid4().hex[:8]}",
                role=role,
                status="pending",
                started_at=now,
            )

    def _run_competitor_role(self, artifact: RunArtifact) -> None:
        """Run the competitor role (optional replay)."""
        step = self._steps[PipelineRole.COMPETITOR]
        step.status = "running"
        step.started_at = datetime.now(UTC)

        try:
            # Competitor role is optional - may replay or re-attempt
            # For now, just mark as completed without action
            step.status = "completed"
            step.completed_at = datetime.now(UTC)
        except Exception as e:
            step.status = "failed"
            step.error = str(e)

    def _run_analyst_role(self, artifact: RunArtifact) -> None:
        """Run the analyst role to analyze failures and generate findings."""
        from ash_hawk.pipeline.analyst import AnalystInput, AnalystRole

        step = self._steps[PipelineRole.ANALYST]
        step.status = "running"
        step.started_at = datetime.now(UTC)

        try:
            analyst = AnalystRole()
            input_data = AnalystInput()
            input_data.artifact = artifact
            input_data.focus_areas = []

            output = analyst.analyze(input_data)
            step.outputs = {
                "findings": [f.model_dump() for f in output.findings],
                "metrics": output.metrics.model_dump(),
                "tool_efficiency": output.tool_efficiency,
                "failure_patterns": output.failure_patterns,
                "risk_areas": output.risk_areas,
            }

            if self._context:
                self._context.outputs["analyst"] = step.outputs

            step.status = "completed"
            step.completed_at = datetime.now(UTC)
        except Exception as e:
            step.status = "failed"
            step.error = str(e)

    def _run_coach_role(self) -> None:
        """Run the coach role to generate policy/playbook proposals."""
        from ash_hawk.pipeline.coach import CoachRole

        step = self._steps[PipelineRole.COACH]
        step.status = "running"
        step.started_at = datetime.now(UTC)

        try:
            if not self._context:
                step.status = "skipped"
                return

            analyst_output = self._context.outputs.get("analyst", {})
            failure_patterns = analyst_output.get("failure_patterns", [])

            coach = CoachRole()
            proposals = coach.generate_proposals(self._context, failure_patterns)

            self._proposals.extend(proposals)
            step.outputs = {
                "proposal_ids": [p.proposal_id for p in proposals],
            }

            self._context.outputs["coach"] = step.outputs
            step.status = "completed"
            step.completed_at = datetime.now(UTC)
        except Exception as e:
            step.status = "failed"
            step.error = str(e)

    def _run_architect_role(self) -> None:
        from ash_hawk.pipeline.architect import ArchitectRole

        step = self._steps[PipelineRole.ARCHITECT]
        step.status = "running"
        step.started_at = datetime.now(UTC)

        try:
            if not self._context:
                step.status = "skipped"
                return

            analyst_output = self._context.outputs.get("analyst", {})
            failure_patterns = analyst_output.get("failure_patterns", [])
            risk_areas = analyst_output.get("risk_areas", [])
            findings = failure_patterns + risk_areas

            architect = ArchitectRole()
            proposals = architect.generate_proposals(self._context, findings)

            self._proposals.extend(proposals)
            step.outputs = {
                "proposal_ids": [p.proposal_id for p in proposals],
            }

            self._context.outputs["architect"] = step.outputs
            step.status = "completed"
            step.completed_at = datetime.now(UTC)
        except Exception as e:
            step.status = "failed"
            step.error = str(e)

    def _run_curator_role(self) -> list[CuratedLesson]:
        """Run the curator role to approve/reject proposals into lessons."""
        from ash_hawk.pipeline.curator import CuratorRole

        step = self._steps[PipelineRole.CURATOR]
        step.status = "running"
        step.started_at = datetime.now(UTC)

        try:
            curator = CuratorRole()
            lessons = curator.curate(self._proposals, auto_appro=True)

            self._lessons = lessons
            step.outputs = {
                "lesson_ids": [lesson.lesson_id for lesson in lessons],
                "approved_count": len(lessons),
                "rejected_count": len(self._proposals) - len(lessons),
            }

            if self._context:
                self._context.outputs["curator"] = step.outputs

            step.status = "completed"
            step.completed_at = datetime.now(UTC)
        except Exception as e:
            step.status = "failed"
            step.error = str(e)

        return self._lessons

    def get_step_result(self, role: PipelineRole) -> PipelineStepResult | None:
        """Get the result for a specific role."""
        return self._steps.get(role)

    def get_all_steps(self) -> dict[PipelineRole, PipelineStepResult]:
        """Get all step results."""
        return self._steps.copy()

    def get_proposals(self) -> list[ImprovementProposal]:
        """Get all generated proposals."""
        return self._proposals.copy()

    def get_lessons(self) -> list[CuratedLesson]:
        """Get all curated lessons."""
        return self._lessons.copy()
