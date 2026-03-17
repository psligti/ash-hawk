from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ash_hawk.contracts import (
    CuratedLesson,
    ImprovementProposal,
    ReviewRequest,
    RunArtifact,
)
from ash_hawk.pipeline.types import PipelineContext, PipelineRole, PipelineStepResult

if TYPE_CHECKING:
    from ash_hawk.integration.post_run_hook import PostRunReviewHook

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the improvement pipeline execution.

    Roles run in sequence:
    1. COMPETITOR: Re-attempts or replays (optional)
    2. ANALYST: Analyzes failures, generates findings
    3. COACH: Generates policy/playbook proposals
    4. ARCHITECT: Generates harness/tool proposals
    5. CURATOR: Approves/rejects proposals into lessons
    """

    def __init__(self, hook: "PostRunReviewHook | None" = None) -> None:
        self._hook = hook
        self._context: PipelineContext | None = None
        self._steps: dict[PipelineRole, PipelineStepResult] = {}
        self._proposals: list[ImprovementProposal] = []
        self._lessons: list[CuratedLesson] = []
        self._comparison_result: dict[str, Any] | None = None

    def run(self, request: ReviewRequest, artifact: RunArtifact) -> list[CuratedLesson]:
        review_id = f"review-{uuid4().hex[:8]}"
        self._context = PipelineContext(
            run_artifact_id=artifact.run_id,
            review_request_id=review_id,
            role=PipelineRole.COMPETITOR,
            target_agent=request.target_agent,
            experiment_id=request.experiment_id,
            inputs={"artifact": artifact.model_dump()},
        )

        self._initialize_steps()

        self._run_competitor_role(artifact)
        self._run_analyst_role(artifact)
        self._run_coach_role()
        self._run_architect_role()
        lessons = self._run_curator_role()

        self._run_comparison_step(request, artifact, lessons)
        self._notify_hook(artifact)

        return lessons

    def _run_comparison_step(
        self,
        request: ReviewRequest,
        artifact: RunArtifact,
        lessons: list[CuratedLesson],
    ) -> None:
        """Run before/after comparison if baseline is provided."""
        if not request.baseline_run_id:
            return

        try:
            from ash_hawk.services.comparison_service import ComparisonService
            from ash_hawk.storage import FileStorage

            storage_path = Path(".ash-hawk")
            storage = FileStorage(str(storage_path))

            baseline_artifact = storage.load_run_artifact(request.baseline_run_id)
            if not baseline_artifact:
                logger.warning(f"Baseline run {request.baseline_run_id} not found")
                return

            lesson_ids = [lesson.lesson_id for lesson in lessons]
            comparison = ComparisonService().compare(
                baseline=baseline_artifact,
                treatment=artifact,
                lessons_applied=lesson_ids,
            )

            self._comparison_result = comparison.model_dump()
            if self._context:
                self._context.outputs["comparison"] = self._comparison_result

            curator_step = self._steps.get(PipelineRole.CURATOR)
            if curator_step:
                curator_step.outputs["comparison"] = self._comparison_result

            logger.info(f"Comparison complete: score_delta={comparison.metrics.score_delta:.3f}")
        except Exception as e:
            logger.error(f"Comparison step failed: {e}")

    def _notify_hook(self, artifact: RunArtifact) -> None:
        """Notify post-run hook if configured."""
        if not self._hook:
            return

        try:
            review_id = self._context.review_request_id if self._context else "unknown"
            self._hook.on_review_complete(artifact, review_id)
            logger.info(f"Hook notified for review {review_id}")
        except Exception as e:
            logger.warning(f"Hook notification failed (non-blocking): {e}")

    def run_with_experiment(
        self,
        request: ReviewRequest,
        artifact: RunArtifact,
        experiment_config: dict[str, Any],
    ) -> list[CuratedLesson]:
        """Run pipeline with experiment tracking.

        Creates experiment in registry, runs pipeline, stores lessons
        in experiment-scoped storage, and updates experiment status.
        """
        from ash_hawk.curation.experiment_store import ExperimentStore
        from ash_hawk.experiments.registry import ExperimentRegistry

        registry = ExperimentRegistry()
        experiment = registry.get_or_create(request.experiment_id or "default", experiment_config)

        registry.increment_trial_count(experiment.experiment_id)

        lessons = self.run(request, artifact)

        if lessons:
            exp_store = ExperimentStore()
            for lesson in lessons:
                exp_store.store(lesson, experiment.experiment_id)

            registry.increment_lesson_count(experiment.experiment_id, len(lessons))

        registry.update_status(experiment.experiment_id, "completed")

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
        from ash_hawk.pipeline.competitor import CompetitorInput, CompetitorRole
        from ash_hawk.services.lesson_service import LessonService

        step = self._steps[PipelineRole.COMPETITOR]
        step.status = "running"
        step.started_at = datetime.now(UTC)

        try:
            if not hasattr(artifact, "is_successful") or artifact.is_successful():
                step.status = "completed"
                step.outputs = {"reason": "Artifact not suitable for replay or already successful"}
                step.completed_at = datetime.now(UTC)
                return

            experiment_id = self._context.experiment_id if self._context else None
            lesson_service = LessonService()
            available_lessons = lesson_service.get_lessons_for_agent(
                getattr(artifact, "agent_name", "unknown"),
                experiment_id=experiment_id,
            )

            competitor = CompetitorRole()

            input_data = CompetitorInput(
                artifact=artifact,
                lessons_to_apply=available_lessons,
            )

            output = competitor.compete(input_data)

            step.outputs = {
                "replay_artifact_id": output.replay_artifact.run_id
                if output.replay_artifact
                else None,
                "improvement_achieved": output.improvement_achieved,
                "comparison_score_delta": output.comparison.metrics.score_delta
                if output.comparison
                else None,
                "lessons_applied": [lesson.lesson_id for lesson in available_lessons]
                if available_lessons
                else [],
                "findings_count": len(output.findings),
                "experiment_id": experiment_id,
            }

            if self._context:
                self._context.outputs["competitor"] = step.outputs
                if output.replay_artifact:
                    self._context.outputs["replay_artifact"] = output.replay_artifact.model_dump()
                if output.comparison:
                    self._context.outputs["comparison"] = output.comparison.model_dump()

            step.status = "completed"
            step.completed_at = datetime.now(UTC)
        except Exception as e:
            step.status = "completed"
            step.error = str(e)
            step.outputs = {"error": str(e), "skipped": True}

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
