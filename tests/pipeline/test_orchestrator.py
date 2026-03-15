"""Tests for PipelineOrchestrator."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ash_hawk.contracts import CuratedLesson, ImprovementProposal, ReviewRequest
from ash_hawk.pipeline.orchestrator import PipelineOrchestrator
from ash_hawk.pipeline.types import PipelineRole

from tests.pipeline.conftest import MockRunArtifact


class TestPipelineOrchestratorInit:
    def test_init_creates_empty_state(self):
        orchestrator = PipelineOrchestrator()
        assert orchestrator.get_proposals() == []
        assert orchestrator.get_lessons() == []
        assert orchestrator.get_all_steps() == {}


class TestPipelineOrchestratorRun:
    def test_run_initializes_context(
        self, review_request: ReviewRequest, mock_run_artifact: MockRunArtifact
    ):
        orchestrator = PipelineOrchestrator()
        orchestrator.run(review_request, mock_run_artifact)

        steps = orchestrator.get_all_steps()
        assert len(steps) == 5
        for role in PipelineRole:
            assert role in steps

    def test_run_initializes_all_steps(
        self, review_request: ReviewRequest, mock_run_artifact: MockRunArtifact
    ):
        orchestrator = PipelineOrchestrator()
        orchestrator.run(review_request, mock_run_artifact)

        steps = orchestrator.get_all_steps()
        for role in PipelineRole:
            step = steps[role]
            assert step.status in ("completed", "pending", "running", "skipped", "failed")
            assert step.role == role

    def test_run_returns_lessons(
        self, review_request: ReviewRequest, mock_run_artifact: MockRunArtifact
    ):
        orchestrator = PipelineOrchestrator()
        lessons = orchestrator.run(review_request, mock_run_artifact)

        assert isinstance(lessons, list)
        for lesson in lessons:
            assert isinstance(lesson, CuratedLesson)

    def test_run_competitor_step_completes(
        self, review_request: ReviewRequest, mock_run_artifact: MockRunArtifact
    ):
        orchestrator = PipelineOrchestrator()
        orchestrator.run(review_request, mock_run_artifact)

        step = orchestrator.get_step_result(PipelineRole.COMPETITOR)
        assert step is not None
        assert step.status == "completed"

    def test_run_analyst_step_completes(
        self, review_request: ReviewRequest, mock_run_artifact: MockRunArtifact
    ):
        orchestrator = PipelineOrchestrator()
        orchestrator.run(review_request, mock_run_artifact)

        step = orchestrator.get_step_result(PipelineRole.ANALYST)
        assert step is not None
        assert step.status == "completed"

    def test_run_curator_step_processes_proposals(
        self, review_request: ReviewRequest, mock_run_artifact: MockRunArtifact
    ):
        orchestrator = PipelineOrchestrator()
        orchestrator.run(review_request, mock_run_artifact)

        step = orchestrator.get_step_result(PipelineRole.CURATOR)
        assert step is not None
        assert "lesson_ids" in step.outputs or step.status in ("completed", "skipped")


class TestPipelineOrchestratorWithFailedRun:
    def test_run_processes_failed_artifact(
        self, review_request: ReviewRequest, mock_failed_run_artifact: MockRunArtifact
    ):
        orchestrator = PipelineOrchestrator()
        lessons = orchestrator.run(review_request, mock_failed_run_artifact)

        assert isinstance(lessons, list)
        step = orchestrator.get_step_result(PipelineRole.ANALYST)
        assert step is not None
        assert step.status == "completed"

    def test_run_generates_findings_for_failures(
        self, review_request: ReviewRequest, mock_failed_run_artifact: MockRunArtifact
    ):
        orchestrator = PipelineOrchestrator()
        orchestrator.run(review_request, mock_failed_run_artifact)

        step = orchestrator.get_step_result(PipelineRole.ANALYST)
        assert step is not None
        assert step.outputs is not None
        assert "failure_patterns" in step.outputs


class TestPipelineOrchestratorStepResults:
    def test_get_step_result_returns_none_for_unset_role(self):
        orchestrator = PipelineOrchestrator()
        result = orchestrator.get_step_result(PipelineRole.ANALYST)
        assert result is None

    def test_get_all_steps_returns_copy(
        self, review_request: ReviewRequest, mock_run_artifact: MockRunArtifact
    ):
        orchestrator = PipelineOrchestrator()
        orchestrator.run(review_request, mock_run_artifact)

        steps1 = orchestrator.get_all_steps()
        steps2 = orchestrator.get_all_steps()

        assert steps1 == steps2
        assert steps1 is not steps2

    def test_get_proposals_returns_copy(
        self, review_request: ReviewRequest, mock_failed_run_artifact: MockRunArtifact
    ):
        orchestrator = PipelineOrchestrator()
        orchestrator.run(review_request, mock_failed_run_artifact)

        proposals1 = orchestrator.get_proposals()
        proposals2 = orchestrator.get_proposals()

        assert proposals1 is not proposals2

    def test_get_lessons_returns_copy(
        self, review_request: ReviewRequest, mock_run_artifact: MockRunArtifact
    ):
        orchestrator = PipelineOrchestrator()
        orchestrator.run(review_request, mock_run_artifact)

        lessons1 = orchestrator.get_lessons()
        lessons2 = orchestrator.get_lessons()

        assert lessons1 is not lessons2


class TestPipelineOrchestratorEdgeCases:
    def test_run_with_empty_tool_calls(self, review_request: ReviewRequest):
        artifact = MockRunArtifact(
            run_id="run-empty-001",
            outcome="success",
            tool_calls=[],
        )
        orchestrator = PipelineOrchestrator()
        lessons = orchestrator.run(review_request, artifact)

        assert isinstance(lessons, list)

    def test_run_with_permission_error(self, review_request: ReviewRequest):
        from tests.pipeline.conftest import MockToolCall

        artifact = MockRunArtifact(
            run_id="run-perm-001",
            outcome="failure",
            tool_calls=[
                MockToolCall(
                    tool_name="delete",
                    outcome="failure",
                    error_message="Permission denied",
                ),
            ],
        )
        orchestrator = PipelineOrchestrator()
        lessons = orchestrator.run(review_request, artifact)

        assert isinstance(lessons, list)
        step = orchestrator.get_step_result(PipelineRole.ANALYST)
        assert step is not None
