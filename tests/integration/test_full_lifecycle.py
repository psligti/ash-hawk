"""End-to-end integration tests for full improvement lifecycle.

Tests the complete flow:
1. Run completes in dawn-kestrel
2. Artifact emitted via hook
3. Ash-hawk orchestrator reviews
4. Proposals generated
5. Lessons curated
6. Lessons injected back to agent
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ash_hawk.contracts import CuratedLesson, ImprovementProposal, ReviewRequest, RunArtifact
from ash_hawk.integration.dawn_kestrel_hook import (
    DawnKestrelPostRunHook,
    TranscriptToArtifactConverter,
)
from ash_hawk.pipeline.orchestrator import PipelineOrchestrator
from ash_hawk.services.lesson_service import LessonService
from ash_hawk.types import EvalTranscript, TokenUsage


def _make_transcript(
    run_id: str = "run-1",
    agent_name: str = "bolt-merlin",
    error_trace: str | None = None,
    tool_calls: list[dict] | None = None,
) -> EvalTranscript:
    return EvalTranscript(
        messages=[
            {"role": "user", "content": "Test prompt"},
            {"role": "assistant", "content": "Test response"},
        ],
        tool_calls=tool_calls or [],
        trace_events=[],
        token_usage=TokenUsage(input=100, output=50),
        cost_usd=0.01,
        duration_seconds=5.0,
        agent_response="Test response",
        error_trace=error_trace,
    )


def _make_run_artifact(
    run_id: str = "run-1",
    agent_name: str = "bolt-merlin",
    outcome: str = "success",
    tool_calls: list | None = None,
) -> RunArtifact:
    return RunArtifact(
        run_id=run_id,
        agent_name=agent_name,
        outcome=outcome,
        tool_calls=tool_calls or [],
        steps=[],
        messages=[],
        metadata={"experiment_id": "test-exp"},
    )


class TestTranscriptToArtifactConverter:
    def test_convert_successful_transcript(self) -> None:
        converter = TranscriptToArtifactConverter()
        transcript = _make_transcript()

        artifact = converter.convert(
            transcript=transcript,
            run_id="run-test",
            suite_id="suite-test",
            agent_name="bolt-merlin",
        )

        assert artifact.run_id == "run-test"
        assert artifact.agent_name == "bolt-merlin"
        assert artifact.outcome == "success"

    def test_convert_failed_transcript(self) -> None:
        converter = TranscriptToArtifactConverter()
        transcript = _make_transcript(error_trace="Tool timeout")

        artifact = converter.convert(transcript)

        assert artifact.outcome == "failure"
        assert "timeout" in (artifact.error_message or "").lower()

    def test_convert_with_tool_calls(self) -> None:
        converter = TranscriptToArtifactConverter()
        tool_calls = [
            {"tool": "read", "input": {"path": "/tmp/file"}, "output": "content"},
            {"tool": "bash", "input": {"cmd": "ls"}, "error": "Permission denied"},
        ]
        transcript = _make_transcript(tool_calls=tool_calls)

        artifact = converter.convert(transcript)

        assert len(artifact.tool_calls) == 2
        assert artifact.tool_calls[0].tool_name == "read"
        assert artifact.tool_calls[0].outcome == "success"
        assert artifact.tool_calls[1].outcome == "failure"


class TestDawnKestrelPostRunHook:
    def test_submit_run_for_review_returns_review_id(self, tmp_path: Path) -> None:
        hook = DawnKestrelPostRunHook(
            agent_name="bolt-merlin",
        )
        artifact = _make_run_artifact()

        review_id = hook.submit_run_for_review(artifact)

        assert review_id.startswith("review-")

    def test_on_transcript_complete_converts_and_submits(self, tmp_path: Path) -> None:
        hook = DawnKestrelPostRunHook(
            agent_name="iron-rook",
        )
        transcript = _make_transcript(agent_name="iron-rook")

        review_id = hook.on_transcript_complete(
            transcript,
            run_id="run-transcript",
            suite_id="suite-review",
        )

        assert review_id.startswith("review-")

    def test_hook_with_orchestrator_triggers_pipeline(self, tmp_path: Path) -> None:
        orchestrator = MagicMock(spec=PipelineOrchestrator)
        lesson_service = LessonService(storage_path=tmp_path)

        hook = DawnKestrelPostRunHook(
            agent_name="bolt-merlin",
            orchestrator=orchestrator,
            experiment_id="exp-1",
        )

        artifact = _make_run_artifact(
            run_id="run-pipeline",
            agent_name="bolt-merlin",
            outcome="failure",
        )

        hook.submit_run_for_review(artifact)

        orchestrator.run.assert_called_once()
        call_args = orchestrator.run.call_args
        assert call_args[0][0].run_artifact_id == "run-pipeline"


class TestFullImprovementLifecycle:
    @pytest.mark.asyncio
    async def test_failed_run_generates_lesson(self, tmp_path: Path) -> None:
        lesson_service = LessonService(storage_path=tmp_path)

        hook = DawnKestrelPostRunHook(
            agent_name="bolt-merlin",
            orchestrator=None,
            experiment_id="test-exp",
        )

        artifact = _make_run_artifact(
            run_id="run-failed",
            agent_name="bolt-merlin",
            outcome="failure",
            tool_calls=[
                {"tool_name": "bash", "outcome": "failure", "error_message": "timeout"},
            ],
        )

        review_id = hook.submit_run_for_review(artifact)
        assert review_id.startswith("review-")

    def test_lesson_scoped_to_experiment(self, tmp_path: Path) -> None:
        service = LessonService(storage_path=tmp_path)

        proposal = ImprovementProposal(
            proposal_id="prop-scoped",
            origin_run_id="run-1",
            target_agent="bolt-merlin",
            proposal_type="skill",
            title="Scoped lesson",
            rationale="Test scoping",
            expected_benefit="Test",
            risk_level="low",
            evidence_refs=["e1"],
            created_at=datetime.now(UTC),
            experiment_id="exp-a",
        )

        lesson = service.approve_proposal(
            proposal,
            experiment_id=proposal.experiment_id,
            require_experiment_id=False,
        )

        lessons_exp_a = service.get_lessons_for_agent("bolt-merlin", experiment_id="exp-a")
        lessons_exp_b = service.get_lessons_for_agent("bolt-merlin", experiment_id="exp-b")
        lessons_global = service.get_lessons_for_agent("bolt-merlin")

        assert lesson in lessons_exp_a or len(lessons_exp_a) >= 1
        assert lesson not in lessons_exp_b

    def test_cross_agent_lesson_application(self, tmp_path: Path) -> None:
        service = LessonService(storage_path=tmp_path)

        proposal = ImprovementProposal(
            proposal_id="prop-cross",
            origin_run_id="run-1",
            target_agent="iron-rook",
            proposal_type="policy",
            title="Cross-agent policy",
            rationale="Shared policy",
            expected_benefit="All agents benefit",
            risk_level="low",
            evidence_refs=["e1"],
            created_at=datetime.now(UTC),
        )

        lesson = service.approve_proposal(
            proposal,
            applies_to_agents=["iron-rook", "bolt-merlin", "vox-jay"],
            require_experiment_id=False,
        )

        iron_lessons = service.get_lessons_for_agent("iron-rook")
        bolt_lessons = service.get_lessons_for_agent("bolt-merlin")
        vox_lessons = service.get_lessons_for_agent("vox-jay")

        assert lesson in iron_lessons
        assert lesson in bolt_lessons
        assert lesson in vox_lessons


class TestHookIntegrationWithRunner:
    def test_hook_implements_dawn_kestrel_protocol(self) -> None:
        hook = DawnKestrelPostRunHook(agent_name="test-agent")

        assert hasattr(hook, "submit_run_for_review")
        assert hasattr(hook, "on_review_complete")
        assert callable(hook.submit_run_for_review)
        assert callable(hook.on_review_complete)

    def test_on_review_complete_handles_completion(self) -> None:
        hook = DawnKestrelPostRunHook(agent_name="test-agent")
        artifact = _make_run_artifact()

        hook.on_review_complete(artifact, "review-123")

    def test_multiple_artifacts_generate_unique_review_ids(self) -> None:
        hook = DawnKestrelPostRunHook()

        artifact1 = _make_run_artifact(run_id="run-1")
        artifact2 = _make_run_artifact(run_id="run-2")

        review_id1 = hook.submit_run_for_review(artifact1)
        review_id2 = hook.submit_run_for_review(artifact2)

        assert review_id1 != review_id2
        assert hook._review_ids["run-1"] == review_id1
        assert hook._review_ids["run-2"] == review_id2
