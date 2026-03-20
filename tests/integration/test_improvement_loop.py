from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ash_hawk.contracts import (
    CuratedLesson,
    ImprovementProposal,
    RunArtifact,
    ToolCallRecord,
)
from ash_hawk.services.lesson_injector import LessonInjector
from ash_hawk.services.lesson_service import LessonService


class TestImprovementLoopIntegration:
    @pytest.fixture
    def temp_storage(self) -> Path:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def lesson_service(self, temp_storage: Path) -> LessonService:
        return LessonService(storage_path=temp_storage / "lessons")

    @pytest.fixture
    def lesson_injector(self, lesson_service: LessonService) -> LessonInjector:
        return LessonInjector(lesson_service=lesson_service)

    @pytest.fixture
    def sample_artifact(self) -> RunArtifact:
        return RunArtifact(
            run_id="run-001",
            suite_id="suite-test",
            agent_name="iron-rook",
            outcome="failure",
            tool_calls=[
                ToolCallRecord(
                    tool_name="read",
                    outcome="success",
                    input_args={"file_path": "/test/file.py"},
                    output="content",
                ),
            ],
            steps=[],
            messages=[],
            total_duration_ms=1000,
            token_usage={"input": 100, "output": 50},
            metadata={"task_type": "pr-review"},
            created_at=datetime.now(UTC),
        )

    @pytest.fixture
    def sample_proposal(self) -> ImprovementProposal:
        return ImprovementProposal(
            proposal_id="prop-001",
            origin_run_id="run-001",
            origin_review_id=None,
            target_agent="iron-rook",
            proposal_type="skill",
            title="Add file existence check before read",
            rationale="Agents should check file exists before reading",
            evidence_refs=[],
            expected_benefit="Reduce file not found errors",
            risk_level="low",
            diff_payload={
                "skill_name": "file_validation",
                "instruction_additions": ["Always check file exists with glob before reading"],
            },
            status="pending",
            created_at=datetime.now(UTC),
        )

    def test_lesson_service_approve_proposal(
        self,
        lesson_service: LessonService,
        sample_proposal: ImprovementProposal,
    ) -> None:
        lesson = lesson_service.approve_proposal(
            proposal=sample_proposal,
            applies_to_agents=["iron-rook", "bolt-merlin"],
            experiment_id="exp-test-001",
            require_experiment_id=False,
        )

        assert lesson.lesson_id == "lesson-prop-001"
        assert lesson.source_proposal_id == "prop-001"
        assert "iron-rook" in lesson.applies_to_agents
        assert "bolt-merlin" in lesson.applies_to_agents
        assert lesson.lesson_type == "skill"
        assert lesson.validation_status == "approved"

    def test_lesson_service_get_lessons_for_agent(
        self,
        lesson_service: LessonService,
        sample_proposal: ImprovementProposal,
    ) -> None:
        lesson_service.approve_proposal(
            proposal=sample_proposal,
            applies_to_agents=["iron-rook"],
            experiment_id="exp-test-001",
            require_experiment_id=False,
        )

        lessons = lesson_service.get_lessons_for_agent("iron-rook")
        assert len(lessons) == 1
        assert lessons[0].lesson_type == "skill"

        lessons_other = lesson_service.get_lessons_for_agent("bolt-merlin")
        assert len(lessons_other) == 0

    def test_lesson_injector_inject_into_prompt(
        self,
        lesson_service: LessonService,
        lesson_injector: LessonInjector,
        sample_proposal: ImprovementProposal,
    ) -> None:
        lesson_service.approve_proposal(
            proposal=sample_proposal,
            applies_to_agents=["iron-rook"],
            experiment_id="exp-test-001",
            require_experiment_id=False,
        )

        base_prompt = "You are a PR reviewer."
        augmented = lesson_injector.inject_into_prompt("iron-rook", base_prompt)

        assert "Learned Lessons" in augmented
        assert "Always check file exists" in augmented

    def test_lesson_injector_get_tool_overrides(
        self,
        lesson_service: LessonService,
        lesson_injector: LessonInjector,
    ) -> None:
        tool_proposal = ImprovementProposal(
            proposal_id="prop-tool-001",
            origin_run_id="run-001",
            origin_review_id=None,
            target_agent="iron-rook",
            proposal_type="tool",
            title="Set read timeout",
            rationale="Prevent hangs on large files",
            evidence_refs=[],
            expected_benefit="Faster response times",
            risk_level="low",
            diff_payload={
                "tool_id": "read",
                "parameter_defaults": {"timeout": 30},
                "usage_hints": ["Use chunked reading for files > 1MB"],
                "timeout_override": 60,
                "preconditions": [],
            },
            status="pending",
            created_at=datetime.now(UTC),
        )

        lesson_service.approve_proposal(
            proposal=tool_proposal,
            applies_to_agents=["iron-rook"],
            experiment_id="exp-test-001",
            require_experiment_id=False,
        )

        overrides = lesson_injector.get_tool_overrides("iron-rook")
        assert "read" in overrides
        assert overrides["read"]["defaults"]["timeout"] == 30
        assert overrides["read"]["timeout"] == 60

    def test_lesson_injector_get_harness_adjustments(
        self,
        lesson_service: LessonService,
        lesson_injector: LessonInjector,
    ) -> None:
        harness_proposal = ImprovementProposal(
            proposal_id="prop-harness-001",
            origin_run_id="run-001",
            origin_review_id=None,
            target_agent="iron-rook",
            proposal_type="harness",
            title="Increase timeout for security reviews",
            rationale="Security scans take longer",
            evidence_refs=[],
            expected_benefit="More thorough security analysis",
            risk_level="low",
            diff_payload={
                "grader_adjustments": {"security": {"weight": 1.5}},
                "fixture_overrides": {},
                "timeout_adjustments": {"security": 120},
                "parallelism_override": 2,
            },
            status="pending",
            created_at=datetime.now(UTC),
        )

        lesson_service.approve_proposal(
            proposal=harness_proposal,
            applies_to_agents=["iron-rook"],
            experiment_id="exp-test-001",
            require_experiment_id=False,
        )

        adjustments = lesson_injector.get_harness_adjustments("iron-rook")
        assert adjustments["grader_adjustments"]["security"]["weight"] == 1.5
        assert adjustments["timeout_adjustments"]["security"] == 120
        assert adjustments["parallelism"] == 2

    def test_full_loop_proposal_to_injection(
        self,
        lesson_service: LessonService,
        lesson_injector: LessonInjector,
        sample_proposal: ImprovementProposal,
    ) -> None:
        lesson = lesson_service.approve_proposal(
            proposal=sample_proposal,
            applies_to_agents=["iron-rook"],
            experiment_id="exp-test-001",
            require_experiment_id=False,
        )

        assert lesson.is_active()

        base_prompt = "Review the PR changes."
        augmented = lesson_injector.inject_into_prompt("iron-rook", base_prompt)

        assert len(augmented) > len(base_prompt)
        assert "check file exists" in augmented.lower()

    def test_lesson_deactivation(
        self,
        lesson_service: LessonService,
        sample_proposal: ImprovementProposal,
    ) -> None:
        lesson = lesson_service.approve_proposal(
            proposal=sample_proposal,
            applies_to_agents=["iron-rook"],
            experiment_id="exp-test-001",
            require_experiment_id=False,
        )

        assert lesson.validation_status == "approved"

        deactivated = lesson_service.deactivate_lesson(lesson.lesson_id)
        assert deactivated is not None
        assert deactivated.validation_status == "rolled_back"

        lessons = lesson_service.get_lessons_for_agent("iron-rook")
        active_lessons = [l for l in lessons if l.is_active()]
        assert len(active_lessons) == 0

    def test_experiment_isolation(
        self,
        lesson_service: LessonService,
        sample_proposal: ImprovementProposal,
    ) -> None:
        lesson_service.approve_proposal(
            proposal=sample_proposal,
            applies_to_agents=["iron-rook"],
            experiment_id="exp-alpha",
            require_experiment_id=False,
        )

        proposal2 = ImprovementProposal(
            proposal_id="prop-002",
            origin_run_id="run-002",
            origin_review_id=None,
            target_agent="iron-rook",
            proposal_type="skill",
            title="Another lesson",
            rationale="Different experiment",
            evidence_refs=[],
            expected_benefit="Test isolation",
            risk_level="low",
            diff_payload={
                "skill_name": "test_skill",
                "instruction_additions": ["Second lesson"],
            },
            status="pending",
            created_at=datetime.now(UTC),
        )
        lesson_service.approve_proposal(
            proposal=proposal2,
            applies_to_agents=["iron-rook"],
            experiment_id="exp-beta",
            require_experiment_id=False,
        )

        alpha_lessons = lesson_service.get_lessons_for_agent("iron-rook", experiment_id="exp-alpha")
        beta_lessons = lesson_service.get_lessons_for_agent("iron-rook", experiment_id="exp-beta")

        assert len(alpha_lessons) == 1
        assert alpha_lessons[0].source_proposal_id == "prop-001"

        assert len(beta_lessons) == 1
        assert beta_lessons[0].source_proposal_id == "prop-002"

    def test_cross_agent_lesson_sharing(
        self,
        lesson_service: LessonService,
        lesson_injector: LessonInjector,
    ) -> None:
        cross_agent_proposal = ImprovementProposal(
            proposal_id="prop-cross-001",
            origin_run_id="run-001",
            origin_review_id=None,
            target_agent="iron-rook",
            proposal_type="skill",
            title="Universal file handling",
            rationale="All agents should handle files consistently",
            evidence_refs=[],
            expected_benefit="Consistent behavior",
            risk_level="low",
            diff_payload={
                "skill_name": "file_handling",
                "instruction_additions": [
                    "Use atomic file operations",
                    "Handle encoding errors gracefully",
                ],
            },
            status="pending",
            created_at=datetime.now(UTC),
        )

        lesson_service.approve_proposal(
            proposal=cross_agent_proposal,
            applies_to_agents=["iron-rook", "bolt-merlin", "vox-jay"],
            experiment_id="exp-cross-001",
            require_experiment_id=False,
        )

        for agent_id in ["iron-rook", "bolt-merlin", "vox-jay"]:
            prompt = lesson_injector.inject_into_prompt(agent_id, "Base prompt")
            assert "atomic file operations" in prompt.lower()

    def test_provenance_tracking(
        self,
        lesson_service: LessonService,
        sample_proposal: ImprovementProposal,
    ) -> None:
        lesson = lesson_service.approve_proposal(
            proposal=sample_proposal,
            applies_to_agents=["iron-rook"],
            experiment_id="exp-test-001",
            require_experiment_id=False,
        )

        provenance = lesson_service.get_provenance(lesson.lesson_id)
        assert provenance is not None
        assert provenance.source_proposal_id == "prop-001"


class TestLessonInjectorStrategy:
    @pytest.fixture
    def temp_storage(self) -> Path:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def lesson_service(self, temp_storage: Path) -> LessonService:
        return LessonService(storage_path=temp_storage / "lessons")

    def test_strategy_filter(
        self,
        lesson_service: LessonService,
    ) -> None:
        proposals = [
            ImprovementProposal(
                proposal_id=f"prop-{i}",
                origin_run_id="run-001",
                origin_review_id=None,
                target_agent="iron-rook",
                proposal_type="skill",
                title=f"Lesson {i}",
                rationale="Test",
                evidence_refs=[],
                expected_benefit="Test benefit",
                risk_level="low",
                diff_payload={
                    "skill_name": f"skill_{i}",
                    "instruction_additions": [f"Instruction {i}"],
                },
                status="pending",
                strategy="skill-quality" if i % 2 == 0 else "tool-quality",
                created_at=datetime.now(UTC),
            )
            for i in range(4)
        ]

        for prop in proposals:
            lesson_service.approve_proposal(
                proposal=prop,
                applies_to_agents=["iron-rook"],
                experiment_id="exp-test",
                require_experiment_id=False,
            )

        injector_skill = LessonInjector(
            lesson_service=lesson_service,
            strategy_filter="skill-quality",
        )

        lessons = injector_skill.get_all_lessons("iron-rook")
        assert len(lessons) == 2

        for lesson in lessons:
            assert lesson.strategy == "skill-quality"
