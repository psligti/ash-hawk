from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ash_hawk.contracts import CuratedLesson, ImprovementProposal, RunArtifact
from ash_hawk.execution.trial import TrialExecutor, _wire_lesson_injector
from ash_hawk.services.lesson_injector import LessonInjector
from ash_hawk.services.lesson_service import LessonService
from ash_hawk.types import (
    EvalOutcome,
    EvalStatus,
    EvalTask,
    EvalTranscript,
    FailureMode,
    RunEnvelope,
    ToolSurfacePolicy,
)


def _make_run_envelope(
    run_id: str = "run-1",
    suite_id: str = "suite-1",
    agent_name: str = "bolt-merlin",
) -> RunEnvelope:
    return RunEnvelope(
        run_id=run_id,
        suite_id=suite_id,
        suite_hash="abc123",
        harness_version="0.1.0",
        agent_name=agent_name,
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        tool_policy_hash="def456",
        python_version="3.12.0",
        os_info="darwin",
        created_at=datetime.now(UTC).isoformat(),
        config_snapshot={},
    )


def _make_proposal(proposal_id: str, agent: str, proposal_type: str) -> ImprovementProposal:
    return ImprovementProposal(
        proposal_id=proposal_id,
        origin_run_id=f"run-{proposal_id}",
        target_agent=agent,
        proposal_type=proposal_type,  # type: ignore[arg-type]
        title=f"Test proposal {proposal_id}",
        rationale="Test rationale",
        expected_benefit="Test benefit",
        risk_level="low",
        evidence_refs=["test_evidence_1"],
        created_at=datetime.now(UTC),
    )


def _make_lesson(
    lesson_id: str,
    agent: str,
    lesson_type: str,
    payload: dict[str, Any],
) -> CuratedLesson:
    return CuratedLesson(
        lesson_id=lesson_id,
        source_proposal_id=f"prop-{lesson_id}",
        applies_to_agents=[agent],
        lesson_type=lesson_type,  # type: ignore[arg-type]
        title=f"Test lesson {lesson_id}",
        description="Test lesson description",
        lesson_payload=payload,
        validation_status="approved",
        version=1,
        created_at=datetime.now(UTC),
    )


class MockAgentRunner:
    def __init__(self) -> None:
        self._lesson_injector: Any = None
        self.run_called = False
        self.received_prompt = ""

    def set_lesson_injector(self, injector: Any) -> None:
        self._lesson_injector = injector

    async def run(
        self,
        task: EvalTask,
        policy_enforcer: Any,
        config: dict[str, object],
    ) -> tuple[EvalTranscript, EvalOutcome]:
        self.run_called = True
        if self._lesson_injector is not None and isinstance(task.input, dict):
            base_prompt = str(task.input.get("prompt", ""))
            self.received_prompt = self._lesson_injector.inject_into_prompt(
                config.get("agent_name", "default"),
                base_prompt,
            )
        return EvalTranscript(), EvalOutcome.success()


class TestLessonInjectorWiring:
    def test_wire_lesson_injector_calls_setter(self) -> None:
        runner = MockAgentRunner()
        injector = LessonInjector()

        _wire_lesson_injector(runner, injector)

        assert runner._lesson_injector is injector

    def test_wire_lesson_injector_handles_missing_setter(self) -> None:
        class RunnerWithoutSetter:
            pass

        runner = RunnerWithoutSetter()
        injector = LessonInjector()

        _wire_lesson_injector(runner, injector)

    def test_trial_executor_accepts_lesson_injector(self) -> None:
        storage = MagicMock()
        policy = ToolSurfacePolicy()
        runner = MockAgentRunner()
        injector = LessonInjector()

        executor = TrialExecutor(
            storage=storage,
            policy=policy,
            agent_runner=runner,
            lesson_injector=injector,
        )

        assert executor.lesson_injector is injector
        assert runner._lesson_injector is injector

    def test_trial_executor_set_lesson_injector_after_init(self) -> None:
        storage = MagicMock()
        policy = ToolSurfacePolicy()
        runner = MockAgentRunner()
        injector = LessonInjector()

        executor = TrialExecutor(
            storage=storage,
            policy=policy,
            agent_runner=runner,
        )

        assert executor.lesson_injector is None
        assert runner._lesson_injector is None

        executor.set_lesson_injector(injector)

        assert executor.lesson_injector is injector
        assert runner._lesson_injector is injector


class TestLessonInjectorPromptAugmentation:
    def test_inject_into_prompt_adds_skill_lessons(self, tmp_path: Path) -> None:
        service = LessonService(storage_path=tmp_path)

        proposal = _make_proposal("prop-skill", "bolt-merlin", "skill")
        proposal.diff_payload = {
            "skill_name": "file-validation",
            "instruction_additions": [
                "Always validate file paths before reading",
                "Use structured error messages",
            ],
        }

        lesson = service.approve_proposal(
            proposal,
            applies_to_agents=["bolt-merlin"],
            require_experiment_id=False,
        )

        injector = LessonInjector(lesson_service=service)
        augmented = injector.inject_into_prompt("bolt-merlin", "Base prompt")

        assert "Learned Lessons" in augmented
        assert "validate file paths" in augmented
        assert "structured error messages" in augmented

    def test_inject_into_prompt_adds_policy_lessons(self, tmp_path: Path) -> None:
        service = LessonService(storage_path=tmp_path)

        proposal = _make_proposal("prop-policy", "iron-rook", "policy")
        proposal.diff_payload = {
            "rule_name": "timeout-short-tasks",
            "rule_type": "engagement",
            "condition": {"task.duration_estimate": "< 5m"},
            "action": {"skip_detailed_review": True},
            "enabled": True,
            "priority": 10,
        }

        lesson = service.approve_proposal(
            proposal,
            applies_to_agents=["iron-rook"],
            require_experiment_id=False,
        )

        injector = LessonInjector(lesson_service=service)
        augmented = injector.inject_into_prompt("iron-rook", "Base prompt")

        assert "timeout-short-tasks" in augmented

    def test_inject_into_prompt_filters_by_agent(self, tmp_path: Path) -> None:
        service = LessonService(storage_path=tmp_path)

        proposal = _make_proposal("prop-1", "bolt-merlin", "skill")
        proposal.diff_payload = {
            "skill_name": "bolt-skill",
            "instruction_additions": ["Bolt Merlin only"],
        }
        service.approve_proposal(
            proposal,
            applies_to_agents=["bolt-merlin"],
            require_experiment_id=False,
        )

        proposal2 = _make_proposal("prop-2", "iron-rook", "skill")
        proposal2.diff_payload = {
            "skill_name": "iron-skill",
            "instruction_additions": ["Iron Rook only"],
        }
        service.approve_proposal(
            proposal2,
            applies_to_agents=["iron-rook"],
            require_experiment_id=False,
        )

        injector = LessonInjector(lesson_service=service)

        bolt_augmented = injector.inject_into_prompt("bolt-merlin", "Base")
        assert "Bolt Merlin only" in bolt_augmented
        assert "Iron Rook only" not in bolt_augmented

        iron_augmented = injector.inject_into_prompt("iron-rook", "Base")
        assert "Iron Rook only" in iron_augmented
        assert "Bolt Merlin only" not in iron_augmented

    def test_inject_into_prompt_returns_unchanged_if_no_lessons(self, tmp_path: Path) -> None:
        service = LessonService(storage_path=tmp_path)
        injector = LessonInjector(lesson_service=service)

        result = injector.inject_into_prompt("unknown-agent", "Base prompt")

        assert result == "Base prompt"


class TestLessonInjectorToolOverrides:
    def test_get_tool_overrides_returns_tool_lessons(self, tmp_path: Path) -> None:
        service = LessonService(storage_path=tmp_path)

        proposal = _make_proposal("prop-tool", "bolt-merlin", "tool")
        proposal.diff_payload = {
            "tool_id": "bash",
            "timeout_override": 60,
            "parameter_defaults": {"timeout": 30},
            "usage_hints": ["Use for quick file operations"],
            "preconditions": ["Check working directory"],
        }

        service.approve_proposal(
            proposal,
            applies_to_agents=["bolt-merlin"],
            require_experiment_id=False,
        )

        injector = LessonInjector(lesson_service=service)
        overrides = injector.get_tool_overrides("bolt-merlin")

        assert "bash" in overrides
        assert overrides["bash"]["timeout"] == 60
        assert overrides["bash"]["defaults"]["timeout"] == 30

    def test_get_tool_overrides_empty_if_no_lessons(self, tmp_path: Path) -> None:
        service = LessonService(storage_path=tmp_path)
        injector = LessonInjector(lesson_service=service)

        overrides = injector.get_tool_overrides("unknown-agent")

        assert overrides == {}


class TestLessonInjectorHarnessAdjustments:
    def test_get_harness_adjustments_returns_harness_lessons(self, tmp_path: Path) -> None:
        service = LessonService(storage_path=tmp_path)

        proposal = _make_proposal("prop-harness", "bolt-merlin", "harness")
        proposal.diff_payload = {
            "suite_id": "suite-1",
            "grader_adjustments": {"llm_judge": {"threshold": 0.8}},
            "fixture_overrides": {"timeout_seconds": 120},
            "timeout_adjustments": {"trial_timeout": 300},
            "parallelism_override": 4,
        }

        service.approve_proposal(
            proposal,
            applies_to_agents=["bolt-merlin"],
            require_experiment_id=False,
        )

        injector = LessonInjector(lesson_service=service)
        adjustments = injector.get_harness_adjustments("bolt-merlin")

        assert adjustments["grader_adjustments"]["llm_judge"]["threshold"] == 0.8
        assert adjustments["fixture_overrides"]["timeout_seconds"] == 120
        assert adjustments["parallelism"] == 4


class TestFullInjectionLoop:
    @pytest.mark.asyncio
    async def test_lesson_injection_into_agent_run(self, tmp_path: Path) -> None:
        service = LessonService(storage_path=tmp_path)

        proposal = _make_proposal("prop-loop", "bolt-merlin", "skill")
        proposal.diff_payload = {
            "skill_name": "file-checking",
            "instruction_additions": ["Always check file existence before reading"],
        }

        service.approve_proposal(
            proposal,
            applies_to_agents=["bolt-merlin"],
            require_experiment_id=False,
        )

        injector = LessonInjector(lesson_service=service)
        runner = MockAgentRunner()
        storage = MagicMock()
        policy = ToolSurfacePolicy()

        executor = TrialExecutor(
            storage=storage,
            policy=policy,
            agent_runner=runner,
            lesson_injector=injector,
        )

        task = EvalTask(
            id="task-1",
            input={"prompt": "Review this code"},
        )

        run_envelope = _make_run_envelope(
            run_id="run-1",
            suite_id="suite-1",
            agent_name="bolt-merlin",
        )

        result = await executor.execute(
            task=task,
            agent_config={"agent_name": "bolt-merlin"},
            run_envelope=run_envelope,
        )

        assert runner.run_called
        assert "check file existence" in runner.received_prompt

    @pytest.mark.asyncio
    async def test_no_injection_when_no_lessons(self, tmp_path: Path) -> None:
        service = LessonService(storage_path=tmp_path)
        injector = LessonInjector(lesson_service=service)
        runner = MockAgentRunner()
        storage = MagicMock()
        policy = ToolSurfacePolicy()

        executor = TrialExecutor(
            storage=storage,
            policy=policy,
            agent_runner=runner,
            lesson_injector=injector,
        )

        task = EvalTask(
            id="task-1",
            input={"prompt": "Base prompt"},
        )

        run_envelope = _make_run_envelope(
            run_id="run-1",
            suite_id="suite-1",
            agent_name="unknown-agent",
        )

        result = await executor.execute(
            task=task,
            agent_config={"agent_name": "unknown-agent"},
            run_envelope=run_envelope,
        )

        assert runner.run_called
        assert runner.received_prompt == "Base prompt"
