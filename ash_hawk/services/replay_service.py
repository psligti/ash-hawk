from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pydantic as pd

from ash_hawk.contracts import CuratedLesson, RunArtifact, ToolCallRecord

if TYPE_CHECKING:
    from dawn_kestrel.contracts.replay_request import LessonInjection, ReplayRequest, ReplayResult


class ReplayConfig(pd.BaseModel):
    max_iterations: int = 3
    timeout_seconds: int = 300
    preserve_seed: bool = True
    use_real_replay: bool = True
    capture_artifact: bool = True

    model_config = pd.ConfigDict(extra="forbid")


class ReplayService:
    """Service for replaying agent runs with lessons applied.

    Supports two modes:
    - Simulation: Estimates replay outcome based on lesson types
    - Real replay: Re-executes via Dawn Kestrel with lesson injection

    For production A/B comparison, use real replay mode.
    """

    def __init__(self, config: ReplayConfig | None = None) -> None:
        self._config = config or ReplayConfig()

    async def replay_with_lessons(
        self,
        artifact: RunArtifact,
        lessons: list[CuratedLesson],
        agent_runner: Any = None,
        experiment_id: str | None = None,
    ) -> RunArtifact:
        if self._config.use_real_replay and agent_runner is not None:
            return await self._real_replay(artifact, lessons, agent_runner, experiment_id)
        return self._simulate_replay(artifact, lessons, experiment_id)

    async def _real_replay(
        self,
        baseline: RunArtifact,
        lessons: list[CuratedLesson],
        agent_runner: Any,
        experiment_id: str | None = None,
    ) -> RunArtifact:
        from dawn_kestrel.contracts.replay_request import (
            LessonInjection,
            ReplayRequest,
        )

        lessons_to_inject = [
            LessonInjection(
                lesson_id=lesson.lesson_id,
                injection_point=self._map_lesson_type_to_injection_point(lesson.lesson_type),
                priority=lesson.version,
                enabled=True,
            )
            for lesson in lessons
        ]

        replay_request = ReplayRequest(
            replay_id=f"replay-{baseline.run_id}-{uuid4().hex[:8]}",
            source_run_id=baseline.run_id,
            agent_id=baseline.agent_name,
            lessons_to_inject=lessons_to_inject,
            timeout_seconds=self._config.timeout_seconds,
            capture_artifact=self._config.capture_artifact,
            experiment_id=experiment_id,
        )

        try:
            if hasattr(agent_runner, "replay"):
                result = await agent_runner.replay(replay_request)
                if result.artifact_path:
                    return self._load_artifact_from_path(result.artifact_path)
            return self._simulate_replay(baseline, lessons, experiment_id)
        except Exception:
            return self._simulate_replay(baseline, lessons, experiment_id)

    def _map_lesson_type_to_injection_point(self, lesson_type: str) -> str:
        mapping = {
            "policy": "policy",
            "skill": "prompt",
            "tool": "tools",
            "harness": "harness",
            "eval": "harness",
        }
        return mapping.get(lesson_type, "prompt")

    def _load_artifact_from_path(self, path: str) -> RunArtifact:
        import json

        with open(path) as f:
            data = json.load(f)
        return RunArtifact.model_validate(data)

    def _simulate_replay(
        self,
        baseline: RunArtifact,
        lessons: list[CuratedLesson],
        experiment_id: str | None = None,
    ) -> RunArtifact:
        simulated_tool_calls = []
        for tc in baseline.tool_calls:
            new_tc = ToolCallRecord(
                tool_name=tc.tool_name,
                outcome=tc.outcome,
                duration_ms=tc.duration_ms,
                error_message=tc.error_message,
            )
            simulated_tool_calls.append(new_tc)

        simulated_outcome = baseline.outcome
        simulated_error = baseline.error_message

        for lesson in lessons:
            payload = lesson.lesson_payload
            if lesson.lesson_type == "tool":
                tool_id = payload.get("tool_id")
                timeout = payload.get("timeout_override")
                if timeout and tool_id:
                    for tc in simulated_tool_calls:
                        if tc.tool_name == tool_id and tc.outcome == "failure":
                            if tc.error_message and "timeout" in tc.error_message.lower():
                                simulated_outcome = "success"
                                simulated_error = None
                                tc.outcome = "success"
                                tc.error_message = None
            if lesson.lesson_type == "policy":
                rule_type = payload.get("rule_type")
                if rule_type in ("engagement", "ranking"):
                    failed_calls = [tc for tc in simulated_tool_calls if tc.outcome == "failure"]
                    if failed_calls and len(failed_calls) < len(simulated_tool_calls) // 2:
                        simulated_outcome = "success"
                        simulated_error = None

        metadata = {
            **baseline.metadata,
            "replay_of": baseline.run_id,
            "lessons_applied": [lesson.lesson_id for lesson in lessons],
            "simulation": True,
        }
        if experiment_id:
            metadata["experiment_id"] = experiment_id

        return RunArtifact(
            run_id=f"replay-{baseline.run_id}-{uuid4().hex[:8]}",
            suite_id=baseline.suite_id,
            agent_name=baseline.agent_name,
            outcome=simulated_outcome,
            tool_calls=simulated_tool_calls,
            steps=baseline.steps,
            messages=baseline.messages,
            total_duration_ms=baseline.total_duration_ms,
            token_usage=baseline.token_usage,
            cost_usd=baseline.cost_usd,
            error_message=simulated_error,
            metadata=metadata,
            created_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )


__all__ = ["ReplayService", "ReplayConfig"]
