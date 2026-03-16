from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pydantic as pd

from ash_hawk.contracts import CuratedLesson, RunArtifact, ToolCallRecord


class ReplayConfig(pd.BaseModel):
    max_iterations: int = 3
    timeout_seconds: int = 300
    preserve_seed: bool = True
    use_real_replay: bool = True

    model_config = pd.ConfigDict(extra="forbid")


class ReplayService:
    def __init__(self) -> None:
        pass

    async def replay_with_lessons(
        self,
        artifact: RunArtifact,
        lessons: list[CuratedLesson],
        agent_runner: Any = None,
    ) -> RunArtifact:
        return self._simulate_replay(artifact, lessons)

    def _simulate_replay(
        self,
        baseline: RunArtifact,
        lessons: list[CuratedLesson],
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
            metadata={
                **baseline.metadata,
                "replay_of": baseline.run_id,
                "lessons_applied": [lesson.lesson_id for lesson in lessons],
                "simulation": True,
            },
            created_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )


__all__ = ["ReplayService", "ReplayConfig"]
