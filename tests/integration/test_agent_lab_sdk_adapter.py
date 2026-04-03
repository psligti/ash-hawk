from __future__ import annotations

import datetime
import enum
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import pytest

if not hasattr(datetime, "UTC"):
    setattr(datetime, "UTC", datetime.UTC)

if not hasattr(enum, "StrEnum"):

    class StrEnum(str, enum.Enum):
        pass

    setattr(enum, "StrEnum", StrEnum)

from ash_hawk.types import EvalOutcome, EvalTask, EvalTranscript, TokenUsage


@dataclass
class MockAgentResult:
    response: str
    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    trace_events: list[dict[str, Any]]
    token_usage: dict[str, int]
    cost_usd: float
    duration_seconds: float
    error: str | None = None


class MockEvaluationAdapter:
    async def to_eval_transcript(self, result: MockAgentResult) -> EvalTranscript:
        return EvalTranscript(
            messages=result.messages,
            tool_calls=result.tool_calls,
            trace_events=result.trace_events,
            token_usage=TokenUsage(**result.token_usage),
            cost_usd=result.cost_usd,
            duration_seconds=result.duration_seconds,
            agent_response=result.response,
            error_trace=result.error,
        )

    def from_eval_task(self, task: EvalTask) -> dict[str, Any]:
        inputs = task.input if isinstance(task.input, dict) else {"prompt": task.input}
        return {
            "id": task.id,
            "description": task.description,
            "input": inputs,
        }

    def to_ash_hawk_tuple(
        self, transcript: EvalTranscript, outcome: EvalOutcome | None = None
    ) -> tuple[
        str | dict[str, Any] | None,
        list[dict[str, Any]],
        dict[str, Any],
        EvalOutcome,
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        resolved_outcome = outcome or EvalOutcome.success()
        artifacts = {"source": "agent_lab_sdk"}
        return (
            transcript.agent_response,
            list(transcript.trace_events),
            artifacts,
            resolved_outcome,
            list(transcript.messages),
            list(transcript.tool_calls),
        )


class MockAgentLabExecution:
    def __init__(self, result: MockAgentResult) -> None:
        self._result = result

    async def run(self) -> MockAgentResult:
        return self._result


@pytest.fixture
def mock_dawn_kestrel_modules(monkeypatch: pytest.MonkeyPatch) -> type[MockAgentResult]:
    dawn_module = ModuleType("dawn_kestrel")
    agents_module = ModuleType("dawn_kestrel.agents")
    results_module = ModuleType("dawn_kestrel.agents.results")
    setattr(results_module, "AgentResult", MockAgentResult)
    setattr(agents_module, "results", results_module)

    monkeypatch.setitem(sys.modules, "dawn_kestrel", dawn_module)
    monkeypatch.setitem(sys.modules, "dawn_kestrel.agents", agents_module)
    monkeypatch.setitem(sys.modules, "dawn_kestrel.agents.results", results_module)
    return MockAgentResult


@pytest.fixture
def mock_agent_result(mock_dawn_kestrel_modules: type[MockAgentResult]) -> MockAgentResult:
    return mock_dawn_kestrel_modules(
        response="Agent response",
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Agent response"},
        ],
        tool_calls=[
            {
                "name": "bash",
                "arguments": {"command": "echo hi"},
                "output": "hi",
            }
        ],
        trace_events=[{"event_type": "ModelMessageEvent", "data": {"role": "user"}}],
        token_usage={
            "input": 10,
            "output": 5,
            "reasoning": 0,
            "cache_read": 0,
            "cache_write": 0,
        },
        cost_usd=0.01,
        duration_seconds=1.2,
    )


@pytest.fixture
def eval_task() -> EvalTask:
    return EvalTask(
        id="task-1",
        description="Agent Lab SDK contract test",
        input={"prompt": "Say hi"},
    )


@pytest.fixture
def evaluation_adapter() -> MockEvaluationAdapter:
    return MockEvaluationAdapter()


@pytest.mark.asyncio
async def test_agent_lab_sdk_adapter_contract(
    evaluation_adapter: MockEvaluationAdapter,
    mock_agent_result: MockAgentResult,
) -> None:
    execution = MockAgentLabExecution(mock_agent_result)
    result = await execution.run()

    transcript = await evaluation_adapter.to_eval_transcript(result)

    assert transcript.agent_response == "Agent response"
    assert transcript.messages == mock_agent_result.messages
    assert transcript.tool_calls == mock_agent_result.tool_calls
    assert transcript.trace_events == mock_agent_result.trace_events

    adapter_tuple = evaluation_adapter.to_ash_hawk_tuple(transcript)
    assert len(adapter_tuple) == 6

    final_output, trace_events, artifacts, outcome, messages, tool_calls = adapter_tuple
    assert final_output == transcript.agent_response
    assert trace_events == transcript.trace_events
    assert artifacts["source"] == "agent_lab_sdk"
    assert outcome.status.value == "completed"
    assert messages == transcript.messages
    assert tool_calls == transcript.tool_calls


def test_agent_lab_sdk_adapter_from_eval_task(
    evaluation_adapter: MockEvaluationAdapter, eval_task: EvalTask
) -> None:
    payload = evaluation_adapter.from_eval_task(eval_task)
    assert payload["id"] == "task-1"
    assert payload["description"] == "Agent Lab SDK contract test"
    assert payload["input"] == {"prompt": "Say hi"}
