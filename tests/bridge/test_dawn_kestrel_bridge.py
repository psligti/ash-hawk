from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ash_hawk.bridge import TelemetrySink
from ash_hawk.bridge.dawn_kestrel import DawnKestrelBridge
from ash_hawk.types import EvalOutcome, EvalStatus, EvalTranscript, TokenUsage


class _SinkRecorder(TelemetrySink):
    def __init__(self) -> None:
        self.starts: list[dict[str, Any]] = []
        self.ends: list[dict[str, Any]] = []
        self.tool_results: list[dict[str, Any]] = []
        self.completions: list[dict[str, Any]] = []

    async def on_iteration_start(self, data: dict[str, Any]) -> None:
        self.starts.append(data)

    async def on_iteration_end(self, data: dict[str, Any]) -> None:
        self.ends.append(data)

    async def on_action_decision(self, data: dict[str, Any]) -> None:
        _ = data

    async def on_tool_result(self, data: dict[str, Any]) -> None:
        self.tool_results.append(data)

    async def on_run_complete(self, data: dict[str, Any]) -> None:
        self.completions.append(data)


@pytest.mark.asyncio
async def test_bridge_applies_fixtures_and_overlays_to_runtime_workdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_source = tmp_path / "fixture.txt"
    fixture_source.write_text("fixture-content", encoding="utf-8")

    sink = _SinkRecorder()
    bridge = DawnKestrelBridge(agent_path=tmp_path, telemetry_sink=sink, workdir=tmp_path)

    captured: dict[str, Any] = {}

    class _FakeRunner:
        def __init__(self, provider: str, model: str) -> None:
            captured["provider"] = provider
            captured["model"] = model

        async def run(
            self, task: Any, policy_enforcer: Any, config: dict[str, Any]
        ) -> tuple[Any, Any]:
            _ = policy_enforcer
            runtime_workdir = Path(config["workdir"])
            captured["workdir"] = runtime_workdir
            captured["task"] = task
            assert (runtime_workdir / "fixtures/fixture.txt").read_text(
                encoding="utf-8"
            ) == "fixture-content"
            assert (runtime_workdir / "overlays/new.txt").read_text(
                encoding="utf-8"
            ) == "overlay-content"

            transcript = EvalTranscript(
                messages=[{"role": "assistant", "content": "ok"}],
                tool_calls=[],
                token_usage=TokenUsage(input=1, output=2, reasoning=0, cache_read=0, cache_write=0),
                agent_response="ok",
            )
            outcome = EvalOutcome(status=EvalStatus.COMPLETED)
            return transcript, outcome

    monkeypatch.setattr("ash_hawk.agents.dawn_kestrel.DawnKestrelAgentRunner", _FakeRunner)
    monkeypatch.setattr(
        bridge,
        "_resolve_provider_model",
        lambda: ("test-provider", "test-model"),
    )

    result = await bridge.run(
        input="test",
        fixtures={"fixtures/fixture.txt": fixture_source},
        overlays={"overlays/new.txt": "overlay-content"},
    )

    assert result.outcome.success is True
    assert sink.starts
    assert sink.completions
    assert sink.starts[0]["payload"]["fixtures_applied"] == 1
    assert sink.starts[0]["payload"]["overlays_applied"] == 1
    assert captured["provider"] == "test-provider"
    assert captured["model"] == "test-model"
    assert isinstance(captured["workdir"], Path)
    assert not captured["workdir"].exists()


@pytest.mark.asyncio
async def test_bridge_emits_run_complete_event_when_run_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _SinkRecorder()
    bridge = DawnKestrelBridge(agent_path=tmp_path, telemetry_sink=sink, workdir=tmp_path)

    async def _raise(*args: Any, **kwargs: Any) -> Any:
        _ = args, kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr(bridge, "_run_agent", _raise)

    result = await bridge.run(input="test-failure")

    assert result.outcome.success is False
    assert result.outcome.error == "boom"
    assert sink.completions
    assert sink.completions[-1]["payload"]["success"] is False
    assert sink.completions[-1]["payload"]["error"] == "boom"
