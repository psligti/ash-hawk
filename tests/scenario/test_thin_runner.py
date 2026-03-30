"""Tests for ThinScenarioRunner and related types."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ash_hawk.bridge import OutcomeData, RunResult, TranscriptData
from ash_hawk.scenario.models import ScenarioGraderSpec, ScenarioV1, SUTConfig
from ash_hawk.scenario.thin_runner import (
    ScenarioTelemetrySink,
    ThinGradedResult,
    ThinScenarioRunner,
)
from ash_hawk.types import GraderResult


def _make_scenario(
    adapter: str = "test-agent",
    agent_path: str | None = None,
    prompt: str | None = None,
    description: str = "Test scenario",
) -> ScenarioV1:
    config: dict[str, Any] = {}
    if agent_path:
        config["agent"] = agent_path
    return ScenarioV1(
        schema_version="v1",
        id="test-scenario",
        description=description,
        sut=SUTConfig(type="coding_agent", adapter=adapter, config=config),
        inputs={"prompt": prompt} if prompt else {},
    )


class TestResolveAgentPath:
    def test_bolt_merlin_returns_dawn_root(self, tmp_path: Path) -> None:
        dawn_root = tmp_path / ".dawn-kestrel"
        dawn_root.mkdir()
        runner = ThinScenarioRunner(workdir=tmp_path)
        scenario = _make_scenario(adapter="bolt_merlin")
        result = runner._resolve_agent_path(scenario)
        assert result == dawn_root

    def test_bolt_merlin_hyphenated_returns_dawn_root(self, tmp_path: Path) -> None:
        dawn_root = tmp_path / ".dawn-kestrel"
        dawn_root.mkdir()
        runner = ThinScenarioRunner(workdir=tmp_path)
        scenario = _make_scenario(adapter="bolt-merlin")
        result = runner._resolve_agent_path(scenario)
        assert result == dawn_root

    def test_dawn_kestrel_agent_dir_found(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / ".dawn-kestrel" / "agents" / "my-agent"
        agent_dir.mkdir(parents=True)
        runner = ThinScenarioRunner(workdir=tmp_path)
        scenario = _make_scenario(adapter="my-agent")
        result = runner._resolve_agent_path(scenario)
        assert result == agent_dir

    def test_opencode_agent_dir_found(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / ".opencode" / "agent"
        agent_dir.mkdir(parents=True)
        agent_file = agent_dir / "my-agent.md"
        agent_file.write_text("agent config", encoding="utf-8")
        runner = ThinScenarioRunner(workdir=tmp_path)
        scenario = _make_scenario(adapter="my-agent")
        result = runner._resolve_agent_path(scenario)
        assert result == agent_dir / "my-agent"

    def test_no_explicit_agent_field_in_sut_config(self, tmp_path: Path) -> None:
        dawn_root = tmp_path / ".dawn-kestrel"
        dawn_root.mkdir()
        custom = tmp_path / "custom-agent"
        custom.mkdir()
        runner = ThinScenarioRunner(workdir=tmp_path)
        scenario = _make_scenario(adapter="anything")
        scenario.sut.config["agent"] = str(custom)
        result = runner._resolve_agent_path(scenario)
        assert result == dawn_root

    def test_fallback_to_dawn_root(self, tmp_path: Path) -> None:
        dawn_root = tmp_path / ".dawn-kestrel"
        dawn_root.mkdir()
        runner = ThinScenarioRunner(workdir=tmp_path)
        scenario = _make_scenario(adapter="nonexistent-agent")
        result = runner._resolve_agent_path(scenario)
        assert result == dawn_root

    def test_fallback_to_workdir(self, tmp_path: Path) -> None:
        runner = ThinScenarioRunner(workdir=tmp_path)
        scenario = _make_scenario(adapter="nonexistent-agent")
        result = runner._resolve_agent_path(scenario)
        assert result == tmp_path


class TestResolveAgentName:
    def test_configured_agent_name(self, tmp_path: Path) -> None:
        runner = ThinScenarioRunner(workdir=tmp_path)
        scenario = _make_scenario(adapter="my-agent", agent_path="/some/path")
        scenario.sut.config["agent"] = "custom-name"
        result = runner._resolve_agent_name(scenario)
        assert result == "custom-name"

    def test_bolt_merlin_returns_orchestrator(self, tmp_path: Path) -> None:
        runner = ThinScenarioRunner(workdir=tmp_path)
        scenario = _make_scenario(adapter="bolt_merlin")
        result = runner._resolve_agent_name(scenario)
        assert result == "orchestrator"

    def test_adapter_name_fallback(self, tmp_path: Path) -> None:
        runner = ThinScenarioRunner(workdir=tmp_path)
        scenario = _make_scenario(adapter="my-agent")
        result = runner._resolve_agent_name(scenario)
        assert result == "my-agent"


class TestBuildInput:
    def test_uses_prompt_from_inputs(self, tmp_path: Path) -> None:
        runner = ThinScenarioRunner(workdir=tmp_path)
        scenario = _make_scenario(prompt="Do the thing")
        result = runner._build_input(scenario)
        assert result == "Do the thing"

    def test_falls_back_to_description(self, tmp_path: Path) -> None:
        runner = ThinScenarioRunner(workdir=tmp_path)
        scenario = _make_scenario(description="Fallback description")
        result = runner._build_input(scenario)
        assert result == "Fallback description"

    def test_empty_prompt_uses_description(self, tmp_path: Path) -> None:
        runner = ThinScenarioRunner(workdir=tmp_path)
        scenario = ScenarioV1(
            schema_version="v1",
            id="test",
            description="Use this",
            sut=SUTConfig(type="coding_agent", adapter="test"),
            inputs={"prompt": ""},
        )
        result = runner._build_input(scenario)
        assert result == "Use this"


class TestScenarioTelemetrySink:
    @pytest.mark.asyncio
    async def test_collects_iteration_starts(self) -> None:
        sink = ScenarioTelemetrySink()
        await sink.on_iteration_start({"iteration": 1})
        await sink.on_iteration_start({"iteration": 2})
        assert len(sink.iterations) == 2
        assert sink.iterations[0]["type"] == "start"

    @pytest.mark.asyncio
    async def test_collects_tool_results(self) -> None:
        sink = ScenarioTelemetrySink()
        await sink.on_tool_result({"tool_name": "bash", "status": "ok"})
        assert len(sink.tool_calls) == 1
        assert sink.tool_calls[0]["tool_name"] == "bash"

    @pytest.mark.asyncio
    async def test_collects_policy_decisions(self) -> None:
        sink = ScenarioTelemetrySink()
        await sink.on_action_decision({"decision": "allow", "risk_level": "low"})
        assert len(sink.policy_decisions) == 1


class TestThinGradedResult:
    def test_all_passed_true(self) -> None:
        run_result = RunResult(
            transcript=TranscriptData(),
            outcome=OutcomeData(success=True),
        )
        result = ThinGradedResult(
            run_result=run_result,
            grader_results=[
                GraderResult(grader_type="a", score=1.0, passed=True),
                GraderResult(grader_type="b", score=0.8, passed=True),
            ],
        )
        assert result.all_passed() is True
        assert result.aggregate_score == 0.9

    def test_all_passed_false(self) -> None:
        run_result = RunResult(
            transcript=TranscriptData(),
            outcome=OutcomeData(success=True),
        )
        result = ThinGradedResult(
            run_result=run_result,
            grader_results=[
                GraderResult(grader_type="a", score=1.0, passed=True),
                GraderResult(grader_type="b", score=0.3, passed=False),
            ],
        )
        assert result.all_passed() is False

    def test_aggregate_score_empty(self) -> None:
        run_result = RunResult(
            transcript=TranscriptData(),
            outcome=OutcomeData(success=True),
        )
        result = ThinGradedResult(run_result=run_result, grader_results=[])
        assert result.aggregate_score == 0.0


class TestRunScenario:
    @pytest.mark.asyncio
    async def test_run_scenario_calls_real_agent(self, tmp_path: Path, monkeypatch: Any) -> None:
        fake_result = RunResult(
            transcript=TranscriptData(),
            outcome=OutcomeData(success=True, message="done"),
            run_id="test-run",
        )

        async def _fake_run_agent(**kwargs: Any) -> RunResult:
            return fake_result

        monkeypatch.setattr("ash_hawk.scenario.thin_runner.run_real_agent", _fake_run_agent)

        runner = ThinScenarioRunner(workdir=tmp_path)
        scenario = _make_scenario()
        result = await runner.run_scenario(scenario, tmp_path / "scenario.yaml")
        assert result.outcome.success is True
        assert result.run_id == "test-run"


class TestRunWithGrading:
    @pytest.mark.asyncio
    async def test_no_graders_returns_empty(self, tmp_path: Path, monkeypatch: Any) -> None:
        fake_result = RunResult(
            transcript=TranscriptData(),
            outcome=OutcomeData(success=True, message="done"),
            run_id="test-run",
        )

        async def _fake_run_agent(**kwargs: Any) -> RunResult:
            return fake_result

        monkeypatch.setattr("ash_hawk.scenario.thin_runner.run_real_agent", _fake_run_agent)

        runner = ThinScenarioRunner(workdir=tmp_path)
        scenario = _make_scenario()
        result = await runner.run_with_grading(scenario, tmp_path / "scenario.yaml")
        assert result.grader_results == []
        assert result.run_result.outcome.success is True

    @pytest.mark.asyncio
    async def test_with_graders_returns_results(self, tmp_path: Path, monkeypatch: Any) -> None:
        fake_result = RunResult(
            transcript=TranscriptData(),
            outcome=OutcomeData(success=True, message="done"),
            run_id="test-run",
        )

        async def _fake_run_agent(**kwargs: Any) -> RunResult:
            return fake_result

        monkeypatch.setattr("ash_hawk.scenario.thin_runner.run_real_agent", _fake_run_agent)

        class _FakeGrader:
            async def grade(self, trial: Any, transcript: Any, spec: Any) -> GraderResult:
                return GraderResult(grader_type=spec.grader_type, score=1.0, passed=True)

        class _FakeRegistry:
            def get(self, name: str) -> _FakeGrader | None:
                return _FakeGrader()

        monkeypatch.setattr(
            "ash_hawk.scenario.thin_runner.get_default_registry",
            lambda: _FakeRegistry(),
        )

        runner = ThinScenarioRunner(workdir=tmp_path)
        scenario = ScenarioV1(
            schema_version="v1",
            id="graded-test",
            sut=SUTConfig(type="coding_agent", adapter="test"),
            graders=[ScenarioGraderSpec(grader_type="string_match")],
        )
        result = await runner.run_with_grading(scenario, tmp_path / "scenario.yaml")
        assert len(result.grader_results) == 1
        assert result.grader_results[0].passed is True
