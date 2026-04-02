from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from _pytest.monkeypatch import MonkeyPatch

import ash_hawk.auto_research.enhanced_cycle_runner as enhanced_cycle_runner
from ash_hawk.auto_research.enhanced_cycle_runner import run_enhanced_cycle
from ash_hawk.auto_research.types import (
    CycleResult,
    CycleStatus,
    EnhancedCycleConfig,
    EvolvableCycleResult,
    MultiTargetResult,
    TargetType,
)


@dataclass
class _DummyTarget:
    name: str
    discovered_path: Path
    target_type: TargetType = TargetType.SKILL


@dataclass
class _FakeLeverConfig:
    name: str

    def to_config_dict(self) -> dict[str, str]:
        return {"agent": self.name}


@pytest.mark.asyncio
async def test_run_enhanced_cycle_single_target_executes_one_target(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text("id: test\n", encoding="utf-8")

    discovered_targets = [
        _DummyTarget(name="first-skill", discovered_path=tmp_path / "first"),
        _DummyTarget(name="second-skill", discovered_path=tmp_path / "second"),
    ]

    class _FakeTargetDiscovery:
        def __init__(self, project_root: Path) -> None:
            self.project_root = project_root

        def discover_all_targets(self) -> list[_DummyTarget]:
            return discovered_targets

    captured: dict[str, list[str]] = {}

    class _FakeMultiTargetCycleRunner:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def run_all_targets(
            self,
            scenarios: list[Path],
            targets: list[_DummyTarget],
            iterations_per_target: int,
            threshold: float,
            storage_path: Path,
        ) -> MultiTargetResult:
            captured["targets"] = [target.name for target in targets]
            target = targets[0]
            return MultiTargetResult(
                agent_name=target.name,
                target_results={
                    target.name: CycleResult(
                        agent_name=target.name,
                        target_path=str(target.discovered_path),
                        scenario_paths=[str(s) for s in scenarios],
                        status=CycleStatus.COMPLETED,
                        initial_score=0.4,
                        final_score=0.5,
                        target_type=TargetType.SKILL,
                    )
                },
                overall_improvement=0.1,
                converged=False,
            )

    monkeypatch.setattr(
        "ash_hawk.auto_research.enhanced_cycle_runner.TargetDiscovery",
        _FakeTargetDiscovery,
    )
    monkeypatch.setattr(
        "ash_hawk.auto_research.enhanced_cycle_runner.MultiTargetCycleRunner",
        _FakeMultiTargetCycleRunner,
    )

    result = await run_enhanced_cycle(
        scenarios=[scenario_path],
        config=EnhancedCycleConfig(
            enable_multi_target=False,
            enable_intent_analysis=False,
            enable_knowledge_promotion=False,
            enable_lever_search=False,
        ),
        project_root=tmp_path,
        storage_path=tmp_path / ".ash-hawk" / "enhanced-auto-research",
        llm_client=None,
    )

    assert captured["targets"] == ["first-skill"]
    assert list(result.target_results.keys()) == ["first-skill"]
    assert result.overall_improvement == 0.1


def test_resolve_agent_startup_details_from_scenario_config(monkeypatch: MonkeyPatch) -> None:
    scenario_stub = SimpleNamespace(
        sut=SimpleNamespace(
            adapter="bolt_merlin",
            config={
                "agent": "orchestrator",
                "run_config": {"agent_name": "bolt-merlin", "agent_version": "v2"},
            },
        )
    )

    def _load_scenario(_: Path) -> SimpleNamespace:
        return scenario_stub

    monkeypatch.setattr(
        "ash_hawk.auto_research.enhanced_cycle_runner.load_scenario",
        _load_scenario,
    )

    resolve_agent = getattr(enhanced_cycle_runner, "_resolve_agent_startup_details")
    agent_name, agent_version = resolve_agent([Path("scenario.yaml")])

    assert agent_name == "bolt-merlin"
    assert agent_version == "v2"


class TestEvolvablePhase:
    @pytest.mark.asyncio
    async def test_evolvable_phase_produces_result_when_enabled(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
    ) -> None:
        scenario_path = tmp_path / "scenario.yaml"
        scenario_path.write_text("id: test\n", encoding="utf-8")

        class _FakeTargetDiscovery:
            def __init__(self, project_root: Path) -> None:
                pass

            def discover_all_targets(self) -> list[_DummyTarget]:
                return [_DummyTarget(name="test-skill", discovered_path=tmp_path)]

        class _FakeMultiTargetCycleRunner:
            def __init__(self, **kwargs: object) -> None:
                pass

            async def run_all_targets(
                self,
                scenarios: list[Path],
                targets: list[_DummyTarget],
                iterations_per_target: int,
                threshold: float,
                storage_path: Path,
            ) -> MultiTargetResult:
                target = targets[0]
                return MultiTargetResult(
                    agent_name=target.name,
                    target_results={
                        target.name: CycleResult(
                            agent_name=target.name,
                            target_path=str(target.discovered_path),
                            scenario_paths=[str(s) for s in scenarios],
                            status=CycleStatus.COMPLETED,
                            initial_score=0.4,
                            final_score=0.5,
                            target_type=TargetType.SKILL,
                        )
                    },
                    overall_improvement=0.1,
                    converged=False,
                )

        class _FakeLeverMatrixSearch:
            def __init__(self, lever_space: dict[str, object] | None = None) -> None:
                self.lever_space = lever_space or {"agent": object()}
                self._scores = [0.45, 0.52]

            def sample_random(self) -> _FakeLeverConfig:
                return _FakeLeverConfig("baseline")

            def sample_neighbors(
                self,
                config: _FakeLeverConfig,
                n: int = 1,
            ) -> list[_FakeLeverConfig]:
                return [_FakeLeverConfig("neighbor")]

            async def evaluate(
                self,
                config: _FakeLeverConfig,
                scenarios: list[Path],
                storage_path: Path,
            ) -> float:
                return self._scores.pop(0)

        monkeypatch.setattr(
            "ash_hawk.auto_research.enhanced_cycle_runner.TargetDiscovery",
            _FakeTargetDiscovery,
        )
        monkeypatch.setattr(
            "ash_hawk.auto_research.enhanced_cycle_runner.MultiTargetCycleRunner",
            _FakeMultiTargetCycleRunner,
        )
        monkeypatch.setattr(
            "ash_hawk.auto_research.enhanced_cycle_runner.LeverMatrixSearch",
            _FakeLeverMatrixSearch,
        )

        result = await run_enhanced_cycle(
            scenarios=[scenario_path],
            config=EnhancedCycleConfig(
                enable_multi_target=False,
                enable_intent_analysis=False,
                enable_knowledge_promotion=False,
                enable_lever_search=True,
                iterations_per_target=2,
            ),
            project_root=tmp_path,
            storage_path=tmp_path / ".ash-hawk" / "evolvable-test",
            llm_client=object(),
        )

        assert result.lever_result is not None
        assert isinstance(result.lever_result, EvolvableCycleResult)
        assert result.lever_result.total_experiments > 0
        assert result.lever_result.baseline_score > 0

    @pytest.mark.asyncio
    async def test_evolvable_phase_skipped_when_disabled(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
    ) -> None:
        scenario_path = tmp_path / "scenario.yaml"
        scenario_path.write_text("id: test\n", encoding="utf-8")

        class _FakeTargetDiscovery:
            def __init__(self, project_root: Path) -> None:
                pass

            def discover_all_targets(self) -> list[_DummyTarget]:
                return [_DummyTarget(name="test-skill", discovered_path=tmp_path)]

        class _FakeMultiTargetCycleRunner:
            def __init__(self, **kwargs: object) -> None:
                pass

            async def run_all_targets(
                self,
                scenarios: list[Path],
                targets: list[_DummyTarget],
                iterations_per_target: int,
                threshold: float,
                storage_path: Path,
            ) -> MultiTargetResult:
                target = targets[0]
                return MultiTargetResult(
                    agent_name=target.name,
                    target_results={
                        target.name: CycleResult(
                            agent_name=target.name,
                            target_path=str(target.discovered_path),
                            scenario_paths=[str(s) for s in scenarios],
                            status=CycleStatus.COMPLETED,
                            initial_score=0.4,
                            final_score=0.5,
                            target_type=TargetType.SKILL,
                        )
                    },
                    overall_improvement=0.1,
                    converged=False,
                )

        monkeypatch.setattr(
            "ash_hawk.auto_research.enhanced_cycle_runner.TargetDiscovery",
            _FakeTargetDiscovery,
        )
        monkeypatch.setattr(
            "ash_hawk.auto_research.enhanced_cycle_runner.MultiTargetCycleRunner",
            _FakeMultiTargetCycleRunner,
        )

        result = await run_enhanced_cycle(
            scenarios=[scenario_path],
            config=EnhancedCycleConfig(
                enable_multi_target=False,
                enable_intent_analysis=False,
                enable_knowledge_promotion=False,
                enable_lever_search=False,
            ),
            project_root=tmp_path,
            storage_path=tmp_path / ".ash-hawk" / "no-evolvable",
            llm_client=object(),
        )

        assert result.lever_result is None

    @pytest.mark.asyncio
    async def test_evolvable_phase_logs_jsonl_trace_events(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
    ) -> None:
        scenario_path = tmp_path / "scenario.yaml"
        scenario_path.write_text("id: test\n", encoding="utf-8")

        class _FakeTargetDiscovery:
            def __init__(self, project_root: Path) -> None:
                pass

            def discover_all_targets(self) -> list[_DummyTarget]:
                return [_DummyTarget(name="trace-skill", discovered_path=tmp_path)]

        class _FakeMultiTargetCycleRunner:
            def __init__(self, **kwargs: object) -> None:
                pass

            async def run_all_targets(
                self,
                scenarios: list[Path],
                targets: list[_DummyTarget],
                iterations_per_target: int,
                threshold: float,
                storage_path: Path,
            ) -> MultiTargetResult:
                target = targets[0]
                return MultiTargetResult(
                    agent_name=target.name,
                    target_results={
                        target.name: CycleResult(
                            agent_name=target.name,
                            target_path=str(target.discovered_path),
                            scenario_paths=[str(s) for s in scenarios],
                            status=CycleStatus.COMPLETED,
                            initial_score=0.4,
                            final_score=0.5,
                            target_type=TargetType.SKILL,
                        )
                    },
                    overall_improvement=0.1,
                    converged=False,
                )

        class _FakeLeverMatrixSearch:
            def __init__(self, lever_space: dict[str, object] | None = None) -> None:
                self.lever_space = lever_space or {"agent": object()}
                self._scores = [0.5, 0.53]

            def sample_random(self) -> _FakeLeverConfig:
                return _FakeLeverConfig("baseline")

            def sample_neighbors(
                self,
                config: _FakeLeverConfig,
                n: int = 1,
            ) -> list[_FakeLeverConfig]:
                return [_FakeLeverConfig("best")]

            async def evaluate(
                self,
                config: _FakeLeverConfig,
                scenarios: list[Path],
                storage_path: Path,
            ) -> float:
                return self._scores.pop(0)

        monkeypatch.setattr(
            "ash_hawk.auto_research.enhanced_cycle_runner.TargetDiscovery",
            _FakeTargetDiscovery,
        )
        monkeypatch.setattr(
            "ash_hawk.auto_research.enhanced_cycle_runner.MultiTargetCycleRunner",
            _FakeMultiTargetCycleRunner,
        )
        monkeypatch.setattr(
            "ash_hawk.auto_research.enhanced_cycle_runner.LeverMatrixSearch",
            _FakeLeverMatrixSearch,
        )

        storage_path = tmp_path / ".ash-hawk" / "trace-logs"
        result = await run_enhanced_cycle(
            scenarios=[scenario_path],
            config=EnhancedCycleConfig(
                enable_multi_target=False,
                enable_intent_analysis=False,
                enable_knowledge_promotion=False,
                enable_lever_search=True,
                iterations_per_target=2,
            ),
            project_root=tmp_path,
            storage_path=storage_path,
            llm_client=object(),
        )

        assert isinstance(result.lever_result, EvolvableCycleResult)
        log_path = storage_path / "evolvable-search" / ".ash-hawk" / "evolvable-experiments.jsonl"
        assert log_path.exists()
        raw_lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line]
        assert raw_lines
        for raw_line in raw_lines:
            payload = json.loads(raw_line)
            assert "event_type" in payload
            assert "ts" in payload

    @pytest.mark.asyncio
    async def test_evolvable_phase_reverts_on_safety_threshold(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
    ) -> None:
        scenario_path = tmp_path / "scenario.yaml"
        scenario_path.write_text("id: test\n", encoding="utf-8")

        class _FakeTargetDiscovery:
            def __init__(self, project_root: Path) -> None:
                pass

            def discover_all_targets(self) -> list[_DummyTarget]:
                return [_DummyTarget(name="safe-skill", discovered_path=tmp_path)]

        class _FakeMultiTargetCycleRunner:
            def __init__(self, **kwargs: object) -> None:
                pass

            async def run_all_targets(
                self,
                scenarios: list[Path],
                targets: list[_DummyTarget],
                iterations_per_target: int,
                threshold: float,
                storage_path: Path,
            ) -> MultiTargetResult:
                target = targets[0]
                return MultiTargetResult(
                    agent_name=target.name,
                    target_results={
                        target.name: CycleResult(
                            agent_name=target.name,
                            target_path=str(target.discovered_path),
                            scenario_paths=[str(s) for s in scenarios],
                            status=CycleStatus.COMPLETED,
                            initial_score=0.4,
                            final_score=0.5,
                            target_type=TargetType.SKILL,
                        )
                    },
                    overall_improvement=0.1,
                    converged=False,
                )

        class _FakeLeverMatrixSearch:
            def __init__(self, lever_space: dict[str, object] | None = None) -> None:
                self.lever_space = lever_space or {"agent": object()}
                self._scores = [0.52, -0.1]

            def sample_random(self) -> _FakeLeverConfig:
                return _FakeLeverConfig("baseline")

            def sample_neighbors(
                self,
                config: _FakeLeverConfig,
                n: int = 1,
            ) -> list[_FakeLeverConfig]:
                return [_FakeLeverConfig("risky")]

            async def evaluate(
                self,
                config: _FakeLeverConfig,
                scenarios: list[Path],
                storage_path: Path,
            ) -> float:
                return self._scores.pop(0)

        monkeypatch.setattr(
            "ash_hawk.auto_research.enhanced_cycle_runner.TargetDiscovery",
            _FakeTargetDiscovery,
        )
        monkeypatch.setattr(
            "ash_hawk.auto_research.enhanced_cycle_runner.MultiTargetCycleRunner",
            _FakeMultiTargetCycleRunner,
        )
        monkeypatch.setattr(
            "ash_hawk.auto_research.enhanced_cycle_runner.LeverMatrixSearch",
            _FakeLeverMatrixSearch,
        )

        result = await run_enhanced_cycle(
            scenarios=[scenario_path],
            config=EnhancedCycleConfig(
                enable_multi_target=False,
                enable_intent_analysis=False,
                enable_knowledge_promotion=False,
                enable_lever_search=True,
                iterations_per_target=2,
            ),
            project_root=tmp_path,
            storage_path=tmp_path / ".ash-hawk" / "safety-threshold",
            llm_client=object(),
        )

        assert isinstance(result.lever_result, EvolvableCycleResult)
        assert result.lever_result.reverted_experiments > 0
