from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from ash_hawk.auto_research.enhanced_cycle_runner import (
    _resolve_agent_startup_details,
    run_enhanced_cycle,
)
from ash_hawk.auto_research.types import (
    CycleResult,
    CycleStatus,
    EnhancedCycleConfig,
    MultiTargetResult,
    TargetType,
)


@dataclass
class _DummyTarget:
    name: str
    discovered_path: Path
    target_type: TargetType = TargetType.SKILL


@pytest.mark.asyncio
async def test_run_enhanced_cycle_single_target_executes_one_target(
    tmp_path: Path,
    monkeypatch,
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
        def __init__(self, **kwargs) -> None:
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


def test_resolve_agent_startup_details_from_scenario_config(monkeypatch) -> None:
    scenario_stub = SimpleNamespace(
        sut=SimpleNamespace(
            adapter="bolt_merlin",
            config={
                "agent": "orchestrator",
                "run_config": {"agent_name": "bolt-merlin", "agent_version": "v2"},
            },
        )
    )

    monkeypatch.setattr(
        "ash_hawk.auto_research.enhanced_cycle_runner.load_scenario",
        lambda _: scenario_stub,
    )

    agent_name, agent_version = _resolve_agent_startup_details([Path("scenario.yaml")])

    assert agent_name == "bolt-merlin"
    assert agent_version == "v2"
