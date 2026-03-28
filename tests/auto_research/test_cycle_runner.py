from __future__ import annotations

from pathlib import Path

import pytest

from ash_hawk.auto_research.cycle_runner import (
    ImprovementTarget,
    TargetType,
    _check_convergence,
    _format_activity_glyphs,
    _run_iteration,
    run_cycle,
)
from ash_hawk.auto_research.types import CycleStatus, IterationResult
from ash_hawk.services.dawn_kestrel_injector import DawnKestrelInjector


@pytest.mark.asyncio
async def test_run_iteration_resets_current_skill_on_revert(tmp_path: Path, monkeypatch) -> None:
    injector = DawnKestrelInjector(project_root=tmp_path)
    base_skill_path = injector.save_skill_content("base-skill", "# Base\n\nbase")
    injector.current_skill_name = "base-skill"

    target = ImprovementTarget(
        target_type=TargetType.SKILL,
        name="base-skill",
        discovered_path=base_skill_path,
        injector=injector,
    )

    async def _fake_generate_improvement(*args, **kwargs) -> str:
        return "---\nname: temp-skill\n---\n# Temp\n\ntemp"

    async def _fake_run_evaluation(*args, **kwargs) -> tuple[float, list[object]]:
        return 0.0, []

    monkeypatch.setattr(
        "ash_hawk.auto_research.cycle_runner.generate_improvement",
        _fake_generate_improvement,
    )
    monkeypatch.setattr(
        "ash_hawk.auto_research.cycle_runner._run_evaluation",
        _fake_run_evaluation,
    )

    result = await _run_iteration(
        iteration_num=1,
        target=target,
        scenarios=[],
        storage=tmp_path,
        llm_client=None,
        score_before=0.1,
        cached_transcripts=[],
        threshold=0.02,
        failed_proposals=None,
        consecutive_failures=0,
        existing_skills=["base-skill"],
    )

    assert result.applied is False
    assert injector.current_skill_name == "base-skill"
    assert not injector.get_skill_path("temp-skill").exists()


@pytest.mark.asyncio
async def test_run_cycle_prefers_explicit_target_over_discovery(
    tmp_path: Path, monkeypatch
) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text("id: explicit-target\n", encoding="utf-8")

    injector = DawnKestrelInjector(project_root=tmp_path)
    explicit_target = injector.save_skill_content("explicit-skill", "# Explicit\n\ncontent")

    def _unexpected_discovery(*args, **kwargs):
        msg = "scenario discovery should not run when explicit target is provided"
        raise AssertionError(msg)

    async def _fake_run_evaluation(*args, **kwargs) -> tuple[float, list[object], dict[str, float]]:
        return 0.6, [], {}

    monkeypatch.setattr(
        "ash_hawk.auto_research.cycle_runner._discover_improvement_target",
        _unexpected_discovery,
    )
    monkeypatch.setattr(
        "ash_hawk.auto_research.cycle_runner._run_evaluation",
        _fake_run_evaluation,
    )

    result = await run_cycle(
        scenarios=[scenario_path],
        iterations=0,
        threshold=0.02,
        storage_path=tmp_path,
        llm_client=object(),
        project_root=tmp_path,
        explicit_targets=[explicit_target],
    )

    assert result.status == CycleStatus.COMPLETED
    assert "explicit-skill" in result.target_path


@pytest.mark.asyncio
async def test_run_cycle_sets_target_type_from_explicit_tool_target(
    tmp_path: Path, monkeypatch
) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text("id: explicit-tool-target\n", encoding="utf-8")

    injector = DawnKestrelInjector(project_root=tmp_path)
    explicit_target = injector.save_tool_content("bash", "# Tool\n\ncontent")

    async def _fake_run_evaluation(*args, **kwargs) -> tuple[float, list[object], dict[str, float]]:
        return 0.6, [], {}

    monkeypatch.setattr(
        "ash_hawk.auto_research.cycle_runner._run_evaluation",
        _fake_run_evaluation,
    )

    result = await run_cycle(
        scenarios=[scenario_path],
        iterations=0,
        threshold=0.02,
        storage_path=tmp_path,
        llm_client=object(),
        project_root=tmp_path,
        explicit_targets=[explicit_target],
    )

    assert result.status == CycleStatus.COMPLETED
    assert result.target_type == TargetType.TOOL


@pytest.mark.asyncio
async def test_run_iteration_keeps_tool_target_type_for_new_named_target(
    tmp_path: Path, monkeypatch
) -> None:
    injector = DawnKestrelInjector(project_root=tmp_path)
    base_tool_path = injector.save_tool_content("base-tool", "# Base Tool\n\nbase")

    target = ImprovementTarget(
        target_type=TargetType.TOOL,
        name="base-tool",
        discovered_path=base_tool_path,
        injector=injector,
    )

    async def _fake_generate_improvement(*args, **kwargs) -> str:
        return "---\nname: alt-tool\n---\n# Alt Tool\n\ncontent"

    async def _fake_run_evaluation(*args, **kwargs) -> tuple[float, list[object]]:
        return 0.0, []

    monkeypatch.setattr(
        "ash_hawk.auto_research.cycle_runner.generate_improvement",
        _fake_generate_improvement,
    )
    monkeypatch.setattr(
        "ash_hawk.auto_research.cycle_runner._run_evaluation",
        _fake_run_evaluation,
    )

    result = await _run_iteration(
        iteration_num=1,
        target=target,
        scenarios=[],
        storage=tmp_path,
        llm_client=None,
        score_before=0.1,
        cached_transcripts=[],
        threshold=0.02,
        failed_proposals=None,
        consecutive_failures=0,
        existing_skills=None,
    )

    assert result.applied is False
    assert not injector.get_tool_path("alt-tool").exists()


def test_format_activity_glyphs_includes_completed_and_running() -> None:
    assert _format_activity_glyphs(2, 4, "/") == "..////"


def test_format_activity_glyphs_caps_and_shows_running_overflow() -> None:
    assert _format_activity_glyphs(9, 7, "|") == ".....|||||+2"


def test_check_convergence_detects_plateau() -> None:
    iterations = [
        IterationResult(iteration_num=0, score_before=0.70, score_after=0.7001, applied=True),
        IterationResult(iteration_num=1, score_before=0.7001, score_after=0.7002, applied=True),
        IterationResult(iteration_num=2, score_before=0.7002, score_after=0.7001, applied=True),
        IterationResult(iteration_num=3, score_before=0.7001, score_after=0.7000, applied=True),
        IterationResult(iteration_num=4, score_before=0.7000, score_after=0.7001, applied=True),
    ]

    assert _check_convergence(iterations) is True


def test_check_convergence_returns_false_when_improving() -> None:
    iterations = [
        IterationResult(iteration_num=0, score_before=0.50, score_after=0.55, applied=True),
        IterationResult(iteration_num=1, score_before=0.55, score_after=0.60, applied=True),
        IterationResult(iteration_num=2, score_before=0.60, score_after=0.65, applied=True),
        IterationResult(iteration_num=3, score_before=0.65, score_after=0.70, applied=True),
        IterationResult(iteration_num=4, score_before=0.70, score_after=0.75, applied=True),
    ]

    assert _check_convergence(iterations) is False
