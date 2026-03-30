from __future__ import annotations

from pathlib import Path

import pytest

from ash_hawk.auto_research.cycle_runner import (
    ImprovementTarget,
    TargetType,
    _check_convergence,
    _discover_improvement_target,
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

    async def _fake_run_evaluation(*args, **kwargs) -> tuple[float, list[object], dict[str, float]]:
        return 0.0, [], {}

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
    _write_auto_research_scenario(scenario_path)

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
    _write_auto_research_scenario(scenario_path)

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

    async def _fake_run_evaluation(*args, **kwargs) -> tuple[float, list[object], dict[str, float]]:
        return 0.0, [], {}

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


def test_discover_tool_target_ignores_scenario_allowed_tools(tmp_path: Path) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text("id: discover-tool\n", encoding="utf-8")

    tool_dir = tmp_path / ".dawn-kestrel" / "tools" / "bash"
    tool_dir.mkdir(parents=True)
    (tool_dir / "TOOL.md").write_text("# Bash tool\n", encoding="utf-8")

    target = _discover_improvement_target(
        scenarios=[scenario_path],
        project_root=tmp_path,
        target_types=[TargetType.TOOL],
    )

    assert target is not None
    assert target.target_type == TargetType.TOOL
    assert target.name == "bash"


@pytest.mark.asyncio
async def test_run_iteration_hill_climb_selects_best_target_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    injector = DawnKestrelInjector(project_root=tmp_path)
    first_skill = injector.save_skill_content("first-skill", "# First\n\nfirst")
    second_skill = injector.save_skill_content("second-skill", "# Second\n\nsecond")

    first_target = ImprovementTarget(
        target_type=TargetType.SKILL,
        name="first-skill",
        discovered_path=first_skill,
        injector=injector,
    )
    second_target = ImprovementTarget(
        target_type=TargetType.SKILL,
        name="second-skill",
        discovered_path=second_skill,
        injector=injector,
    )

    async def _fake_filter(
        transcripts: list[object],
    ) -> tuple[list[object], list[dict[str, object]]]:
        return transcripts, []

    async def _fake_generate_improvement(
        llm_client: object,
        original_content: str,
        transcripts: list[object],
        failed_proposals: list[str] | None,
        consecutive_failures: int,
        **kwargs,
    ) -> str:
        _ = llm_client, transcripts, failed_proposals, consecutive_failures, kwargs
        if "# First" in original_content:
            return "# First\n\nfirst-improved"
        return "# Second\n\nsecond-improved"

    async def _fake_run_evaluation(*args, **kwargs) -> tuple[float, list[object], dict[str, float]]:
        _ = args, kwargs
        first_content = injector.get_skill_path("first-skill").read_text(encoding="utf-8")
        second_content = injector.get_skill_path("second-skill").read_text(encoding="utf-8")
        if "second-improved" in second_content:
            return 0.66, [], {"quality": 0.66}
        if "first-improved" in first_content:
            return 0.58, [], {"quality": 0.58}
        return 0.50, [], {"quality": 0.50}

    monkeypatch.setattr(
        "ash_hawk.auto_research.cycle_runner._filter_valid_transcripts", _fake_filter
    )
    monkeypatch.setattr(
        "ash_hawk.auto_research.cycle_runner.generate_improvement",
        _fake_generate_improvement,
    )
    monkeypatch.setattr("ash_hawk.auto_research.cycle_runner._run_evaluation", _fake_run_evaluation)

    result = await _run_iteration(
        iteration_num=1,
        target=first_target,
        candidate_targets=[first_target, second_target],
        max_candidate_updates=2,
        scenarios=[],
        storage=tmp_path,
        llm_client=object(),
        score_before=0.50,
        cached_transcripts=[object()],
        threshold=0.02,
        failed_proposals=None,
        consecutive_failures=0,
        existing_skills=["first-skill", "second-skill"],
        category_scores={"quality": 0.50},
    )

    assert result.applied is True
    assert result.score_after == pytest.approx(0.66)
    assert (
        injector.get_skill_path("second-skill").read_text(encoding="utf-8")
        == "# Second\n\nsecond-improved"
    )
    assert injector.get_skill_path("first-skill").read_text(encoding="utf-8") == "# First\n\nfirst"


def _write_auto_research_scenario(path: Path, extra_block: str = "") -> None:
    path.write_text(
        (
            'schema_version: "v1"\n'
            "id: auto-research-minimal\n"
            "description: Minimal scenario for auto-research\n"
            "sut:\n"
            '  type: "agentic_sdk"\n'
            '  adapter: "dawn_kestrel"\n'
            "  config: {}\n"
            "inputs:\n"
            '  intent: "Fix Python bug and validate with tests"\n'
            '  prompt: "Fix the bug with minimal changes"\n'
            "graders:\n"
            '  - grader_type: "test_runner"\n'
            "    config:\n"
            '      test_path: "./tests/test_bug.py"\n'
            f"{extra_block}"
        ),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_run_cycle_accepts_minimal_intent_plus_graders_scenario(
    tmp_path: Path, monkeypatch
) -> None:
    scenario_path = tmp_path / "minimal.scenario.yaml"
    _write_auto_research_scenario(scenario_path)

    injector = DawnKestrelInjector(project_root=tmp_path)
    explicit_target = injector.save_skill_content("minimal-skill", "# Minimal\n\ncontent")

    async def _fake_run_evaluation(*args, **kwargs) -> tuple[float, list[object], dict[str, float]]:
        _ = args, kwargs
        return 0.75, [], {}

    monkeypatch.setattr("ash_hawk.auto_research.cycle_runner._run_evaluation", _fake_run_evaluation)

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
    assert result.initial_score == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_run_cycle_rejects_extra_scenario_spec_sections(tmp_path: Path) -> None:
    scenario_path = tmp_path / "extra-spec.scenario.yaml"
    _write_auto_research_scenario(
        scenario_path,
        extra_block=("tools:\n  allowed_tools:\n    - bash\n"),
    )

    with pytest.raises(ValueError, match="disallowed top-level keys"):
        await run_cycle(
            scenarios=[scenario_path],
            iterations=0,
            storage_path=tmp_path,
            llm_client=object(),
            project_root=tmp_path,
        )


@pytest.mark.asyncio
async def test_run_cycle_accepts_benign_metadata_key(tmp_path: Path, monkeypatch) -> None:
    scenario_path = tmp_path / "metadata-ok.scenario.yaml"
    _write_auto_research_scenario(
        scenario_path,
        extra_block="metadata:\n  owner: auto-research\n",
    )

    injector = DawnKestrelInjector(project_root=tmp_path)
    explicit_target = injector.save_skill_content("metadata-skill", "# Metadata\n\ncontent")

    async def _fake_run_evaluation(*args, **kwargs) -> tuple[float, list[object], dict[str, float]]:
        _ = args, kwargs
        return 0.70, [], {}

    monkeypatch.setattr("ash_hawk.auto_research.cycle_runner._run_evaluation", _fake_run_evaluation)

    result = await run_cycle(
        scenarios=[scenario_path],
        iterations=0,
        storage_path=tmp_path,
        llm_client=object(),
        project_root=tmp_path,
        explicit_targets=[explicit_target],
    )

    assert result.status == CycleStatus.COMPLETED


@pytest.mark.asyncio
async def test_run_cycle_accepts_empty_grader_config(tmp_path: Path, monkeypatch) -> None:
    scenario_path = tmp_path / "empty-grader-config.scenario.yaml"
    scenario_path.write_text(
        (
            'schema_version: "v1"\n'
            "id: empty-grader-config\n"
            "description: Empty config grader is valid\n"
            "sut:\n"
            '  type: "agentic_sdk"\n'
            '  adapter: "dawn_kestrel"\n'
            "  config: {}\n"
            "inputs:\n"
            '  intent: "Validate behavior with empty grader config"\n'
            '  prompt: "Run and validate"\n'
            "graders:\n"
            '  - grader_type: "trace_schema"\n'
            "    config: {}\n"
        ),
        encoding="utf-8",
    )

    injector = DawnKestrelInjector(project_root=tmp_path)
    explicit_target = injector.save_skill_content("trace-skill", "# Trace\n\ncontent")

    async def _fake_run_evaluation(*args, **kwargs) -> tuple[float, list[object], dict[str, float]]:
        _ = args, kwargs
        return 0.64, [], {}

    monkeypatch.setattr("ash_hawk.auto_research.cycle_runner._run_evaluation", _fake_run_evaluation)

    result = await run_cycle(
        scenarios=[scenario_path],
        iterations=0,
        storage_path=tmp_path,
        llm_client=object(),
        project_root=tmp_path,
        explicit_targets=[explicit_target],
    )

    assert result.status == CycleStatus.COMPLETED


@pytest.mark.asyncio
async def test_run_cycle_requires_inputs_intent(tmp_path: Path) -> None:
    scenario_path = tmp_path / "missing-intent.scenario.yaml"
    scenario_path.write_text(
        (
            'schema_version: "v1"\n'
            "id: missing-intent\n"
            "description: Missing intent should fail\n"
            "sut:\n"
            '  type: "agentic_sdk"\n'
            '  adapter: "dawn_kestrel"\n'
            "  config: {}\n"
            "inputs:\n"
            '  prompt: "Fix the bug"\n'
            "graders:\n"
            '  - grader_type: "test_runner"\n'
            "    config:\n"
            '      test_path: "./tests/test_bug.py"\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="inputs.intent"):
        await run_cycle(
            scenarios=[scenario_path],
            iterations=0,
            storage_path=tmp_path,
            llm_client=object(),
            project_root=tmp_path,
        )
