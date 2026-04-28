from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from ash_hawk.thin_runtime import RuntimeGoal, create_default_harness
from ash_hawk.thin_runtime.context import RuntimeContextAssembler
from ash_hawk.thin_runtime.models import AgentSpec, ContextSnapshot, SkillSpec, ToolSpec
from ash_hawk.thin_runtime.planner import PlannerDecision, build_planner_prompt, plan_next_tool


def _planner_inputs() -> tuple[
    RuntimeGoal,
    AgentSpec,
    list[SkillSpec],
    list[SkillSpec],
    list[ToolSpec],
    ContextSnapshot,
]:
    harness = create_default_harness(console_output=False)
    goal = RuntimeGoal(goal_id="goal-planner", description="Choose the next thin-runtime tool")
    agent = harness.agents.get("coordinator")
    active_skills = [harness.skills.get("baseline-evaluation"), harness.skills.get("verification")]
    candidate_skills = active_skills + [harness.skills.get("workspace-governance")]
    candidate_tools = [
        harness.tools.get("run_baseline_eval"),
        harness.tools.get("verify_outcome"),
        harness.tools.get("load_workspace_state"),
    ]
    context = RuntimeContextAssembler().assemble(
        goal=goal,
        agent=agent,
        skills=active_skills,
        tools=candidate_tools,
        memory_snapshot=harness.memory.snapshot(),
        workdir=harness.workdir,
    )
    return goal, agent, active_skills, candidate_skills, candidate_tools, context


def test_plan_next_tool_accepts_valid_model_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    goal, agent, active_skills, candidate_skills, candidate_tools, context = _planner_inputs()

    def fake_call_model_structured(*_args: object, **_kwargs: object) -> PlannerDecision:
        return PlannerDecision(
            selected_tool="verify_outcome",
            activate_skills=["verification"],
            source="model_planner",
            rationale="Verification is the correct next step.",
            considered_tools=["run_baseline_eval", "verify_outcome", "load_workspace_state"],
            confidence=0.82,
        )

    monkeypatch.setattr(
        "ash_hawk.thin_runtime.planner.call_model_structured", fake_call_model_structured
    )

    decision = plan_next_tool(
        goal=goal,
        agent=agent,
        active_skills=active_skills,
        candidate_skills=candidate_skills,
        candidate_tools=candidate_tools,
        context=context,
        tool_execution_order=None,
    )

    assert decision.source == "model_planner"
    assert decision.selected_tool == "verify_outcome"
    assert decision.activate_skills == ["verification"]


def test_plan_next_tool_rejects_unknown_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    goal, agent, active_skills, candidate_skills, candidate_tools, context = _planner_inputs()

    def fake_call_model_structured(*_args: object, **_kwargs: object) -> PlannerDecision:
        return PlannerDecision(
            selected_tool="bash",
            activate_skills=[],
            source="model_planner",
            rationale="Use bash.",
            considered_tools=["run_baseline_eval", "verify_outcome", "load_workspace_state"],
        )

    monkeypatch.setattr(
        "ash_hawk.thin_runtime.planner.call_model_structured", fake_call_model_structured
    )

    decision = plan_next_tool(
        goal=goal,
        agent=agent,
        active_skills=active_skills,
        candidate_skills=candidate_skills,
        candidate_tools=candidate_tools,
        context=context,
        tool_execution_order=None,
    )

    assert decision.source == "invalid_model_selection"
    assert decision.selected_tool is None


def test_plan_next_tool_rejects_unknown_skill_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    goal, agent, active_skills, candidate_skills, candidate_tools, context = _planner_inputs()

    def fake_call_model_structured(*_args: object, **_kwargs: object) -> PlannerDecision:
        return PlannerDecision(
            selected_tool="load_workspace_state",
            activate_skills=["nonexistent-skill"],
            source="model_planner",
            rationale="Need a custom skill.",
            considered_tools=["run_baseline_eval", "verify_outcome", "load_workspace_state"],
        )

    monkeypatch.setattr(
        "ash_hawk.thin_runtime.planner.call_model_structured", fake_call_model_structured
    )

    decision = plan_next_tool(
        goal=goal,
        agent=agent,
        active_skills=active_skills,
        candidate_skills=candidate_skills,
        candidate_tools=candidate_tools,
        context=context,
        tool_execution_order=None,
    )

    assert decision.source == "invalid_model_skills"
    assert decision.selected_tool is None


def test_plan_next_tool_reports_model_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    goal, agent, active_skills, candidate_skills, candidate_tools, context = _planner_inputs()

    def fake_call_model_structured(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "ash_hawk.thin_runtime.planner.call_model_structured",
        fake_call_model_structured,
    )

    decision = plan_next_tool(
        goal=goal,
        agent=agent,
        active_skills=active_skills,
        candidate_skills=candidate_skills,
        candidate_tools=candidate_tools,
        context=context,
        tool_execution_order=None,
    )

    assert decision.source == "model_unavailable"
    assert decision.selected_tool is None


def test_plan_next_tool_uses_bolt_fallback_when_dawn_client_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    goal, agent, active_skills, candidate_skills, candidate_tools, context = _planner_inputs()

    fake_execute_module = ModuleType("bolt_merlin.agent.execute")

    async def fake_execute(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            response=(
                '{"selected_tool": "verify_outcome", '
                '"activate_skills": ["verification"], '
                '"rationale": "Use verification after the baseline context is available.", '
                '"confidence": 0.91}'
            )
        )

    fake_execute_module.execute = fake_execute  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bolt_merlin", ModuleType("bolt_merlin"))
    monkeypatch.setitem(sys.modules, "bolt_merlin.agent", ModuleType("bolt_merlin.agent"))
    monkeypatch.setitem(sys.modules, "bolt_merlin.agent.execute", fake_execute_module)

    def fake_import_module(name: str) -> object:
        if name in {"dawn_kestrel.base.config", "dawn_kestrel.provider.llm_client"}:
            raise ImportError(name)
        raise AssertionError(f"Unexpected importlib request: {name}")

    monkeypatch.setattr(
        "ash_hawk.thin_runtime.llm_client.importlib.import_module",
        fake_import_module,
    )

    decision = plan_next_tool(
        goal=goal,
        agent=agent,
        active_skills=active_skills,
        candidate_skills=candidate_skills,
        candidate_tools=candidate_tools,
        context=context,
        tool_execution_order=None,
    )

    assert decision.source == "model_planner"
    assert decision.selected_tool == "verify_outcome"
    assert decision.activate_skills == ["verification"]


def test_build_planner_prompt_compacts_large_context() -> None:
    goal, agent, active_skills, candidate_skills, candidate_tools, context = _planner_inputs()
    huge_chunk = "path/to/run.json " * 10000
    context.memory["session"] = {
        "transcripts": [huge_chunk for _ in range(20)],
        "traces": [huge_chunk for _ in range(20)],
        "delegations": [{"agent": "child", "details": huge_chunk} for _ in range(10)],
    }
    context.audit["events"] = [{"preview": huge_chunk} for _ in range(50)]
    context.audit["transcripts"] = [huge_chunk for _ in range(30)]
    context.workspace["scenario_summary"] = huge_chunk

    prompt = build_planner_prompt(
        goal=goal,
        agent=agent,
        active_skills=active_skills,
        candidate_skills=candidate_skills,
        candidate_tools=candidate_tools,
        context=context,
        feedback=None,
    )

    assert len(prompt) < 30_000
    assert huge_chunk not in prompt
    assert '"transcript_count": 30' in prompt
