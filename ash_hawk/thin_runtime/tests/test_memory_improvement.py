from __future__ import annotations

from pathlib import Path

import pytest

from ash_hawk.thin_runtime import RuntimeGoal, create_default_harness
from ash_hawk.thin_runtime.context import RuntimeContextAssembler
from ash_hawk.thin_runtime.models import ToolCall, ToolResult
from ash_hawk.thin_runtime.planner import PlannerDecision, plan_next_tool
from ash_hawk.thin_runtime.tool_impl.search_knowledge import run as search_knowledge_run
from ash_hawk.thin_runtime.tool_types import (
    AuditRunResult,
    AuditToolContext,
    EvaluationToolContext,
    FailureToolContext,
    RankedHypothesis,
    RuntimeToolContext,
    ScoreSummary,
    ToolExecutionPayload,
    WorkspaceToolContext,
)


def test_search_knowledge_surfaces_relevant_memory_entries() -> None:
    call = ToolCall.model_validate(
        {
            "tool_name": "search_knowledge",
            "goal_id": "goal-memory-search",
            "agent_text": "Use targeted tests and clear status updates for people.",
            "context": {
                "memory": {
                    "semantic_memory": {
                        "rules": ["Prefer targeted tests for fast feedback"],
                        "boosts": [],
                        "penalties": ["Avoid broad reruns when a narrower check exists"],
                    },
                    "personal_memory": {
                        "preferences": ["Use plain language for human-facing status updates"]
                    },
                    "episodic_memory": {"episodes": ["Targeted validation fixed the issue faster"]},
                },
                "failure": {
                    "explanations": ["targeted tests should run before broader validation"]
                },
            },
        }
    )

    result = search_knowledge_run(call)

    assert result.success is True
    assert result.payload.memory_updates.search_results
    assert any(
        "Prefer targeted tests for fast feedback" in item
        for item in result.payload.memory_updates.search_results
    )
    assert any(
        "Use plain language for human-facing status updates" in item
        for item in result.payload.memory_updates.search_results
    )


def test_agent_text_uses_persisted_memory_after_dream_state(tmp_path: Path) -> None:
    writer = create_default_harness(workdir=Path.cwd(), storage_root=tmp_path)
    writer.memory.write_scope(
        "personal_memory",
        {"preferences": ["Meet the user where they are and explain decisions plainly"]},
    )
    writer.persistence.save_dream_queue(
        [
            {
                "scope": "semantic_memory",
                "key": "rules",
                "value": "Prefer targeted tests for faster human feedback loops",
            }
        ]
    )
    writer.runner.dream_state.run()

    reader = create_default_harness(workdir=Path.cwd(), storage_root=tmp_path)
    result = reader.execute(
        RuntimeGoal(goal_id="goal-dream-read", description="Use what was learned before"),
        tool_execution_order=["load_workspace_state"],
    )

    assert "Memory guidance:" in result.context.runtime["agent_text"]
    assert (
        "Meet the user where they are and explain decisions plainly"
        in result.context.runtime["agent_text"]
    )
    assert (
        "Prefer targeted tests for faster human feedback loops"
        in result.context.runtime["agent_text"]
    )
    assert (
        "Prefer targeted tests for faster human feedback loops"
        in result.memory_snapshot["semantic_memory"]["rules"]
    )


def test_model_planner_selects_valid_tool_from_allowed_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = create_default_harness(workdir=Path.cwd())
    harness.memory.write_scope(
        "semantic_memory",
        {"rules": ["Prefer targeted tests for fast feedback"], "boosts": [], "penalties": []},
    )
    harness.memory.write_scope(
        "personal_memory",
        {"preferences": ["Use plain language for people reading the output"]},
    )

    goal = RuntimeGoal(goal_id="goal-policy-memory", description="Choose the next best tool")
    agent = harness.agents.get("coordinator")
    active_skills = [harness.skills.get("baseline-evaluation"), harness.skills.get("verification")]
    active_tools = [harness.tools.get("run_baseline_eval"), harness.tools.get("verify_outcome")]
    context = RuntimeContextAssembler().assemble(
        goal=goal,
        agent=agent,
        skills=active_skills,
        tools=active_tools,
        memory_snapshot=harness.memory.snapshot(),
        workdir=Path.cwd(),
    )
    context.runtime["agent_text"] = (
        "Memory guidance:\n- Prefer targeted tests for fast feedback\n"
        "- Use plain language for people reading the output"
    )

    def fake_call_model_structured(*_args: object, **_kwargs: object) -> PlannerDecision:
        return PlannerDecision(
            selected_tool="verify_outcome",
            activate_skills=["verification"],
            source="model_planner",
            rationale="Verification is the best next action for this run.",
            considered_tools=["run_baseline_eval", "verify_outcome"],
            confidence=0.83,
        )

    monkeypatch.setattr(
        "ash_hawk.thin_runtime.planner.call_model_structured",
        fake_call_model_structured,
    )

    decision = plan_next_tool(
        goal=goal,
        agent=agent,
        active_skills=active_skills,
        candidate_skills=active_skills,
        candidate_tools=active_tools,
        context=context,
        tool_execution_order=None,
    )

    assert decision.source == "model_planner"
    assert decision.selected_tool == "verify_outcome"
    assert decision.activate_skills == ["verification"]
    assert decision.considered_tools == ["run_baseline_eval", "verify_outcome"]


def test_normal_execution_persists_canonical_memory_for_future_runs(tmp_path: Path) -> None:
    writer = create_default_harness(workdir=Path.cwd(), storage_root=tmp_path)
    memory_manager = writer.agents.get("memory_manager")
    memory_manager.available_tools = ["search_knowledge"]
    memory_update_skill = writer.skills.get("memory-update")
    memory_update_skill.tool_names = ["search_knowledge"]
    memory_update_skill.memory_write_scopes = ["episodic_memory"]

    writer.execute(
        RuntimeGoal(goal_id="goal-write-memory", description="Learn from a memory search run"),
        agent_name="memory_manager",
        requested_skills=["memory-update"],
        tool_execution_order=["search_knowledge"],
    )

    reader = create_default_harness(workdir=Path.cwd(), storage_root=tmp_path)
    result = reader.execute(
        RuntimeGoal(goal_id="goal-read-memory", description="Use learned memory guidance"),
        tool_execution_order=["load_workspace_state"],
    )

    assert result.memory_snapshot["episodic_memory"]["episodes"]
    assert "Memory guidance:" in result.context.runtime["agent_text"]
    assert "goal-write-memory" in result.context.runtime["agent_text"]


test_agent_text_uses_persisted_memory_after_dream_state = pytest.mark.e2e(
    test_agent_text_uses_persisted_memory_after_dream_state
)
test_normal_execution_persists_canonical_memory_for_future_runs = pytest.mark.e2e(
    test_normal_execution_persists_canonical_memory_for_future_runs
)


def test_improver_records_survivor_memory_for_future_iterations(tmp_path: Path) -> None:
    harness = create_default_harness(workdir=tmp_path, storage_root=tmp_path / ".ash-hawk-test")
    improver = harness.agents.get("improver")
    active_skills = [harness.skills.get("context-assembly"), harness.skills.get("improvement-loop")]
    active_tools = [
        harness.tools.get("call_llm_structured"),
        harness.tools.get("run_eval_repeated"),
        harness.tools.get("sync_workspace_changes"),
    ]
    context = RuntimeContextAssembler().assemble(
        goal=RuntimeGoal(goal_id="goal-improver-memory", description="Record survivor genes"),
        agent=improver,
        skills=active_skills,
        tools=active_tools,
        memory_snapshot=harness.memory.snapshot(),
        workdir=tmp_path,
    )
    context.failure["failure_family"] = "needs_improvement"
    context.failure["ranked_hypotheses"] = [
        RankedHypothesis(
            name="Teach direct-context priority",
            score=0.91,
            rationale="Direct file reads are the missing behavior.",
            target_files=["agent.md"],
            ideal_outcome="The coding agent reads the named file before broad search.",
        )
    ]
    context.workspace["allowed_target_files"] = ["agent.md"]
    context.workspace["mutated_files"] = ["agent.md"]
    context.runtime["last_delegation_summary"] = "executor mutated agent.md"
    context.evaluation["baseline_summary"] = ScoreSummary(
        score=0.55,
        status="completed",
        tool="run_baseline_eval",
    ).model_dump()
    context.evaluation["repeat_eval_summary"] = ScoreSummary(
        score=0.78,
        status="completed",
        tool="run_eval_repeated",
    ).model_dump()

    harness.runner.record_improver_learning(
        agent=improver,
        tool_name="call_llm_structured",
        context=context,
        result=ToolResult(
            tool_name="call_llm_structured", success=True, payload=ToolExecutionPayload()
        ),
    )
    harness.runner.record_improver_learning(
        agent=improver,
        tool_name="run_eval_repeated",
        context=context,
        result=ToolResult(
            tool_name="run_eval_repeated", success=True, payload=ToolExecutionPayload()
        ),
    )
    harness.runner.record_improver_learning(
        agent=improver,
        tool_name="sync_workspace_changes",
        context=context,
        result=ToolResult(
            tool_name="sync_workspace_changes", success=True, payload=ToolExecutionPayload()
        ),
    )
    queued_items = harness.memory.read_scope("session_memory").get("dream_queue", [])
    assert isinstance(queued_items, list) and queued_items
    harness.persistence.save_dream_queue([item for item in queued_items if isinstance(item, dict)])
    dream_result = harness.runner.dream_state.run()

    assert dream_result["applied"] >= 1
    memory_snapshot = harness.memory.snapshot()
    working_memory = memory_snapshot["working_memory"]
    assert working_memory["active_hypotheses"][0]["name"] == "Teach direct-context priority"
    assert working_memory["last_result"]["status"] == "improved"
    assert working_memory["last_result"]["score_delta"] == pytest.approx(0.23)

    validations = memory_snapshot["session_memory"]["validations"]
    assert validations[-1]["hypothesis"] == "Teach direct-context priority"

    semantic_rules = memory_snapshot["semantic_memory"]["rules"]
    assert any(
        isinstance(item, dict)
        and item.get("entry_type") == "survivor_gene"
        and item.get("hypothesis") == "Teach direct-context priority"
        for item in semantic_rules
    )
