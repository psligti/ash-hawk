from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from ash_hawk.thin_runtime import RuntimeGoal, create_default_harness
from ash_hawk.thin_runtime.models import ToolResult
from ash_hawk.thin_runtime.tool_types import (
    AuditRunResult,
    AuditToolContext,
    EvaluationToolContext,
    ScoreSummary,
    ToolExecutionPayload,
    VerificationStatus,
    WorkspaceToolContext,
)


def test_harness_executes_agentic_run_with_skills_tools_and_context() -> None:
    harness = create_default_harness(workdir=Path.cwd())
    goal = RuntimeGoal(goal_id="goal-1", description="Exercise thin runtime objects")

    result = harness.execute(goal)

    assert result.agent.name == "coordinator"
    assert result.active_skills
    assert result.tools
    assert result.selected_tool_names
    assert result.available_contexts
    assert result.context.goal["goal_id"] == "goal-1"
    assert result.context.runtime["active_agent"] == "coordinator"
    assert "Agent: coordinator" in result.context.runtime["agent_text"]
    assert "Tool control surface" not in result.context.runtime["agent_text"]
    assert isinstance(result.tool_results[0].payload, ToolExecutionPayload)


def test_harness_execution_emits_hooks_and_records_tool_results() -> None:
    harness = create_default_harness(workdir=Path.cwd())
    goal = RuntimeGoal(goal_id="goal-2", description="Run default execution")

    result = harness.execute(goal, tool_execution_order=["run_baseline_eval"])

    emitted_hook_names = [event.hook_name for event in result.emitted_hooks]
    assert emitted_hook_names[0] == "before_run"
    assert emitted_hook_names[-1] == "after_dream_state"
    assert "after_run" in emitted_hook_names
    assert result.success is False
    assert result.selected_tool_names == ["run_baseline_eval"]
    assert [tool_result.tool_name for tool_result in result.tool_results] == ["run_baseline_eval"]
    assert result.tool_results[0].error == "missing_scenario_path"
    assert all(
        isinstance(tool_result.payload, ToolExecutionPayload) for tool_result in result.tool_results
    )
    assert result.memory_snapshot["artifact_memory"]["events"]
    assert result.memory_snapshot["session_memory"]["traces"]
    assert result.memory_snapshot["session_memory"]["transcripts"]
    assert result.memory_snapshot["session_memory"]["dream_queue"] == []


def test_harness_dynamically_selects_initial_tool_when_no_order_provided() -> None:
    harness = create_default_harness(workdir=Path.cwd())
    goal = RuntimeGoal(goal_id="goal-5", description="Need to evaluate and then inspect workspace")

    result = harness.execute(
        goal,
        agent_name="coordinator",
        requested_skills=["baseline-evaluation", "workspace-governance"],
    )

    assert result.selected_tool_names[0] in {"run_eval", "run_baseline_eval"}


def test_harness_stops_when_max_iterations_is_reached() -> None:
    harness = create_default_harness(workdir=Path.cwd())
    goal = RuntimeGoal(goal_id="goal-6", description="Plan repeatedly", max_iterations=1)

    result = harness.execute(goal, agent_name="coordinator")

    assert result.success is False
    assert result.error == "Reached max iterations: 1"


def test_improver_iteration_budget_counts_completed_loops_not_tools() -> None:
    harness = create_default_harness(workdir=Path.cwd())
    goal = RuntimeGoal(
        goal_id="goal-loop", description="Improve through one loop", max_iterations=1
    )

    result = harness.execute(
        goal,
        agent_name="improver",
        tool_execution_order=["load_workspace_state", "run_eval_repeated", "read"],
    )

    assert result.success is False
    assert result.selected_tool_names == ["load_workspace_state", "read"]
    assert result.context.runtime["completed_iterations"] == 0
    assert result.context.runtime["remaining_iterations"] == 1
    assert result.error == "bolt_merlin_unavailable"


def test_improver_agent_text_comes_from_clear_loop_body() -> None:
    harness = create_default_harness(workdir=Path.cwd())
    goal = RuntimeGoal(goal_id="goal-agent-text", description="Use improver guidance")

    result = harness.execute(
        goal,
        agent_name="improver",
        tool_execution_order=["load_workspace_state"],
    )

    agent_text = result.context.runtime["agent_text"]
    assert "# Improvement Loop" in agent_text
    assert (
        "Only the completed re-evaluation step counts against the iteration budget." in agent_text
    )
    assert "# High-Quality Signal" in agent_text
    assert "# Weak Signal" in agent_text
    assert "Primary objectives:" not in agent_text


def test_improver_active_tools_exclude_low_signal_workspace_tools() -> None:
    harness = create_default_harness(workdir=Path.cwd())
    result = harness.execute(
        RuntimeGoal(goal_id="goal-improver-tools", description="Inspect improver tool surface"),
        agent_name="improver",
        tool_execution_order=["load_workspace_state"],
    )
    active_tool_names = {tool.name for tool in result.tools}

    assert "bash" not in active_tool_names
    assert "glob" not in active_tool_names
    assert "test" not in active_tool_names
    assert "write" not in active_tool_names
    assert "edit" not in active_tool_names
    assert "audit_claims" not in active_tool_names
    assert "run_eval" not in active_tool_names
    assert "todoread" not in active_tool_names
    assert "todowrite" not in active_tool_names
    assert "grep" in active_tool_names
    assert "read" in active_tool_names
    assert "mutate_agent_files" in active_tool_names


def test_tool_call_iterations_track_loops_separately_from_tool_calls() -> None:
    harness = create_default_harness(workdir=Path.cwd())
    captured_call: dict[str, object] = {}

    def recording_handler(call: object) -> ToolResult:
        captured_call["call"] = call
        return ToolResult(tool_name="load_workspace_state", success=True)

    harness.tools.register_handler("load_workspace_state", recording_handler)

    result = harness.execute(
        RuntimeGoal(goal_id="goal-call-count", description="Track loop and tool counts"),
        agent_name="improver",
        tool_execution_order=["load_workspace_state"],
    )

    del result
    call = captured_call["call"]
    assert getattr(call, "iterations") == 0
    assert getattr(call, "tool_call_count") == 1


def test_improver_loop_mode_reuses_tools_after_completed_iteration() -> None:
    harness = create_default_harness(workdir=Path.cwd())

    def load_handler(_call: object) -> ToolResult:
        return ToolResult(
            tool_name="load_workspace_state",
            success=True,
            payload=ToolExecutionPayload(
                workspace_updates=WorkspaceToolContext(allowed_target_files=["agent.md"]),
            ),
        )

    def mutate_handler(_call: object) -> ToolResult:
        return ToolResult(
            tool_name="mutate_agent_files",
            success=True,
            payload=ToolExecutionPayload(
                workspace_updates=WorkspaceToolContext(mutated_files=["agent.md"]),
            ),
        )

    def repeat_handler(_call: object) -> ToolResult:
        return ToolResult(
            tool_name="run_eval_repeated",
            success=True,
            payload=ToolExecutionPayload(
                evaluation_updates=EvaluationToolContext(
                    repeat_eval_summary=ScoreSummary(
                        score=0.8,
                        status="completed",
                        tool="run_eval_repeated",
                    )
                )
            ),
        )

    harness.tools.register_handler("load_workspace_state", load_handler)
    harness.tools.register_handler("mutate_agent_files", mutate_handler)
    harness.tools.register_handler("run_eval_repeated", repeat_handler)

    result = harness.execute(
        RuntimeGoal(
            goal_id="goal-loop-reuse",
            description="Complete two improvement loops",
            max_iterations=2,
        ),
        agent_name="improver",
        tool_execution_order=["load_workspace_state", "mutate_agent_files", "run_eval_repeated"],
    )

    assert result.success is False
    assert result.error == "Reached max iterations: 2"
    assert result.selected_tool_names == [
        "load_workspace_state",
        "mutate_agent_files",
        "run_eval_repeated",
        "load_workspace_state",
        "mutate_agent_files",
        "run_eval_repeated",
    ]
    assert result.context.runtime["completed_iterations"] == 2
    assert result.context.runtime["remaining_iterations"] == 0


def test_workspace_tools_emit_richer_outputs(tmp_path: Path) -> None:
    (tmp_path / "alpha.py").write_text("print('a')\n", encoding="utf-8")
    (tmp_path / ".hidden").write_text("secret\n", encoding="utf-8")
    (tmp_path / ".dawn-kestrel").mkdir()
    (tmp_path / ".dawn-kestrel" / "agent_config.yaml").write_text("agent: test\n", encoding="utf-8")
    (tmp_path / "demo_pkg").mkdir()
    (tmp_path / "demo_pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "demo_pkg" / "agent").mkdir()
    (tmp_path / "demo_pkg" / "agent" / "coding_agent.py").write_text(
        "PROMPT = 'demo'\n", encoding="utf-8"
    )
    (tmp_path / "demo_pkg" / "agent" / "prompts").mkdir(parents=True)
    (tmp_path / "demo_pkg" / "agent" / "prompts" / "coding.md").write_text(
        "# coding prompt\n", encoding="utf-8"
    )
    (tmp_path / "demo_pkg" / "agent" / "tools").mkdir(parents=True)
    (tmp_path / "demo_pkg" / "agent" / "tools" / "bash.py").write_text(
        "def run():\n    return None\n", encoding="utf-8"
    )
    (tmp_path / "demo_pkg" / "cli").mkdir()
    (tmp_path / "demo_pkg" / "cli" / "repl.py").write_text("class REPL: ...\n", encoding="utf-8")
    scenario_dir = tmp_path / "evals" / "scenarios" / "thin_runtime"
    scenario_dir.mkdir(parents=True)
    scenario_file = scenario_dir / "probe.scenario.yaml"
    scenario_file.write_text(
        "schema_version: v1\nid: probe\ndescription: Pack capability test\ninputs:\n  intent: Teach python repl capability\ngraders:\n  - grader_type: repo_diff\n    config:\n      required_file_changes:\n        - path: reports/repl-notes.md\n        - path: src/sequence.py\n",
        encoding="utf-8",
    )
    pack = tmp_path / "evals" / "scenarios" / "pack.yaml"
    pack.write_text(
        "schema_version: v1\nid: pack\ndescription: REPL capability pack\nscenarios:\n  - scenario: ./thin_runtime/probe.scenario.yaml\n    focus: [python_repl, recursive_learning]\n",
        encoding="utf-8",
    )

    harness = create_default_harness(workdir=tmp_path)
    result = harness.execute(
        RuntimeGoal(goal_id="goal-rich-workspace", description="Inspect workspace outputs"),
        tool_execution_order=["load_workspace_state", "detect_agent_config", "scope_workspace"],
        scenario_path=str(pack),
    )

    by_tool = {tool_result.tool_name: tool_result for tool_result in result.tool_results}
    assert (
        "Loaded workspace state (1 files: alpha.py)"
        == by_tool["load_workspace_state"].payload.message
    )
    assert (
        by_tool["load_workspace_state"].payload.audit_updates.run_summary["changed_file_count"]
        == "1"
    )
    assert by_tool["load_workspace_state"].payload.workspace_updates.workdir == str(
        tmp_path.resolve()
    )
    assert by_tool["load_workspace_state"].payload.workspace_updates.scenario_targets == [
        str(scenario_file.resolve())
    ]
    assert by_tool["load_workspace_state"].payload.workspace_updates.scenario_required_files == [
        "reports/repl-notes.md",
        "src/sequence.py",
    ]
    assert "REPL capability pack" in (
        by_tool["load_workspace_state"].payload.workspace_updates.scenario_summary or ""
    )

    assert by_tool["scope_workspace"].payload.message.startswith("Scoped workspace to")
    assert (
        by_tool["scope_workspace"].payload.audit_updates.run_summary["allowed_target_file_count"]
        == "3"
    )
    assert by_tool["scope_workspace"].payload.workspace_updates.allowed_target_files[:2] == [
        "reports/repl-notes.md",
        "src/sequence.py",
    ]
    assert by_tool["scope_workspace"].payload.workspace_updates.allowed_target_files[2:] == [
        "alpha.py"
    ]

    detect_result = by_tool["detect_agent_config"]
    assert "Detected agent config at" in detect_result.payload.message
    assert detect_result.payload.audit_updates.run_summary["agent_config"].endswith(
        ".dawn-kestrel/agent_config.yaml"
    )
    assert detect_result.payload.audit_updates.run_summary["package_name"] == "demo_pkg"


def test_mutate_agent_files_records_diff_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target.py"
    target.write_text("value = 1\n", encoding="utf-8")
    (tmp_path / ".dawn-kestrel").mkdir()
    (tmp_path / ".dawn-kestrel" / "agent_config.yaml").write_text("agent: test\n", encoding="utf-8")

    fake_module = ModuleType("bolt_merlin.agent.execute")

    async def fake_execute(**kwargs: object) -> SimpleNamespace:
        del kwargs
        target.write_text("value = 2\n", encoding="utf-8")
        return SimpleNamespace(session_id="mutation-session", response="Changed value to 2")

    fake_module.execute = fake_execute  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bolt_merlin.agent.execute", fake_module)

    harness = create_default_harness(workdir=tmp_path)
    result = harness.execute(
        RuntimeGoal(goal_id="goal-mutate-rich", description="Mutate with diff output"),
        agent_name="improver",
        tool_execution_order=["load_workspace_state", "scope_workspace", "mutate_agent_files"],
    )

    tool_result = result.tool_results[-1]
    assert tool_result.tool_name == "mutate_agent_files"
    assert tool_result.success is True
    assert "changed" in tool_result.payload.message
    assert tool_result.payload.audit_updates.run_summary["target_file"] == "target.py"
    assert tool_result.payload.audit_updates.run_summary["changed"] == "True"
    assert tool_result.payload.audit_updates.diff_report["files"] == 1
    assert tool_result.payload.audit_updates.diff_report["diff_line_count"] > 0
    assert "value = 2" in tool_result.payload.audit_updates.run_summary["diff_preview"]
    assert result.context.runtime["preferred_tool"] == "run_eval_repeated"


def test_tool_handlers_update_context_and_unlock_outputs() -> None:
    harness = create_default_harness(workdir=Path.cwd())
    goal = RuntimeGoal(goal_id="goal-8", description="Establish baseline and verify it")

    def baseline_handler(_call: object) -> ToolResult:
        return ToolResult(
            tool_name="run_baseline_eval",
            success=True,
            payload=ToolExecutionPayload(
                evaluation_updates=EvaluationToolContext(
                    baseline_summary=ScoreSummary(
                        score=0.91,
                        status="completed",
                        tool="run_baseline_eval",
                    )
                ),
                audit_updates=AuditToolContext(
                    run_result=AuditRunResult(aggregate_passed=True),
                    validation_tools=["run_baseline_eval"],
                ),
                message="Completed baseline evaluation",
            ),
        )

    harness.tools.register_handler("run_baseline_eval", baseline_handler)

    result = harness.execute(
        goal,
        agent_name="verifier",
        requested_skills=["baseline-evaluation", "verification"],
        tool_execution_order=["run_baseline_eval", "verify_outcome"],
    )

    assert result.success is True
    assert result.context.evaluation["baseline_summary"]["status"] == "completed"
    assert result.context.evaluation["verification"]["verified"] is True
    assert "evaluation_context" in result.available_contexts
    assert result.selected_tool_names[0] in {"run_eval", "run_baseline_eval"}


def test_read_falls_back_to_scenario_target_when_workspace_has_no_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario_dir = tmp_path / "evals" / "scenarios" / "thin_runtime"
    scenario_dir.mkdir(parents=True)
    scenario_file = scenario_dir / "probe.scenario.yaml"
    scenario_file.write_text(
        "schema_version: v1\nid: probe\ndescription: Scenario body\ninputs:\n  prompt: hello\n",
        encoding="utf-8",
    )
    pack = tmp_path / "evals" / "scenarios" / "pack.yaml"
    pack.write_text(
        "schema_version: v1\nid: pack\ndescription: Read the pack first\nscenarios:\n  - scenario: ./thin_runtime/probe.scenario.yaml\n",
        encoding="utf-8",
    )

    fake_read_module = ModuleType("bolt_merlin.agent.tools.read")

    class FakeReadTool:
        async def execute(self, payload: dict[str, object], _ctx: object) -> SimpleNamespace:
            file_path = str(payload["file_path"])
            return SimpleNamespace(output=Path(file_path).read_text(encoding="utf-8"), error=None)

    fake_framework_module = ModuleType("dawn_kestrel.tools.framework")

    class FakeToolContext:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

    fake_read_module.ReadTool = FakeReadTool  # type: ignore[attr-defined]
    fake_framework_module.ToolContext = FakeToolContext  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bolt_merlin.agent.tools.read", fake_read_module)
    monkeypatch.setitem(sys.modules, "dawn_kestrel.tools.framework", fake_framework_module)

    harness = create_default_harness(workdir=tmp_path)
    result = harness.execute(
        RuntimeGoal(goal_id="goal-read-pack", description="Read scenario pack"),
        agent_name="improver",
        tool_execution_order=["read"],
        scenario_path=str(pack),
    )

    tool_result = result.tool_results[0]
    assert tool_result.tool_name == "read"
    assert tool_result.success is True
    assert tool_result.payload.message == "Read the pack first"


def test_harness_persists_memory_and_execution_artifacts(tmp_path: Path) -> None:
    harness = create_default_harness(workdir=Path.cwd(), storage_root=tmp_path)
    goal = RuntimeGoal(goal_id="goal-9", description="Persist this run")

    result = harness.execute(goal, tool_execution_order=["load_workspace_state"])

    assert (tmp_path / "memory" / "snapshot.json").exists()
    assert not (tmp_path / "memory" / "dream_queue.json").exists()
    assert (Path(result.artifact_dir) / "execution.json").exists()
    assert (Path(result.artifact_dir) / "summary.json").exists()
    assert result.memory_snapshot["session_memory"]["traces"]
    assert result.memory_snapshot["session_memory"]["transcripts"]
    artifact_paths = {
        result.artifact_dir,
        str(Path(result.artifact_dir) / "execution.json"),
        str(Path(result.artifact_dir) / "summary.json"),
    }
    assert artifact_paths.issubset(set(result.memory_snapshot["artifact_memory"]["artifacts"]))
    assert any(
        result.run_id in item for item in result.memory_snapshot["episodic_memory"]["episodes"]
    )

    reloaded_harness = create_default_harness(workdir=Path.cwd(), storage_root=tmp_path)
    assert reloaded_harness.memory.snapshot()["artifact_memory"]["events"]
    assert reloaded_harness.memory.snapshot()["session_memory"]["traces"]
    assert reloaded_harness.memory.snapshot()["session_memory"]["transcripts"]
    assert artifact_paths.issubset(
        set(reloaded_harness.memory.snapshot()["artifact_memory"]["artifacts"])
    )


def test_harness_persists_tool_run_ids_into_episodic_memory(tmp_path: Path) -> None:
    harness = create_default_harness(workdir=Path.cwd(), storage_root=tmp_path)

    def run_result_handler(_call: object) -> ToolResult:
        return ToolResult(
            tool_name="load_workspace_state",
            success=True,
            payload=ToolExecutionPayload(
                audit_updates=AuditToolContext(
                    run_result=AuditRunResult(
                        run_id="external-run-123",
                        aggregate_score=0.91,
                        aggregate_passed=False,
                    )
                ),
                message="Loaded workspace state",
            ),
        )

    harness.tools.register_handler("load_workspace_state", run_result_handler)

    result = harness.execute(
        RuntimeGoal(goal_id="goal-run-result-memory", description="Persist observed run ids"),
        tool_execution_order=["load_workspace_state"],
    )

    assert any(
        "external-run-123" in item for item in result.memory_snapshot["episodic_memory"]["episodes"]
    )


def test_dream_state_consolidates_deferred_memory_on_final_hook() -> None:
    harness = create_default_harness(workdir=Path.cwd())
    goal = RuntimeGoal(goal_id="goal-10", description="Dream state consolidation")
    harness.skills.get("lesson-consolidation").input_contexts = ["memory_context"]

    result = harness.execute(
        goal,
        agent_name="memory_manager",
        requested_skills=["lesson-consolidation"],
        tool_execution_order=["save_lesson"],
    )

    assert result.success is True
    assert result.memory_snapshot["semantic_memory"]["entries"]
    emitted_hook_names = [event.hook_name for event in result.emitted_hooks]
    assert "after_dream_state" in emitted_hook_names
    assert result.memory_snapshot["session_memory"]["dream_queue"] == []


def test_skill_input_contexts_gate_tool_eligibility() -> None:
    harness = create_default_harness(workdir=Path.cwd())
    goal = RuntimeGoal(goal_id="goal-7", description="Need verification after baseline")

    def baseline_handler(_call: object) -> ToolResult:
        return ToolResult(
            tool_name="run_baseline_eval",
            success=True,
            payload=ToolExecutionPayload(
                evaluation_updates=EvaluationToolContext(
                    baseline_summary=ScoreSummary(
                        score=0.88,
                        status="completed",
                        tool="run_baseline_eval",
                    )
                ),
                audit_updates=AuditToolContext(
                    run_result=AuditRunResult(aggregate_passed=True),
                    validation_tools=["run_baseline_eval"],
                ),
                message="Completed baseline evaluation",
            ),
        )

    harness.tools.register_handler("run_baseline_eval", baseline_handler)

    result = harness.execute(
        goal,
        agent_name="verifier",
        requested_skills=["verification", "baseline-evaluation"],
        tool_execution_order=["run_baseline_eval", "verify_outcome"],
    )

    assert result.selected_tool_names[0] in {"run_eval", "run_baseline_eval"}
    assert "verify_outcome" in result.selected_tool_names
    first_eval_index = min(
        result.selected_tool_names.index(tool_name)
        for tool_name in {"run_eval", "run_baseline_eval"}
        if tool_name in result.selected_tool_names
    )
    assert first_eval_index < result.selected_tool_names.index("verify_outcome")


def test_eval_tool_fails_honestly_without_scenario_path() -> None:
    harness = create_default_harness(workdir=Path.cwd())

    result = harness.execute(
        RuntimeGoal(goal_id="goal-no-scenario", description="Need a baseline"),
        tool_execution_order=["run_baseline_eval"],
    )

    assert result.success is False
    assert result.tool_results[0].success is False
    assert result.tool_results[0].error == "missing_scenario_path"
    assert (
        result.tool_results[0].payload.evaluation_updates.baseline_summary.status
        == "missing_scenario"
    )


def test_tool_context_requirements_are_enforced() -> None:
    harness = create_default_harness(workdir=Path.cwd())
    goal = RuntimeGoal(goal_id="goal-3", description="Check context enforcement")
    harness.tools.get("run_baseline_eval").required_contexts.append("missing_context")

    result = harness.execute(
        goal,
        agent_name="verifier",
        requested_skills=["baseline-evaluation"],
        tool_execution_order=["run_baseline_eval"],
    )

    assert result.success is False
    assert result.tool_results[0].error == "Missing required contexts: missing_context"


def test_memory_write_permissions_are_enforced() -> None:
    harness = create_default_harness(workdir=Path.cwd())
    goal = RuntimeGoal(goal_id="goal-4", description="Check policy-only skill path")
    harness.skills.get("lesson-consolidation").input_contexts = ["memory_context"]

    result = harness.execute(
        goal,
        agent_name="coordinator",
        requested_skills=["lesson-consolidation"],
    )

    assert result.success is True
    assert result.selected_tool_names == []
