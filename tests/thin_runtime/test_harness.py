from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Callable, cast

import pytest
from dawn_kestrel.provider.llm_client import LLMResponse
from dawn_kestrel.provider.provider_types import TokenUsage

from ash_hawk.thin_runtime import RuntimeGoal, create_default_harness
from ash_hawk.thin_runtime.models import ContextSnapshot, ToolCall, ToolResult
from ash_hawk.thin_runtime.tool_impl.mutate_agent_files import run as mutate_agent_files_run
from ash_hawk.thin_runtime.tool_impl.prepare_isolated_workspace import (
    run as prepare_isolated_workspace_run,
)
from ash_hawk.thin_runtime.tool_types import (
    AuditRunResult,
    AuditToolContext,
    EvaluationToolContext,
    FailureToolContext,
    RankedHypothesis,
    RuntimeToolContext,
    ScoreSummary,
    ToolCallContext,
    ToolExecutionPayload,
    VerificationStatus,
    WorkspaceToolContext,
)

pytestmark = pytest.mark.e2e


class _FakeDkClient:
    provider_id = "test"

    def __init__(self, chooser: Callable[[list[dict[str, object]]], str | None]) -> None:
        self._chooser = chooser

    async def complete(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        options: object = None,
    ) -> LLMResponse:
        del messages
        del options
        tool_defs = tools or []
        tool_name = self._chooser(tool_defs)
        if tool_name is None:
            return LLMResponse(
                text="No tool call",
                usage=TokenUsage(input=0, output=0, reasoning=0),
                finish_reason="stop",
                cost=Decimal("0"),
                tool_calls=None,
            )
        return LLMResponse(
            text=f"Choosing {tool_name}",
            usage=TokenUsage(input=0, output=0, reasoning=0),
            finish_reason="tool_use",
            cost=Decimal("0"),
            tool_calls=[
                {
                    "id": f"call-{tool_name}",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                    "tool": tool_name,
                    "input": {},
                }
            ],
        )


def test_harness_executes_agentic_run_with_skills_tools_and_context() -> None:
    harness = create_default_harness(workdir=Path.cwd())
    harness.runner.set_client_factory(
        lambda: _FakeDkClient(lambda _tool_defs: "load_workspace_state")
    )
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
    assert "Available tools now:" not in result.context.runtime["agent_text"]
    assert "Objective:" in result.context.runtime["agent_text"]
    assert isinstance(result.tool_results[0].payload, ToolExecutionPayload)


def test_harness_execution_emits_hooks_and_records_tool_results(tmp_path: Path) -> None:
    harness = create_default_harness(
        workdir=tmp_path,
        storage_root=tmp_path / ".ash-hawk-test",
    )
    goal = RuntimeGoal(goal_id="goal-2", description="Run default execution")

    result = harness.execute(goal, tool_execution_order=["run_baseline_eval"])

    emitted_hook_names = [event.hook_name for event in result.emitted_hooks]
    assert emitted_hook_names[0] == "before_run"
    assert emitted_hook_names[-1] == "after_dream_state"
    assert "after_run" in emitted_hook_names
    assert result.selected_tool_names == ["run_baseline_eval"]
    assert all(
        isinstance(tool_result.payload, ToolExecutionPayload) for tool_result in result.tool_results
    )


def test_harness_dynamically_selects_initial_tool_when_no_order_provided() -> None:
    harness = create_default_harness(workdir=Path.cwd())
    harness.runner.set_client_factory(lambda: _FakeDkClient(lambda _tool_defs: "run_baseline_eval"))
    goal = RuntimeGoal(goal_id="goal-5", description="Need to evaluate and then inspect workspace")

    result = harness.execute(
        goal,
        agent_name="coordinator",
        requested_skills=["baseline-evaluation", "workspace-governance"],
    )

    assert result.selected_tool_names[0] == "run_baseline_eval"


def test_harness_prints_skill_activation_to_console(capsys: pytest.CaptureFixture[str]) -> None:
    harness = create_default_harness(workdir=Path.cwd(), console_output=True)

    harness.execute(
        RuntimeGoal(goal_id="goal-skill-console", description="Show skill activation"),
        agent_name="coordinator",
        requested_skills=["context-assembly"],
        tool_execution_order=["load_workspace_state"],
    )

    captured = capsys.readouterr()
    assert "Skill activated: context-assembly (preloaded)" in captured.out


def test_improver_bootstraps_without_explicit_tool_order() -> None:
    harness = create_default_harness(workdir=Path.cwd(), console_output=False)
    harness.runner.set_client_factory(
        lambda: _FakeDkClient(lambda _tool_defs: "load_workspace_state")
    )

    result = harness.execute(
        RuntimeGoal(
            goal_id="goal-improver-bootstrap", description="Bootstrap improver without order"
        ),
        agent_name="improver",
    )

    assert result.selected_tool_names
    assert result.selected_tool_names[0] == "load_workspace_state"


def test_improver_does_not_repeat_detect_agent_config_after_context_is_known(
    tmp_path: Path,
) -> None:
    def choose_tool(tool_defs: list[dict[str, object]]) -> str | None:
        tool_names: list[str] = []
        for tool_def in tool_defs:
            function = tool_def.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str):
                tool_names.append(name)
        if "load_workspace_state" in tool_names:
            return "load_workspace_state"
        if "run_baseline_eval" in tool_names:
            return "run_baseline_eval"
        if "detect_agent_config" in tool_names:
            return "detect_agent_config"
        if "call_llm_structured" in tool_names:
            return "call_llm_structured"
        return None

    def baseline_handler(_call: object) -> ToolResult:
        return ToolResult(
            tool_name="run_baseline_eval",
            success=True,
            payload=ToolExecutionPayload(
                evaluation_updates=EvaluationToolContext(
                    baseline_summary=ScoreSummary(
                        score=0.76,
                        status="completed",
                        tool="run_baseline_eval",
                    )
                ),
                failure_updates=FailureToolContext(
                    failure_family="needs_improvement",
                    explanations=["The agent searched broadly instead of using direct context."],
                ),
            ),
        )

    def detect_handler(_call: object) -> ToolResult:
        return ToolResult(
            tool_name="detect_agent_config",
            success=True,
            payload=ToolExecutionPayload(
                workspace_updates=WorkspaceToolContext(
                    agent_config="/repo/.dawn-kestrel/agent_config.yaml",
                    source_root="/repo",
                    package_name="bolt_merlin",
                )
            ),
        )

    def diagnosis_handler(_call: object) -> ToolResult:
        return ToolResult(
            tool_name="call_llm_structured",
            success=True,
            payload=ToolExecutionPayload(
                runtime_updates=RuntimeToolContext(stop_reason="diagnosis reached"),
                stop=True,
            ),
        )

    harness = create_default_harness(
        workdir=tmp_path,
        storage_root=tmp_path / ".ash-hawk-test",
        console_output=False,
    )
    harness.runner.set_client_factory(lambda: _FakeDkClient(choose_tool))
    harness.tools.register_handler(
        "load_workspace_state",
        lambda _call: ToolResult(
            tool_name="load_workspace_state",
            success=True,
            payload=ToolExecutionPayload(
                workspace_updates=WorkspaceToolContext(changed_files=[]),
            ),
        ),
    )
    harness.tools.register_handler("run_baseline_eval", baseline_handler)
    harness.tools.register_handler("detect_agent_config", detect_handler)
    harness.tools.register_handler("call_llm_structured", diagnosis_handler)

    result = harness.execute(
        RuntimeGoal(goal_id="goal-no-repeat-config", description="Avoid repeated config detection"),
        agent_name="improver",
    )

    assert result.selected_tool_names == [
        "load_workspace_state",
        "detect_agent_config",
        "run_baseline_eval",
        "call_llm_structured",
    ]
    assert result.success is True


def test_improver_does_not_offer_delegate_task_before_baseline() -> None:
    class DelegationGuardClient:
        provider_id = "test"

        def __init__(self) -> None:
            self.calls = 0

        async def complete(
            self,
            messages: list[dict[str, object]],
            tools: list[dict[str, object]] | None = None,
            options: object = None,
        ) -> LLMResponse:
            del messages
            del options
            tool_defs = tools or []
            tool_names = [
                cast(dict[str, object], tool_def["function"])["name"]
                for tool_def in tool_defs
                if isinstance(tool_def.get("function"), dict)
            ]
            if self.calls == 0:
                assert "delegate_task" not in tool_names
                self.calls += 1
                first_tool = tool_names[0]
                return LLMResponse(
                    text="Start with the first safe bootstrap tool.",
                    usage=TokenUsage(input=0, output=0, reasoning=0),
                    finish_reason="tool_use",
                    cost=Decimal("0"),
                    tool_calls=[
                        {
                            "id": "call-bootstrap",
                            "type": "function",
                            "function": {"name": first_tool, "arguments": "{}"},
                            "tool": first_tool,
                            "input": {},
                        }
                    ],
                )
            return LLMResponse(
                text="Stop.",
                usage=TokenUsage(input=0, output=0, reasoning=0),
                finish_reason="stop",
                cost=Decimal("0"),
                tool_calls=None,
            )

    harness = create_default_harness(workdir=Path.cwd(), console_output=False)
    harness.runner.set_client_factory(lambda: DelegationGuardClient())
    harness.tools.register_handler(
        "run_baseline_eval",
        lambda _call: ToolResult(
            tool_name="run_baseline_eval",
            success=True,
            payload=ToolExecutionPayload(
                evaluation_updates=EvaluationToolContext(
                    baseline_summary=ScoreSummary(
                        score=0.76,
                        status="completed",
                        tool="run_baseline_eval",
                    )
                )
            ),
        ),
    )

    result = harness.execute(
        RuntimeGoal(goal_id="goal-delegation-guard", description="Improve after baseline"),
        agent_name="improver",
    )

    assert result.selected_tool_names == ["load_workspace_state"]
    assert result.error == "No completed improvement loops recorded"
    assert result.success is False


def test_goal_scoped_audit_events_ignore_previous_run_history() -> None:
    harness = create_default_harness(workdir=Path.cwd(), console_output=False)

    first = harness.execute(
        RuntimeGoal(goal_id="shared-goal", description="First run"),
        agent_name="coordinator",
        tool_execution_order=["load_workspace_state"],
    )
    second = harness.execute(
        RuntimeGoal(goal_id="shared-goal", description="Second run"),
        agent_name="coordinator",
        tool_execution_order=["load_workspace_state"],
    )

    assert first.run_id != second.run_id
    assert second.context.audit["events"]
    assert all(
        isinstance(item, dict) and item.get("run_id") == second.run_id
        for item in second.context.audit["events"]
    )
    assert all(
        isinstance(item, dict) and item.get("run_id") == second.run_id
        for item in second.context.audit["transcripts"]
    )
    session_memory = second.context.memory["session"]
    assert all(
        isinstance(item, dict) and item.get("run_id") == second.run_id
        for item in session_memory["traces"]
    )
    assert all(
        isinstance(item, dict) and item.get("run_id") == second.run_id
        for item in session_memory["transcripts"]
    )


def test_harness_stops_when_max_iterations_is_reached() -> None:
    harness = create_default_harness(workdir=Path.cwd(), console_output=False)
    goal = RuntimeGoal(goal_id="goal-6", description="Plan repeatedly", max_iterations=1)

    def load_handler(_call: object) -> ToolResult:
        return ToolResult(tool_name="load_workspace_state", success=True)

    harness.tools.register_handler("load_workspace_state", load_handler)

    result = harness.execute(
        goal,
        agent_name="coordinator",
        tool_execution_order=["load_workspace_state"],
    )

    assert result.success is False
    assert result.error == "Reached max iterations: 1"


def test_improver_iteration_budget_counts_completed_loops_not_tools() -> None:
    harness = create_default_harness(workdir=Path.cwd(), console_output=False)
    goal = RuntimeGoal(
        goal_id="goal-loop", description="Improve through one loop", max_iterations=1
    )

    result = harness.execute(
        goal,
        agent_name="improver",
        tool_execution_order=["load_workspace_state", "run_eval_repeated", "delegate_task"],
    )

    assert result.success is False
    assert result.selected_tool_names == ["load_workspace_state"]
    assert result.context.runtime["completed_iterations"] == 0
    assert result.context.runtime["remaining_iterations"] == 1
    assert (
        result.error
        == "No eligible tools available for a complete diagnosis-mutation-re-evaluation loop"
    )


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
    assert "Use `run_eval_repeated` after a mutation to close the loop." in agent_text
    assert "# High-Quality Signal" in agent_text
    assert "# Weak Signal" in agent_text
    assert "Primary objectives:" not in agent_text


def test_harness_builds_richer_context_and_agent_text(tmp_path: Path) -> None:
    harness = create_default_harness(
        workdir=tmp_path,
        storage_root=tmp_path / ".ash-hawk-test",
        console_output=False,
    )
    remaining_sequence = [
        "load_workspace_state",
        "detect_agent_config",
        "run_baseline_eval",
        "call_llm_structured",
    ]

    def choose_tool(tool_defs: list[dict[str, object]]) -> str | None:
        tool_names = [
            function.get("name")
            for tool_def in tool_defs
            if isinstance((function := tool_def.get("function")), dict)
            and isinstance(function.get("name"), str)
        ]
        for tool_name in list(remaining_sequence):
            if tool_name in tool_names:
                remaining_sequence.remove(tool_name)
                return tool_name
        return None

    harness.runner.set_client_factory(lambda: _FakeDkClient(choose_tool))

    def load_handler(_call: object) -> ToolResult:
        return ToolResult(
            tool_name="load_workspace_state",
            success=True,
            payload=ToolExecutionPayload(
                workspace_updates=WorkspaceToolContext(
                    scenario_path="evals/sample.scenario.yaml",
                    scenario_required_files=["agent.md", "ash_hawk/thin_runtime/context.py"],
                    changed_files=["ash_hawk/thin_runtime/context.py"],
                    agent_config="/repo/.dawn-kestrel/agent_config.yaml",
                )
            ),
        )

    def baseline_handler(_call: object) -> ToolResult:
        return ToolResult(
            tool_name="run_baseline_eval",
            success=True,
            payload=ToolExecutionPayload(
                evaluation_updates=EvaluationToolContext(
                    baseline_summary=ScoreSummary(
                        score=0.41,
                        status="completed",
                        tool="run_baseline_eval",
                    )
                )
            ),
        )

    def detect_handler(_call: object) -> ToolResult:
        return ToolResult(
            tool_name="detect_agent_config",
            success=True,
            payload=ToolExecutionPayload(
                workspace_updates=WorkspaceToolContext(
                    agent_config="/repo/.dawn-kestrel/agent_config.yaml",
                    source_root="/repo",
                    package_name="bolt_merlin",
                )
            ),
        )

    def diagnosis_handler(_call: object) -> ToolResult:
        return ToolResult(
            tool_name="call_llm_structured",
            success=True,
            payload=ToolExecutionPayload(
                failure_updates=FailureToolContext(
                    failure_family="missing_context",
                    explanations=["The agent did not keep an updated progress summary."],
                    ranked_hypotheses=[
                        RankedHypothesis(
                            name="publish runtime checkpoint",
                            score=0.92,
                            target_files=["ash_hawk/thin_runtime/agent_text.py"],
                            rationale="Expose current run state to the model.",
                            ideal_outcome="The model sees concise live progress artifacts.",
                        )
                    ],
                ),
                runtime_updates=RuntimeToolContext(stop_reason="diagnosis complete"),
                stop=True,
            ),
        )

    harness.tools.register_handler("load_workspace_state", load_handler)
    harness.tools.register_handler("detect_agent_config", detect_handler)
    harness.tools.register_handler("run_baseline_eval", baseline_handler)
    harness.tools.register_handler("call_llm_structured", diagnosis_handler)

    result = harness.execute(
        RuntimeGoal(goal_id="goal-power-context", description="Build a powerhouse context"),
        agent_name="improver",
    )

    assert isinstance(result.context.runtime.get("goal_intent"), str)
    assert result.context.runtime.get("phase") == "completed"
    assert isinstance(result.context.runtime.get("progress_summary"), str)
    assert result.context.runtime.get("active_skill_summaries")
    assert result.context.runtime.get("recent_steps")
    assert result.context.runtime.get("latest_evidence")
    assert result.context.runtime.get("next_pressure")
    assert result.context.evaluation.get("recent_eval_summaries")
    assert result.context.failure.get("diagnosed_issues")
    assert result.context.failure.get("top_hypothesis") == "publish runtime checkpoint (0.92)"
    assert result.context.workspace.get("actionable_files")
    assert result.context.workspace.get("reference_files")
    assert result.context.workspace.get("blocked_files")
    assert result.context.audit.get("artifact_index")

    summary_names = {
        item.split(":", 1)[0]
        for item in result.context.runtime["active_skill_summaries"]
        if isinstance(item, str)
    }
    assert summary_names == set(result.context.runtime["active_skills"])

    agent_text = result.context.runtime["agent_text"]
    assert "Current run context:" in agent_text
    assert "Objective:" in agent_text
    assert "Phase:" in agent_text
    assert "Recent steps:" in agent_text
    assert "Latest evidence:" in agent_text
    assert "Actionable files:" in agent_text
    assert "Blocked files:" in agent_text
    assert "Next pressure:" in agent_text
    assert "Artifact refs:" in agent_text
    assert "Available tools now:" not in agent_text
    assert "Tool surface:" not in agent_text
    assert result.selected_tool_names == [
        "load_workspace_state",
        "detect_agent_config",
        "run_baseline_eval",
        "call_llm_structured",
    ]


def test_load_workspace_state_uses_required_targets_not_repo_docs_for_actionable_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "AGENTS.md").write_text("docs", encoding="utf-8")
    (tmp_path / "ARCHITECTURE.md").write_text("docs", encoding="utf-8")
    (tmp_path / "agent.md").write_text("agent", encoding="utf-8")
    (tmp_path / "src" / "http").mkdir(parents=True)
    (tmp_path / "src" / "http" / "auth.py").write_text("pass\n", encoding="utf-8")
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text(
        "\n".join(
            [
                'description: "workspace targeting"',
                "graders:",
                "  - config:",
                "      required_file_changes:",
                '        - path: "src/http/auth.py"',
                '        - path: "agent.md"',
                '        - path: "reports/parallel-plan.md"',
            ]
        ),
        encoding="utf-8",
    )

    harness = create_default_harness(
        workdir=tmp_path,
        storage_root=tmp_path / ".ash-hawk-test",
        console_output=False,
    )
    harness.runner.set_client_factory(
        lambda: _FakeDkClient(lambda _tool_defs: "load_workspace_state")
    )
    result = harness.execute(
        RuntimeGoal(goal_id="goal-workspace-targets", description="Inspect workspace targets"),
        agent_name="improver",
        scenario_path=str(scenario),
    )

    actionable_files = result.context.workspace["actionable_files"]
    reference_files = result.context.workspace["reference_files"]
    assert any("src/http/auth.py" in item for item in actionable_files)
    assert any("agent.md" in item for item in actionable_files)
    assert not any(item.startswith("AGENTS.md") for item in actionable_files)
    assert not any(item.startswith("ARCHITECTURE.md") for item in actionable_files)
    assert not any("reports/parallel-plan.md" in item for item in actionable_files)
    assert any("reports/parallel-plan.md" in item for item in reference_files)


def test_merge_seed_context_keeps_parent_runtime_identity_and_audit() -> None:
    harness = create_default_harness(workdir=Path.cwd(), console_output=False)
    parent = ContextSnapshot(
        runtime={
            "run_id": "parent-run",
            "phase": "diagnosis",
            "recent_steps": ["parent step"],
            "last_decision": "parent decision",
            "progress_summary": "parent summary",
        },
        workspace={"allowed_target_files": ["src/parent.py"]},
        evaluation={"baseline_summary": {"score": 0.5, "status": "completed"}},
        failure={"failure_family": "needs_improvement"},
        memory={"session": {"traces": [{"run_id": "parent-run", "tool": "load_workspace_state"}]}},
        audit={
            "tool_results": [{"tool": "load_workspace_state", "success": True}],
            "artifacts": ["parent-artifact.json"],
        },
    )
    seed = ContextSnapshot(
        runtime={
            "run_id": "child-run",
            "phase": "mutation",
            "recent_steps": ["child read"],
            "last_decision": "child decision",
            "progress_summary": "child summary",
        },
        workspace={
            "mutated_files": ["src/child.py"],
            "allowed_target_files": ["src/child.py"],
            "scenario_path": "child-scenario.yaml",
        },
        evaluation={"repeat_eval_summary": {"score": 0.8, "status": "completed"}},
        failure={"top_hypothesis": "child hypothesis"},
        memory={"session": {"traces": [{"run_id": "child-run", "tool": "read"}]}},
        audit={
            "tool_results": [{"tool": "read", "success": True}],
            "artifacts": ["child-artifact.json"],
            "diff_report": {"files_changed": 1},
        },
    )

    getattr(harness.runner, "_merge_seed_context")(parent, seed)

    assert parent.runtime["run_id"] == "parent-run"
    assert parent.runtime["phase"] == "diagnosis"
    assert parent.runtime["recent_steps"] == ["parent step"]
    assert parent.runtime["last_decision"] == "parent decision"
    assert parent.runtime["progress_summary"] == "parent summary"
    assert parent.workspace["mutated_files"] == ["src/child.py"]
    assert parent.workspace["allowed_target_files"] == ["src/child.py"]
    assert parent.workspace.get("scenario_path") == "child-scenario.yaml"
    assert parent.evaluation["repeat_eval_summary"]["score"] == 0.8
    assert parent.failure["top_hypothesis"] == "child hypothesis"
    assert parent.memory["session"]["traces"][0]["run_id"] == "parent-run"
    assert parent.audit["tool_results"] == [{"tool": "load_workspace_state", "success": True}]
    assert parent.audit["artifacts"] == ["child-artifact.json"]
    assert parent.audit["diff_report"] == {"files_changed": 1}


def test_merge_seed_context_preserves_parent_baseline_summary() -> None:
    harness = create_default_harness(workdir=Path.cwd(), console_output=False)
    parent = ContextSnapshot(
        evaluation={
            "baseline_summary": {"score": 0.5, "status": "completed", "tool": "run_baseline_eval"}
        }
    )
    seed = ContextSnapshot(
        evaluation={
            "baseline_summary": {"score": 0.9, "status": "completed", "tool": "child_baseline"},
            "repeat_eval_summary": {
                "score": 0.8,
                "status": "completed",
                "tool": "run_eval_repeated",
            },
        }
    )

    getattr(harness.runner, "_merge_seed_context")(parent, seed)

    assert parent.evaluation["baseline_summary"] == {
        "score": 0.5,
        "status": "completed",
        "tool": "run_baseline_eval",
    }
    assert parent.evaluation["repeat_eval_summary"] == {
        "score": 0.8,
        "status": "completed",
        "tool": "run_eval_repeated",
    }


def test_scoped_non_durable_files_are_reference_only(tmp_path: Path) -> None:
    harness = create_default_harness(
        workdir=tmp_path,
        storage_root=tmp_path / ".ash-hawk-test",
        console_output=False,
    )
    harness.runner.set_client_factory(
        lambda: _FakeDkClient(lambda _tool_defs: "load_workspace_state")
    )

    def load_handler(_call: object) -> ToolResult:
        return ToolResult(
            tool_name="load_workspace_state",
            success=True,
            payload=ToolExecutionPayload(
                workspace_updates=WorkspaceToolContext(
                    allowed_target_files=[
                        "src/status_payload.py",
                        "evals/scenarios/example_pack.yaml",
                        "reports/parallel-plan.md",
                        "agent.md",
                    ]
                )
            ),
        )

    harness.tools.register_handler("load_workspace_state", load_handler)
    result = harness.execute(
        RuntimeGoal(goal_id="goal-scoped-reference", description="Check scoped targets"),
        agent_name="improver",
    )

    actionable_files = result.context.workspace["actionable_files"]
    reference_files = result.context.workspace["reference_files"]
    assert any("src/status_payload.py" in item for item in actionable_files)
    assert any("agent.md" in item for item in actionable_files)
    assert not any("example_pack.yaml" in item for item in actionable_files)
    assert not any("parallel-plan.md" in item for item in actionable_files)
    assert any("example_pack.yaml" in item for item in reference_files)
    assert any("parallel-plan.md" in item for item in reference_files)


def test_mutated_non_durable_files_are_reference_only(tmp_path: Path) -> None:
    harness = create_default_harness(
        workdir=tmp_path,
        storage_root=tmp_path / ".ash-hawk-test",
        console_output=False,
    )
    harness.runner.set_client_factory(
        lambda: _FakeDkClient(lambda _tool_defs: "load_workspace_state")
    )

    def load_handler(_call: object) -> ToolResult:
        return ToolResult(
            tool_name="load_workspace_state",
            success=True,
            payload=ToolExecutionPayload(
                workspace_updates=WorkspaceToolContext(
                    mutated_files=[
                        "config/runtime.yaml",
                        "reports/parallel-plan.md",
                        "src/status_payload.py",
                    ]
                )
            ),
        )

    harness.tools.register_handler("load_workspace_state", load_handler)
    result = harness.execute(
        RuntimeGoal(goal_id="goal-mutated-reference", description="Check mutated targets"),
        agent_name="improver",
    )

    actionable_files = result.context.workspace["actionable_files"]
    reference_files = result.context.workspace["reference_files"]
    assert any("src/status_payload.py" in item for item in actionable_files)
    assert not any("runtime.yaml" in item for item in actionable_files)
    assert not any("parallel-plan.md" in item for item in actionable_files)
    assert any("runtime.yaml" in item for item in reference_files)
    assert any("parallel-plan.md" in item for item in reference_files)


def test_dk_runner_appends_live_context_checkpoint_after_tool_results() -> None:
    class FakeCheckpointClient:
        provider_id = "test"

        def __init__(self) -> None:
            self.calls: list[list[dict[str, object]]] = []
            self.used = False

        async def complete(
            self,
            messages: list[dict[str, object]],
            tools: list[dict[str, object]] | None = None,
            options: object = None,
        ) -> LLMResponse:
            del options
            self.calls.append(messages)
            if self.used:
                return LLMResponse(
                    text="Done.",
                    usage=TokenUsage(input=0, output=0, reasoning=0),
                    finish_reason="stop",
                    cost=Decimal("0"),
                    tool_calls=None,
                )
            self.used = True
            tool_defs = tools or []
            load_tool = next(
                tool_def
                for tool_def in tool_defs
                if isinstance(tool_def.get("function"), dict)
                and cast(dict[str, object], tool_def["function"]).get("name")
                == "load_workspace_state"
            )
            load_function = cast(dict[str, object], load_tool["function"])
            return LLMResponse(
                text="Bootstrapping workspace context.",
                usage=TokenUsage(input=0, output=0, reasoning=0),
                finish_reason="tool_use",
                cost=Decimal("0"),
                tool_calls=[
                    {
                        "id": "call-load-workspace",
                        "type": "function",
                        "function": {
                            "name": load_function["name"],
                            "arguments": "{}",
                        },
                        "tool": load_function["name"],
                        "input": {},
                    }
                ],
            )

    harness = create_default_harness(workdir=Path.cwd(), console_output=False)
    client = FakeCheckpointClient()
    harness.runner.set_client_factory(lambda: client)

    def load_handler(_call: object) -> ToolResult:
        return ToolResult(
            tool_name="load_workspace_state",
            success=True,
            payload=ToolExecutionPayload(
                workspace_updates=WorkspaceToolContext(
                    scenario_path="evals/sample.scenario.yaml",
                    scenario_required_files=["src/service.py"],
                    changed_files=["src/service.py"],
                ),
                message="Loaded workspace state for src/service.py",
            ),
        )

    harness.tools.register_handler("load_workspace_state", load_handler)

    result = harness.execute(
        RuntimeGoal(goal_id="goal-dk-checkpoint", description="Observe live checkpoint updates"),
        agent_name="improver",
    )

    assert result.selected_tool_names[0] == "load_workspace_state"
    assert len(client.calls) >= 2
    second_call_messages = client.calls[1]
    tool_messages = [
        message
        for message in second_call_messages
        if message.get("role") == "tool" and isinstance(message.get("content"), str)
    ]
    assert tool_messages
    latest_tool_message = cast(str, tool_messages[-1]["content"])
    assert "Live context update after load_workspace_state:" in latest_tool_message
    assert "Recent steps:" in latest_tool_message
    assert "Available tools now:" not in latest_tool_message


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
    assert "grep" not in active_tool_names
    assert "read" not in active_tool_names
    assert "mutate_agent_files" not in active_tool_names
    assert "delegate_task" in active_tool_names


def test_tool_call_iterations_track_loops_separately_from_tool_calls(tmp_path: Path) -> None:
    harness = create_default_harness(
        workdir=tmp_path,
        storage_root=tmp_path / ".ash-hawk-test",
        console_output=False,
    )
    harness.runner.set_client_factory(
        lambda: _FakeDkClient(lambda _tool_defs: "load_workspace_state")
    )
    captured_call: dict[str, object] = {}

    def recording_handler(call: object) -> ToolResult:
        captured_call["call"] = call
        return ToolResult(tool_name="load_workspace_state", success=True)

    harness.tools.register_handler("load_workspace_state", recording_handler)

    result = harness.execute(
        RuntimeGoal(
            goal_id="goal-call-count",
            description="Track loop and tool counts",
            max_iterations=1,
        ),
        agent_name="improver",
    )

    del result
    call = captured_call["call"]
    assert getattr(call, "iterations") == 0
    assert getattr(call, "tool_call_count") == 1


def test_improver_loop_mode_requires_full_reduced_contract() -> None:
    harness = create_default_harness(workdir=Path.cwd(), console_output=False)

    def load_handler(_call: object) -> ToolResult:
        return ToolResult(
            tool_name="load_workspace_state",
            success=True,
            payload=ToolExecutionPayload(
                workspace_updates=WorkspaceToolContext(allowed_target_files=["agent.md"]),
            ),
        )

    def baseline_handler(_call: object) -> ToolResult:
        return ToolResult(
            tool_name="run_baseline_eval",
            success=True,
            payload=ToolExecutionPayload(
                evaluation_updates=EvaluationToolContext(
                    baseline_summary=ScoreSummary(
                        score=0.7,
                        status="completed",
                        tool="run_baseline_eval",
                    )
                )
            ),
        )

    def delegate_handler(_call: object) -> ToolResult:
        return ToolResult(
            tool_name="delegate_task",
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
    harness.tools.register_handler("run_baseline_eval", baseline_handler)
    harness.tools.register_handler("delegate_task", delegate_handler)
    harness.tools.register_handler("run_eval_repeated", repeat_handler)

    result = harness.execute(
        RuntimeGoal(
            goal_id="goal-loop-reuse",
            description="Complete two improvement loops",
            max_iterations=2,
        ),
        agent_name="improver",
        tool_execution_order=[
            "load_workspace_state",
            "run_baseline_eval",
            "delegate_task",
            "run_eval_repeated",
        ],
    )

    assert result.success is False
    assert (
        result.error
        == "No eligible tools available for a complete diagnosis-mutation-re-evaluation loop"
    )
    assert result.context.runtime["completed_iterations"] == 0
    assert result.context.runtime["remaining_iterations"] == 2


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

    class FakeWorkspaceClient:
        provider_id = "test"

        def __init__(self) -> None:
            self.calls = 0

        async def complete(
            self,
            messages: list[dict[str, object]],
            tools: list[dict[str, object]] | None = None,
            options: object = None,
        ) -> LLMResponse:
            del messages
            del tools
            del options
            self.calls += 1
            if self.calls == 1:
                tool_call = {
                    "id": "call-load",
                    "type": "function",
                    "function": {"name": "load_workspace_state", "arguments": "{}"},
                    "tool": "load_workspace_state",
                    "input": {},
                }
            elif self.calls == 2:
                tool_call = {
                    "id": "call-detect",
                    "type": "function",
                    "function": {"name": "detect_agent_config", "arguments": "{}"},
                    "tool": "detect_agent_config",
                    "input": {},
                }
            elif self.calls == 3:
                tool_call = {
                    "id": "call-scope",
                    "type": "function",
                    "function": {
                        "name": "scope_workspace",
                        "arguments": '{"target_files": ["reports/repl-notes.md", "src/sequence.py"]}',
                    },
                    "tool": "scope_workspace",
                    "input": {"target_files": ["reports/repl-notes.md", "src/sequence.py"]},
                }
            else:
                return LLMResponse(
                    text="Workspace inspection complete.",
                    usage=TokenUsage(input=0, output=0, reasoning=0),
                    finish_reason="stop",
                    cost=Decimal("0"),
                    tool_calls=None,
                )
            return LLMResponse(
                text=f"Calling {tool_call['tool']}",
                usage=TokenUsage(input=0, output=0, reasoning=0),
                finish_reason="tool_use",
                cost=Decimal("0"),
                tool_calls=[tool_call],
            )

    harness = create_default_harness(workdir=tmp_path)
    harness.runner.set_client_factory(lambda: FakeWorkspaceClient())
    result = harness.execute(
        RuntimeGoal(goal_id="goal-rich-workspace", description="Inspect workspace outputs"),
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
        == "2"
    )
    assert by_tool["scope_workspace"].payload.workspace_updates.allowed_target_files[:2] == [
        str((tmp_path / "reports" / "repl-notes.md").resolve()),
        str((tmp_path / "src" / "sequence.py").resolve()),
    ]
    assert by_tool["scope_workspace"].payload.workspace_updates.allowed_target_files[2:] == []

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
        workdir = Path(str(kwargs["working_dir"]))
        (workdir / "target.py").write_text("value = 2\n", encoding="utf-8")
        return SimpleNamespace(session_id="mutation-session", response="Changed value to 2")

    fake_module.execute = fake_execute  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bolt_merlin.agent.execute", fake_module)

    prepared = prepare_isolated_workspace_run(
        ToolCall(
            tool_name="prepare_isolated_workspace",
            goal_id="goal-mutate-rich",
            context=ToolCallContext(
                runtime=RuntimeToolContext(active_agent="executor"),
                workspace=WorkspaceToolContext(
                    workdir=str(tmp_path),
                    allowed_target_files=["target.py"],
                    agent_config=".dawn-kestrel/agent_config.yaml",
                ),
            ),
        )
    )
    assert prepared.success is True

    tool_result = mutate_agent_files_run(
        ToolCall(
            tool_name="mutate_agent_files",
            goal_id="goal-mutate-rich",
            tool_args={"target_file": "target.py"},
            context=ToolCallContext(
                runtime=RuntimeToolContext(active_agent="executor"),
                workspace=WorkspaceToolContext(
                    workdir=prepared.payload.workspace_updates.workdir,
                    isolated_workspace=True,
                    isolated_workspace_path=prepared.payload.workspace_updates.isolated_workspace_path,
                    allowed_target_files=["target.py"],
                    agent_config=".dawn-kestrel/agent_config.yaml",
                    source_root=str(tmp_path),
                    package_name="bolt_merlin",
                ),
                evaluation=EvaluationToolContext(
                    baseline_summary=ScoreSummary(
                        score=0.41, status="completed", tool="run_baseline_eval"
                    )
                ),
            ),
        )
    )

    assert tool_result.tool_name == "mutate_agent_files"
    assert tool_result.success is True
    assert "changed" in tool_result.payload.message
    assert tool_result.payload.audit_updates.run_summary["target_file"] == "target.py"
    assert tool_result.payload.audit_updates.run_summary["changed"] == "True"
    assert tool_result.payload.audit_updates.diff_report["files"] == 1
    assert tool_result.payload.audit_updates.diff_report["diff_line_count"] > 0
    assert "value = 2" in tool_result.payload.audit_updates.run_summary["diff_preview"]
    assert (
        tool_result.payload.runtime_updates.stop_reason
        == "candidate mutation completed in isolated workspace"
    )
    assert tool_result.payload.stop is True


def test_tool_handlers_update_context_and_unlock_outputs() -> None:
    harness = create_default_harness(workdir=Path.cwd())
    goal = RuntimeGoal(goal_id="goal-8", description="Establish baseline and verify it")
    remaining_sequence = ["run_baseline_eval", "verify_outcome"]

    def choose_tool(tool_defs: list[dict[str, object]]) -> str | None:
        tool_names = [
            function.get("name")
            for tool_def in tool_defs
            if isinstance((function := tool_def.get("function")), dict)
            and isinstance(function.get("name"), str)
        ]
        for tool_name in list(remaining_sequence):
            if tool_name in tool_names:
                remaining_sequence.remove(tool_name)
                return tool_name
        return None

    harness.runner.set_client_factory(lambda: _FakeDkClient(choose_tool))

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
    )

    assert result.success is True
    assert result.context.evaluation["baseline_summary"]["status"] == "completed"
    assert result.context.evaluation["verification"]["verified"] is True
    assert "evaluation_context" in result.available_contexts
    assert result.selected_tool_names[0] in {"run_eval", "run_baseline_eval"}


def test_read_uses_explicit_file_path_when_model_calls_it(
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

    class FakeReadClient:
        provider_id = "test"

        def __init__(self) -> None:
            self.used = False

        async def complete(
            self,
            messages: list[dict[str, object]],
            tools: list[dict[str, object]] | None = None,
            options: object = None,
        ) -> LLMResponse:
            del messages
            del tools
            del options
            if self.used:
                return LLMResponse(
                    text="Read complete.",
                    usage=TokenUsage(input=0, output=0, reasoning=0),
                    finish_reason="stop",
                    cost=Decimal("0"),
                    tool_calls=None,
                )
            self.used = True
            read_path = str(pack)
            return LLMResponse(
                text="Reading the pack file directly.",
                usage=TokenUsage(input=0, output=0, reasoning=0),
                finish_reason="tool_use",
                cost=Decimal("0"),
                tool_calls=[
                    {
                        "id": "call-read-pack",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": f'{{"file_path": "{read_path}"}}',
                        },
                        "tool": "read",
                        "input": {"file_path": read_path},
                    }
                ],
            )

    harness = create_default_harness(workdir=tmp_path)
    harness.runner.set_client_factory(lambda: FakeReadClient())
    result = harness.execute(
        RuntimeGoal(goal_id="goal-read-pack", description="Read scenario pack"),
        agent_name="researcher",
        scenario_path=str(pack),
    )

    tool_result = result.tool_results[0]
    assert tool_result.tool_name == "read"
    assert tool_result.success is True
    assert "description: Read the pack first" in tool_result.payload.message


def test_harness_persists_memory_and_execution_artifacts(tmp_path: Path) -> None:
    harness = create_default_harness(workdir=Path.cwd(), storage_root=tmp_path)
    harness.runner.set_client_factory(
        lambda: _FakeDkClient(lambda _tool_defs: "load_workspace_state")
    )
    goal = RuntimeGoal(goal_id="goal-9", description="Persist this run")

    result = harness.execute(goal)

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

    summary = (Path(result.artifact_dir) / "summary.json").read_text(encoding="utf-8")
    assert "live_brief" in summary
    assert "artifact_index" in summary
    assert "progress_summary" in summary
    assert "available_tool_summaries" in summary


def test_harness_persists_tool_run_ids_into_episodic_memory(tmp_path: Path) -> None:
    harness = create_default_harness(workdir=Path.cwd(), storage_root=tmp_path)
    harness.runner.set_client_factory(
        lambda: _FakeDkClient(lambda _tool_defs: "load_workspace_state")
    )

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
    )

    assert any(
        "external-run-123" in item for item in result.memory_snapshot["episodic_memory"]["episodes"]
    )


def test_dream_state_consolidates_deferred_memory_on_final_hook() -> None:
    harness = create_default_harness(workdir=Path.cwd())
    goal = RuntimeGoal(goal_id="goal-10", description="Dream state consolidation")
    harness.agents.get("memory_manager").available_tools = ["search_knowledge"]
    harness.skills.get("lesson-consolidation").tool_names = ["search_knowledge"]
    harness.skills.get("lesson-consolidation").input_contexts = ["memory_context"]

    result = harness.execute(
        goal,
        agent_name="memory_manager",
        requested_skills=["lesson-consolidation"],
        tool_execution_order=["search_knowledge"],
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
    harness.runner.set_client_factory(lambda: _FakeDkClient(lambda _tool_defs: "run_baseline_eval"))

    result = harness.execute(
        RuntimeGoal(goal_id="goal-no-scenario", description="Need a baseline"),
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
    harness.runner.set_client_factory(lambda: _FakeDkClient(lambda _tool_defs: "run_baseline_eval"))
    goal = RuntimeGoal(goal_id="goal-3", description="Check context enforcement")
    harness.tools.get("run_baseline_eval").required_contexts.append("missing_context")

    result = harness.execute(
        goal,
        agent_name="verifier",
        requested_skills=["baseline-evaluation"],
    )

    assert result.success is False
    assert result.tool_results[0].error == "Missing required contexts: missing_context"


def test_memory_write_permissions_are_enforced() -> None:
    harness = create_default_harness(workdir=Path.cwd())
    harness.runner.set_client_factory(lambda: _FakeDkClient(lambda _tool_defs: "search_knowledge"))
    goal = RuntimeGoal(goal_id="goal-4", description="Check policy-only skill path")
    harness.skills.get("lesson-consolidation").tool_names = ["search_knowledge"]
    harness.skills.get("lesson-consolidation").input_contexts = ["memory_context"]

    result = harness.execute(
        goal,
        agent_name="coordinator",
        requested_skills=["lesson-consolidation"],
    )

    assert result.success is False
    assert result.selected_tool_names == ["search_knowledge"]
    assert result.error == "Actor 'coordinator' cannot write memory scope 'semantic_memory'"
