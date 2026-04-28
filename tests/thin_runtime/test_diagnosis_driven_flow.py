from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from ash_hawk.thin_runtime import RuntimeGoal, create_default_harness
from ash_hawk.thin_runtime.models import ToolCall, ToolResult
from ash_hawk.thin_runtime.tool_impl.call_llm_structured import (
    StructuredHypothesis,
    StructuredLLMResponse,
)
from ash_hawk.thin_runtime.tool_impl.call_llm_structured import (
    run as call_llm_structured_run,
)
from ash_hawk.thin_runtime.tool_impl.mutate_agent_files import run as mutate_agent_files_run
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


def test_call_llm_structured_derives_ranked_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    response = StructuredLLMResponse(
        diagnosis="The improver is stopping after baseline instead of deriving a mutation target.",
        blocker="No durable target file has been derived from the failure evidence.",
        ideal_outcome="The improver should mutate the durable coding-agent prompt and re-evaluate.",
        hypotheses=[
            StructuredHypothesis(
                name="Update coding prompt",
                score=0.92,
                rationale="The eval is grading direct-context behavior owned by the coding prompt.",
                target_files=["bolt_merlin/agent/prompts/coding.md"],
                ideal_outcome="The coding agent reads named files directly before broad search.",
            )
        ],
    )

    def fake_call_model_structured(*_args: object, **_kwargs: object) -> StructuredLLMResponse:
        return response

    monkeypatch.setattr(
        "ash_hawk.thin_runtime.tool_impl.call_llm_structured.call_model_structured",
        fake_call_model_structured,
    )

    call = ToolCall.model_validate(
        {
            "tool_name": "call_llm_structured",
            "goal_id": "goal-diagnosis",
            "context": {
                "workspace": {"allowed_target_files": []},
                "failure": {"failure_family": "needs_improvement"},
            },
        }
    )

    result = call_llm_structured_run(call)

    assert result.success is True
    assert result.payload.runtime_updates.preferred_tool == "mutate_agent_files"
    assert result.payload.workspace_updates.allowed_target_files == [
        "bolt_merlin/agent/prompts/coding.md"
    ]
    assert result.payload.failure_updates.ranked_hypotheses == [
        RankedHypothesis(
            name="Update coding prompt",
            score=0.92,
            rationale="The eval is grading direct-context behavior owned by the coding prompt.",
            target_files=["bolt_merlin/agent/prompts/coding.md"],
            ideal_outcome="The coding agent reads named files directly before broad search.",
        )
    ]


def test_call_llm_structured_includes_prior_mutation_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def fake_call_model_structured(*_args: object, **kwargs: object) -> StructuredLLMResponse:
        user_prompt = kwargs.get("user_prompt")
        assert isinstance(user_prompt, str)
        captured["user_prompt"] = user_prompt
        return StructuredLLMResponse(
            diagnosis="Build on the last survivor gene.",
            hypotheses=[
                StructuredHypothesis(
                    name="Extend direct-context prompt",
                    score=0.88,
                    target_files=["agent.md"],
                )
            ],
        )

    monkeypatch.setattr(
        "ash_hawk.thin_runtime.tool_impl.call_llm_structured.call_model_structured",
        fake_call_model_structured,
    )

    call = ToolCall.model_validate(
        {
            "tool_name": "call_llm_structured",
            "goal_id": "goal-prior-memory",
            "context": {
                "workspace": {"workdir": "/repo", "repo_root": "/repo"},
                "failure": {
                    "failure_family": "needs_improvement",
                    "explanations": ["The agent keeps searching broadly."],
                },
                "memory": {
                    "working_memory": {
                        "active_hypotheses": [
                            {
                                "name": "Teach direct-context priority",
                                "score": 0.91,
                                "target_files": ["agent.md"],
                            }
                        ],
                        "last_result": {
                            "status": "improved",
                            "score_delta": 0.14,
                            "target_files": ["agent.md"],
                        },
                    },
                    "session_memory": {
                        "validations": [
                            {
                                "hypothesis": "Teach direct-context priority",
                                "status": "improved",
                                "score_delta": 0.14,
                            }
                        ]
                    },
                    "semantic_memory": {
                        "rules": [
                            {
                                "entry_type": "survivor_gene",
                                "hypothesis": "Teach direct-context priority",
                                "status": "improved",
                                "score_delta": 0.14,
                            }
                        ]
                    },
                },
            },
        }
    )

    result = call_llm_structured_run(call)

    assert result.success is True
    prompt = captured["user_prompt"]
    assert "Prior mutation memory" in prompt
    assert "Teach direct-context priority" in prompt
    assert "Last validation outcome: improved delta=+0.1400 targets=agent.md" in prompt


def test_mutate_agent_files_builds_rich_diagnosis_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "agent.md"
    target.write_text("# Agent\n", encoding="utf-8")
    (tmp_path / ".dawn-kestrel").mkdir()
    (tmp_path / ".dawn-kestrel" / "agent_config.yaml").write_text("agent: test\n", encoding="utf-8")

    captured: dict[str, str] = {}
    fake_module = ModuleType("bolt_merlin.agent.execute")

    async def fake_execute(**kwargs: object) -> SimpleNamespace:
        prompt = str(kwargs["prompt"])
        captured["prompt"] = prompt
        target.write_text("# Agent\nUpdated\n", encoding="utf-8")
        return SimpleNamespace(session_id="mutation-session", response="Updated target")

    fake_module.execute = fake_execute  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bolt_merlin.agent.execute", fake_module)

    call = ToolCall.model_validate(
        {
            "tool_name": "mutate_agent_files",
            "goal_id": "goal-rich-prompt",
            "tool_args": {"target_file": "agent.md"},
            "context": {
                "workspace": {
                    "workdir": str(tmp_path),
                    "isolated_workspace": True,
                    "isolated_workspace_path": str(tmp_path),
                    "scenario_summary": "Teach direct-context-first behavior.",
                    "scenario_required_files": ["reports/out.md"],
                },
                "evaluation": {
                    "baseline_summary": {
                        "score": 0.7619,
                        "status": "completed",
                        "tool": "run_baseline_eval",
                    },
                    "regressions": ["repeat evaluation regressed after broad search"],
                },
                "failure": {
                    "failure_family": "needs_improvement",
                    "explanations": [
                        "The agent ignored the named active implementation path and searched broadly."
                    ],
                    "concepts": ["prefer direct context over broad search"],
                    "ranked_hypotheses": [
                        {
                            "name": "Teach direct-context priority",
                            "score": 0.93,
                            "rationale": "The failure is owned by the durable coding prompt, not a scenario-local output.",
                            "target_files": ["agent.md"],
                            "ideal_outcome": "The coding agent reads named target files before exploring.",
                        }
                    ],
                },
                "audit": {
                    "decision_trace": [
                        "Baseline showed direct-context failure.",
                        "Diagnosis selected the durable prompt surface.",
                    ],
                    "events": [
                        {
                            "event_type": "tool_result",
                            "tool": "run_baseline_eval",
                            "rationale": "baseline showed direct-context failure",
                        }
                    ],
                    "transcripts": [
                        {
                            "speaker": "grader",
                            "message": "The agent broad-searched instead of reading the named active file.",
                        }
                    ],
                    "run_summary": {"previous": "none"},
                    "diff_report": {"files": 0},
                },
            },
        }
    )

    result = mutate_agent_files_run(call)

    assert result.success is True
    assert result.payload.workspace_updates.mutated_files == ["agent.md"]
    prompt = captured["prompt"]
    assert "## Current baseline" in prompt
    assert "Baseline score: 0.7619" in prompt
    assert "Failure family: needs_improvement" in prompt
    assert "## Ranked hypotheses" in prompt
    assert "Teach direct-context priority" in prompt
    assert "## Trace analysis" in prompt
    assert "## Transcript evidence" in prompt
    assert "## Decision trace" in prompt
    assert "agent.md" in prompt


@pytest.mark.e2e
def test_improver_fails_when_no_eligible_tool_can_start_loop() -> None:
    harness = create_default_harness(workdir=Path.cwd())

    result = harness.execute(
        RuntimeGoal(
            goal_id="goal-no-soft-success", description="Do not soft-succeed without a loop"
        ),
        agent_name="improver",
        requested_skills=["verification"],
        tool_execution_order=["verify_outcome"],
    )

    assert result.success is False
    assert (
        result.error
        == "No eligible tools available for a complete diagnosis-mutation-re-evaluation loop"
    )


@pytest.mark.e2e
def test_improver_can_reach_diagnosis_and_derive_targets_without_scoped_inputs() -> None:
    harness = create_default_harness(workdir=Path.cwd())

    def load_handler(_call: object) -> ToolResult:
        return ToolResult(
            tool_name="load_workspace_state",
            success=True,
            payload=ToolExecutionPayload(
                workspace_updates=WorkspaceToolContext(changed_files=[]),
            ),
        )

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
                    explanations=["The agent searched broadly instead of using the named path."],
                ),
            ),
        )

    def diagnosis_handler(_call: object) -> ToolResult:
        return ToolResult(
            tool_name="call_llm_structured",
            success=True,
            payload=ToolExecutionPayload(
                runtime_updates=RuntimeToolContext(preferred_tool="delegate_task"),
                workspace_updates=WorkspaceToolContext(allowed_target_files=["agent.md"]),
                failure_updates=FailureToolContext(
                    explanations=["Mutate the durable prompt instead of searching broadly."],
                    ranked_hypotheses=[
                        RankedHypothesis(
                            name="Teach direct-context priority",
                            score=0.91,
                            rationale="The prompt owns the broad-search behavior.",
                            target_files=["agent.md"],
                            ideal_outcome="The coding agent reads named files directly.",
                        )
                    ],
                ),
            ),
        )

    harness.tools.register_handler("load_workspace_state", load_handler)
    harness.tools.register_handler("run_baseline_eval", baseline_handler)
    harness.tools.register_handler("call_llm_structured", diagnosis_handler)

    result = harness.execute(
        RuntimeGoal(goal_id="goal-full-loop", description="Reach diagnosis and derive targets"),
        agent_name="improver",
        tool_execution_order=[
            "load_workspace_state",
            "run_baseline_eval",
            "call_llm_structured",
        ],
    )

    assert result.success is False
    assert result.selected_tool_names == [
        "load_workspace_state",
        "run_baseline_eval",
        "call_llm_structured",
    ]
