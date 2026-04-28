from __future__ import annotations

from pathlib import Path

import pytest

from ash_hawk.thin_runtime.models import ToolCall
from ash_hawk.thin_runtime.tool_impl import (
    run_baseline_eval,
    run_eval_repeated,
    scope_workspace,
    sync_workspace_changes,
)
from ash_hawk.thin_runtime.tool_impl.prepare_isolated_workspace import (
    run as prepare_isolated_workspace_run,
)
from ash_hawk.thin_runtime.tool_types import (
    AuditRunResult,
    AuditToolContext,
    EvaluationToolContext,
    RuntimeToolContext,
    ScoreSummary,
    ToolCallContext,
    ToolExecutionPayload,
    WorkspaceToolContext,
)


def test_prepare_isolated_workspace_copies_target_and_scenario(tmp_path: Path) -> None:
    target = tmp_path / "agent.md"
    target.write_text("# Agent\n", encoding="utf-8")
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text("description: test\n", encoding="utf-8")

    result = prepare_isolated_workspace_run(
        ToolCall(
            tool_name="prepare_isolated_workspace",
            goal_id="goal-isolation",
            context=ToolCallContext(
                runtime=RuntimeToolContext(active_agent="executor"),
                workspace=WorkspaceToolContext(
                    workdir=str(tmp_path),
                    allowed_target_files=["agent.md"],
                    scenario_path=str(scenario),
                ),
            ),
        )
    )

    assert result.success is True
    isolated_root = Path(result.payload.workspace_updates.isolated_workspace_path or "")
    assert isolated_root.exists()
    assert result.payload.workspace_updates.workdir == str(isolated_root)
    assert (isolated_root / "agent.md").read_text(encoding="utf-8") == "# Agent\n"
    isolated_scenario = Path(result.payload.workspace_updates.scenario_path or "")
    assert isolated_scenario.exists()
    assert isolated_scenario.read_text(encoding="utf-8") == "description: test\n"


def test_scope_workspace_normalizes_absolute_targets_to_relative_paths(tmp_path: Path) -> None:
    target = tmp_path / "agent.md"
    target.write_text("# Agent\n", encoding="utf-8")

    result = scope_workspace.run(
        ToolCall(
            tool_name="scope_workspace",
            goal_id="goal-scope-normalize",
            tool_args={"target_files": [str(target)]},
            context=ToolCallContext(
                workspace=WorkspaceToolContext(workdir=str(tmp_path), repo_root=str(tmp_path)),
            ),
        )
    )

    assert result.success is True
    assert result.payload.workspace_updates.allowed_target_files == ["agent.md"]


def test_sync_workspace_changes_requires_clean_validation_and_syncs_back(tmp_path: Path) -> None:
    primary = tmp_path / "agent.md"
    primary.write_text("old\n", encoding="utf-8")
    isolated_root = tmp_path / "isolated"
    isolated_root.mkdir()
    (isolated_root / "agent.md").write_text("new\n", encoding="utf-8")

    result = sync_workspace_changes.run(
        ToolCall(
            tool_name="sync_workspace_changes",
            goal_id="goal-sync",
            context=ToolCallContext(
                workspace=WorkspaceToolContext(
                    workdir=str(tmp_path),
                    primary_workdir=str(tmp_path),
                    isolated_workspace=True,
                    isolated_workspace_path=str(isolated_root),
                    mutated_files=["agent.md"],
                    scenario_path="candidate.yaml",
                    source_scenario_path="source.yaml",
                ),
                evaluation=EvaluationToolContext(
                    baseline_summary=ScoreSummary(
                        score=0.4,
                        status="completed",
                        tool="run_baseline_eval",
                    ),
                    repeat_eval_summary=ScoreSummary(
                        score=0.9,
                        status="completed",
                        tool="run_eval_repeated",
                    ),
                ),
                audit=AuditToolContext(run_result=AuditRunResult(aggregate_passed=True)),
            ),
        )
    )

    assert result.success is True
    assert primary.read_text(encoding="utf-8") == "new\n"
    assert result.payload.stop is True
    assert result.payload.workspace_updates.changed_files == ["agent.md"]
    assert result.payload.workspace_updates.isolated_workspace is False
    assert not isolated_root.exists()


def test_run_eval_repeated_rejects_candidate_and_cleans_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_scenario = tmp_path / "scenario.yaml"
    source_scenario.write_text("description: test\n", encoding="utf-8")
    isolated_root = tmp_path / "isolated"
    isolated_root.mkdir()
    isolated_scenario = isolated_root / "scenario.yaml"
    isolated_scenario.write_text("description: test\n", encoding="utf-8")
    (isolated_root / "agent.md").write_text("candidate\n", encoding="utf-8")

    manifest_path, manifest_hash = run_baseline_eval.write_eval_manifest(
        workdir=tmp_path,
        run_id="goal-repeat",
        scenario_path=str(source_scenario),
        scenario_required_files=[],
        repetitions=2,
    )
    assert manifest_path is not None
    assert manifest_hash is not None

    def fake_run_live_scenario_eval(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
        return (
            True,
            ToolExecutionPayload(
                evaluation_updates=EvaluationToolContext(
                    repeat_eval_summary=ScoreSummary(
                        score=0.2,
                        status="completed",
                        tool="run_eval_repeated",
                    )
                ),
                audit_updates=AuditToolContext(run_result=AuditRunResult(aggregate_passed=False)),
            ),
            "ok",
            [],
        )

    monkeypatch.setattr(run_eval_repeated, "run_live_scenario_eval", fake_run_live_scenario_eval)

    result = run_eval_repeated.run(
        ToolCall(
            tool_name="run_eval_repeated",
            goal_id="goal-repeat",
            context=ToolCallContext(
                workspace=WorkspaceToolContext(
                    workdir=str(tmp_path),
                    isolated_workspace=True,
                    isolated_workspace_path=str(isolated_root),
                    mutated_files=["agent.md"],
                    scenario_path=str(isolated_scenario),
                    source_scenario_path=str(source_scenario),
                ),
                evaluation=EvaluationToolContext(
                    baseline_summary=ScoreSummary(
                        score=0.5,
                        status="completed",
                        tool="run_baseline_eval",
                    ),
                    eval_manifest_path=str(manifest_path),
                    eval_manifest_hash=manifest_hash,
                ),
            ),
        )
    )

    assert result.success is True
    assert result.payload.runtime_updates.preferred_tool == "call_llm_structured"
    assert result.payload.workspace_updates.mutated_files == []
    assert result.payload.workspace_updates.isolated_workspace is False
    assert result.payload.workspace_updates.scenario_path == str(source_scenario)
    assert not isolated_root.exists()
