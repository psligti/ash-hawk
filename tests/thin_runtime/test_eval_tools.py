from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

from ash_hawk.thin_runtime.models import ToolCall
from ash_hawk.thin_runtime.tool_impl import (
    run_eval,
    run_eval_repeated,
    run_integrity_validation,
    run_targeted_validation,
)
from ash_hawk.thin_runtime.tool_types import (
    EvaluationToolContext,
    ToolCallContext,
    ToolExecutionPayload,
    WorkspaceToolContext,
)


@pytest.mark.parametrize(
    ("module", "tool_name", "summary_field", "repetitions"),
    [
        (run_eval, "run_eval", "last_eval_summary", 1),
        (run_eval_repeated, "run_eval_repeated", "repeat_eval_summary", 2),
        (run_targeted_validation, "run_targeted_validation", "targeted_validation_summary", 1),
        (run_integrity_validation, "run_integrity_validation", "integrity_summary", 1),
    ],
)
def test_eval_tools_pass_correct_summary_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    tool_name: str,
    summary_field: str,
    repetitions: int,
) -> None:
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text("description: test\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run_live_scenario_eval(
        observed_tool_name: str,
        scenario_path: Path,
        *,
        summary_field: str = "baseline_summary",
        repetitions: int = 1,
    ) -> tuple[bool, ToolExecutionPayload, str, list[str]]:
        captured.update(
            {
                "tool_name": observed_tool_name,
                "scenario_path": scenario_path,
                "summary_field": summary_field,
                "repetitions": repetitions,
            }
        )
        return True, ToolExecutionPayload(), "ok", []

    monkeypatch.setattr(module, "run_live_scenario_eval", fake_run_live_scenario_eval)
    if module is run_eval_repeated:

        def fake_verify_eval_manifest(
            *, manifest_path: str | None, manifest_hash: str | None, scenario_path: str | None
        ) -> tuple[bool, str | None]:
            del manifest_path, manifest_hash, scenario_path
            return True, None

        monkeypatch.setattr(module, "verify_eval_manifest", fake_verify_eval_manifest)

    success, _payload, _message, _errors = module._execute(
        ToolCall(
            tool_name=tool_name,
            goal_id="goal-eval-tool",
            context=ToolCallContext(
                workspace=WorkspaceToolContext(scenario_path=str(scenario)),
                evaluation=EvaluationToolContext(
                    eval_manifest_path=str(scenario),
                    eval_manifest_hash="placeholder",
                ),
            ),
        )
    )

    assert success is True
    assert captured == {
        "tool_name": tool_name,
        "scenario_path": scenario,
        "summary_field": summary_field,
        "repetitions": repetitions,
    }
