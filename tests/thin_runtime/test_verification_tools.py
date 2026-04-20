from __future__ import annotations

from ash_hawk.thin_runtime.models import ToolCall
from ash_hawk.thin_runtime.tool_impl.detect_regressions import run as detect_regressions_run


def _tool_call(
    *,
    evaluation: dict[str, object] | None = None,
    failure: dict[str, object] | None = None,
    audit: dict[str, object] | None = None,
) -> ToolCall:
    return ToolCall.model_validate(
        {
            "tool_name": "tool-under-test",
            "goal_id": "goal-verification",
            "context": {
                "evaluation": evaluation or {},
                "failure": failure or {},
                "audit": audit or {},
            },
        }
    )


def test_detect_regressions_reports_repeat_score_drop() -> None:
    result = detect_regressions_run(
        _tool_call(
            evaluation={
                "baseline_summary": {
                    "score": 0.90,
                    "status": "completed",
                    "tool": "run_baseline_eval",
                },
                "repeat_eval_summary": {
                    "score": 0.84,
                    "status": "completed",
                    "tool": "run_eval_repeated",
                },
            }
        )
    )

    assert result.success is True
    assert len(result.payload.evaluation_updates.regressions) == 1
    assert "regressed from 0.900 to 0.840" in result.payload.evaluation_updates.regressions[0]


def test_detect_regressions_preserves_existing_regressions() -> None:
    result = detect_regressions_run(
        _tool_call(
            evaluation={"regressions": ["existing regression"]},
        )
    )

    assert result.success is True
    assert result.payload.evaluation_updates.regressions == ["existing regression"]
