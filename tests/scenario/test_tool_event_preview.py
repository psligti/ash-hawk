from __future__ import annotations

from ash_hawk.scenario.tool_event_preview import tool_action_preview, tool_event_preview


def test_tool_event_preview_prefers_action_summary_for_operator_tools() -> None:
    preview = tool_event_preview(
        "bash",
        {"command": "uv run pytest tests/thin_runtime/test_harness.py"},
        {"stdout": "all good"},
    )

    assert preview == "uv run pytest tests/thin_runtime/test_harness.py"


def test_tool_action_preview_uses_path_for_read_like_tools() -> None:
    preview = tool_action_preview("read", {"path": "src/api.py"})

    assert preview == "src/api.py"


def test_tool_event_preview_falls_back_to_result_when_input_is_uninformative() -> None:
    preview = tool_event_preview("unknown", {}, {"message": "loaded workspace state"})

    assert preview == "loaded workspace state"
