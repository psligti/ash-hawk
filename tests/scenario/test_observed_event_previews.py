from __future__ import annotations

from pathlib import Path

from ash_hawk.scenario.agent_runner import ToolingHarnessRecorder
from ash_hawk.scenario.tooling import ToolingHarness


def test_tooling_harness_recorder_emits_action_preview(tmp_path: Path) -> None:
    harness = ToolingHarness(mode="mock", root=tmp_path)
    harness.register_mock("glob", {"pattern": "src/**/*.py"}, {"status": "ok"})
    observed_events: list[dict[str, object]] = []

    recorder = ToolingHarnessRecorder(harness, event_callback=observed_events.append)
    recorder.call("glob", {"pattern": "src/**/*.py"})

    assert observed_events == [
        {
            "tool": "glob",
            "event_type": "tool_result",
            "success": True,
            "preview": "src/**/*.py",
            "error": None,
        }
    ]
