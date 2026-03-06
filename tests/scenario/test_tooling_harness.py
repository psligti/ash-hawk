from __future__ import annotations

from pathlib import Path

import pytest

from ash_hawk.scenario.tooling import ToolTimeoutError, ToolingHarness


def test_tooling_harness_mock_hit_miss() -> None:
    harness = ToolingHarness(mode="mock", root=Path("scenario-1"))
    harness.register_mock("read", {"path": "/tmp/file.txt"}, {"status": "ok"})

    assert harness.call("read", {"path": "/tmp/file.txt"}) == {"status": "ok"}

    with pytest.raises(KeyError):
        harness.call("read", {"path": "/tmp/other.txt"})


def test_tooling_harness_timeout_injection() -> None:
    harness = ToolingHarness(mode="mock", root=Path("scenario-1"))
    harness.register_mock("write", {"path": "/tmp/out.txt"}, {"status": "ok"})
    harness.inject_timeout("write")

    with pytest.raises(ToolTimeoutError):
        harness.call("write", {"path": "/tmp/out.txt"})

    assert harness.call("write", {"path": "/tmp/out.txt"}) == {"status": "ok"}


def test_tooling_harness_record_replay_roundtrip(tmp_path) -> None:
    root = tmp_path / "scenario-1"
    root.mkdir()

    record = ToolingHarness(mode="record", root=root)
    record.register_mock("read", {"path": "a.txt"}, {"status": "ok"})
    record.register_mock("write", {"path": "b.txt"}, {"status": "written"})

    assert record.call("read", {"path": "a.txt"}) == {"status": "ok"}
    assert record.call("write", {"path": "b.txt"}) == {"status": "written"}

    trace_path = root / "tool_mocks" / "scenario-1" / "trace.jsonl"
    assert trace_path.exists()

    replay = ToolingHarness(mode="replay", root=root)
    assert replay.call("read", {"path": "a.txt"}) == {"status": "ok"}
    assert replay.call("write", {"path": "b.txt"}) == {"status": "written"}


def test_tooling_harness_replay_order_mismatch(tmp_path) -> None:
    root = tmp_path / "scenario-1"
    root.mkdir()

    record = ToolingHarness(mode="record", root=root)
    record.register_mock("read", {"path": "a.txt"}, {"status": "ok"})
    record.register_mock("write", {"path": "b.txt"}, {"status": "written"})

    record.call("read", {"path": "a.txt"})
    record.call("write", {"path": "b.txt"})

    replay = ToolingHarness(mode="replay", root=root)
    with pytest.raises(ValueError):
        replay.call("write", {"path": "b.txt"})
