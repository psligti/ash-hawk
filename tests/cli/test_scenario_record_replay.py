import re
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

import ash_hawk.scenario.registry as registry_module
from ash_hawk.cli.main import cli
from ash_hawk.config import reload_config
from ash_hawk.scenario.adapters import ScenarioAdapter
from ash_hawk.scenario.registry import ScenarioAdapterRegistry


class ToolingScenarioAdapter:
    @property
    def name(self) -> str:
        return "test_adapter"

    def run_scenario(
        self,
        scenario: dict[str, Any],
        workdir: Path,
        tooling_harness: dict[str, Any],
        budgets: dict[str, Any],
    ) -> tuple[Any, list[Any], dict[str, Any]]:
        del scenario, workdir, budgets
        tooling_harness["call"]("read", {"path": "input.txt"})
        return "ok", [], {}


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def temp_storage(monkeypatch: pytest.MonkeyPatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "storage"
        storage_path.mkdir()
        monkeypatch.setenv("ASH_HAWK_STORAGE_BACKEND", "file")
        monkeypatch.setenv("ASH_HAWK_STORAGE_PATH", str(storage_path))
        reload_config()
        yield storage_path


@pytest.fixture
def scenario_file(tmp_path: Path) -> Path:
    scenario_root = tmp_path / "record_replay"
    scenario_root.mkdir(parents=True, exist_ok=True)
    scenario_path = scenario_root / "record.scenario.yaml"
    scenario = {
        "schema_version": "v1",
        "id": "record-001",
        "description": "Record/replay test",
        "sut": {"type": "coding_agent", "adapter": "test_adapter", "config": {}},
        "inputs": {"prompt": "Hello"},
        "tools": {
            "allowed_tools": ["read"],
            "mocks": {
                "read": {
                    "input": {"path": "input.txt"},
                    "result": {"status": "ok"},
                }
            },
            "fault_injection": {},
        },
        "budgets": {
            "max_steps": 3,
            "max_tool_calls": 5,
            "max_tokens": 100,
            "max_time_seconds": 10.0,
        },
        "expectations": {
            "must_events": [],
            "must_not_events": [],
            "ordering_rules": [],
            "diff_assertions": [],
            "output_assertions": [],
        },
        "graders": [
            {
                "grader_type": "string_match",
                "config": {"expected": "ok"},
            }
        ],
    }
    scenario_path.write_text(yaml.safe_dump(scenario, sort_keys=False), encoding="utf-8")
    return scenario_path


def test_scenario_record_and_replay(
    runner: CliRunner,
    scenario_file: Path,
    temp_storage: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ScenarioAdapterRegistry()
    adapter = ToolingScenarioAdapter()
    assert isinstance(adapter, ScenarioAdapter)
    registry.register(adapter)
    monkeypatch.setattr(registry_module, "_default_registry", registry)

    record_result = runner.invoke(
        cli,
        [
            "scenario",
            "record",
            str(scenario_file),
            "--sut",
            "test_adapter",
        ],
    )

    assert record_result.exit_code == 0
    run_match = re.search(r"Run ID: (run-[a-f0-9]+)", record_result.output)
    assert run_match
    run_id = run_match.group(1)

    trace_path = scenario_file.parent / "tool_mocks" / scenario_file.parent.name / "trace.jsonl"
    assert trace_path.exists()

    replay_result = runner.invoke(cli, ["scenario", "replay", "--run", run_id])
    assert replay_result.exit_code == 0
    assert "Replay summary hash" in replay_result.output
