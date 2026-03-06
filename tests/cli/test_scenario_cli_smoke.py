import re
import tempfile
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from ash_hawk.cli.main import cli
from ash_hawk.config import reload_config


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_storage(temp_dir, monkeypatch):
    storage_path = Path(temp_dir) / "storage"
    storage_path.mkdir()
    monkeypatch.setenv("ASH_HAWK_STORAGE_BACKEND", "file")
    monkeypatch.setenv("ASH_HAWK_STORAGE_PATH", str(storage_path))
    reload_config()
    return storage_path


@pytest.fixture
def scenario_file(temp_dir):
    scenario_path = Path(temp_dir) / "scenarios" / "simple.scenario.yaml"
    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    scenario = {
        "schema_version": "v1",
        "id": "scenario-001",
        "description": "Scenario CLI smoke test",
        "sut": {"type": "coding_agent", "adapter": "mock_adapter", "config": {}},
        "inputs": {"prompt": "Hello"},
        "graders": [
            {
                "grader_type": "string_match",
                "config": {"expected": "OK"},
            }
        ],
    }
    scenario_path.write_text(yaml.safe_dump(scenario, sort_keys=False), encoding="utf-8")
    return scenario_path


def test_scenario_validate_smoke(runner, scenario_file, temp_storage):
    result = runner.invoke(cli, ["scenario", "validate", str(scenario_file)])
    assert result.exit_code == 0
    assert "PASS" in result.output


def test_scenario_run_and_report_smoke(runner, scenario_file, temp_storage):
    run_result = runner.invoke(
        cli,
        ["scenario", "run", str(scenario_file), "--sut", "mock_adapter"],
    )
    assert run_result.exit_code == 0
    match = re.search(r"Run ID: (run-[a-f0-9]+)", run_result.output)
    assert match

    run_id = match.group(1)
    report_result = runner.invoke(
        cli,
        ["scenario", "report", "--run", run_id],
    )
    assert report_result.exit_code == 0
    assert "Scenario Run Report" in report_result.output
