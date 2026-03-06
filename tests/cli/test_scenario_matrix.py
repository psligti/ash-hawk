import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from ash_hawk.cli.main import cli
from ash_hawk.config import reload_config


@pytest.fixture
def runner():
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
def scenario_file(tmp_path: Path):
    scenario_path = tmp_path / "matrix" / "matrix.scenario.yaml"
    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    scenario: dict[str, Any] = {
        "schema_version": "v1",
        "id": "matrix-001",
        "description": "Scenario matrix test",
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


def test_scenario_matrix_runs_all_combinations(
    runner: CliRunner, scenario_file: Path, temp_storage: Path
):
    result = runner.invoke(
        cli,
        [
            "scenario",
            "matrix",
            str(scenario_file),
            "--sut",
            "mock_adapter",
            "--policies",
            "p1,p2",
            "--models",
            "m1,m2",
        ],
    )

    assert result.exit_code == 0
    assert "Policy" in result.output
    assert "Model" in result.output
    assert "p1" in result.output
    assert "p2" in result.output
    assert "m1" in result.output
    assert "m2" in result.output
