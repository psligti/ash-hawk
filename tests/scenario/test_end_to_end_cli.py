import re
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from ash_hawk.cli.main import cli
from ash_hawk.config import reload_config


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def temp_dir() -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_storage(temp_dir: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    storage_path = Path(temp_dir) / "storage"
    storage_path.mkdir()
    monkeypatch.setenv("ASH_HAWK_STORAGE_BACKEND", "file")
    monkeypatch.setenv("ASH_HAWK_STORAGE_PATH", str(storage_path))
    reload_config()
    return storage_path


@pytest.fixture
def examples_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "examples" / "scenarios"


def test_scenario_validate_examples(
    runner: CliRunner, temp_storage: Path, examples_dir: Path
) -> None:
    result = runner.invoke(cli, ["scenario", "validate", str(examples_dir)])

    assert result.exit_code == 0
    assert "PASS" in result.output
    assert "examples" in result.output  # Path contains 'examples'


def test_scenario_run_success_example(
    runner: CliRunner, temp_storage: Path, examples_dir: Path
) -> None:
    scenario_path = examples_dir / "hello_world.scenario.yaml"
    result = runner.invoke(cli, ["scenario", "run", str(scenario_path)])

    assert result.exit_code == 0
    assert "PASS" in result.output
    assert "trace_schema=" in result.output


def test_scenario_run_failure_report_includes_trace_excerpt(
    runner: CliRunner, temp_storage: Path, examples_dir: Path
) -> None:
    run_result = runner.invoke(cli, ["scenario", "run", str(examples_dir)])

    assert run_result.exit_code != 0
    match = re.search(r"Run ID: (run-[a-f0-9]+)", run_result.output)
    assert match is not None

    run_id = match.group(1)
    report_result = runner.invoke(cli, ["scenario", "report", "--run", run_id])

    assert report_result.exit_code == 0
    assert "Scenario Run Report" in report_result.output
    assert "diff_constraints=" in report_result.output
    assert "Trace excerpt: coding_agent_smoke" in report_result.output
    assert "ModelMessageEvent" in report_result.output
